"""Run full CAD Intelligence routes from clean, SHA-pinned checkouts."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

import psutil
import trimesh

from geometry_acceptance import Tolerance, distances, evaluate, file_hash, load_step, step_mesh, write_json

PACKAGES = ("cadquery", "cadquery-ocp", "numpy", "scipy", "trimesh", "rtree",
            "psutil", "fast-simplification", "pymeshfix", "networkx", "open3d")


def fingerprint():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "logical_cpus": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
            "packages": versions}


def verify_checkout(path, sha):
    path = path.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("the manifest must contain a full SHA for each variant")
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    untracked = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "--", "claude/cad-demo"],
        text=True)
    if actual != sha or dirty or untracked:
        raise ValueError(f"expected a clean demo checkout at {sha}, found {actual}")
    if not (path / "claude/cad-demo/demo.py").is_file():
        raise ValueError(f"missing demo.py in checkout {path}")
    return path


def process_tree(root):
    try:
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def rss_bytes(tree):
    total = 0
    for process in tree:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def stop_process_tree(process, root):
    for p in reversed(process_tree(root)):
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait()


def final_step(out):
    files = list(out.glob("*/model.step"))
    if len(files) != 1:
        return None
    path = files[0]
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 256))
            if handle.read().rstrip().endswith(b"END-ISO-10303-21;"):
                return path
    except (FileNotFoundError, PermissionError):
        pass
    return None


def json_file(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_observations(log, directory):
    route = re.findall(r"Выбран путь (\S+)", log)
    last = next((line for line in reversed(log.splitlines()) if "время по этапам:" in line), "")
    workers = {}
    if directory is not None:
        for name in ("geometric", "surface", "smooth", "walls"):
            report = json_file(directory / f"{name}.json")
            if report.get("timings"):
                workers[name] = report["timings"]
    smooth = workers.get("smooth", {})
    steps = smooth.get("nurbs_steps", [])

    def from_log(label):
        hit = re.search(re.escape(label) + r" (\d+(?:\.\d+)?) с", last)
        return float(hit.group(1)) if hit else None

    def total_steps(key):
        values = [step.get("times", {}).get(key) for step in steps]
        available = [x for x in values if x is not None]
        return sum(available) if available else None

    return {
        "selected_route": route[-1] if route else None,
        "regional_stage_observed": any("cylinders" in step.get("times", {}) for step in steps),
        "stages_seconds": {
            "load_and_analysis": from_log("загрузка и замеры"),
            "cleanup_smooth": smooth.get("cleanup"),
            "segmentation_only": total_steps("segmentation"),
            "regions_and_surface_tagging": total_steps("cylinders"),
            "coarsen": total_steps("coarse"), "surface_fit": total_steps("fit"),
            "sewing": total_steps("sewing"), "solid": total_steps("solid"),
            "step_revalidation": from_log("перечитывание STEP"),
            "export_only": smooth.get("export"),
        },
        "worker_timings": workers, "demo_timing_line": last.split("время по этапам: ", 1)[-1] or None,
        "operation_counts": {
            name: {kind: {"attempted": timing.get(f"{kind}_ops"),
                          "rolled_back": timing.get(f"{kind}_rolled_back_ops")}
                   for kind in ("fillet", "hole")} for name, timing in workers.items()
        },
        "nonaccepted_nurbs_steps": (
            sum(step.get("status") != "accepted" for step in steps) if steps else None
        ),
    }


def run_case(checkout, case, scan, reference, folder, variant, config):
    folder.mkdir(parents=True, exist_ok=False)
    out = folder / "demo"
    path_log = folder / "stdout.log"
    demo = checkout / "claude/cad-demo/demo.py"
    flags = ["--mode", "geometric", "--smooth", "auto", "--stepcad", "off", "--cadena", "off",
             "--p2s", "off", "--units", case["units"], "--tol", str(config["pipeline_tol_mm"]),
             "--no-open", "--out", str(out)]
    cmd = [sys.executable, "-u", str(demo), str(scan), *flags]
    # Keep route-specific host variables out of both runs. Exclude all optional
    # model providers. The explicit environment is recorded with the manifest.
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith("CAD_") and k.upper() not in
           {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    applied = {"CAD_REGIONS": "1" if variant == "B" else "0", "PYTHONUTF8": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": str(config["threads"]),
               "OMP_NUM_THREADS": str(config["threads"])}
    env.update(applied)
    peak = 0
    to_step = None
    timed_out = False
    started = time.perf_counter()
    with path_log.open("wb") as log:
        process = subprocess.Popen(cmd, cwd=demo.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                peak = max(peak, rss_bytes(process_tree(root)))
                elapsed = time.perf_counter() - started
                if to_step is None and final_step(out) is not None:
                    to_step = elapsed
                if elapsed >= config["timeout_s"]:
                    timed_out = True
                    stop_process_tree(process, root)
                    break
                time.sleep(config["sample_interval_s"])
        except BaseException:
            stop_process_tree(process, root)
            raise
    wall = time.perf_counter() - started
    step = final_step(out)
    if to_step is None and step is not None:
        to_step = wall
    directories = sorted(out.glob("*/sheet.json"))
    directory = directories[0].parent if len(directories) == 1 else None
    result = {"status": "timeout" if timed_out else "failed" if process.returncode or step is None else "ok",
              "exit_code": process.returncode, "wall_s": wall, "to_final_step_s": to_step,
              "sampled_peak_tree_rss_bytes": peak or None, "flags": flags[:-2],
              "env": applied, "step_bytes": step.stat().st_size if step else None,
              **run_observations(path_log.read_text(encoding="utf-8", errors="replace"), directory)}
    if step is None:
        return result
    tol = Tolerance(**config["measurement"])
    sheet = json_file(step.parent / "sheet.json")
    frame = case.get("reference_frame", "aligned")
    transform = sheet.get("transform_from_scan_4x4") if reference and frame == "scan" else None
    if reference and frame == "scan" and transform is None:
        result["reference_error"] = "missing scan-to-model transform"
        reference = None
    try:
        result["geometry"] = evaluate(step, reference, transform=transform, tol=tol)
        aligned = step.parent / "scan_aligned.stl"
        if aligned.is_file():
            m = step_mesh(load_step(step), tol)
            s = trimesh.load(aligned, force="mesh")
            result["scan_surface"] = distances(s, m, tol)
    except Exception as exc:  # a damaged artifact must not cancel the remaining runs
        result["measurement_error"] = f"{type(exc).__name__}: {exc}"
    return result


def report(data):
    def mean_values(rows, field):
        values = [r[field] for r in rows if r.get(field) is not None]
        return statistics.median(values) if values else None

    def fmt(x, digits=2):
        return f"{x:.{digits}f}" if x is not None else "—"

    lines = ["# Полный бенчмарк CAD", "", f"Режим — {data['mode']}.",
             "", f"SHA A — {data['variants']['A']}. SHA B — {data['variants']['B'] or 'ожидается'}.",
             "", "В таблице медианы завершившихся прогонов. RAM — максимум суммарного RSS дерева процессов.",
             "Повторы — новые процессы и каталоги, без очистки системного файлового кэша.",
             "", "| Деталь | Вариант | Успешно / всего | Время до STEP, с | Полный запуск, с | RAM, МиБ | STEP, КиБ |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for case in data["cases"]:
        if case.get("status") == "unavailable":
            lines.append(f"| {case['id']} | — | скан недоступен | — | — | — | — |")
            continue
        for v in ("A", "B"):
            runs = [r for r in case["runs"] if r["variant"] == v]
            if not runs:
                continue
            ok = [r for r in runs if r["status"] == "ok"]
            ram = max((r["sampled_peak_tree_rss_bytes"] for r in runs
                       if r.get("sampled_peak_tree_rss_bytes")), default=0)
            size = mean_values(ok, "step_bytes")
            lines.append(f"| {case['id']} | {v} | {len(ok)} / {len(runs)} | "
                         f"{fmt(mean_values(ok, 'to_final_step_s'))} | {fmt(mean_values(ok, 'wall_s'))} | "
                         f"{fmt(ram / 2**20 if ram else None, 0)} | "
                         f"{fmt(size / 1024 if size is not None else None, 1)} |")
    lines.extend(["", "| Деталь / прогон | Путь | Грани / рёбра / тела | Валидно / замкнуто / связно | Скан → STEP | STEP → скан | P95 в двух направлениях, мм | Ошибки элементов |",
                  "| --- | --- | --- | --- | ---: | ---: | --- | --- |"])
    for case in data["cases"]:
        for r in case.get("runs", ()):
            g = r.get("geometry") or {}
            top = g.get("model_topology")
            a = (r.get("scan_surface") or {}).get("reference_to_model") or {}
            b = (r.get("scan_surface") or {}).get("model_to_reference") or {}
            counts = g.get("outcomes")
            error = (" / ".join(str(counts.get(k, 0)) for k in ("missing", "extra", "parameter_error"))
                     if counts is not None else "—")
            lines.append(f"| {case['id']} {r['variant']}/{r['repeat']} | {r.get('selected_route') or r['status']} | "
                         f"{'/'.join(str(top[k]) for k in ('faces','edges','solids')) if top else '—'} | "
                         f"{'/'.join('да' if top[k] else 'нет' for k in ('valid','closed','connected_solid')) if top else '—'} | "
                         f"{fmt(100 * a['within_share'], 1) + '%' if a else '—'} | "
                         f"{fmt(100 * b['within_share'], 1) + '%' if b else '—'} | "
                         f"{fmt(a.get('p95_mm'))} / {fmt(b.get('p95_mm'))} | {error} |")
    lines += ["", "Ошибки элементов приведены как пропуски / лишние / вне допуска. "
              "Подробные стадии, откаты, максимумы отклонений и статус регионального этапа есть в JSON. "
              "У недоступных стадий значение отсутствует. Наличие CAD_REGIONS=1 само по себе не доказывает запуск 0к.", ""]
    if data["mode"] == "full_AB":
        lines += ["", "## Сравнение", ""]
        for case in data["cases"]:
            completed = {v: [r for r in case["runs"] if r["variant"] == v and r["status"] == "ok"]
                         for v in ("A", "B")}
            a, b = (mean_values(completed[v], "to_final_step_s") for v in ("A", "B"))
            if a is None or b is None:
                lines.append(f"- {case['id']} — нет успешной пары A/B.")
                continue
            observed = sum(bool(r["regional_stage_observed"]) for r in completed["B"])
            delta = 100 * (b / a - 1)
            lines.append(f"- {case['id']} — изменение медианы времени до STEP {delta:+.1f}%. "
                         f"Региональный этап наблюдался в {observed} из {len(completed['B'])} "
                         "успешных прогонов B.")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--checkout-a", type=Path, required=True)
    p.add_argument("--checkout-b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke-case", help="One real A run to verify the harness; does not produce an A/B comparison")
    args = p.parse_args()
    config = json_file(args.config)
    variants = config["variants"]
    checkouts = {"A": verify_checkout(args.checkout_a, variants["A"])}
    if not args.smoke_case:
        if not args.checkout_b or not variants["B"]:
            p.error("full A/B requires the integrated stage B SHA and --checkout-b")
        if config["repeats"] < 2:
            p.error("full A/B requires at least two repeats")
        checkouts["B"] = verify_checkout(args.checkout_b, variants["B"])
    ids = [case["id"] for case in config["cases"]]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", i) for i in ids):
        p.error("case IDs must be unique and contain only letters, digits, underscores or hyphens")
    if args.smoke_case and args.smoke_case not in ids:
        p.error("unknown smoke case")
    if config["threads"] < 1:
        p.error("threads must be at least one")
    Tolerance(**config["measurement"])
    if config["timeout_s"] <= 0 or config["sample_interval_s"] <= 0:
        p.error("timeout and RSS sampling interval must be positive")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    data = {"mode": "smoke_A_only" if args.smoke_case else "full_AB",
            "variants": variants, "environment": fingerprint(), "config": config, "cases": []}
    for case in config["cases"]:
        if args.smoke_case and args.smoke_case != case["id"]:
            continue
        scan = (args.inputs / case["scan"]).resolve() if case.get("scan") else None
        ref = (args.inputs / case["reference"]).resolve() if case.get("reference") else None
        ref = ref if ref and ref.is_file() else None
        row = {"id": case["id"], "reference_status": "available" if ref else "unavailable", "runs": []}
        data["cases"].append(row)
        if scan is None or not scan.is_file():
            row.update(status="unavailable", reason=case.get("unavailable_reason", "scan file missing"))
            continue
        row["scan_sha256"] = file_hash(scan)
        reference_hash = file_hash(ref) if ref else None
        row["reference_sha256"] = reference_hash
        for repeat in range(1, 2 if args.smoke_case else config["repeats"] + 1):
            order = list(checkouts) if repeat % 2 else list(reversed(checkouts))
            for variant in order:
                if file_hash(scan) != row["scan_sha256"] or (ref and file_hash(ref) != reference_hash):
                    raise RuntimeError("input changed during the experiment")
                checkout = verify_checkout(checkouts[variant], variants[variant])
                print(case["id"], variant, repeat, flush=True)
                run = run_case(checkout, case, scan, ref,
                               args.out / case["id"] / f"{variant}-{repeat}", variant, config)
                row["runs"].append({"variant": variant, "repeat": repeat, **run})
                write_json(args.out / "results.json", data)
                verify_checkout(checkout, variants[variant])
                if file_hash(scan) != row["scan_sha256"] or (ref and file_hash(ref) != reference_hash):
                    raise RuntimeError("input changed during the experiment")
    write_json(args.out / "results.json", data)
    (args.out / "report.md").write_text(report(data), encoding="utf-8")


if __name__ == "__main__":
    main()
