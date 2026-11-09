"""Проверка на случайных деталях: python bench.py [кол-во] [seed] [семейства] [-j 3]

Генерирует случайные детали шести семейств (плита с отверстиями и бобышкой,
фланец, втулка с буртом и пазом, ступенчатая ось, уголок со стенкой, точёная с конусом), случайно откалывает кусок
края, добавляет шум сканера и случайно поворачивает. Затем строит модель
алгоритмом без ИИ и сравнивает с целой деталью до поломки. Так алгоритм нельзя
подогнать под три известные детали.
"""
from __future__ import annotations

import json
import math
import sys

# Windows: вывод в канал по умолчанию в cp1251, а страница и проверки читают UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
import tempfile
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

import compare as cmpmod
import demo
import gates
import geometric
import scan_analysis as sa


def r_(rng, a, b, step=0.5):
    if b < a:  # узкая деталь: верхняя граница ниже нижней (seed 114, нашёл Север)
        b = a
    return round(float(rng.uniform(a, b)) / step) * step


def plate(rng):
    L, W, T = r_(rng, 50, 120), r_(rng, 30, 70), r_(rng, 5, 12)
    hd = r_(rng, 4, 9)
    m = r_(rng, hd + 3, min(L, W) / 4)
    body = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    if rng.random() < 0.7:
        body = body.edges("|Z").fillet(r_(rng, 2, min(6, m - hd / 2 - 0.5), 0.5))
    body = body.faces(">Z").workplane().rect(L - 2 * m, W - 2 * m, forConstruction=True).vertices().hole(hd)
    if rng.random() < 0.6:
        bd = r_(rng, 14, min(W - 2 * m - hd, 40)) if W - 2 * m - hd > 15 else 14
        bh = r_(rng, 6, 25)
        body = body.union(cq.Workplane("XY").workplane(offset=T).circle(bd / 2).extrude(bh))
        body = body.faces(">Z").workplane().hole(r_(rng, 5, bd - 6))
    return body, f"плита {L}×{W}×{T}, 4×Ø{hd}"


def flange(rng):
    D, T = r_(rng, 60, 140), r_(rng, 6, 14)
    hub_d = r_(rng, D * 0.3, D * 0.5)
    bore = r_(rng, hub_d * 0.35, hub_d * 0.7)
    n = int(rng.choice([3, 4, 5, 6, 8]))
    hd = r_(rng, 5, 11)
    pcd = r_(rng, (hub_d + D) / 2 - 2, (hub_d + D) / 2 + 2)
    body = cq.Workplane("XY").circle(D / 2).extrude(T)
    body = body.union(cq.Workplane("XY").workplane(offset=T).circle(hub_d / 2).extrude(r_(rng, 6, 25)))
    body = body.faces(">Z").workplane().hole(bore)
    a0 = float(rng.uniform(0, 360 / n))
    pts = [(pcd / 2 * math.cos(math.radians(a0 + 360 * k / n)), pcd / 2 * math.sin(math.radians(a0 + 360 * k / n))) for k in range(n)]
    body = body.cut(cq.Workplane("XY").pushPoints(pts).circle(hd / 2).extrude(T + 1).translate((0, 0, -0.5)))
    return body, f"фланец Ø{D}, {n}×Ø{hd} на Ø{pcd}"


def bushing(rng):
    Dc, Tc = r_(rng, 35, 70), r_(rng, 4, 10)
    Ds = r_(rng, Dc * 0.45, Dc * 0.75)
    H = r_(rng, 15, 45)
    bore = r_(rng, Ds * 0.35, Ds * 0.6)
    body = cq.Workplane("XY").circle(Dc / 2).extrude(Tc).union(
        cq.Workplane("XY").workplane(offset=Tc).circle(Ds / 2).extrude(H))
    body = body.cut(cq.Workplane("XY").circle(bore / 2).extrude(Tc + H))
    if rng.random() < 0.6:
        kw = r_(rng, 3, max(3.5, bore * 0.35))
        kd = r_(rng, 1.5, min(4, (Ds - bore) / 4))
        body = body.cut(cq.Workplane("XY").center(0, bore / 2 + kd / 2 - 0.3).rect(kw, kd + 0.6).extrude(Tc + H))
    return body, f"втулка Ø{Dc}/Ø{Ds}×{Tc + H}"


def shaft(rng):
    """Ступенчатая ось: лежит на торце, 2–4 ступени."""
    n = int(rng.integers(2, 5))
    d = sorted([r_(rng, 12, 50) for _ in range(n)], reverse=True)
    z, body = 0.0, None
    for di in d:
        h = r_(rng, 6, 25)
        s = cq.Workplane("XY").workplane(offset=z).circle(di / 2).extrude(h)
        body = s if body is None else body.union(s)
        z += h
    if rng.random() < 0.5:
        body = body.faces(">Z").workplane().hole(r_(rng, 4, d[-1] * 0.5))
    return body, f"ось {n} ступ., Ø{d[0]}…Ø{d[-1]}"


def angle(rng):
    """Уголок: основание + стенка, в стенке отверстия с горизонтальной осью (сложно для слоёв)."""
    L, W, T = r_(rng, 40, 90), r_(rng, 30, 60), r_(rng, 5, 10)
    H = r_(rng, 25, 60)
    base = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    wall = cq.Workplane("XY").box(T, W, H, centered=(False, True, False)).translate((-L / 2, 0, 0))
    body = base.union(wall)
    hd = r_(rng, 5, 10)
    body = body.faces(">Z").workplane().pushPoints([(L / 4, W / 4), (L / 4, -W / 4)]).hole(hd)
    zc = T + (H - T) * 0.6
    side = cq.Workplane("YZ").workplane(offset=-L / 2 - 1).pushPoints([(W / 4, zc), (-W / 4, zc)]).circle(hd / 2).extrude(T + 2)
    body = body.cut(side)
    if rng.random() < 0.6:  # косынка
        g = r_(rng, 4, 8)
        gus = (cq.Workplane("XZ").polyline([(-L / 2 + T, T), (-L / 2 + T + (L - T) * 0.45, T), (-L / 2 + T, T + (H - T) * 0.45)])
               .close().extrude(g / 2, both=True))
        body = body.union(gus)
    return body, f"уголок {L}×{W}, стенка {H}"


def turned(rng):
    """Точёная деталь: конусный переход, фаски, скругление у буртика (сложно для слоёв)."""
    D1 = r_(rng, 30, 60)
    D2 = r_(rng, 14, D1 - 8)
    h1, hc, h2 = r_(rng, 6, 15), r_(rng, 6, 18), r_(rng, 8, 25)
    pts = [(0, 0), (D1 / 2, 0), (D1 / 2, h1), (D2 / 2, h1 + hc), (D2 / 2, h1 + hc + h2), (0, h1 + hc + h2)]
    body = cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0))
    c = r_(rng, 0.8, 2.0)
    try:
        body = body.faces(">Z").edges().chamfer(c)
        body = body.faces("<Z").edges().chamfer(c)
    except Exception:  # noqa: BLE001
        pass
    if rng.random() < 0.6:
        body = body.faces(">Z").workplane().hole(r_(rng, 4, D2 * 0.5))
    return body, f"точёная Ø{D1}→Ø{D2}, конус {hc}"


FAMILIES = [plate, flange, bushing, shaft, angle, turned]


def break_part(part, rng):
    """Откол: клин с краю детали в нижней части (там, где деталь держится)."""
    bb = part.val().BoundingBox()
    cx, cy = (bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2
    R = max(bb.xlen, bb.ylen) / 2
    ang = float(rng.uniform(0, 2 * math.pi))
    depth = float(rng.uniform(0.12, 0.25)) * R
    w = float(rng.uniform(0.35, 0.7)) * R
    ux, uy = math.cos(ang), math.sin(ang)
    vx, vy = -uy, ux
    p0 = (cx + ux * (R - depth), cy + uy * (R - depth))
    poly = [(p0[0] + vx * w * s + ux * t, p0[1] + vy * w * s + uy * t)
            for s, t in ((-1, R), (-0.4, depth * 0.2), (0.3, -depth * 0.1), (1, R))]
    poly.append((p0[0] + ux * 3 * R, p0[1] + uy * 3 * R))
    hcut = float(rng.uniform(0.5, 1.0)) * bb.zlen
    br = cq.Workplane("XY").polyline(poly).close().extrude(hcut + 1).translate((0, 0, bb.zmin - 0.5))
    return part.cut(br)


def fracture(m: trimesh.Trimesh, truth: trimesh.Trimesh, rng) -> trimesh.Trimesh:
    """Излом шероховатый, а не ровный срез: поверхность скола (точки дальше 0.05 мм от
    целой детали) гуляет по нормали на 0.1–0.4 мм плавными буграми.
    Без этого скол по сетке неотличим от фрезерованного выреза (живой тест 2, 09.11)."""
    m2 = m.copy()
    _, dv, _ = trimesh.proximity.closest_point(truth, m2.vertices)
    on = dv > 0.05
    if not on.any():
        return m
    amp = float(rng.uniform(0.3, 0.7))
    x = m2.vertices[on]
    bump = np.zeros(len(x))
    for _ in range(6):
        k = rng.normal(size=3)
        k *= 2 * np.pi / float(rng.uniform(3.0, 8.0)) / np.linalg.norm(k)
        bump += np.sin(x @ k + rng.uniform(0, 2 * np.pi))
    nv = m2.vertex_normals[on]
    m2.vertices[on] = x + nv * (amp * bump / 3)[:, None]
    return m2


def make(seed: int, d: Path):
    rng = np.random.default_rng(seed)
    fams = [f for f in FAMILIES if f.__name__ in ONLY] if ONLY else FAMILIES
    fam = fams[seed % len(fams)]
    part, desc = fam(rng)
    broken = break_part(part, rng)
    cq.exporters.export(part, str(d / "truth.stl"), tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(broken, str(d / "broken.stl"), tolerance=0.02, angularTolerance=0.1)
    m = trimesh.load(d / "broken.stl").subdivide_to_size(float(rng.uniform(1.5, 3.0)))
    m = fracture(m, trimesh.load(d / "truth.stl"), rng)
    m.vertices += rng.normal(0, float(rng.uniform(0.01, 0.06)), m.vertices.shape)
    T = trimesh.transformations.random_rotation_matrix(rng.random(3))
    T[:3, 3] = rng.uniform(-200, 200, 3)
    m.apply_transform(T)
    m.export(d / "scan.stl")
    truth = trimesh.load(d / "truth.stl")
    truth.apply_transform(T)  # эталон в тех же координатах, что и «скан»
    return desc, truth


ONLY: list[str] = []


def one(args):
    """Одна деталь: генерация, построение без ИИ, сравнение. Отдельно, чтобы детали
    считались параллельно."""
    global ONLY
    seed, only, tmp = args
    ONLY = only
    d = Path(tmp) / f"p{seed}"
    d.mkdir()
    t0 = time.time()
    try:
        desc, truth = make(seed, d)
    except Exception as e:  # noqa: BLE001 — неудачная генерация не считается
        return {"seed": seed, "skip": f"генерация не удалась ({type(e).__name__})"}
    row = {"seed": seed, "family": desc.split()[0], "part": desc}
    try:
        sheet, scan = sa.measurement_sheet(str(d / "scan.stl"))
        code, meta = geometric.build(scan, sheet)
        run = demo.run_code(code, d)
        row["valid"] = bool(run.get("ok") and run.get("valid"))
        if run.get("ok"):
            model = trimesh.load(d / "model.stl", force="mesh")
            truth.apply_transform(np.array(sheet["transform_from_scan_4x4"]))
            tol = max(0.3, 3 * sheet["diagnostics"].get("noise_mm", 0))
            c_scan = cmpmod.compare(scan, model, tol=tol)
            c_truth = cmpmod.compare(truth, model, tol=tol)
            row["gates"] = gates.check(run, model, sheet, scan)
            c_fixed = cmpmod.compare(truth, model, tol=0.3)
            # «наивное копирование»: сам сломанный скан как модель. Если метрика даёт ему
            # высокий балл, она не отличает восстановление от копии скола
            c_copy = cmpmod.compare(truth, scan, tol=tol)
            # Главная метрика восстановления: только там, где скан НЕ видел целую деталь
            # (точки эталона дальше 1 мм от скана — отколотое). Копия скола здесь даёт ~0.
            tp = c_truth["_scan_pts"]
            d_scan = trimesh.proximity.closest_point(scan, tp)[1]
            lost = d_scan > 1.0
            if lost.sum() >= 20:
                row["lost_share"] = round(float(lost.mean()), 4)
                row["restore_fit"] = round(float((np.abs(c_truth["_scan_dev"][lost]) <= tol).mean()), 4)
                row["restore_fit_copy"] = round(float((np.abs(c_copy["_scan_dev"][lost]) <= tol).mean()), 4)
            row.update(scan_fit=c_scan["within_tol_share"],
                       # симметрично: и эталон близко к модели, и модель не выходит за эталон
                       truth_fit=min(c_truth["within_tol_share"], 1 - c_truth["added_surface_share"]),
                       truth_fit_03=min(c_fixed["within_tol_share"], 1 - c_fixed["added_surface_share"]),
                       copy_fit=min(c_copy["within_tol_share"], 1 - c_copy["added_surface_share"]),
                       tol=round(tol, 3), truth_p95=c_truth["p95_abs_dev_mm"], truth_max=c_truth["max_abs_dev_mm"],
                       restored=len([r for r in meta["restored"] if "ничего" not in r]))
        else:
            row["error"] = run.get("error", "")[:120]
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:160]
    row["sec"] = round(time.time() - t0, 1)
    return row


def main():
    """python bench.py [кол-во] [seed] [семейства через запятую] [-j потоков]"""
    import warnings

    warnings.filterwarnings("ignore")
    argv = sys.argv[1:]
    jobs = 3
    if "-j" in argv:
        i = argv.index("-j")
        jobs = int(argv[i + 1])
        del argv[i:i + 2]
    wargs = []
    if "--worker" in argv:
        i = argv.index("--worker")
        wargs = argv[i:i + 4]
        del argv[i:i + 4]
    n = int(argv[0]) if len(argv) > 0 else 12
    seed0 = int(argv[1]) if len(argv) > 1 else 100
    only = argv[2].split(",") if len(argv) > 2 else []
    tmp = tempfile.mkdtemp()
    rows = []
    tasks = [(seed0 + k, only, tmp) for k in range(n)]
    # каждая деталь в своём процессе (maxtasksperchild=1): память ядра CAD не копится
    import os
    import subprocess as sp

    if wargs:  # дочерний процесс: одна деталь (по номеру part в списке), строка в файл
        part, outf = int(wargs[1]), wargs[3]
        with open(outf, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(one(tasks[part]), ensure_ascii=False) + "\n")
        return
    # Каждая деталь в своём процессе, до jobs одновременно. Упавший процесс теряет только
    # свою деталь, а причина (код выхода, конец stderr) попадает в отчёт. Раньше процесс
    # вёл треть деталей, и его падение молча теряло все оставшиеся (bench-600-36: 15 из 36).
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    base = [sys.executable, str(Path(__file__).resolve()), str(n), str(seed0)] + ([",".join(only)] if only else [])
    queue = list(range(len(tasks)))
    running: dict = {}
    while queue or running:
        while queue and len(running) < jobs:
            k = queue.pop(0)
            f, ef = Path(tmp) / f"r{k}.jsonl", Path(tmp) / f"r{k}.err"
            p = sp.Popen(base + ["--worker", str(k), "1", str(f)], env=env, stderr=open(ef, "w"))
            running[k] = (p, f, ef, time.time())
        for k, (p, f, ef, t0) in list(running.items()):
            if p.poll() is None and time.time() - t0 < 900:
                continue
            if p.poll() is None:
                p.kill()
                p.wait()
            del running[k]
            seed = tasks[k][0]
            if f.exists() and f.read_text(encoding="utf-8").strip():
                row = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
            else:
                tail = ef.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:] if ef.exists() else []
                why = "убит (нехватка памяти?)" if p.returncode in (-9, 137) else f"код выхода {p.returncode}"
                row = {"seed": seed, "family": "?", "part": "?",
                       "error": f"процесс проверки упал: {why}; " + " | ".join(tail)[-400:]}
            rows.append(row)
            print(f"{row['seed']}: {row['skip']}, пропуск" if "skip" in row else json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1)
    rows.sort(key=lambda r: r["seed"])
    skipped = len([r for r in rows if "skip" in r])
    rows_g = [r for r in rows if "skip" not in r]
    ok = [r for r in rows_g if r.get("truth_fit") is not None and r.get("valid")]
    bad = [r for r in rows_g if r not in ok]

    def stat(key, rr):
        v = np.array([r[key] for r in rr])
        return f"медиана {np.median(v)*100:.1f}%, худшая {v.min()*100:.1f}%, ≥95%: {(v >= 0.95).sum()} из {len(v)}"

    gate_bad = [r for r in ok if r.get("gates")]
    lines = [f"Команда: python bench.py {n} {seed0}" + (f" {','.join(only)}" if only else ""),
             f"Сгенерировано {len(rows_g)} (генерация не удалась: {skipped}), построено {len(ok)}, "
             f"не построено или битое тело: {len(bad)}, не прошли обязательные проверки: {len(gate_bad)}"]
    if ok:
        lines += [f"С целой деталью (допуск 0.3 мм или 3× шум): {stat('truth_fit', ok)}",
                  f"С целой деталью (жёсткий допуск 0.3 мм):   {stat('truth_fit_03', ok)}",
                  f"Для сравнения, сломанный скан как есть:     {stat('copy_fit', ok)}"]
        rf = [r for r in ok if "restore_fit" in r]
        if rf:
            lines += [f"Только отколотое место (эталон дальше 1 мм от скана), {len(rf)} деталей:",
                      f"  модель:                {stat('restore_fit', rf)}",
                      f"  сломанный скан как есть: {stat('restore_fit_copy', rf)}"]
        for fam in sorted({r["family"] for r in rows_g}):
            rr = [r for r in ok if r["family"] == fam]
            nb = len([r for r in bad if r["family"] == fam])
            if rr:
                lines.append(f"  {fam}: {stat('truth_fit', rr)}" + (f", не построено {nb}" if nb else ""))
            elif nb:
                lines.append(f"  {fam}: не построено {nb}")
    print("\nИТОГ:\n" + "\n".join(lines))
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    tag = f"bench-{seed0}-{n}" + (f"-{'-'.join(only)}" if only else "")
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        sha = ""
    (out / f"{tag}.json").write_text(json.dumps({"commit": sha, "summary": lines, "rows": rows},
                                                ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
