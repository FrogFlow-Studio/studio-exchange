"""Negative controls for Compass CAD demo, without LLM calls.

Run with the same dependencies as claude/cad-demo:
  python cad_demo_review_probes.py --source PATH_TO_COPIED_CAD_DEMO --out OUTPUT_DIR

The source must be a disposable copy: run_code writes only inside --out.
No source modules are modified. This script records counterexamples, not a
general estimate of reconstruction accuracy. Exit 0 means probes completed;
inspect flagged false accepts in results.json.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-cli", action="store_true", help="Also run the tower case through the scripted CLI")
    ap.add_argument("--run-intact", action="store_true", help="Run a pristine part intentionally designed with three holes")
    args = ap.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(name, None)
    sys.path.insert(0, str(source))
    import cadquery as cq
    import compare as cmpmod
    import demo
    import scan_analysis as sa
    import trimesh

    # Extract the exact nested acceptance predicate; avoid maintaining a subtly
    # different copy of the production rule in this review script.
    tree = ast.parse((source / "demo.py").read_text(encoding="utf-8"))
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    good_node = next(n for n in ast.walk(main_node) if isinstance(n, ast.FunctionDef) and n.name == "good")
    ns: dict = {"a": SimpleNamespace(target=0.95)}
    exec(compile(ast.Module(body=[good_node], type_ignores=[]), "acceptance_predicate", "exec"), ns)
    good = ns["good"]

    def mesh(shape, name):
        path = out / f"{name}.stl"
        cq.exporters.export(shape, str(path), tolerance=0.02)
        return trimesh.load(path, force="mesh")

    rows = []
    base = cq.Workplane("XY").box(100, 100, 10, centered=(True, True, False))
    base_mesh = mesh(base, "base")
    # A legitimate small through-hole is completely filled in the candidate.
    drilled = base.faces(">Z").workplane().hole(2)
    scan = mesh(drilled, "scan_with_2mm_hole")
    comparison = cmpmod.compare(scan, base_mesh, tol=0.3)
    candidate = {"run": {"ok": True, "valid": True, "solids": 1}, "cmp": comparison}
    rows.append({"probe": "filled_functional_hole", "hole_diameter_mm": 2,
                 "accepted_by_demo_good": good(candidate),
                 "score": demo.score(candidate["run"], comparison),
                 "bench_truth_fit": min(comparison["within_tol_share"], 1 - comparison["added_surface_share"]),
                 "comparison": cmpmod.public(comparison)})

    # An unwanted, connected 100mm tower occupies only a small patch of the
    # observed base; extra geometry is measured but ignored by score/good.
    tower = cq.Workplane("XY").workplane(offset=10).rect(2, 2).extrude(100)
    wrong = base.union(tower)
    wrong_mesh = mesh(wrong, "candidate_with_unwanted_tower")
    comparison = cmpmod.compare(base_mesh, wrong_mesh, tol=0.3)
    candidate = {"run": {"ok": True, "valid": bool(wrong.val().isValid()), "solids": 1}, "cmp": comparison}
    rows.append({"probe": "unwanted_connected_tower", "tower_height_mm": 100,
                 "accepted_by_demo_good": good(candidate),
                 "score": demo.score(candidate["run"], comparison),
                 "bench_truth_fit": min(comparison["within_tol_share"], 1 - comparison["added_surface_share"]),
                 "comparison": cmpmod.public(comparison)})

    # A known 1mm object triggers unconditional 1000x scaling. Its dimensions
    # are known by construction, but no explicit units can be supplied here.
    tiny = trimesh.creation.box(extents=[1.0, 0.8, 0.5])
    tiny_path = out / "known_millimetre_part.stl"
    tiny.export(tiny_path)
    sheet, processed = sa.measurement_sheet(str(tiny_path))
    rows.append({"probe": "known_mm_part_rescaled", "input_extents": tiny.extents.tolist(),
                 "processed_extents": processed.extents.tolist(),
                 "diagnostics": sheet["diagnostics"], "bounds_mm": sheet["bounds_mm"]})

    # The runner accepts a compound with two disconnected solids, despite
    # the prompt's requirement of one solid. No untrusted/generated code runs.
    runner_dir = out / "compound_runner"
    runner_dir.mkdir(exist_ok=True)
    code = ("import cadquery as cq\n"
            "a = cq.Workplane('XY').box(10,10,10).val()\n"
            "b = cq.Workplane('XY').box(10,10,10).translate((100,0,0)).val()\n"
            "result = cq.Compound.makeCompound([a,b])\n")
    run = demo.run_code(code, runner_dir)
    rows.append({"probe": "runner_accepts_two_solids", "run": run})
    if run.get("ok"):
        restored = cq.importers.importStep(str(runner_dir / "model.step")).val()
        rows.append({"probe": "compound_step_round_trip", "valid": bool(restored.isValid()),
                     "solids": len(restored.Solids()), "volume_mm3": restored.Volume()})

    # Exact score/acceptance policy can prefer a failing candidate over a
    # passing one. Demonstrates the ordering mismatch, not a full LLM run.
    failing = {"run": {"ok": True, "valid": True},
               "cmp": {"within_tol_share": .99, "outside_share": .006}}
    passing = {"run": {"ok": True, "valid": True},
               "cmp": {"within_tol_share": .965, "outside_share": .004}}
    rows.append({"probe": "score_and_acceptance_disagree",
                 "failing_good": good(failing), "passing_good": good(passing),
                 "failing_score": demo.score(failing["run"], failing["cmp"]),
                 "passing_score": demo.score(passing["run"], passing["cmp"])})

    if args.run_cli:
        replay = out / "negative_control_scripted.json"
        answer = ("```python\nimport cadquery as cq\n"
                  "base = cq.Workplane('XY').box(100,100,10,centered=(True,True,False))\n"
                  "tower = cq.Workplane('XY').workplane(offset=10).rect(2,2).extrude(100)\n"
                  "result = base.union(tower)\n```\n"
                  '```json\n{"plan":"Review negative control: unwanted tower",'
                  '"assumptions":[],"restored":[],"check_by_engineer":["Reject tower"]}\n```')
        replay.write_text(json.dumps([{"answer": answer}]), encoding="utf-8")
        cmd = [sys.executable, str(source / "demo.py"), str(out / "base.stl"),
               "--mode", "scripted", "--replay", str(replay), "--no-geometric",
               "--iters", "1", "--no-open", "--out", str(out / "cli_tower")]
        started = time.perf_counter()
        cli = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        wall = time.perf_counter() - started
        (out / "cli_tower.log").write_text(cli.stdout + cli.stderr, encoding="utf-8")
        reports = list((out / "cli_tower").glob("*/report.html"))
        report = reports[-1].read_text(encoding="utf-8") if reports else ""
        reported = re.search(r"Время<b>(\d+) с</b>", report)
        rows.append({"probe": "scripted_cli_accepts_unwanted_tower", "exit_code": cli.returncode,
                     "converged_log": "сошлось на попытке" in cli.stdout,
                     "converged_report": "Модель сошлась со сканом." in report,
                     "wall_seconds": round(wall, 2),
                     "reported_seconds": int(reported.group(1)) if reported else None})

    if args.run_intact:
        intact_dir = out / "intact_three_holes"
        intact_dir.mkdir(exist_ok=True)
        intact = (cq.Workplane("XY").box(80, 50, 8, centered=(True, True, False))
                  .faces(">Z").workplane().pushPoints([(-30, -15), (30, -15), (-30, 15)]).hole(4))
        cq.exporters.export(intact, str(intact_dir / "intact.stl"), tolerance=0.02)
        cq.exporters.export(intact, str(intact_dir / "truth.step"))
        cmd = [sys.executable, str(source / "demo.py"), str(intact_dir / "intact.stl"),
               "--mode", "geometric", "--notes", "Деталь целая, три отверстия, четвертого не должно быть",
               "--truth", str(intact_dir / "truth.step"), "--no-open", "--out", str(intact_dir / "runs")]
        cli = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        (intact_dir / "run.log").write_text(cli.stdout + cli.stderr, encoding="utf-8")
        reports = list((intact_dir / "runs").glob("*/report.html"))
        report = reports[-1].read_text(encoding="utf-8") if reports else ""
        rows.append({"probe": "intact_three_holes_modified", "exit_code": cli.returncode,
                     "fourth_hole_added_log": "найдено 3 из 4" in cli.stdout,
                     "converged_report": "Модель сошлась со сканом." in report})

    payload = {"python": sys.version, "platform": platform.platform(),
               "versions": {n: importlib.metadata.version(n) for n in
                            ("cadquery", "cadquery-ocp", "trimesh", "numpy", "scipy", "shapely")},
               "llm_calls": 0, "probes": rows}
    target = out / "results.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k not in ("comparison", "diagnostics")}, ensure_ascii=False))
    print(f"Evidence: {target}")


if __name__ == "__main__":
    main()
