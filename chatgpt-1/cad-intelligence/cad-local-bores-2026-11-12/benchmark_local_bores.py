"""Compare local cylindrical wall replacement with the existing 0k bore cuts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad-analytic-bores-2026-11-12"))
from benchmark_bores import (SAMPLES, SYNTHETIC_CASES, TOL, cq, extract, load_step, match,
                             prepare, run_boolean, topology, trimesh, write_json)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect(path, gate, targets, reference, strict_gate=None):
    import nurbs
    from bore_stitch import exact_wall

    body = load_step(path)
    mesh = nurbs.body_mesh(body, .03)
    walls = [exact_wall(body, bore, span, .3) for bore, span in targets]
    result = {"topology": topology(body), "coverage": gate.measure(mesh),
              "detected_exact": sum(wall is not None for wall in walls), "walls": walls,
              "signed_mesh_volume_mm3": float(mesh.volume)}
    if strict_gate is not None:
        result["coverage_at_0_2_mm"] = strict_gate.measure(mesh)
    if reference is not None:
        with patch.object(body, "tessellate", return_value=([], [])):
            rows = match(reference, extract(body, TOL), TOL)
        bores = [row for row in rows if row["reference"] and row["reference"]["role"] == "bore"]
        result["reference_exact"] = sum(row["status"] == "ok" for row in bores)
        result["reference_bands"] = len(bores)
    return result


def checks(before, after, cap, target):
    geom = after["topology"]
    return {
        "more_exact": after["detected_exact"] > before["detected_exact"],
        "minimum_exact": after["detected_exact"] >= target,
        "faces": geom["faces"] <= cap,
        "topology": (geom["valid"] and geom["closed"] and geom["connected_solid"]
                     and after["signed_mesh_volume_mm3"] > 0),
        "coverage": all(after["coverage"][k] >= v for k, v in before["coverage"].items()),
        "time": after["seconds"] <= before["seconds"],
        **({"coverage_at_0_2_mm": all(after["coverage_at_0_2_mm"][k] >= v
                                     for k, v in before["coverage_at_0_2_mm"].items())}
           if "coverage_at_0_2_mm" in before else {}),
    }


def worker(args, name, stitch):
    env = dict(os.environ, CAD_BORE_STITCH=str(int(stitch)), CAD_ANALYTIC_BORES="0",
               CAD_HOLE_PLUG="0", CAD_SURFACE_CACHE="0", CAD_HOLE_CACHE="0",
               CAD_SMOOTH_BACKGROUND="0")
    answer = args.work / f"{name}.json"
    with (args.work / f"{name}.log").open("wb") as log:
        start = perf_counter()
        subprocess.run([sys.executable, str(args.demo / "geo_worker.py"), str(args.scan),
                        str(args.sheet), "1", str(answer), "smooth"], cwd=args.demo,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        seconds = perf_counter() - start
    return answer.with_suffix(".step"), seconds, read(answer)


def evaluate(args):
    import nurbs
    import surface
    from bore_stitch import Bore, Coverage, observed_span, repair_bores

    if args.scan:
        stem, ref_path = args.work / "studio", args.reference
        scan = trimesh.load(args.scan, force="mesh")
        specs = surface.hole_parameters(scan)
    else:
        stem = args.work / args.case
        scan_path, ref_path = stem.with_suffix(".ply"), stem.with_suffix(".step")
        if not args.reuse:
            prepare(args.case, args.demo, args.work, nurbs, surface, args.uniform)
        scan = trimesh.load(scan_path, force="mesh")
        specs = read(stem.with_name(stem.name + "-specs.json"))

    clean = surface.flatten_planes(surface.surface_mesh(scan))
    isolated = not args.scan or args.stage_step is not None or args.stage_index is not None
    reference = extract(load_step(ref_path), TOL) if ref_path else None
    gate = Coverage(scan, clean, .3)
    strict_gate = Coverage(scan, clean, .2) if args.scan else None
    noise = float(read(args.sheet).get("diagnostics", {}).get("noise_mm", 0) or 0) if args.scan else 0.
    targets = [(bore, observed_span(scan, bore, max(.3, noise))) for bore in map(Bore.from_spec, specs)]
    supported = [(bore, span) for bore, span in targets if span is not None]

    if not isolated:
        old_step, old_time, old_worker = worker(args, "baseline", False)
        new_step, new_time, new_worker = worker(args, "local", True)
        stage = None
        details = {"baseline_worker": old_worker, "local_worker": new_worker}
        cap = args.face_cap or 1776
    else:
        def artifact(suffix):
            return stem.with_name(f"{stem.name}-{suffix}")

        stage_step = args.stage_step or artifact("stage.step")
        if args.stage_step:
            stage = {"provided_step": True}
        elif args.stage_index is not None:
            nurbs._SCAN = scan
            wide = os.environ.get("CAD_FIT_TOL") == "wide"
            nurbs.FACE_TOL = max(.3, (3 if wide else 1) * noise)
            triangles, nurbs.K, nurbs.NP, nurbs.PAIR = nurbs.STEPS[args.stage_index]
            projector = nurbs._Projector(clean)
            started = perf_counter()
            body, _, _ = nurbs._build(clean, triangles, projector, profile={}, stage={})
            if body is None:
                raise RuntimeError("requested 0k stage did not build a solid")
            stage = {"seconds": perf_counter() - started, "stage_index": args.stage_index}
            cq.exporters.export(body, str(stage_step))
        else:
            stage = read(artifact("stage.json"))
        old_step, old_data = artifact("boolean.step"), artifact("boolean.json")
        new_step = artifact("local.step")
        if args.refresh_baseline or not args.reuse or not (old_step.is_file() and old_data.is_file()):
            run_boolean(load_step(stage_step), scan, specs, surface, old_step, old_data)
        old_time = read(old_data)["seconds"]
        new, repair = repair_bores(load_step(stage_step), scan, specs, clean=clean, noise_mm=noise)
        new_time = repair["seconds"]
        fallback = not repair["accepted"]
        if fallback:
            # A rejected experiment returns the same input to the same cut routine.
            new_step = old_step
            new_time += old_time
        else:
            cq.exporters.export(new, str(new_step))
        stage.update(inspect(stage_step, gate, supported, reference))
        details = {"repair": repair, "fallback_reuses_baseline_measurement": fallback}
        cap = args.face_cap or (1776 if args.scan else int(stage["topology"]["faces"] * 1.1))

    before = inspect(old_step, gate, supported, reference, strict_gate)
    after = dict(before) if new_step == old_step else inspect(new_step, gate, supported, reference, strict_gate)
    before["seconds"], after["seconds"] = old_time, new_time
    return {"case": args.case or "studio", "specs": len(specs), "supported_spans": len(supported),
            "stage": stage, "face_cap": cap, "baseline": before, "local": after, **details,
            "checks": checks(before, after, cap, args.minimum_exact)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", choices=(*SYNTHETIC_CASES, *SAMPLES))
    mode.add_argument("--scan", type=Path, help="aligned scan from a 0k run")
    parser.add_argument("--sheet", type=Path, help="the diagnostics JSON from the same run")
    parser.add_argument("--reference", type=Path, help="optional reference STEP for --scan")
    stage_mode = parser.add_mutually_exclusive_group()
    stage_mode.add_argument("--stage-step", type=Path, help="optional STEP before any hole cuts")
    stage_mode.add_argument("--stage-index", type=int, choices=range(5),
                            help="run only this zero-based stage from nurbs.STEPS")
    parser.add_argument("--uniform", action="store_true", help="use Gmsh 1 mm synthetic mesh")
    parser.add_argument("--reuse", action="store_true", help="reuse local stage and optional baseline")
    parser.add_argument("--refresh-baseline", action="store_true", help="rerun boolean cuts on the cached stage")
    parser.add_argument("--face-cap", type=int)
    parser.add_argument("--minimum-exact", type=int, default=1)
    args = parser.parse_args()
    if args.scan and not args.sheet:
        parser.error("--scan requires --sheet")
    if not args.scan and (args.stage_step or args.stage_index is not None):
        parser.error("isolated studio stage options require --scan")
    for key in ("demo", "work", "scan", "sheet", "reference", "stage_step"):
        if (value := getattr(args, key)) is not None:
            setattr(args, key, value.resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.demo))
    report = evaluate(args)
    output = args.output or args.work / f"{args.case or 'studio'}-local-report.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), **report["checks"]}))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
