"""Acceptance of coarse cylindrical walls with an independent STEP reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS

HERE = Path(__file__).resolve().parent
for folder in ("cad-coarse-bores", "cad-local-bores"):
    sys.path.insert(0, str(HERE.parent / f"{folder}-2026-11-12"))
from benchmark_coarse_bores import fixture, mesh_step
from benchmark_local_bores import inspect, worker


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def generate(work):
    import nurbs
    import surface

    body, truth = fixture()
    cq.exporters.export(body, str(work / "truth.step"))
    (work / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    scan = mesh_step(work / "truth.step")
    scan.export(work / "scan.ply")
    scan = trimesh.load(work / "scan.ply", force="mesh")
    clean = surface.flatten_planes(surface.surface_mesh(scan))
    nurbs._SCAN, nurbs.FACE_TOL = scan, .3
    triangles, nurbs.K, nurbs.NP, nurbs.PAIR = nurbs.STEPS[0]
    stage, _, _ = nurbs._build(clean, triangles, nurbs._Projector(clean), profile={}, stage={})
    if stage is None:
        raise RuntimeError("normal smooth stage failed")
    cq.exporters.export(stage, str(work / "stage.step"))


def exact_distance(point, shape):
    vertex = cq.Vertex.makeVertex(*point)
    distance = BRepExtrema_DistShapeShape(vertex.wrapped, shape.Shells()[0].wrapped)
    support = distance.SupportOnShape2(1)
    kind = cq.Face(TopoDS.Face_s(support)).geomType() if support.ShapeType() == TopAbs_FACE else str(support.ShapeType())
    return {"distance_mm": float(distance.Value()), "closest_support": kind}


def missing_rims(index, bore, span):
    rows = []
    for factor in (.3, .6, .85, 1.05):
        reach = max(.6, factor * bore.radius)
        low = index.rims(bore, span[0], reach, .3)
        high = index.rims(bore, span[1], reach, .3)
        pairs = [(a, b) for a in low for b in high]
        disjoint = sum(not ({u for u, _, _ in a.path} & {u for u, _, _ in b.path}) for a, b in pairs)
        viable = sum(index.between(a, b) is not None for a, b in pairs)
        rows.append({"reach_mm": reach, "low_loops": len(low), "high_loops": len(high),
                     "pairs": len(pairs), "disjoint": disjoint, "valid_removed_region": viable})
    return {"diameter_mm": 2 * bore.radius, "radial_span_mm": span, "loops": rows}


def negative_controls(walls, plans):
    import nurbs
    from bore_stitch import Bore
    from coarse_bore_evidence import RadialWall, explain_losses

    plate = cq.Workplane("XY").box(55, 35, 6).faces(">Z").workplane().slot2D(36, 10).cutThruAll().val()
    slot = nurbs.body_mesh(plate, .03)
    spec = ((0, 0, 3.3), (0, 0, -1), 5., 6.3)
    false_slot = RadialWall(slot, Bore.from_spec(spec), .3, .05).span(.3)
    point = np.array([[0., 0., 1000.]])
    unrelated = explain_losses(point, plans, walls, .3)
    return {"slot_rejected": false_slot is None, "unrelated_loss_rejected": not unrelated[0]["accepted"]}


def run(args):
    import surface
    from bore_stitch import Bore, Coverage, ShellIndex, _plans, repair_bores
    from coarse_bore_evidence import RadialWall

    if not args.scan and not args.reuse:
        generate(args.work)
    path = args.scan or args.work / "scan.ply"
    scan = trimesh.load(path, force="mesh")
    clean = surface.flatten_planes(surface.surface_mesh(scan))
    noise = float(read(args.sheet).get("diagnostics", {}).get("noise_mm", 0) or 0) if args.scan else .05
    parameters = surface.hole_parameters(scan)
    specs = [Bore.from_spec(spec) for spec in parameters]
    evidence = [RadialWall(scan, bore, .3, noise) for bore in specs]
    supported = [(wall.bore, span) for wall in evidence if (span := wall.span(.3)) is not None]
    gate = Coverage(scan, clean, .3)
    if not args.scan:
        truth = read(args.work / "truth.json")
        targets = [(Bore.from_spec(spec), (0., spec[3])) for spec in truth]
    else:
        targets = supported

    if args.scan and args.stage_step is None:
        with patch.dict("os.environ", {"CAD_BORE_COARSE": "1"}):
            old_step, old_time, _ = worker(args, "baseline", False)
            new_step, new_time, new_worker = worker(args, "coarse", True)
        audit = new_worker["meta"].get("bore_stitch") or {}
        timing = {"baseline_0k": old_time, "coarse_0k": new_time}
    else:
        old_step = args.stage_step or args.work / "stage.step"
        stage = cq.importers.importStep(str(old_step)).val()
        result, audit = repair_bores(stage, scan, parameters, clean=clean,
                                     noise_mm=noise, coarse=True)
        new_step = args.work / "coarse.step"
        cq.exporters.export(result, str(new_step))
        timing = {"coarse_holes": audit["seconds"]}

    before = inspect(old_step, gate, targets, None)
    after = inspect(new_step, gate, targets, None)
    topology = after["topology"]
    limit = args.face_cap or (1776 if args.scan else int(before["topology"]["faces"] * 1.1))
    raw_before, raw_after = before["coverage"], after["coverage"]
    losses = audit.get("batch", {}).get("clean_losses", [])
    checks = {"exact_walls": after["detected_exact"] >= args.minimum_exact,
              "more_exact": after["detected_exact"] > before["detected_exact"],
              "one_valid_closed_solid": topology["valid"] and topology["closed"] and topology["connected_solid"]
                                        and after["signed_mesh_volume_mm3"] > 0,
              "face_cap": topology["faces"] <= limit,
              "raw_coverage": all(raw_after[k] >= raw_before[k] for k in ("scan_to_body", "body_to_scan")),
              "batch_accepted": audit.get("batch", {}).get("reason") == "accepted"}
    output = {"baseline": before, "coarse": after, "face_cap": limit, "seconds": timing,
              "checks": checks, "audit": audit, "edge_mm_p50_p90": np.quantile(scan.edges_unique_length, [.5, .9]).tolist()}
    if not args.scan:
        ref = cq.importers.importStep(str(args.work / "truth.step")).val()
        candidate = cq.importers.importStep(str(new_step)).val()
        for row in losses:
            row["truth"] = exact_distance(row["point_mm"], ref)
            row["candidate"] = exact_distance(row["point_mm"], candidate)
        index = ShellIndex(stage)
        plans, missing = [], []
        for bore, span in supported:
            plan = next(_plans(index, bore, span, .3), None)
            if plan is None:
                missing.append(missing_rims(index, bore, span))
            else:
                plans.append(plan)
        output["unrepaired"] = missing
        output["negative_controls"] = negative_controls({id(wall.bore): wall for wall in evidence}, plans)
        checks.update(output["negative_controls"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--stage-step", type=Path)
    parser.add_argument("--face-cap", type=int)
    parser.add_argument("--minimum-exact", type=int)
    args = parser.parse_args()
    if args.scan and not args.sheet:
        parser.error("--scan requires --sheet")
    if not args.scan and args.stage_step:
        parser.error("--stage-step requires --scan")
    args.minimum_exact = args.minimum_exact or (1 if args.scan else 4)
    for key in ("demo", "work", "scan", "sheet", "stage_step"):
        if (value := getattr(args, key)) is not None:
            setattr(args, key, value.resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.demo))
    result = run(args)
    (args.work / "task23-report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"]), flush=True)
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
