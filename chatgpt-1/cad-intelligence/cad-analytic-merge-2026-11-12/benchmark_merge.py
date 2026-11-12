"""Task 14 regression using task 12 acceptance and task 9/11 reference bodies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert

HERE = Path(__file__).resolve().parent
for name in ("cad-acceptance", "cad-regions", "cad-region-layout"):
    sys.path.insert(0, str(HERE.parent / f"{name}-2026-11-12"))

from benchmark_layout import EXTRA, extra_fixture
from benchmark_regions import FIXTURES, SAMPLES, fixture
from geometry_acceptance import (Tolerance, evaluate, extract, load_step, match, topology,
                                 transform_shape, write_json)


def split_and_join(shape, direction):
    box = shape.BoundingBox()
    plane = cq.Face.makePlane(3 * box.DiagonalLength, 3 * box.DiagonalLength, box.center, direction)
    solids = shape.split(plane).Solids()
    if len(solids) < 2:
        raise AssertionError("the test plane did not split the solid")
    return solids[0].fuse(*solids[1:])


def call(merge, shape, tol):
    started = perf_counter()
    candidate, report = merge(shape, tol)
    report["seconds"] = round(perf_counter() - started, 4)
    return candidate, report


def regression(module, tol):
    box = cq.Solid.makeBox(10, 10, 10)
    cases = {
        "PLANE": box,
        "CYLINDER": cq.Solid.makeCylinder(10, 10),
        "CONE": cq.Solid.makeCone(10, 7, 10),
        "SPHERE": cq.Solid.makeSphere(10, angleDegrees1=-90),
    }
    reports = {}
    for kind, shape in cases.items():
        split = split_and_join(shape, cq.Vector(0, 0, 1))
        result, report = call(module.merge_analytic, split, tol)
        assert report["status"] == "merged", (kind, report)
        assert report["faces_after"][kind] < report["faces_before"][kind], report
        assert Counter(face.geomType() for face in result.Faces()) == Counter(
            face.geomType() for face in shape.Faces())
        assert abs(result.Volume() - shape.Volume()) < 1e-6
        assert module.merge_analytic(result, tol)[0] is result
        reports[kind] = report

    split = split_and_join(box, cq.Vector(0, 0, 1))
    spline = cq.Shape.cast(BRepBuilderAPI_NurbsConvert(box.translate((30, 0, 0)).wrapped, True).Shape())
    mixed = cq.Compound.makeCompound([split, spline])
    spline_faces = {f for f in mixed.Faces() if f.geomType() == "BSPLINE"}
    assert spline_faces, "fixture must have actual B-spline surfaces"
    result, reports["protected_spline"] = call(module.merge_analytic, mixed, tol)
    assert reports["protected_spline"]["status"] == "merged", reports["protected_spline"]
    assert {f for f in result.Faces() if f.geomType() == "BSPLINE"} == spline_faces

    curved = (cq.Workplane("XY").moveTo(-10, -5).lineTo(10, -5).lineTo(10, 5)
              .spline([(6, 6), (0, 7), (-6, 6), (-10, 5)], includeCurrent=True)
              .close().extrude(10).val())
    same_solid = split_and_join(curved, cq.Vector(0, 0, 1))
    unsupported = {f for f in same_solid.Faces() if f.geomType() == "EXTRUSION"}
    assert unsupported
    result, reports["protected_same_solid"] = call(module.merge_analytic, same_solid, tol)
    assert reports["protected_same_solid"]["status"] == "merged"
    assert unsupported <= set(result.Faces())

    # KeepShape cannot turn coplanar but disconnected faces into one face.
    separate = cq.Compound.makeCompound([box, box.translate((30, 0, 0))])
    result, reports["disconnected"] = call(module.merge_analytic, separate, tol)
    assert result is separate and reports["disconnected"]["eligible_edges"] == 0

    # Adjacent cylinders with different radii must remain separate.
    lower = cq.Solid.makeCylinder(10, 5)
    upper = cq.Solid.makeCylinder(10 + 4 * tol, 5, cq.Vector(0, 0, 5))
    step = lower.fuse(upper)
    result, reports["different_radii"] = call(module.merge_analytic, step, tol)
    assert sum(f.geomType() == "CYLINDER" for f in result.Faces()) == 2

    # Force validator failures to exercise both rollback paths.
    closed_solids = module._closed_solids
    with patch.object(module, "_closed_solids", side_effect=[closed_solids(split), []]):
        result, reports["invalid_candidate"] = call(module.merge_analytic, split, tol)
    assert result is split and reports["invalid_candidate"]["reason"] == "output_not_closed_valid_solid"

    original_volume = cq.Solid.Volume
    with patch.object(cq.Solid, "Volume",
                      lambda self, *args, **kwargs: original_volume(self, *args, **kwargs)
                      * (1.01 if len(self.Faces()) == 6 else 1)):
        result, reports["volume_limit"] = call(module.merge_analytic, split, tol)
    assert result is split and reports["volume_limit"]["reason"] == "volume_change_above_limit"
    return reports


def inspect(model, reference, matrix, *, surface):
    if surface:
        result = evaluate(model, reference, transform=matrix, tol=Tolerance())
        return {key: result[key] for key in ("model_topology", "outcomes", "surface")}
    shape, ref = load_step(model), load_step(reference)
    ref = transform_shape(ref, matrix) if matrix is not None else ref
    tol = Tolerance()
    # The topology-only run needs trims of analytic faces, not a mesh of all splines.
    with patch.object(shape, "tessellate", return_value=([], [])):
        rows = match(extract(ref, tol), extract(shape, tol), tol)
    return {"model_topology": topology(shape), "outcomes": dict(Counter(r["status"] for r in rows)),
            "surface": None}


def compare(merge, shape, before_step, reference, matrix, scratch, tol, *, surface=True):
    result, report = call(merge, shape, tol)
    before = inspect(before_step, reference, matrix, surface=surface)
    if result is shape:
        after = before
    else:
        after_step = scratch / "after.step"
        cq.exporters.export(result, str(after_step))
        after = inspect(after_step, reference, matrix, surface=surface)
    top = after["model_topology"]
    assert top["valid"] and top["closed"], (before_step, top)
    assert abs(top["volume_mm3"] / before["model_topology"]["volume_mm3"] - 1) <= 0.001
    return {"merge": report, "before": before, "after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--sample-runs", type=Path, required=True)
    parser.add_argument("--smooth-stage", type=Path)
    parser.add_argument("--tol", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    demo = args.demo.resolve()
    sys.path.insert(0, str(demo))
    import merge_analytic

    result = {
        "baseline_commit": subprocess.check_output(["git", "-C", str(demo), "rev-parse", "HEAD"],
                                                   text=True).strip(),
        "runtime": {"python": sys.version.split()[0], "cadquery": metadata.version("cadquery"),
                    "cadquery-ocp": metadata.version("cadquery-ocp")},
        "merge_tolerance_mm": args.tol,
        "acceptance_tolerance": vars(Tolerance()),
        "regressions": regression(merge_analytic, args.tol),
        "samples": {},
        "synthetic": {},
    }
    with TemporaryDirectory(prefix="cad-merge-acceptance-") as temp:
        scratch = Path(temp)
        for name in SAMPLES:
            runs = sorted((args.sample_runs / name).glob("*/sheet.json"))
            if len(runs) != 1:
                raise ValueError(f"expected exactly one run for {name}, got {runs}")
            sheet = runs[0]
            matrix = json.loads(sheet.read_text(encoding="utf-8"))["transform_from_scan_4x4"]
            step = sheet.parent / "iter0\u0436" / "model.step"
            ref = demo / "samples" / f"{name}_truth.step"
            result["samples"][name] = compare(merge_analytic.merge_analytic, load_step(step),
                                               step, ref, matrix, scratch, args.tol)
            print(name, result["samples"][name]["merge"], flush=True)
            if name == "bracket":
                baseline = load_step(step)
                result["samples"][name]["tolerance_sweep"] = {
                    str(value): call(merge_analytic.merge_analytic, baseline, value)[1]
                    for value in (0.005, 0.01, 0.02, 0.05, 0.1)
                }
            if name == "flange" and args.smooth_stage:
                stage = args.smooth_stage
                key = "flange_smooth_stage"
                smooth = load_step(stage)
                result["samples"][key] = compare(merge_analytic.merge_analytic, smooth, stage, ref,
                                                 matrix, scratch, args.tol, surface=False)
                r45 = {f for f in smooth.Faces() if f.geomType() == "CYLINDER"
                       and abs(merge_analytic._surface(f, "CYLINDER").Radius() - 45) < 0.01}
                incidence = defaultdict(int)
                for face in r45:
                    for edge in face.Edges():
                        incidence[edge] += 1
                result["samples"][key]["r45"] = {
                    "faces": len(r45), "edges_between_r45_faces": sum(n > 1 for n in incidence.values())
                }
                print(key, result["samples"][key]["merge"], flush=True)
        for name in (*FIXTURES, *EXTRA):
            shape = (fixture if name in FIXTURES else extra_fixture)(name)
            split = split_and_join(shape, cq.Vector(0, 0, 1))
            ref, step = scratch / "reference.step", scratch / "before.step"
            cq.exporters.export(shape, str(ref))
            cq.exporters.export(split, str(step))
            result["synthetic"][name] = compare(merge_analytic.merge_analytic, load_step(step),
                                                step, ref, None, scratch, args.tol)
            print(name, result["synthetic"][name]["merge"], flush=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
