"""Compare task 21a against the standard smooth body on identical scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool


def inspect(path, probes, corners):
    from analytic_bores import coverage_share
    from nurbs import body_mesh

    shape = cq.importers.importStep(str(path)).val()
    mesh = body_mesh(shape, .03)
    _, distance, _ = trimesh.proximity.closest_point(mesh, corners)
    shells = shape.Shells()
    return {"faces": len(shape.Faces()), "valid": bool(shape.isValid()),
            "one_closed_solid": bool(len(shape.Solids()) == 1 and len(shells) == 1
                                     and BRep_Tool.IsClosed_s(shells[0].wrapped)),
            "reference_share_0_3": coverage_share(mesh, probes, .3),
            "corners_mm": distance.tolist(), "max_corner_mm": float(distance.max())}


def measure(folder, noise, reuse):
    import nurbs
    import sharp_layout
    import surface

    scan = trimesh.load(folder / f"scan-{noise}.ply", force="mesh")
    target = cq.importers.importStep(str(folder / "truth.step")).val()
    probes = trimesh.sample.sample_surface(nurbs.body_mesh(target, .03), 8000, seed=21)[0]
    corners = np.load(folder / "corners.npy")
    clean = surface.flatten_planes(surface.surface_mesh(scan))
    baseline_path = folder / f"base-{noise}.step"
    baseline_seconds = None
    if reuse and baseline_path.is_file():
        baseline = cq.importers.importStep(str(baseline_path)).val()
    else:
        start = perf_counter()
        baseline, _ = nurbs.smooth_body(clean, scan, tol=.3)
        baseline_seconds = perf_counter() - start
        if baseline is None:
            raise RuntimeError(f"{folder.name} {noise}: normal smooth body failed")
        cq.exporters.export(baseline, str(baseline_path))
        baseline = cq.importers.importStep(str(baseline_path)).val()

    start = perf_counter()
    body, audit = sharp_layout.choose(baseline, scan, clean, noise)
    elapsed = perf_counter() - start
    result = {"case": folder.name, "noise_mm": noise,
              "edge_mm_p50": float(np.median(scan.edges_unique_length)),
              "seconds_baseline": baseline_seconds, "seconds_trial": elapsed, "decision": audit,
              "baseline": inspect(baseline_path, probes, corners)}
    if body is baseline:
        trial, candidate_audit = sharp_layout.build(scan, noise)
        result["trial_build"] = candidate_audit
        if trial is None:
            result["passed"] = False
            return result
    else:
        trial = body
    trial_path = folder / f"sharp-{noise}.step"
    cq.exporters.export(trial, str(trial_path))
    inspected = inspect(trial_path, probes, corners)
    result["trial"] = inspected
    result["passed"] = bool(audit["status"] == "accepted" and inspected["valid"]
                            and inspected["one_closed_solid"] and inspected["reference_share_0_3"] >= .99
                            and inspected["max_corner_mm"] <= .3
                            and inspected["faces"] <= int(1.15 * result["baseline"]["faces"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--case", nargs="+", choices=("plate-1.4", "plate-2.3", "button-1.4", "button-2.3"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.demo.resolve()))
    if not args.reuse:
        from sharp_fixtures import generate

        generate(args.work)
    cases = args.case or ("plate-1.4", "plate-2.3", "button-1.4", "button-2.3")
    rows = []
    for name in cases:
        for noise in (.05, .2):
            row = measure(args.work / name, noise, args.reuse)
            rows.append(row)
            trial = row.get("trial", {})
            print(name, noise, row["decision"]["status"], trial.get("reference_share_0_3"),
                  trial.get("max_corner_mm"), row["passed"], flush=True)
    (args.work / "report.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n", encoding="utf8")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
