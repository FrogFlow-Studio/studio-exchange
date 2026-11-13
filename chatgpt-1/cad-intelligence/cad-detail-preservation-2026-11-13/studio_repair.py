"""Measure the cached geodesic repair on a local aligned studio scan."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

import fairing
import idealize
import planar_brep
import regions
import scan_uncertainty
from patch_cache import CachedBuilder
from repair_geodesic import repair_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scan = trimesh.load(args.scan, force="mesh")
    if not scan.is_watertight:
        parser.error("this benchmark requires a closed scan")
    noise = scan_uncertainty.estimate(scan).working_mm
    fit = idealize.fit_tol(.3, noise)
    step = np.median(scan.edges_unique_length)
    seg = regions.segment(scan, tol=fit, min_area=max(6., 4*step**2*.43), scan=scan)
    if regions.summary(seg)["plane"] >= .3:
        from straight_plus import augment

        seg, _ = augment(scan, .3, fit/2, seg=seg)
    mesh, info = idealize.idealize(scan, noise=noise, seg=seg)
    vertices, faces, label = planar_brep.collapse_degenerate(
        mesh.vertices, mesh.faces, info["label"], .001)
    vertices, _ = fairing.fair(vertices, faces, label, info["regions"], .8*fit)
    print(f"Stage ready: {len(scan.faces)} scan triangles", flush=True)
    cache = CachedBuilder()

    def build(*state, smooth_budget=180., **_):
        return cache(*state, smooth_tol=fit*2/3, smooth_edge_tol=fit, smooth_budget=smooth_budget)

    def progress(stage, seconds, value):
        print(f"{stage}: {seconds:.1f} s, {value:.2f}", flush=True)

    _, report = repair_build(vertices, faces, label, info["regions"], scan, fit, build=build, overhead_limit=.5,
                             progress=progress)
    report["scan_triangles"] = len(scan.faces)
    report["repair_overhead_ratio"] = report["repair_overhead_seconds"] / report["baseline_build_seconds"]
    report["fold_area_mm2"] = sum(r["area_mm2"] for r in report["defects"] if r["type"] == "смятая поверхность")
    report["lost_area_mm2"] = sum(r["area_mm2"] for r in report["defects"] if r["type"] == "потерянный элемент")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
