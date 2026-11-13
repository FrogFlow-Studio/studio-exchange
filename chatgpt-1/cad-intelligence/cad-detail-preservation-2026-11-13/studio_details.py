"""Measure the combined tasks 42–43 experiment against a local aligned STEP."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cadquery as cq
import numpy as np
import trimesh

from compare import compare
from defects import analyze
from detail_defects import find_details
import fairing
import idealize
from nurbs import body_mesh
import planar_brep
from preserve import curvature, relief
import regions
import scan_uncertainty


def score(scan, body):
    mesh = body_mesh(body, .04)
    mesh.merge_vertices()
    coverage = compare(scan, mesh, .3, n=12000, seed=42)
    details = find_details(mesh, scan)
    areas = Counter()
    counts = Counter()
    for row in details + analyze(mesh, scan, .3):
        areas[row["type"]] += row["area_mm2"]
        counts[row["type"]] += 1
    return {"coverage": [coverage["within_tol_share"], 1-coverage["added_surface_share"]],
            "valid": bool(body.isValid() and len(body.Solids()) == 1),
            "defect_count": dict(counts), "defect_area_mm2": dict(areas)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scan = trimesh.load(args.scan, force="mesh")
    stock = cq.importers.importStep(str(args.baseline)).val()
    before = score(scan, stock)
    model = body_mesh(stock, .04)
    model.merge_vertices()
    started = perf_counter()
    noise = scan_uncertainty.estimate(scan).working_mm
    fit = idealize.fit_tol(.3, noise)
    step = np.median(scan.edges_unique_length)
    seg = regions.segment(scan, tol=fit, min_area=max(6., 4*step**2*.43), scan=scan)
    changed, released = curvature(model, scan, seg, find_details(model, scan))
    changed, patches = relief(scan, changed, noise)
    report = {"triangles": len(scan.faces), "noise_mm": noise,
              "curvature_faces_released": len(released), "relief_patches": len(patches),
              "relief_area_mm2": sum(row["area_mm2"] for row in patches), "before": before}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if len(released) or patches:
        mesh, info = idealize.idealize(scan, noise=noise, seg=changed)
        vertices, faces, label = planar_brep.collapse_degenerate(
            mesh.vertices, mesh.faces, info["label"], .001)
        vertices, _ = fairing.fair(vertices, faces, label, info["regions"], .8*fit)
        body, audit = planar_brep.build(vertices, faces, label, info["regions"], smooth_tol=fit*2/3,
                                       smooth_edge_tol=fit, smooth_budget=180.)
        with TemporaryDirectory() as temp:
            path = Path(temp)/"candidate.step"
            cq.exporters.export(body, str(path))
            report["after"] = score(scan, cq.importers.importStep(str(path)).val())
        report["free_edges"] = audit["free_edges"]
        report["planar_area_share"] = audit["planar_area_share"]
    report["seconds"] = perf_counter()-started
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
