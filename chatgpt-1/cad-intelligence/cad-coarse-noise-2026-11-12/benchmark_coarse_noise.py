"""Prepare variable-density CAD scans and compare noise estimators."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
for folder in ("cad-regions", "cad-region-layout", "cad-scan-noise", "cad-coarse-bores"):
    sys.path.insert(0, str(HERE.parent / f"{folder}-2026-11-12"))

from benchmark_layout import EXTRA, extra_fixture
from benchmark_regions import FIXTURES, fixture
import scan_noise as previous


def variable_mesh(shape, path):
    import cadquery as cq
    import gmsh

    cq.exporters.export(shape, str(path))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 4.55)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.add("coarse")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
        field = gmsh.model.mesh.field
        coarse = field.add("Box")
        for key, value in dict(VIn=4.55, VOut=2.0, XMin=x0-2, XMax=(x0+x1)*.5,
                               YMin=y0-2, YMax=y1+2, ZMin=z0-2, ZMax=z1+2, Thickness=2.).items():
            field.setNumber(coarse, key, value)
        field.setAsBackgroundMesh(coarse)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, nodes = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    order = np.argsort(tags)
    faces = order[np.searchsorted(tags[order], nodes)].reshape(-1, 3)
    mesh = trimesh.Trimesh(xyz.reshape(-1, 3), faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def mesh_for(name, work, reuse):
    from scan_analysis import align

    path = work / f"{name}-clean.ply"
    if reuse and path.is_file():
        return trimesh.load(path, force="mesh")
    if name == "coarse_bracket":
        from benchmark_coarse_bores import fixture as bracket_fixture
        shape, _ = bracket_fixture()
    else:
        shape = (fixture if name in FIXTURES else extra_fixture)(name)
    mesh = variable_mesh(shape, work / f"{name}-truth.step")
    if name != "coarse_bracket":
        mesh, _ = align(mesh)
    mesh.export(path)
    return trimesh.load(path, force="mesh")


def previous_support_rejection(baseline, result, legacy):
    if baseline.method == "support_plane":
        return "verified"
    if result.support_faces < 30:
        return "too_few_legacy_faces"
    if baseline.support_area_share < .025:
        return "legacy_area_below_2_5_percent"
    if baseline.support_curvature_gain is None:
        return "global_support_fit_rejected"
    if baseline.support_curvature_gain > .2:
        return "global_quadratic_fits_better"
    if baseline.support_patches < 12:
        return "too_few_local_support_patches"
    if baseline.support_flat_share is not None and baseline.support_flat_share < .5:
        return "too_few_flat_support_patches"
    if baseline.vertex_sigma_mm is not None and legacy > max(.01, 1.5 * baseline.vertex_sigma_mm):
        return "legacy_height_spread_exceeds_local_noise"
    return "unverified"


def clean_distances(mesh):
    from surface import surface_mesh

    clean = surface_mesh(mesh)
    rng = np.random.default_rng(200115)
    vertices = mesh.vertices[rng.choice(len(mesh.vertices), min(4096, len(mesh.vertices)), replace=False)]
    distance = np.concatenate([trimesh.proximity.closest_point(clean, vertices[i:i+128])[1]
                               for i in range(0, len(vertices), 128)])
    return {"vertices": len(vertices), "clean_watertight": bool(clean.is_watertight),
            "p50_mm": float(np.median(distance)),
            "p95_mm": float(np.quantile(distance, .95))}


def compare(mesh, samples):
    import scan_analysis
    import coarse_noise

    legacy = scan_analysis.noise_estimate(mesh)
    baseline = previous.estimate(mesh, samples=samples)
    started = perf_counter()
    result = coarse_noise.estimate(mesh, samples=samples)
    seconds = perf_counter() - started
    return {"legacy_mm": legacy, "task18": asdict(baseline), "task20a": asdict(result),
            "task18_support_status": previous_support_rejection(baseline, result, legacy),
            "seconds": seconds, "faces": len(mesh.faces),
            "edge_mm_p50_p90": np.quantile(mesh.edges_unique_length, [.5, .9]).tolist()}


def controls(samples):
    import coarse_noise
    from benchmark_scan_noise import large_shell

    # A small actual plane amid much more curved area tests the old 2.5% gate.
    x, y = np.meshgrid(np.linspace(-7, 7, 13), np.linspace(-7, 7, 13))
    xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    grid = np.arange(len(xyz)).reshape(x.shape)
    a, b, c, d = (v.ravel() for v in (grid[:-1, :-1], grid[:-1, 1:],
                                      grid[1:, :-1], grid[1:, 1:]))
    faces = np.vstack((np.column_stack((a, c, b)), np.column_stack((b, c, d))))
    xyz += np.random.default_rng(200121).normal(0, .05, xyz.shape)
    plane = trimesh.Trimesh(xyz, faces, process=False)
    shell = trimesh.creation.icosphere(subdivisions=4, radius=90).apply_translation((0, 0, 110))
    combined = trimesh.util.concatenate((plane, shell))
    support = compare(combined, samples)
    support["pass"] = (support["task20a"]["support_area_share"] < .004
                       and support["task18"]["method"] != "support_plane"
                       and support["task20a"]["method"] == "support_plane"
                       and support["task20a"]["noise_mm"] == support["legacy_mm"])

    large = []
    for sigma in (.05, .2):
        mesh = large_shell(sigma)
        start = perf_counter()
        result = coarse_noise.estimate(mesh, samples=samples)
        row = {"faces": len(mesh.faces), "sigma_mm": sigma, "seconds": perf_counter() - start,
               "result": asdict(result),
               "pass": result.noise_mm is not None and abs(result.noise_mm - sigma) <= .2 * sigma}
        large.append(row)
        print("million", sigma, result.noise_mm, row["seconds"], flush=True)
    return {"small_support": support, "large": large}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = (*FIXTURES, *EXTRA, "coarse_bracket")
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=cases, nargs="+", default=cases)
    parser.add_argument("--scan", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=200112)
    parser.add_argument("--clean-p95", action="store_true", help="measure raw vertices to cleaned mesh on --scan inputs")
    parser.add_argument("--controls", action="store_true", help="measure small support and million-triangle controls")
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()
    if args.seeds < 1 or args.samples < 32:
        parser.error("--seeds must be positive and --samples at least 32")
    if args.skip_synthetic and not (args.scan or args.controls):
        parser.error("--skip-synthetic requires --scan or --controls")
    sys.path.insert(0, str(args.demo.resolve()))
    import scan_analysis

    args.work.mkdir(parents=True, exist_ok=True)
    synthetic, scans = [], []
    if not args.skip_synthetic:
        for name in args.case:
            base = mesh_for(name, args.work, args.reuse)
            for sigma in (0., .05, .2):
                for trial in range(args.seeds):
                    tilt = (6., 7.5, 9.)[trial % 3]
                    mesh = base.copy()
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(tilt), (1, 2, 0)))
                    mesh.vertices += np.random.default_rng(args.seed + trial).normal(0., sigma, mesh.vertices.shape)
                    row = compare(mesh, args.samples)
                    estimate = row["task20a"]["noise_mm"]
                    accuracy = estimate is not None and abs(estimate - sigma) <= (.02 if sigma == 0 else .2 * sigma)
                    row.update(name=name, sigma_mm=sigma, trial=trial, tilt_deg=tilt, accuracy=accuracy)
                    synthetic.append(row)
                    print(name, sigma, tilt, row["task18"]["vertex_sigma_mm"], estimate, accuracy, flush=True)

    for name, file in args.scan:
        mesh = scan_analysis.load_mesh(file)
        if not args.aligned:
            mesh, _ = scan_analysis.align(mesh)
        row = compare(mesh, args.samples)
        row["name"] = name
        if args.clean_p95:
            try:
                row["raw_to_clean"] = clean_distances(mesh)
            except (ValueError, RuntimeError, MemoryError) as error:
                row["raw_to_clean"] = {"error": f"{type(error).__name__}: {error}"}
        scans.append(row)
        print(name, row["legacy_mm"], row["task18"]["noise_mm"], row["task20a"]["noise_mm"], flush=True)
    data = {"synthetic": synthetic, "scans": scans, "samples": args.samples, "seed": args.seed}
    if args.controls:
        data["controls"] = controls(args.samples)
    report = args.output or args.work / "results.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    passed = all(row["accuracy"] for row in synthetic if row["sigma_mm"])
    if args.controls:
        checks = data["controls"]
        passed &= checks["small_support"]["pass"] and all(row["pass"] for row in checks["large"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
