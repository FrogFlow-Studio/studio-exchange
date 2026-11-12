"""Evaluate task 18 on a uniform, labelled synthetic suite and optional studio scans."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import trimesh

from scan_noise import estimate

HERE = Path(__file__).resolve().parent
for directory in ("cad-region-layout", "cad-regions"):
    sys.path.insert(0, str(HERE.parent / f"{directory}-2026-11-12"))

from benchmark_layout import EXTRA, extra_fixture
from benchmark_regions import FIXTURES, SAMPLES, fixture

CURVED = {"curved_cover", "toroidal_ring", "full_sphere"}
LEVELS = (0., .05, .2)


def uniform_mesh(shape, target_mm):
    import cadquery as cq
    import gmsh

    with TemporaryDirectory() as temp:
        path = Path(temp) / "fixture.step"
        cq.exporters.export(shape, str(path))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.option.setNumber("Mesh.RandomSeed", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMin", target_mm)
            gmsh.option.setNumber("Mesh.MeshSizeMax", target_mm)
            gmsh.model.add("fixture")
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.removeDuplicateNodes()
            tags, coords, _ = gmsh.model.mesh.getNodes()
            _, nodes = gmsh.model.mesh.getElementsByType(2)
        finally:
            gmsh.finalize()
    order = tags.argsort()
    faces = order[np.searchsorted(tags[order], nodes)].reshape(-1, 3)
    mesh = trimesh.Trimesh(coords.reshape(-1, 3), faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def large_shell(noise):
    # Open double-curved scan with 1,000,000 triangles and roughly 1 mm spacing.
    x, y = np.meshgrid(np.linspace(-500., 500., 1001), np.linspace(-250., 250., 501))
    z = 24 * np.cos(x / 310) * np.cos(y / 160) + .00007 * x * y
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    vertices += np.random.default_rng(181120).normal(0., noise, vertices.shape)
    grid = np.arange(len(vertices)).reshape(x.shape)
    a, b, c, d = (v.ravel() for v in (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]))
    faces = np.vstack((np.column_stack((a, c, b)), np.column_stack((b, c, d))))
    return trimesh.Trimesh(vertices, faces, process=False)


def measure(mesh, *, samples):
    import scan_analysis

    started = perf_counter()
    result = asdict(estimate(mesh, samples=samples))
    result["seconds"] = perf_counter() - started
    result["legacy_mm"] = scan_analysis.noise_estimate(mesh)
    result["watertight"] = bool(mesh.is_watertight)
    result["faces"] = len(mesh.faces)
    result["vertices"] = len(mesh.vertices)
    return result


def accuracy(row, level):
    value = row["vertex_sigma_mm"]
    return value is not None and abs(value - level) <= (.012 if level == 0 else .20 * level)


def legacy_match(row):
    return row["method"] == "support_plane" and abs(row["noise_mm"] - row["legacy_mm"]) <= .2 * row["legacy_mm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=(*SAMPLES, *FIXTURES, *EXTRA, "full_sphere"), default=(*SAMPLES, *FIXTURES, *EXTRA, "full_sphere"))
    parser.add_argument("--scan", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    parser.add_argument("--expect-support", action="append", default=[], metavar="NAME")
    parser.add_argument("--skip-suite", action="store_true", help="evaluate only explicitly provided scans")
    parser.add_argument("--aligned", action="store_true", help="provided scans are in the 0k aligned frame")
    parser.add_argument("--million", action="store_true", help="also measure a one-million-triangle shell")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=1.)
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()
    if args.seeds < 1 or args.spacing <= 0:
        parser.error("--seeds and --spacing must be positive")
    names = {name for name, _ in args.scan}
    if set(args.expect_support) - names:
        parser.error("--expect-support must refer to a named --scan")
    if args.skip_suite and not (args.scan or args.million):
        parser.error("--skip-suite requires --scan or --million")
    sys.path.insert(0, str(args.demo.resolve()))
    import scan_analysis

    known, unknown, large = [], [], []
    if not args.skip_suite:
        for name in args.cases:
            if name in SAMPLES:
                mesh = scan_analysis.load_mesh(str(args.demo / "samples" / f"{name}_scan.stl"))
                aligned, _ = scan_analysis.align(mesh)
                row = dict(measure(aligned, samples=args.samples), name=name, source="repository_sample")
                row["within_legacy_20pct"] = legacy_match(row)
                unknown.append(row)
                print(name, row["method"], row["legacy_mm"], row["noise_mm"], flush=True)
                continue
            if name == "full_sphere":
                import cadquery as cq

                shape = cq.Solid.makeSphere(21, angleDegrees1=-90)
            else:
                shape = (fixture if name in FIXTURES else extra_fixture)(name)
            base = uniform_mesh(shape, args.spacing)
            for level in LEVELS:
                for trial in range(args.seeds):
                    rng = np.random.default_rng(181112 + trial)
                    mesh = base.copy()
                    mesh.vertices += rng.normal(0., level, mesh.vertices.shape)
                    aligned, _ = scan_analysis.align(mesh)
                    row = dict(measure(aligned, samples=args.samples), name=name, sigma_mm=level, trial=trial)
                    row["vertex_accuracy"] = accuracy(row, level)
                    if name in CURVED:
                        row["curved_accuracy"] = row["vertex_accuracy"] and row["method"] == "local_quadratic"
                    known.append(row)
                print(name, level, row["method"], round(row["legacy_mm"], 4),
                      round(row["vertex_sigma_mm"], 4) if row["vertex_sigma_mm"] is not None else None, flush=True)
    for name, path in args.scan:
        mesh = scan_analysis.load_mesh(path)
        if not args.aligned:
            mesh, _ = scan_analysis.align(mesh)
        row = dict(measure(mesh, samples=args.samples), name=name, source="local_scan")
        if name in args.expect_support:
            row["within_legacy_20pct"] = legacy_match(row)
        unknown.append(row)
        print(name, row["method"], row["legacy_mm"], row["noise_mm"], flush=True)
    if args.million:
        for level in (.05, .2):
            row = dict(measure(large_shell(level), samples=args.samples), name="million_shell", sigma_mm=level)
            row["vertex_accuracy"] = accuracy(row, level)
            row["seconds_at_million_pass"] = row["seconds"] <= 5.
            large.append(row)
            print(row["name"], level, row["noise_mm"], f"{row['seconds']:.2f} s", flush=True)

    result = {"environment": {"platform": platform.system(), "python": platform.python_version()},
              "spacing_mm": args.spacing, "samples": args.samples, "noise_seeds": args.seeds,
              "synthetic": known,
              "reference_unknown": unknown, "million_faces": large}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    passes = [r["curved_accuracy"] for r in known if "curved_accuracy" in r]
    passes.extend(r["within_legacy_20pct"] for r in unknown if "within_legacy_20pct" in r)
    passes.extend(r["vertex_accuracy"] and r["seconds_at_million_pass"] for r in large)
    return 0 if all(passes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
