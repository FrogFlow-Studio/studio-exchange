"""Validate the task 22 estimate on task 20a fixtures and local scans."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import struct
import sys
from time import perf_counter

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad-coarse-noise-2026-11-12"))
from benchmark_coarse_noise import EXTRA, FIXTURES, mesh_for
from scan_uncertainty import binary_stl_roundoff, estimate


def read_roundoff(path, mesh):
    if path.suffix.lower() != ".stl":
        return None
    with path.open("rb") as stream:
        header = stream.read(84)
    if len(header) != 84:
        return None
    count = struct.unpack_from("<I", header, 80)[0]
    return binary_stl_roundoff(mesh) if path.stat().st_size == 84 + 50 * count else None


def measure(mesh, samples, roundoff=None):
    start = perf_counter()
    row = asdict(estimate(mesh, samples=samples, roundoff_mm=roundoff))
    row["seconds"] = perf_counter() - start
    row["faces"] = len(mesh.faces)
    return row


def controls(work, samples):
    shell = trimesh.creation.icosphere(subdivisions=3, radius=20)
    cases = {"binary_sphere": shell, "binary_planar_box": trimesh.creation.box(extents=(80, 40, 8))}
    rows = []
    for name, shape in cases.items():
        path = work / f"{name}.stl"
        shape.export(path)
        mesh = trimesh.load(path, force="mesh")
        row = dict(measure(mesh, samples, read_roundoff(path, mesh)), name=name)
        assert row["working_mm"] > 0 and row["working_mm"] >= row["floor_mm"]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", nargs="+", choices=(*FIXTURES, *EXTRA, "coarse_bracket"))
    parser.add_argument("--scan", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()
    if args.samples < 32:
        parser.error("--samples must be at least 32")
    sys.path.insert(0, str(args.demo.resolve()))
    import scan_analysis

    args.work.mkdir(parents=True, exist_ok=True)
    synthetic, scans = [], []
    if not args.skip_synthetic:
        names = args.case or (*FIXTURES, *EXTRA, "coarse_bracket")
        fixtures = args.fixture_root or args.work
        fixtures.mkdir(parents=True, exist_ok=True)
        for name in names:
            base = mesh_for(name, fixtures, reuse=True)
            for seed in (200112, 200212):
                for trial, tilt in enumerate((6., 7.5, 9.)):
                    for target in (.05, .2):
                        mesh = base.copy()
                        pose = trimesh.transformations.rotation_matrix(np.radians(tilt), (1, 2, 0))
                        mesh.apply_transform(pose)
                        mesh.vertices += np.random.default_rng(seed + trial).normal(0, target, mesh.vertices.shape)
                        row = dict(measure(mesh, args.samples), name=name, target_mm=target, tilt_deg=tilt, seed=seed+trial)
                        row["passed"] = (row["working_mm"] is not None
                                         and abs(row["working_mm"] - target) <= .2 * target)
                        synthetic.append(row)
            rows = [row for row in synthetic if row["name"] == name]
            print(name, sum(row["passed"] for row in rows), len(rows), flush=True)

    for name, filename in args.scan:
        path = Path(filename)
        mesh = scan_analysis.load_mesh(str(path))
        roundoff = read_roundoff(path, mesh)
        if not args.aligned:
            mesh, _ = scan_analysis.align(mesh)
        row = dict(measure(mesh, args.samples, roundoff), name=name,
                   legacy_mm=scan_analysis.noise_estimate(mesh))
        scans.append(row)
        print(name, row["measured_mm"], row["working_mm"], row["limiting_component"], flush=True)
    data = {"synthetic": synthetic, "scans": scans, "controls": controls(args.work, args.samples)}
    output = args.output or args.work / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if (all(row["passed"] for row in synthetic) and all(
        row["working_mm"] is not None and row["working_mm"] >= row["floor_mm"] for row in scans)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
