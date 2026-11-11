"""Compare task 9 segmentation and the noise-aware patch on matched CAD scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

import cadquery as cq
import numpy as np
import trimesh

TASK9 = Path(__file__).resolve().parent.parent / "cad-regions-2026-11-12"
sys.path.insert(0, str(TASK9))
from benchmark_regions import FIXTURES, SAMPLES, fixture, measure, scanned, tessellate, truth_ids


def task9_segment(repo):
    source = subprocess.check_output(
        ["git", "-C", str(repo), "show", "da7feb1:claude/cad-demo/regions.py"])
    with tempfile.TemporaryDirectory(prefix="regions-task9-") as temp:
        root = Path(temp)
        path = root / "claude/cad-demo/regions.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(source)
        subprocess.run(["git", "apply", str(TASK9 / "regions.patch")], cwd=root, check=True)
        source = path.read_bytes()
    baseline = types.ModuleType("regions_task9")
    exec(compile(source, "regions_task9.py", "exec"), baseline.__dict__)
    return baseline.segment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=(*SAMPLES, *FIXTURES), default=(*SAMPLES, *FIXTURES))
    p.add_argument("--noise", nargs="+", type=float, default=(0.05, 0.2, 0.4, 0.7))
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--tol", type=float, default=0.3)
    p.add_argument("--without-raw", action="store_true",
                   help="Run the compatible segment(mesh) call without raw scan samples")
    args = p.parse_args()
    demo = args.repo.resolve() / "claude/cad-demo"
    sys.path.insert(0, str(demo))
    import regions
    import surface

    baseline = task9_segment(args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.cases:
        existing = name in SAMPLES
        shape = (cq.importers.importStep(str(demo / "samples" / f"{name}_truth.step")).val()
                 if existing else fixture(name))
        ref, face_ids, truth = tessellate(shape, 0.015, 0.1)
        for noise in ([None] if existing else args.noise):
            start = time.perf_counter()
            print(name, noise, "scan / cleanup", flush=True)
            if existing:
                scan = trimesh.load(demo / "samples" / f"{name}_scan.stl", force="mesh")
            else:
                rng = np.random.default_rng(args.seed + FIXTURES.index(name))
                scan = scanned(shape, rng, noise)
            mesh = surface.surface_mesh(scan)
            print(name, noise, "reference projection", len(mesh.faces),
                  round(time.perf_counter() - start, 2), flush=True)
            ids = truth_ids(mesh, ref, face_ids, args.tol)

            def candidate(clean, tol):
                return regions.segment(clean, tol=tol, scan=None if args.without_raw else scan)

            for method, segment in (("task9", baseline), ("task10", candidate)):
                print(name, noise, method, "segment", flush=True)
                row = measure(mesh, ids, truth, segment, args.tol)
                row.update(case=name, noise=noise, method=method, seed=args.seed, tol=args.tol,
                           raw_scan=not args.without_raw and method == "task10")
                rows.append(row)
                args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
                area = sum(t["matched_area"] for t in row["truth"])
                visible = sum(t["visible_area"] for t in row["truth"] if t["kind"] != "free")
                recall = round(100 * area / visible, 1) if visible else None
                false_share = round(100 * sum(row["false_area"].values()) / row["mesh_area"], 1)
                print(name, noise, method, "recall", recall, "false area %", false_share,
                      "seconds", round(row["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
