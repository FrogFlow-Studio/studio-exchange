"""Benchmark the closed, region-aware layout against the current NURBS layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import fast_simplification
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad-regions-2026-11-12"))
from benchmark_regions import FIXTURES, SAMPLES, fixture, scanned

EXTRA = ("slotted_block", "counterbored_plate", "filleted_step")
CASES = (*SAMPLES, *FIXTURES, *EXTRA)


def extra_fixture(name):
    import cadquery as cq

    if name == "slotted_block":
        return (cq.Workplane("XY").box(62, 42, 12)
                .faces(">Z").workplane().slot2D(36, 10).cutThruAll().val())
    if name == "counterbored_plate":
        return (cq.Workplane("XY").box(64, 44, 10)
                .faces(">Z").workplane().pushPoints([(-14, 0), (14, 0)])
                .cboreHole(7, 12, 4).val())
    if name == "filleted_step":
        base = cq.Workplane("XY").box(56, 36, 8).edges("|Z").fillet(3)
        step = cq.Workplane("XY", origin=(5, 0, 4)).box(25, 22, 9, centered=(True, True, False))
        return base.union(step).val()
    raise ValueError(name)


def load_mesh(demo, name, seed):
    import surface

    if name in SAMPLES:
        scan = trimesh.load(demo / "samples" / f"{name}_scan.stl", force="mesh")
    else:
        shape = fixture(name) if name in FIXTURES else extra_fixture(name)
        index = (*FIXTURES, *EXTRA).index(name)
        scan = scanned(shape, np.random.default_rng(seed + index), 0.05)
    return surface.surface_mesh(scan), scan


def closest(mesh, points):
    # Bound temporary arrays even when CAD triangles have wide bounding boxes.
    rows = [trimesh.proximity.closest_point(mesh, points[i:i + 128])
            for i in range(0, len(points), 128)]
    return tuple(np.concatenate([row[j] for row in rows]) for j in range(3))


def probe_points(mesh):
    weights = np.array([[1/3, 1/3, 1/3], [0.7, 0.15, 0.15], [0.15, 0.7, 0.15],
                        [0.15, 0.15, 0.7], [0.45, 0.45, 0.1], [0.45, 0.1, 0.45],
                        [0.1, 0.45, 0.45]])
    return np.einsum("ij,fjk->fik", weights, mesh.triangles)


def pair_probes(reference, label, area, pairs):
    classes = label[reference][pairs]
    weights = np.broadcast_to(area[pairs, None] / reference.shape[1], classes.shape).reshape(len(pairs), -1)
    classes = classes.reshape(len(pairs), -1)
    best = np.zeros(len(pairs))
    for j in range(classes.shape[1]):
        best = np.maximum(best, np.sum((classes == classes[:, j, None]) * weights, axis=1))
    return best / weights.sum(axis=1)


def score(mesh, labels, coarse, coarse_labels, elapsed, target, samples):
    import nurbs

    inherited = coarse_labels is not None
    out = dict(triangles=len(coarse.faces), closed=bool(coarse.is_watertight),
               winding=bool(coarse.is_winding_consistent), euler=int(coarse.euler_number),
               coarse_seconds=elapsed, pairing_seconds=None, pair_same=None,
               pair_convex=None, pair_projected_pure=None, pair_projected_80=None, agreement=None,
               reverse_agreement=None, p95=None, maximum=None)
    if not len(coarse.faces):
        return out
    points, idx = trimesh.sample.sample_surface(coarse, samples, seed=481)
    back, source_idx = trimesh.sample.sample_surface(mesh, samples, seed=482)
    _, d_out, i_out = closest(mesh, points)
    _, d_back, i_back = closest(coarse, back)
    distances = np.r_[d_out, d_back]
    out.update(p95=float(np.quantile(distances, 0.95)), maximum=float(distances.max()))
    if coarse_labels is None:
        _, _, face_ids = closest(mesh, coarse.triangles_center)
        coarse_labels = labels[face_ids]
    out["agreement"] = float(np.mean(coarse_labels[idx] == labels[i_out]))
    out["reverse_agreement"] = float(np.mean(labels[source_idx] == coarse_labels[i_back]))
    # Avoid launching the cubic matching step on a failed, oversized baseline.
    if not out["closed"] or len(coarse.faces) > max(2 * target, 3000):
        return out
    start = time.perf_counter()
    pairing = nurbs._paired_quads(coarse, coarse_labels if inherited else None,
                                 return_pairs=True)
    out["pairing_seconds"] = time.perf_counter() - start
    if pairing is None:
        return out
    quads, pairs = pairing
    out['pair_convex'] = float(np.mean([q[-1] > 0 for q in quads]))
    pairs = np.asarray(pairs)
    out["pair_same"] = float(np.mean(coarse_labels[pairs[:, 0]] == coarse_labels[pairs[:, 1]]))
    probes = probe_points(coarse)
    _, _, reference = closest(mesh, probes.reshape(-1, 3))
    purity = pair_probes(reference.reshape(probes.shape[:2]), labels, coarse.area_faces, pairs)
    out["pair_projected_pure"] = float(np.mean(purity >= 1 - 1e-12))
    out["pair_projected_80"] = float(np.mean(purity >= 0.8 - 1e-12))
    return out


def measure(mesh, label, target, samples):
    import region_layout
    import surface

    rows = {}
    start = time.perf_counter()
    if len(mesh.faces) > target:
        v, f = fast_simplification.simplify(mesh.vertices, mesh.faces,
                                            1 - target / len(mesh.faces))
        base = surface.repair(trimesh.Trimesh(v, f, process=True))
    else:
        base = mesh
    rows["baseline"] = score(mesh, label, base, None, time.perf_counter() - start, target, samples)
    start = time.perf_counter()
    coarse, coarse_label = region_layout.coarsen_by_regions(mesh, label, target)
    rows["regions"] = score(mesh, label, coarse, coarse_label, time.perf_counter() - start, target, samples)
    rows["regions"].update(coarse.metadata.get("region_layout", {}))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    p.add_argument("--triangles", type=int, default=1500)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20261112)
    args = p.parse_args()
    demo = args.repo.resolve() / "claude/cad-demo"
    sys.path.insert(0, str(demo))
    import regions

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.cases:
        print(name, "cleanup / segmentation", flush=True)
        mesh, scan = load_mesh(demo, name, args.seed)
        label = regions.segment(mesh, scan=scan)["label"]
        print(name, "coarsening / pairing", len(mesh.faces), flush=True)
        row = dict(case=name, source_faces=len(mesh.faces), source_euler=int(mesh.euler_number),
                   source_closed=bool(mesh.is_watertight), seed=args.seed,
                   **measure(mesh, label, args.triangles, args.samples))
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(name, row["baseline"]["triangles"], row["regions"]["triangles"],
              row["regions"]["pair_same"], flush=True)


if __name__ == "__main__":
    main()
