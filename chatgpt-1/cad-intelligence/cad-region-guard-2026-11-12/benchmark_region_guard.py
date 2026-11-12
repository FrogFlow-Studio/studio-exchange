"""A/B one-stage diagnosis for task 19. Private scans are read locally."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import cadquery as cq
import numpy as np
from scipy.spatial import cKDTree
import trimesh

HERE = Path(__file__).resolve().parent
for name in ("cad-analytic-bores", "cad-smooth-acceptance", "cad-acceptance"):
    sys.path.insert(0, str(HERE.parent / f"{name}-2026-11-12"))

from benchmark_bores import uniform_input
from benchmark_smooth import CASES, analytic_features, generate_input
from geometry_acceptance import Tolerance, load_step, match, topology, transform_shape

MODES = ("off", "stock", "layout_only", "projected_splines", "guarded")


def _segment_distance(points, segments):
    if not len(segments):
        return np.full(len(points), np.inf)
    tree = cKDTree(segments.mean(axis=1))
    idx = tree.query(points, k=min(8, len(segments)))[1].reshape(len(points), -1)
    a, b = segments[idx, 0], segments[idx, 1]
    delta = b - a
    fraction = np.clip(np.sum((points[:, None] - a) * delta, axis=-1)
                       / np.maximum(np.sum(delta * delta, axis=-1), 1e-20), 0, 1)
    return np.linalg.norm(points[:, None] - a - fraction[..., None] * delta, axis=-1).min(axis=1)


def localize(clean, labels, probes, before, after, tol):
    lost = np.flatnonzero((before <= tol) & (after > tol))
    gained = np.flatnonzero((before > tol) & (after <= tol))
    result = {"lost": len(lost), "recovered": len(gained), "examples": []}
    if not len(lost):
        return result
    pts = probes[lost]
    adj = clean.face_adjacency
    segments = clean.vertices[clean.face_adjacency_edges]
    sharp = clean.face_adjacency_angles >= np.radians(30)
    boundary = labels[adj[:, 0]] != labels[adj[:, 1]] if labels is not None else np.zeros(len(adj), bool)
    sharp_distance = _segment_distance(pts, segments[sharp])
    boundary_distance = _segment_distance(pts, segments[boundary])
    # Opposite normals on nearby triangles are evidence of a possible thin wall.
    tree = cKDTree(clean.triangles_center)
    _, idx = tree.query(pts, k=min(16, len(clean.faces)))
    nearest = idx[:, 0]
    normals = clean.face_normals[nearest]
    vec = clean.triangles_center[idx] - pts[:, None]
    lengths = np.linalg.norm(vec, axis=-1)
    opposed = np.einsum("ij,ikj->ik", normals, clean.face_normals[idx]) < -.7
    along = np.abs(np.einsum("ij,ikj->ik", normals, vec)) >= .5 * lengths
    thin = np.any(opposed & along & (lengths <= 4 * tol), axis=1)
    flags = {"region_boundary": boundary_distance <= 2 * tol,
             "sharp_edge": sharp_distance <= 2 * tol, "opposing_normals": thin}
    result["near"] = {key: int(np.count_nonzero(values)) for key, values in flags.items()}
    for i in np.argsort((after - before)[lost])[-10:][::-1]:
        j = lost[i]
        result["examples"].append({"xyz_mm": pts[i].tolist(), "before_mm": float(before[j]),
                                   "after_mm": float(after[j]),
                                   "near": [key for key, values in flags.items() if values[i]]})
    return result


def one_stage(nurbs, clean, scan, number, mode, tol):
    os.environ.update(CAD_REGIONS="0" if mode == "off" else "1",
                      CAD_REGIONS_GUARD="1" if mode == "guarded" else "0")
    nurbs._NO_REGIONS = False
    nurbs._SCAN = scan
    nurbs.FACE_TOL = min(tol, .5)
    triangles, nurbs.K, nurbs.NP, nurbs.PAIR = nurbs.STEPS[number - 1]
    report, profile, trace = {}, {}, {}
    projector = nurbs._Projector(clean)
    original_merge = nurbs._merge_planes

    def legacy_merge(*args, **kwargs):
        with patch.object(nurbs, "_NO_REGIONS", True):
            return original_merge(*args, **kwargs)

    with ExitStack() as stack:
        if mode == "layout_only":
            stack.enter_context(patch.object(nurbs, "_cyl_tags",
                                             side_effect=lambda mesh, q, raw, proj, w, tag, fit, **kw: (tag, fit)))
            stack.enter_context(patch.object(nurbs, "_merge_planes", side_effect=legacy_merge))
        if mode == "projected_splines":
            stack.enter_context(patch.object(nurbs, "_merge_planes",
                                             side_effect=lambda sewn, *args, **kwargs: (sewn, 0)))
        start = perf_counter()
        body, faces, mids = nurbs._build(clean, triangles, projector, profile=profile, stage={}, audit=trace)
        report.update(seconds=perf_counter() - start, faces=faces, times=profile, trace=trace)
    if body is not None:
        report["legacy_near"] = nurbs.near_share(mids, projector, nurbs.GOAL_TOL)
    return body, report


def analyse(body, output, ref_features, gate, nurbs):
    from regions_guard import bounded_distances

    cq.exporters.export(body, str(output))
    rebuilt = load_step(output)
    topo = topology(rebuilt)
    report = {"topology": topo}
    if ref_features is not None:
        features = match(ref_features, analytic_features(rebuilt), Tolerance())
        report["reference_outcomes"] = dict(Counter(r["status"] for r in features))
    mesh = nurbs.body_mesh(rebuilt, min(.05, gate.tol / 6))
    report["coverage"] = gate.measure_mesh(mesh)
    forward = bounded_distances(mesh, gate.clean_probes, gate.limit)
    return report, forward


def checkpoint(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--scan", type=Path)
    source.add_argument("--case", choices=CASES)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--sheet", type=Path, help="optional demo sheet for noise and reference transform")
    ap.add_argument("--tol", type=float)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--steps", type=int, nargs="+", default=(1, 3))
    ap.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    ap.add_argument("--cascade", action="store_true", help="also run the three actual cascades")
    ap.add_argument("--trace-only", action="store_true", help="record stage diagnostics without STEP analysis")
    ap.add_argument("--budget", type=float, default=900)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    demo = args.demo.resolve()
    sys.path.insert(0, str(demo))
    os.environ.update(CAD_NURBS_FAST="1", CAD_EDGE_KEEP="0")
    import nurbs
    import regions
    import surface
    from regions_guard import WholeBodyGate

    if any(not 1 <= n <= len(nurbs.STEPS) for n in args.steps):
        ap.error("stage index out of range")
    base = subprocess.check_output(["git", "-C", str(demo), "rev-parse", "HEAD"], text=True).strip()
    sheet = json.loads(args.sheet.read_text(encoding="utf-8")) if args.sheet else {}
    if args.case:
        if args.uniform:
            scan = uniform_input(args.case, args.work)
            ref = load_step(args.work / f"{args.case}.step")
        else:
            ref, scan, noise = generate_input(args.case, 15012 + CASES.index(args.case), args.work)
            scan.vertices += .05 * noise
    else:
        scan = trimesh.load(args.scan, force="mesh")
        ref = load_step(args.reference) if args.reference else None
        if ref is not None and "transform_from_scan_4x4" in sheet:
            ref = transform_shape(ref, sheet["transform_from_scan_4x4"])
    noise = float(sheet.get("diagnostics", {}).get("noise_mm", 0) or 0)
    tol = args.tol if args.tol is not None else max(.3, noise)
    if not np.isfinite(tol) or tol <= 0:
        ap.error("tolerance must be positive and finite")
    clean = surface.flatten_planes(surface.surface_mesh(scan))
    gate = WholeBodyGate(clean, scan, nurbs.body_mesh, min(tol, .5))
    ref_features = analytic_features(ref) if ref is not None else None
    result = {"base": base, "case": args.case or args.scan.stem,
              "tol_mm": gate.tol, "raw_triangles": len(scan.faces), "clean_triangles": len(clean.faces),
              "uniform": args.uniform, "steps": {}}
    segment = regions.segment
    cache = None

    def memoized(mesh, **kw):
        nonlocal cache
        if cache is None:
            start = perf_counter()
            cache = segment(mesh, **kw)
            result["segmentation_seconds"] = perf_counter() - start
        return cache

    path = args.work / "results.json"
    with patch.object(regions, "segment", side_effect=memoized):
        for n in args.steps:
            section = result["steps"][str(n)] = {"variants": {}}
            d = {}
            for mode in args.modes:
                print(n, mode, "starting", flush=True)
                body, report = one_stage(nurbs, clean, scan, n, mode, tol)
                report["built"] = body is not None
                section["variants"][mode] = report
                checkpoint(path, result)
                if body is not None and not args.trace_only:
                    checked, d[mode] = analyse(body, args.work / f"stage{n}-{mode}.step",
                                               ref_features, gate, nurbs)
                    report.update(checked)
                    checkpoint(path, result)
                print(n, mode, report["built"], report["faces"],
                      report.get("coverage", {}).get("clean_to_body"), flush=True)
            section["localization"] = {}
            for a, b in (("off", "stock"), ("off", "layout_only"),
                         ("projected_splines", "stock"), ("off", "guarded")):
                if a in d and b in d:
                    section["localization"][f"{a}_to_{b}"] = localize(
                        clean, cache["label"] if cache is not None else None,
                        gate.clean_probes, d[a], d[b], gate.tol)
            checkpoint(path, result)
    if args.cascade:
        result["cascade"] = {}
        for mode in ("off", "stock", "guarded"):
            os.environ.update(CAD_REGIONS="0" if mode == "off" else "1",
                              CAD_REGIONS_GUARD="1" if mode == "guarded" else "0")
            body, info = nurbs.smooth_body(clean, scan, budget=args.budget, tol=tol)
            if body is not None:
                checked, _ = analyse(body, args.work / f"cascade-{mode}.step",
                                     ref_features, gate, nurbs)
                info.update(checked)
            result["cascade"][mode] = info
            checkpoint(path, result)
            print("cascade", mode, info.get("step"), info.get("faces"), flush=True)


if __name__ == "__main__":
    main()
