"""Generate deterministic CAD controls and test the straight B-rep pipeline."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
from time import perf_counter

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
import trimesh

from cad_quality import quality


PARTS = ("ribbed", "angle", "bosses", "shaft", "slope", "thin")
PROBES = ("subtol_wall", "near_parallel", "near_slope")


def box(x, y, z, shift=(0, 0, 0)):
    return cq.Workplane("XY").box(x, y, z, centered=(True, True, False)).translate(shift)


def fixture(kind, seed):
    rng = np.random.default_rng(seed)
    width, depth = (round(float(x), 2) for x in rng.uniform([30, 24], [42, 34]))
    params = {"width": width, "depth": depth}
    base = box(width, depth, 4.)
    if kind == "ribbed":
        thickness = rng.uniform(.8, 2.9, 3).round(2).tolist()
        height = round(float(rng.uniform(4., 8.)), 2)
        params.update(thickness=thickness, height=height)
        for x, t in zip((-width / 3, 0., width / 3), thickness):
            base = base.union(box(t, .8 * depth, height, (x, 0., 4.)))
    elif kind == "angle":
        radius = round(float(rng.uniform(1.4, 2.25)), 2)
        params.update(fillet=radius, web_height=16., hole_diameter=4.)
        base = base.union(box(width, 5., 16., (0, depth / 2 - 2.5, 4.))).edges("|Z").fillet(radius)
        for x in (-width / 4, width / 4):
            drill = cq.Workplane("XY").center(x, -depth / 4).circle(2).extrude(5)
            base = base.cut(drill)
    elif kind == "bosses":
        radius = round(float(rng.uniform(2.8, 4.6)), 2)
        height = round(float(rng.uniform(4, 9)), 2)
        params.update(boss_radius=radius, boss_height=height)
        for x in (-width / 4, width / 4):
            for y in (-depth / 4, depth / 4):
                post = cq.Workplane("XY").workplane(offset=4).center(x, y).circle(radius).circle(radius / 2).extrude(height)
                base = base.union(post)
    elif kind == "shaft":
        radii = rng.uniform([7.5, 5, 3.5], [10, 6.5, 4.6]).round(2).tolist()
        lengths = rng.uniform(5, 11, 3).round(2).tolist()
        params = {"radii": radii, "lengths": lengths}
        z = lengths[0]
        base = cq.Workplane("XY").circle(radii[0]).extrude(lengths[0])
        for r, h in zip(radii[1:], lengths[1:]):
            base = base.union(cq.Workplane("XY").workplane(offset=z).circle(r).extrude(h))
            z += h
    elif kind in ("slope", "near_slope"):
        angle = 2.5 if kind == "near_slope" else round(float(rng.uniform(10, 40)), 2)
        params.update(slope_deg=angle)
        rise = width * np.tan(np.radians(angle))
        profile = [(-width / 2, 0), (width / 2, 0), (width / 2, 4 + rise), (-width / 2, 4)]
        base = cq.Workplane("XZ").polyline(profile).close().extrude(depth / 2, both=True)
    elif kind == "thin":
        thickness = round(float(rng.uniform(.6, 2.)), 2)
        params.update(thickness=thickness)
        for x in (-width / 3, width / 3):
            base = base.union(box(thickness, .85 * depth, 10., (x, 0., 4.)))
    elif kind == "subtol_wall":
        params.update(thickness=.18)
        base = box(width, depth, params["thickness"])
    elif kind == "near_parallel":
        params.update(offset=.18)
        base = base.union(box(width / 2, depth, params["offset"], (width / 4, 0., 4.)))
    else:
        raise ValueError(kind)
    return base.val(), params


def scan_mesh(path, noise, seed):
    import gmsh

    gmsh.initialize()
    try:
        for key, value in {"General.Terminal": 0, "General.NumThreads": 1, "Mesh.RandomSeed": seed % 10000 + 1,
                           "Mesh.MeshSizeMin": .8, "Mesh.MeshSizeMax": 2., "Mesh.Smoothing": 4,
                           "Mesh.MeshSizeFromCurvature": 0}.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc:
                                        .8 + 1.2 * (.5 + .5 * np.sin(.17 * x + .09 * y + phase)))
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, connectivity = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    order = np.argsort(tags)
    faces = order[np.searchsorted(tags[order], connectivity)].reshape(-1, 3)
    clean = trimesh.Trimesh(xyz.reshape(-1, 3), faces)
    trimesh.repair.fix_normals(clean)
    mesh = clean.copy()
    rng = np.random.default_rng(seed + 294)
    mesh.vertices += rng.normal(0, noise, (len(mesh.vertices), 1)) * mesh.vertex_normals
    return mesh


def _plane_features(shape):
    planes = []
    edge_normals = defaultdict(list)
    for face in shape.Faces():
        if face.geomType() != "PLANE":
            continue
        center = np.array(face.Center().toTuple())
        normal = np.array(face.normalAt().toTuple())
        planes.append((center, normal, face.Area()))
        for edge in face.Edges():
            if edge.geomType() == "LINE":
                edge_normals[edge].append(normal)
    edges = []
    for edge, normals in edge_normals.items():
        if len(normals) == 2 and normals[0] @ normals[1] < np.cos(np.radians(12.)):
            edges.append([edge.startPoint().toTuple(), edge.endPoint().toTuple()])
    return planes, np.asarray(edges, dtype=float).reshape(-1, 2, 3)


def plane_and_edge_errors(reference, model, step):
    expected, truth_edges = _plane_features(reference)
    planes, model_edges = _plane_features(model)
    centers = np.array([p[0] for p in planes]).reshape(-1, 3)
    normals = np.array([p[1] for p in planes]).reshape(-1, 3)
    areas = np.array([p[2] for p in planes])
    data = []
    for point, normal, area in expected:
        # Parallel surfaces on opposite sides of thin stock have opposite outward normals.
        angles = np.degrees(np.arccos(np.clip(normals @ normal, -1, 1)))
        offsets = np.abs((centers - point) @ normal)
        ids = np.flatnonzero((angles < 15) & (offsets < max(.6, step)) &
                             (areas >= min(4 * step**2, .2 * area)))
        if not len(ids):
            data.append({"reference_area_mm2": area, "matched": False})
            continue
        chosen = ids[np.argmin(offsets[ids] + step * angles[ids] / 15 + .15 * step * np.abs(np.log(areas[ids] / area)))]
        data.append({"reference_area_mm2": area, "matched": True, "angle_deg": float(angles[chosen]),
                     "offset_mm": float(abs((centers[chosen] - point) @ normals[chosen]))})
    delta = model_edges[:, 1] - model_edges[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    keep = lengths > 1e-9
    starts, lengths = model_edges[keep, 0], lengths[keep]
    directions = delta[keep] / lengths[:, None]
    errors = []
    for start, end in truth_edges:
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 1e-9:
            continue
        eligible = np.abs(directions @ (direction / length)) >= np.cos(np.radians(10.))
        if not np.any(eligible):
            errors.extend([None] * 5)
            continue
        pos = start + np.linspace(.1, .9, 5)[:, None] * direction
        vector = pos[:, None, :] - starts[eligible]
        axes = directions[eligible]
        along = np.clip(np.einsum("ijk,jk->ij", vector, axes), 0., lengths[eligible])
        dists = np.linalg.norm(vector - along[..., None] * axes, axis=2).min(axis=1)
        errors.extend(map(float, dists))
    valid = [d for d in errors if d is not None]
    return {"planes": data, "sharp_edges": len(truth_edges), "sharp_probe_count": len(errors),
            "sharp_probe_within_0p3": sum(d <= .3 for d in valid),
            "sharp_p95_mm": float(np.percentile(valid, 95)) if valid else None}


def _probe(kind, params, mesh):
    if kind not in PROBES:
        return {}
    width = params["width"]
    xs = np.linspace(-.39, .39, 9) * width
    if kind == "near_parallel":
        # The exact CAD vertical face lies at x=0, where ray/edge hits are ambiguous.
        xs = np.sort(np.r_[xs[xs != 0], [-.05, -.01, .01, .05]])
    origins = np.column_stack((xs, np.full(len(xs), 1.234), np.full(len(xs), -5.)))
    points, rays, _ = mesh.ray.intersects_location(origins, np.tile([0, 0, 1.], (len(xs), 1)))
    hits = [np.unique(np.round(points[rays == i, 2], 5)) for i in range(len(xs))]
    if any(len(h) != 2 for h in hits):
        return {"passed": False, "reason": "ray_did_not_find_two_surfaces", "hit_counts": [len(h) for h in hits]}
    if kind == "subtol_wall":
        gaps = [float(h[1] - h[0]) for h in hits]
        return {"passed": min(gaps) > .09, "min_thickness_mm": min(gaps), "max_thickness_mm": max(gaps)}
    top = [h[1] for h in hits]
    if kind == "near_parallel":
        top = np.asarray(top)
        separation = float(np.median(top[xs > width / 10]) - np.median(top[xs < -width / 10]))
        rise = float(top[xs == .05][0] - top[xs == -.05][0])
        return {"passed": separation >= .09, "step_height_mm": separation, "transition_rise_mm_at_pm_0p05": rise}
    angle = float(np.degrees(np.arctan(np.polyfit(xs, top, 1)[0])))
    return {"passed": abs(angle - params["slope_deg"]) <= .6, "measured_angle_deg": angle}


def build_scan(mesh, noise, *, plus=False, tol=.3):
    import idealize
    import planar_brep

    if plus:
        from straight_plus import build

        return build(mesh, tol, noise)
    started = perf_counter()
    adjusted, info = idealize.idealize(mesh, tol, noise)
    idealized = perf_counter()
    vertices, faces, labels = planar_brep.collapse_degenerate(adjusted.vertices, adjusted.faces, info["label"], 1e-3)
    body, built = planar_brep.build(vertices, faces, labels, info["regions"])
    return body, {**built, "aligned_planes": info["aligned_planes"],
                  "idealize_seconds": idealized - started, "brep_seconds": perf_counter() - idealized}


def evaluate(truth, mesh, kind, params, noise, *, plus=False):
    from compare import compare
    from nurbs import body_mesh

    started = perf_counter()
    try:
        body, built = build_scan(mesh, noise, plus=plus)
    except (RuntimeError, ValueError) as exc:
        return {"built": False, "error": f"{type(exc).__name__}: {exc}", "total_seconds": perf_counter() - started}
    built["closed"] = all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in body.Shells()) and bool(body.Shells())
    built["volume_ratio"] = body.Volume() / truth.Volume()
    step = float(np.median(mesh.edges_unique_length))
    score = quality(body, step=step)
    reference = body_mesh(truth, .04)
    output = body_mesh(body, .04)
    reference.merge_vertices()
    output.merge_vertices()
    coverage = {}
    for name, tol in (("0p3", .3), ("3_noise", 3 * noise)):
        if tol <= 0:
            coverage[name] = None
            continue
        rep = compare(reference, output, tol, n=1500, seed=331)
        coverage[name] = {"cad_to_result": rep["within_tol_share"], "result_to_cad": 1 - rep["added_surface_share"]}
    return {"built": True, **built, "score": score["score"], "coverage": coverage,
            **plane_and_edge_errors(truth, body, step), "probe": _probe(kind, params, output),
            "total_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--case", choices=PARTS + PROBES, action="append")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--noise", type=float, action="append")
    parser.add_argument("--plus", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for kind in args.case or PARTS + PROBES:
        truth, params = fixture(kind, args.seed)
        path = args.work / f"{kind}-{args.seed}.step"
        cq.exporters.export(truth, str(path))
        for noise in args.noise or ((0.,) if kind in PROBES else (0., .05, .2)):
            mesh = scan_mesh(path, noise, args.seed)
            row = {"case": kind, "seed": args.seed, "params": params, "noise_mm": noise, "source_sha": sha,
                   "scan_triangles": len(mesh.faces), "scan_watertight": bool(mesh.is_watertight),
                   "step_median_mm": float(np.median(mesh.edges_unique_length))}
            row["baseline"] = evaluate(truth, mesh, kind, params, noise)
            if args.plus:
                row["plus"] = evaluate(truth, mesh, kind, params, noise, plus=True)
            destination = args.work / f"{kind}-{args.seed}-{noise:g}.json"
            destination.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(kind, noise, {label: {k: row[label].get(k) for k in
                                        ("valid", "free_edges", "planar_area_share", "triangle_faces", "score", "probe")}
                                for label in ("baseline", "plus") if label in row}, flush=True)


if __name__ == "__main__":
    main()
