"""Candidate integration of narrow planes and tangent cylinders into the straight build."""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax3, gp_Cylinder, gp_Dir, gp_Pnt

from narrow_planes import narrow_planes
from plane_fillets import fillets


def _sector(blend, regions):
    center = np.asarray(blend["origin"])
    axis = np.asarray(blend["axis"])
    directions = []
    for i in blend["planes"]:
        plane = regions[i]
        n = np.array(plane["normal"], dtype=float)
        n /= np.linalg.norm(n)
        distance = float((np.asarray(plane["origin"]) - center) @ n)
        if abs(abs(distance) - blend["radius"]) > 1e-5 or abs(axis @ n) > 1e-6:
            raise ValueError("cylinder is not tangent to the support planes")
        directions.append(np.sign(distance) * n)
    x, end = directions
    y = np.cross(axis, x)
    sweep = float(np.arctan2(end @ y, end @ x))
    low, high = blend["axis_bounds_mm"]
    if sweep < 0:
        axis, y, sweep, low, high = -axis, -y, -sweep, -high, -low
    if not 0 < sweep < np.pi or low >= high:
        raise ValueError("invalid cylinder sector")
    return center, axis, x, y, sweep, low, high


def cylindrical_face(blend, regions):
    """Create two straight tangent rails and two analytic arcs, suitable for shared wires."""
    center, axis, x, _, sweep, low, high = _sector(blend, regions)
    system = gp_Ax3(gp_Pnt(*center), gp_Dir(*axis), gp_Dir(*x))
    cylinder = gp_Cylinder(system, blend["radius"])
    maker = BRepBuilderAPI_MakeFace(cylinder, 0., sweep, low, high)
    if not maker.IsDone() or not BRepCheck_Analyzer(maker.Face()).IsValid():
        raise RuntimeError("cylinder sector construction failed")
    face = maker.Face()
    return TopoDS.Face_s(face.Reversed()) if blend["sense"] == "concave" else face


def _grow_to_tangency(mesh, seg, blend, claimed, neighbors, limit, step):
    center, axis, x, y, sweep, low, high = _sector(blend, seg["regions"])
    radius = blend["radius"]
    triangles = mesh.vertices[mesh.faces] - center
    axial = triangles @ axis
    tx, ty = triangles @ x, triangles @ y
    error = np.abs(np.hypot(tx, ty) - radius)
    angles = np.arctan2(ty, tx)
    mid_angle = np.arctan2(ty.mean(axis=1), tx.mean(axis=1))
    margin = min(np.arctan2(limit, radius), sweep / 5)
    labels = seg["label"]
    allowed = [*blend["planes"], *(i for i, r in enumerate(seg["regions"]) if r["kind"] == "cylinder")]
    mask = ((labels < 0) | np.isin(labels, allowed)) & ~claimed
    mask &= (error <= limit).all(axis=1)
    mask &= ((angles >= -margin) & (angles <= sweep + margin)).all(axis=1)
    mask &= (mid_angle >= 0) & (mid_angle <= sweep)
    mask &= (axial.min(axis=1) >= low - step) & (axial.max(axis=1) <= high + step)
    seed = np.asarray(blend["faces"])
    seed = seed[~claimed[seed]]
    group = list(map(int, seed))
    mask[seed] = False
    for i in group:
        for j in neighbors[i]:
            if mask[j]:
                mask[j] = False
                group.append(j)
    ids = np.array(sorted(group), dtype=int)
    if not len(ids):
        return ids, {}
    points = mesh.vertices[np.unique(mesh.faces[ids])] - center
    heights = points @ axis
    profile = points - heights[:, None] * axis
    tangent = [radius * x, radius * (np.cos(sweep) * x + np.sin(sweep) * y)]
    gaps = []
    for direction in tangent:
        sections = []
        for a, b in zip(np.linspace(low, high, 6)[:-1], np.linspace(low, high, 6)[1:]):
            close = (heights >= a - step) & (heights <= b + step)
            sections.append(float(np.min(np.linalg.norm(profile[close] - direction, axis=1))) if close.any() else None)
        gaps.append(sections)
    return ids, {"seed_triangles": len(seed), "extended_triangles": len(ids) - len(seed),
                 "rail_gaps_mm": gaps,
                 "rail_endpoints": [[(center + a * axis + u).tolist() for a in (low, high)] for u in tangent]}


def _insert(seg, region, ids, areas):
    if not len(ids):
        return
    labels = seg["label"]
    prior = np.unique(labels[ids])
    labels[ids] = len(seg["regions"])
    seg["regions"].append({**region, "faces": ids, "area": float(areas[ids].sum())})
    for i in prior[prior >= 0]:
        remaining = np.flatnonzero(labels == i)
        seg["regions"][i].update(faces=remaining, area=float(areas[remaining].sum()))


def _ring_guides(mesh, regions):
    centers = mesh.triangles_center
    guides = []
    for region in regions:
        if region["kind"] != "cylinder":
            continue
        center, axis = np.asarray(region["origin"]), np.asarray(region["axis"])
        relative = centers - center
        heights = relative @ axis
        radial = relative - heights[:, None] * axis
        radius = np.linalg.norm(radial, axis=1)
        align = np.abs(np.einsum("ij,ij->i", mesh.face_normals, radial)) >= .8 * radius
        x = np.cross(axis, [1, 0, 0] if abs(axis[0]) < .9 else [0, 1, 0])
        x /= np.linalg.norm(x)
        y = np.cross(axis, x)
        angles = np.arctan2(radial @ y, radial @ x)
        sector = np.minimum(11, ((angles + np.pi) * (6 / np.pi)).astype(int))
        guides.append((center, axis, heights, radius, align, sector))
    return guides


def _on_ring(mesh, region, guides, step, limit):
    points = mesh.vertices[np.unique(mesh.faces[region["faces"]])]
    for center, axis, heights, radii, alignment, sectors in guides:
        if region["kind"] == "plane" and abs(np.dot(region["normal"], axis)) > .35:
            continue
        if region["kind"] == "fillet" and abs(np.dot(region["axis"], axis)) < .92:
            continue
        relative = points - center
        h = relative @ axis
        radial = np.linalg.norm(relative - h[:, None] * axis, axis=1)
        radius = np.median(radial)
        if np.percentile(np.abs(radial - radius), 95) > limit:
            continue
        support = alignment & (radii > step) & (np.abs(radii - radius) <= limit)
        support &= (heights >= h.min() - step / 2) & (heights <= h.max() + step / 2)
        if np.count_nonzero(np.bincount(sectors[support], minlength=12) >= 2) >= 7:
            return True
    return False


def augment(mesh, tol, noise, seg=None):
    """Update a copy of the segmentation and report how far evidence reaches toward each rail.
    seg — готовая разметка regions.segment с теми же параметрами (чтобы не считать её второй раз)."""
    from regions import segment

    step = float(np.median(mesh.edges_unique_length))
    limit = max(tol, 2 * noise)
    source = seg if seg is not None else segment(mesh, tol=limit, min_area=max(6., 4 * step * step * .43), scan=mesh)
    seg = {**source, "label": np.array(source["label"], copy=True), "regions": [dict(r) for r in source["regions"]]}
    areas = mesh.area_faces
    strips = narrow_planes(mesh, seg, tol, noise)
    guides = _ring_guides(mesh, seg["regions"])
    added = 0
    rejected = {"below_noise_resolution": 0, "supported_ring": 0}
    for region in strips:
        # Effective transverse width stays meaningful when extreme vertices are noisy.
        if region["area"] / region["length_mm"] <= 5 * noise:
            rejected["below_noise_resolution"] += 1
            continue
        if _on_ring(mesh, region, guides, step, limit):
            rejected["supported_ring"] += 1
            continue
        _insert(seg, region, region["faces"], areas)
        added += 1
    neighbors = defaultdict(list)
    for i, j in mesh.face_adjacency:
        neighbors[i].append(j)
        neighbors[j].append(i)
    blends = fillets(mesh, seg, tol, noise)
    claimed = np.zeros(len(mesh.faces), dtype=bool)
    cylinders = []
    rejected_blends = 0
    for blend in blends:
        if _on_ring(mesh, blend, guides, step, limit):
            rejected_blends += 1
            continue
        ids, audit = _grow_to_tangency(mesh, seg, blend, claimed, neighbors, limit, step)
        if len(ids):
            claimed[ids] = True
            region = {"kind": "cylinder", "origin": blend["origin"], "axis": blend["axis"], "r": blend["radius"],
                      "fillet_planes": list(map(int, blend["planes"]))}
            _insert(seg, region, ids, areas)
            cylinders.append({**audit, "planes": list(map(int, blend["planes"])), "radius_mm": blend["radius"],
                              "sense": blend["sense"]})
    active = np.unique(seg["label"][seg["label"] >= 0])
    remap = np.full(len(seg["regions"]), -1, dtype=int)
    remap[active] = np.arange(len(active))
    occupied = seg["label"] >= 0
    seg["label"][occupied] = remap[seg["label"][occupied]]
    seg["regions"] = [seg["regions"][i] for i in active]
    for blend in cylinders:
        blend["planes"] = remap[blend["planes"]].tolist()
    for region in seg["regions"]:
        if "fillet_planes" in region:
            pair = remap[region["fillet_planes"]].tolist()
            if min(pair) < 0:
                del region["fillet_planes"]
            else:
                region["fillet_planes"] = pair
    return seg, {"narrow_planes_added": added, "narrow_planes_rejected": rejected,
                 "fillets_added": len(cylinders), "fillets_rejected_supported_ring": rejected_blends,
                 "fillet_growth": cylinders}


def build(mesh, tol=.3, noise=0.):
    """Run the candidate straight pipeline with augmented segmentation."""
    import idealize
    import planar_brep

    started = perf_counter()
    seg, audit = augment(mesh, tol, noise)
    adjusted, info = idealize.idealize(mesh, tol, noise, seg=seg)
    idealized = perf_counter()
    vertices, faces, labels = planar_brep.collapse_degenerate(adjusted.vertices, adjusted.faces, info["label"], 1e-3)
    body, built = planar_brep.build(vertices, faces, labels, info["regions"])
    return body, {**built, **audit, "aligned_planes": info["aligned_planes"],
                  "idealize_seconds": idealized - started, "brep_seconds": perf_counter() - idealized}
