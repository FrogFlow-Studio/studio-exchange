"""Grow tangent cylinders through supported scan triangles up to their end contacts."""
from collections import defaultdict

import numpy as np
import trimesh

from straight_plus import _sector
from crease_strips import _groups
from planar_brep import _loops


def extend(V, F, label, regions, tol=.3, noise=0.):
    """Return candidate labels and audit. Keep planar end triangles untouched."""
    mesh = trimesh.Trimesh(V, F, process=False)
    tags = np.array(label, dtype=int, copy=True)
    regs = [dict(row) for row in regions]
    step = float(np.median(mesh.edges_unique_length))
    eps = max(tol, 2*noise)
    links = defaultdict(list)
    for a, b in mesh.face_adjacency:
        links[a].append(b)
        links[b].append(a)
    rows = []
    for i, region in enumerate(regs):
        if region["kind"] != "cylinder" or "fillet_planes" not in region:
            continue
        sides = list(region["fillet_planes"])
        for k, side in enumerate(sides):
            while side >= 0 and regs[side]["kind"] == "merged":
                side = regs[side]["into"]
            sides[k] = side
        if min(sides) < 0 or len(set(sides)) != 2 or any(regs[k]["kind"] != "plane" for k in sides):
            continue
        seed = np.flatnonzero(tags == i)
        if not len(seed):
            continue
        n = np.array([regs[k]["normal"] for k in sides])
        n /= np.linalg.norm(n, axis=1)[:, None]
        axis = np.cross(*n)
        if np.linalg.norm(axis) < .18:
            continue
        axis /= np.linalg.norm(axis)
        radius = region["r"]
        center = np.asarray(region["origin"])
        offsets = np.array([n[j] @ regs[k]["origin"] for j, k in enumerate(sides)])
        signs = np.sign(n @ center - offsets)
        if np.any(signs == 0):
            continue
        center = center + n.T @ np.linalg.solve(n @ n.T, offsets + signs*radius - n@center)
        fit = {"origin": center, "axis": axis, "radius": radius, "planes": sides, "axis_bounds_mm": (-1., 1.)}
        try:
            center, axis, x, y, sweep, _, _ = _sector(fit, regs)
        except ValueError:
            continue
        points = mesh.vertices[mesh.faces] - center
        axial = points @ axis
        cx, cy = points @ x, points @ y
        angles = np.arctan2(cy, cx)
        radial = np.abs(np.hypot(cx,cy)-radius)
        slack = min(sweep/5, np.arctan2(eps,radius))
        free = tags < 0
        fragments = []
        for j, other in enumerate(regs):
            if j == i or other["kind"] != "cylinder" or set(other.get("fillet_planes", ())) != set(region["fillet_planes"]):
                continue
            drift = np.linalg.norm(np.cross(np.asarray(other["origin"])-center, axis))
            if drift <= eps and abs(other["r"]-radius) <= eps and abs(np.dot(other["axis"], axis)) >= .985:
                fragments.append(j)
        candidates = free | np.isin(tags, [*sides, *fragments, i])
        candidates &= np.abs(mesh.face_normals @ axis) < .60
        candidates &= (radial <= eps).all(axis=1)
        candidates &= ((angles >= -slack) & (angles <= sweep+slack)).all(axis=1)
        candidates &= (np.arctan2(cy.mean(axis=1), cx.mean(axis=1)) >= 0)
        candidates &= (np.arctan2(cy.mean(axis=1), cx.mean(axis=1)) <= sweep)
        # Claim compatible fragments only when they are geometrically connected.
        candidates[seed] = True
        groups = list(_groups(np.flatnonzero(candidates), links))
        supported = [g for g in groups if np.intersect1d(g,seed).size]
        group = max(supported, key=len)
        border = _loops(F, group)
        own = set(group)
        support = {int(tags[j]) for k in group for j in links[k] if j not in own}
        old_span = float(np.ptp(axial[seed]))
        new_span = float(np.ptp(axial[group]))
        accepted = bool(border and len(border) == 1 and len(group) >= len(seed)
                        and set(sides) <= support and (new_span >= old_span or len(supported) > 1))
        if accepted:
            tags[seed] = -1
            merged = sorted(set(tags[group]).intersection(fragments))
            tags[group] = i
            region.update(origin=center, axis=axis, fillet_planes=sides)
        else:
            merged = []
        rows.append({"region": i, "before_triangles": len(seed), "after_triangles": len(group),
                     "axis_span_mm": [old_span, new_span], "boundary_loops": len(border or ()),
                     "accepted": accepted, "end_contacts": sorted(support - set(sides)),
                     "merged_fragments": merged})
    for i, region in enumerate(regs):
        ids = np.flatnonzero(tags == i)
        region.update(faces=ids, area=float(mesh.area_faces[ids].sum()))
    return tags, regs, rows
