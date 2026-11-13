"""Remove planar labels on smoothly turning wall chains using the original scan."""
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from crease_strips import _groups


def _unit(v):
    return v / np.linalg.norm(v)


def _fit(mesh, label, regions, ids, step, noise, tol):
    normals = np.array([_unit(np.asarray(regions[i]["normal"], float)) for i in ids])
    _, singular, basis = np.linalg.svd(normals, full_matrices=False)
    axis = basis[-1]
    if singular[1] < .015 * singular[0] or singular[2] > .30 * singular[1]:
        return None
    faces = np.flatnonzero(np.isin(label, ids))
    p = mesh.vertices[np.unique(mesh.faces[faces])]
    origin = p.mean(axis=0)
    basis = np.array([basis[0], np.cross(axis, basis[0])])
    q = (p - origin) @ basis.T
    matrix = np.column_stack((2 * q, np.ones(len(q))))
    center = np.linalg.lstsq(matrix, np.sum(q*q, axis=1), rcond=None)[0][:2]
    initial_radius = np.median(np.linalg.norm(q - center, axis=1))
    estimate = least_squares(lambda x: np.linalg.norm(q-x[:2], axis=1)-x[2],
                             np.r_[center, initial_radius], loss="soft_l1",
                             f_scale=max(noise, .03 * step), max_nfev=50)
    center = estimate.x[:2]
    distances = np.linalg.norm(q - center, axis=1)
    radius = float(np.median(distances))
    error = np.abs(distances - radius)
    if radius < 3 * step or np.percentile(error, 95) > max(tol, 2.5 * noise):
        return None
    _, _, directions = np.linalg.svd(q - q.mean(axis=0), full_matrices=False)
    along = q @ directions[0]
    width = np.ptp(along)
    sag = radius - np.sqrt(max(0., radius**2 - (width/2)**2))
    plane_error = np.percentile(np.abs((q - q.mean(axis=0)) @ directions[1]), 95)
    # Mean square error uses the repeated axial sections; per-point p95 is dominated by noise.
    circle_rms = np.sqrt(np.mean(error**2))
    plane_rms = np.sqrt(np.mean(((q - q.mean(axis=0)) @ directions[1])**2))
    if sag <= max(3 * noise, .5 * tol) or plane_error <= 2 * noise or plane_rms <= 1.06 * circle_rms:
        return None
    # Signed normal rotation must follow the radial field from one end to the other.
    radial = np.array([(np.mean(mesh.triangles_center[label == i], axis=0) - origin) @ basis.T
                       for i in ids]) - center
    theta = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    normals_2d = normals @ basis.T
    phi = np.unwrap(np.arctan2(normals_2d[:, 1], normals_2d[:, 0]))
    delta = np.diff(theta)
    turn = np.diff(phi)
    if not np.all(delta * turn > 0) or not (np.all(delta > 0) or np.all(delta < 0)):
        return None
    if np.any(np.abs(turn - delta) > np.maximum(.08, .5 * np.abs(delta))):
        return None
    return {"faces": faces, "planes": ids, "radius_mm": radius, "axis": axis,
            "origin": origin + center @ basis, "p95_mm": float(np.percentile(error, 95))}


def unlabel_curved_walls(mesh, seg, tol, noise):
    """Return a new label array and accepted wall groups. Evaluate before idealize."""
    label = np.asarray(seg["label"])
    regions = seg["regions"]
    step = float(np.median(mesh.edges_unique_length))
    plane = {i for i, r in enumerate(regions) if r["kind"] == "plane" and np.count_nonzero(label == i) >= 3}
    shared = defaultdict(float)
    neighbors = defaultdict(list)
    for (a, b), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        i, j = sorted((label[a], label[b]))
        if i != j and i in plane and j in plane:
            shared[i, j] += np.linalg.norm(mesh.vertices[edge[1]] - mesh.vertices[edge[0]])
    for (i, j), length in shared.items():
        if length < 3 * step:
            continue
        a, b = (_unit(np.asarray(regions[k]["normal"], float)) for k in (i, j))
        turn = np.arccos(np.clip(a @ b, -1, 1))
        if not .012 < turn < .30:
            continue
        neighbors[i].append(j)
        neighbors[j].append(i)
    accepted = []
    for group in _groups(neighbors, neighbors):
        if len(group) < 3 or any(len(neighbors[i]) > 2 for i in group):
            continue
        ends = [i for i in group if len(neighbors[i]) == 1]
        if len(ends) != 2:
            continue
        chain, previous = [int(ends[0])], None
        while len(chain) < len(group):
            current = chain[-1]
            nxt = next(i for i in neighbors[current] if i != previous)
            previous = current
            chain.append(int(nxt))
        row = _fit(mesh, label, regions, chain, step, noise, tol)
        if row is None:
            continue
        # Reject a real polygon: its rails have a coherent, abrupt slope jump
        # compared with pairs one edge inside each region.
        if _has_crease(mesh, label, chain, row["axis"], step, noise):
            continue
        accepted.append(row)
    changed = label.copy()
    for row in accepted:
        changed[row["faces"]] = -1
    return changed, accepted


def cylindrical_regions(mesh, label, regions, groups):
    """Assign a supported wall a cylinder tag, preserving planar neighbors and caps."""
    label = np.array(label, copy=True)
    regions = [dict(row) for row in regions]
    for row in groups:
        own = np.zeros(len(label), dtype=bool)
        own[row["faces"]] = True
        adjacent = mesh.face_adjacency
        a, b = adjacent.T
        neighbors = np.r_[label[a[~own[a] & own[b]]], label[b[own[a] & ~own[b]]]]
        sides = [int(i) for i in np.unique(neighbors) if i >= 0 and regions[i]["kind"] == "plane"
                 and abs(np.asarray(regions[i]["normal"]) @ row["axis"]) < .25]
        if len(sides) != 2:
            continue
        label[own] = len(regions)
        regions.append({"kind": "cylinder", "curved_wall": True, "fillet_planes": sides, "r": row["radius_mm"],
                        "axis": row["axis"], "origin": row["origin"], "area": float(mesh.area_faces[own].sum())})
    return label, regions


def _has_crease(mesh, label, chain, axis, step, noise):
    pairs = {tuple(sorted(p)) for p in zip(chain, chain[1:])}
    boundaries = defaultdict(list)
    for (i, j), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        key = tuple(sorted((label[i], label[j])))
        if key in pairs:
            boundaries[key].extend(edge)
    for (i, j), edges in boundaries.items():
        points = mesh.vertices[np.unique(mesh.faces[(label == i) | (label == j)])]
        border = mesh.vertices[np.unique(edges)].mean(axis=0)
        radial = mesh.face_normals[(label == i) | (label == j)].mean(axis=0)
        radial = _unit(radial - axis * (radial @ axis))
        transverse = np.cross(axis, radial)
        q = points - border
        x, y = q @ transverse, q @ radial
        keep = np.abs(x) <= 3 * step
        x, y = x[keep] / step, y[keep]
        if np.count_nonzero(x < -.3) < 5 or np.count_nonzero(x > .3) < 5:
            return True
        model = np.column_stack((np.ones(len(x)), x, x*x, np.maximum(x, 0)))
        fit, *_ = np.linalg.lstsq(model, y, rcond=None)
        err = y - model @ fit
        sigma = max(noise, np.sqrt(np.mean(err**2)))
        variance = np.linalg.pinv(model.T @ model)[3, 3]
        jump = abs(fit[3])
        if jump > max(.015 * step, 2.8 * sigma * np.sqrt(variance)):
            return True
    return False
