"""Find constant-radius cylindrical blends tangent to two labelled planes."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from narrow_planes import _components


def _plane(region, mesh, ids):
    normal = np.array(region["normal"], dtype=float, copy=True)
    normal /= np.linalg.norm(normal)
    if mesh.area_faces[ids] @ (mesh.face_normals[ids] @ normal) < 0:
        normal = -normal
    return normal, float(normal @ region["origin"])


def _frame(first, second):
    normals = np.stack((first[0], second[0]))
    axis = np.cross(*normals)
    length = np.linalg.norm(axis)
    if length < np.sin(np.radians(15.)):
        return None
    gram = normals @ normals.T
    corner = normals.T @ np.linalg.solve(gram, [first[1], second[1]])
    shift = normals.T @ np.linalg.solve(gram, np.ones(2))
    return normals, axis / length, corner, shift


def _p95(values):
    return float(np.percentile(np.abs(values), 95))


def _phase(radial, radius, shift, sign, normal, tangent):
    outward = -sign * (radial - radius * shift)
    return np.arctan2(outward @ tangent, outward @ normal)


def _fit(mesh, ids, pair, rails, frame, step, tol, noise):
    normals, axis, corner, shift = frame
    dot = float(normals[0] @ normals[1])
    vertices = mesh.vertices[np.unique(mesh.faces[ids])] - corner
    heights = vertices @ axis
    if len(vertices) < 8 or np.ptp(heights) < 3 * step:
        return None
    radial = vertices - heights[:, None] * axis
    # Both tangent rails must persist along the same axis interval.
    line = [mesh.vertices[np.unique(rail)] - corner for rail in rails]
    longitudinal = [q @ axis for q in line]
    overlap = min(h.max() for h in longitudinal) - max(h.min() for h in longitudinal)
    if overlap < max(2 * step, .55 * np.ptp(heights)):
        return None
    estimate = np.array([np.median(q @ normals[1 - k]) / (1 - dot) for k, q in enumerate(line)])
    limit = max(tol, 2 * noise)
    if estimate[0] * estimate[1] <= 0 or np.ptp(estimate) > max(2 * limit, .3 * np.abs(estimate).mean()):
        return None
    sign = np.sign(estimate[0])
    initial = float(np.abs(estimate).mean())
    center_shift = sign * shift
    width_per_radius = np.linalg.norm(normals[0] - normals[1])
    if initial * width_per_radius <= 2 * limit:
        return None

    def residual(params, points):
        radius = params[0]
        return np.linalg.norm(points - radius * center_shift, axis=1) - radius

    bounds = ([.45 * initial], [2.2 * initial])
    scale = max(noise, .05 * step)
    fit = least_squares(residual, [initial], args=(radial,), bounds=bounds, loss="soft_l1", f_scale=scale)
    radius = float(fit.x[0])
    tangent = np.cross(axis, normals[0])
    sweep = np.arccos(np.clip(dot, -1., 1.))
    # A cylinder label can absorb tangent planar triangles. Trim by the supported
    # angular sector before evaluating residuals and publishing the triangle set.
    triangles = mesh.vertices[mesh.faces[ids]] - corner
    projected = triangles - (triangles @ axis)[..., None] * axis
    phase = _phase(projected, radius, center_shift, sign, normals[0], tangent)
    slack = np.arctan2(limit, radius)
    keep = ((phase >= -slack) & (phase <= sweep + slack)).all(axis=1)
    ids = ids[keep]
    if len(ids) < 4:
        return None
    vertices = mesh.vertices[np.unique(mesh.faces[ids])] - corner
    heights = vertices @ axis
    radial = vertices - heights[:, None] * axis
    if np.ptp(heights) < max(3 * step, .6 * overlap):
        return None

    fit = least_squares(residual, [radius], args=(radial,), bounds=bounds, loss="soft_l1", f_scale=scale)
    radius = float(fit.x[0])
    error = _p95(fit.fun)
    if radius * width_per_radius <= 2 * limit or error > limit:
        return None
    flat = radial - radial.mean(axis=0)
    _, _, basis = np.linalg.svd(flat, full_matrices=False)
    plane_error = _p95(flat @ basis[1])
    if plane_error <= max(1.25 * error, 2 * noise, .08 * step):
        return None
    phase = _phase(radial, radius, center_shift, sign, normals[0], tangent)
    slack = np.arctan2(limit, radius)
    if np.mean((phase >= -slack) & (phase <= sweep + slack)) < .9:
        return None
    low, high = np.quantile(phase, [.05, .95])
    if high - low < .55 * sweep:
        return None

    jacobian = np.abs(fit.jac[:, 0])
    sensitivity = max(.1, float(np.median(jacobian)))
    residual_mad = 1.4826 * np.median(np.abs(fit.fun - np.median(fit.fun)))
    effective_sections = max(1., overlap / step)
    sigma = float(max(residual_mad, noise) / (sensitivity * np.sqrt(effective_sections)))
    snapped = round(radius * 2) / 2
    proposed = snapped if snapped > 0 and abs(radius - snapped) + 2 * sigma < .25 else None
    axial_origin = float((heights.min() + heights.max()) * .5)
    return {"kind": "fillet", "planes": pair, "radius": radius, "radius_sigma_mm": sigma,
            "suggested_radius_mm": proposed, "axis": axis,
            "origin": corner + radius * center_shift + axial_origin * axis,
            "axis_bounds_mm": [float(heights.min() - axial_origin), float(heights.max() - axial_origin)],
            "faces": ids, "p95_mm": error, "planar_p95_mm": plane_error,
            "sense": "convex" if sign < 0 else "concave"}


def fillets(mesh, seg, tol, noise) -> list[dict]:
    """Detect a blend using original triangles and the indices of its two planar supports."""
    label = np.asarray(seg["label"])
    regions = seg["regions"]
    if not len(label) or not len(mesh.edges_unique):
        return []
    plane_ids = {i for i, r in enumerate(regions) if r["kind"] == "plane"}
    if len(plane_ids) < 2:
        return []
    step = float(np.median(mesh.edges_unique_length))
    if step <= 0 or max(tol, 2 * noise) <= 0:
        return []
    planes = {}
    for i in plane_ids:
        ids = np.asarray(regions[i]["faces"], dtype=int) if "faces" in regions[i] else np.flatnonzero(label == i)
        ids = ids[label[ids] == i]
        if len(ids):
            planes[i] = _plane(regions[i], mesh, ids)
    cylinders = [i for i, r in enumerate(regions) if r["kind"] == "cylinder"]
    eligible = (label < 0) | np.isin(label, cylinders)
    neighbors = defaultdict(list)
    contacts = defaultdict(list)
    for (i, j), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        if eligible[i] and eligible[j]:
            neighbors[i].append(j)
            neighbors[j].append(i)
        elif eligible[i] and label[j] in planes:
            contacts[i].append((label[j], edge))
        elif eligible[j] and label[i] in planes:
            contacts[j].append((label[i], edge))
    candidates = []
    for group in _components(np.flatnonzero(eligible), neighbors):
        if len(group) < 4:
            continue
        supports = defaultdict(list)
        for face in group:
            for plane, edge in contacts[face]:
                supports[plane].append(edge)
        normals = mesh.face_normals[group]
        for pair in combinations(sorted(k for k, edges in supports.items() if len(edges) >= 2), 2):
            frame = _frame(planes[pair[0]], planes[pair[1]])
            if frame is None:
                continue
            ns, axis, _, _ = frame
            # Restrict graph traversal to normals in the wedge between supports.
            weights = normals @ ns.T @ np.linalg.inv(ns @ ns.T)
            mask = (np.abs(normals @ axis) < .5) & (weights.min(axis=1) >= -.27)
            for patch in _components(group[mask], neighbors):
                rails = [[], []]
                for face in patch:
                    for i, edge in contacts[face]:
                        if i in pair:
                            rails[pair.index(i)].append(edge)
                if len(patch) < 4 or min(map(len, rails)) < 2:
                    continue
                row = _fit(mesh, patch, pair, rails, frame, step, tol, noise)
                if row:
                    candidates.append(row)
    claimed = np.zeros(len(label), dtype=bool)
    output = []
    for row in sorted(candidates, key=lambda r: (-len(r["faces"]), r["p95_mm"])):
        if not claimed[row["faces"]].any():
            claimed[row["faces"]] = True
            output.append(row)
    return output
