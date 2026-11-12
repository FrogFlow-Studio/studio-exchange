"""Recover connected narrow planar strips left free by regions.segment."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _directions(mesh, seg):
    from idealize import regularize

    planes = [dict(r) for r in seg["regions"]]
    _, axes = regularize(planes, mesh.vertices, mesh.faces, seg["label"])
    seeds = [] if axes is None else list(axes)
    seeds.extend(np.asarray(r["normal"], float) for r in planes if r["kind"] == "plane")
    return seeds


def _thickness_offsets(regions, direction, limit, step):
    offsets = [[], []]
    for region in regions:
        if region["kind"] != "plane":
            continue
        alignment = float(np.dot(region["normal"], direction))
        if abs(alignment) >= .98:
            offsets[int(alignment > 0)].append(float(np.dot(region["origin"], direction)))
    negative, positive = (np.unique(side) for side in offsets)
    thickness = positive[:, None] - negative
    valid = thickness[(thickness > 2 * limit) & (thickness <= 4.5 * step)]
    if not len(valid):
        return []
    bins, counts = np.unique(np.rint(valid / limit), return_counts=True)
    modes = bins[np.argsort(-counts, kind="stable")[:3]]
    widths = [np.median(valid[np.rint(valid / limit) == mode]) for mode in modes]
    return [offset for width in widths for offset in np.r_[negative + width, positive - width]]


def _components(ids, neighbors):
    remaining = set(map(int, ids))
    while remaining:
        root = remaining.pop()
        component = [root]
        for i in component:
            for j in neighbors[i]:
                if j in remaining:
                    remaining.remove(j)
                    component.append(j)
        yield np.array(sorted(component))


def _frame(points, normal, origin):
    xy = points - origin
    xy -= (xy @ normal)[:, None] * normal
    _, _, basis = np.linalg.svd(xy, full_matrices=False)
    long, across = basis[:2]
    return long, across, np.ptp(xy @ long), np.ptp(xy @ across)


def _rolling_component(mesh, ids, noise, step):
    """Test for a cylindrical cross-section across an entire free strip."""
    points = mesh.vertices[np.unique(mesh.faces[ids])]
    if len(points) < 12:
        return False
    xy = points - points.mean(axis=0)
    _, _, basis = np.linalg.svd(xy, full_matrices=False)
    longitudinal = np.ptp(xy @ basis[0])
    section = xy @ basis[1:].T
    width = np.ptp(section[:, 0])
    if width < step or longitudinal < 2.5 * width:
        return False
    matrix = np.column_stack((2 * section, np.ones(len(section))))
    center = np.linalg.lstsq(matrix, (section * section).sum(axis=1), rcond=None)[0][:2]
    radial = np.linalg.norm(section - center, axis=1)
    radius = np.median(radial)
    if radius < width / 2 or radius > 2 * width:
        return False
    plane_error = np.percentile(np.abs(section[:, 1]), 95)
    circle_error = np.percentile(np.abs(radial - radius), 95)
    sagitta = radius - np.sqrt(max(0., radius**2 - (width / 2)**2))
    return (sagitta > max(2.5 * noise, .10 * step) and
            plane_error > max(1.18 * circle_error, 1.8 * noise, .08 * step) and
            circle_error <= max(2.5 * noise, .15 * step))


def _curving_rails(mesh, ids, normal, origin, long, across, boundaries, noise, step):
    """Detect coherent rolling across both long sides of a putative plane."""
    rails = [[], []]
    inside = set(ids)
    oriented = mesh.face_normals
    for i in ids:
        for j, vertices in boundaries[i]:
            if j in inside:
                continue
            a, b = mesh.vertices[vertices]
            edge = b - a
            length = np.linalg.norm(edge)
            if not length or abs(edge @ long) < .75 * length:
                continue
            midpoint = (a + b) * .5
            side = int((midpoint - origin) @ across >= 0)
            n = oriented[j]
            if n @ normal < 0:
                n = -n
            rails[side].append((length, n))
    if min(map(len, rails)) < 2:
        return False
    turns = []
    for rail in rails:
        mean = sum(length * n for length, n in rail)
        signed = np.degrees(np.arctan2(mean @ across, mean @ normal))
        turns.append(signed)
    floor = max(4., np.degrees(np.arctan2(2 * noise, step)))
    return turns[0] * turns[1] < 0 and all(floor <= abs(t) <= 65 for t in turns)


def _strip(mesh, ids, seed, limit, noise, step, boundaries):
    vertices = mesh.vertices[np.unique(mesh.faces[ids])]
    if len(vertices) < 6:
        return None
    origin = vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(vertices - origin, full_matrices=False)
    fitted = basis[-1] * np.sign(basis[-1] @ seed)
    best = None
    for normal in (seed, fitted):
        if normal @ seed < .90:
            continue
        offset = np.median(vertices @ normal)
        error = np.abs(vertices @ normal - offset)
        if error.max() > limit or (best and np.median(error) >= best[0]):
            continue
        best = (float(np.median(error)), normal, origin + (offset - origin @ normal) * normal)
    if best is None:
        return None
    residual, normal, origin = best
    long, across, length, width = _frame(vertices, normal, origin)
    if length < 4 * step or width > min(4.5 * step, length / 2.5):
        return None
    mean = np.average(mesh.face_normals[ids], weights=mesh.area_faces[ids], axis=0)
    if mean @ normal < 0:
        normal = -normal
    if _curving_rails(mesh, ids, normal, origin, long, across, boundaries, noise, step):
        return None
    return {"kind": "plane", "normal": normal, "origin": origin, "area": float(mesh.area_faces[ids].sum()),
            "faces": ids, "width_mm": float(width), "length_mm": float(length), "median_error_mm": residual}


def narrow_planes(mesh, seg, tol, noise) -> list[dict]:
    """Return nonoverlapping planar strips on unassigned original face indices."""
    label = np.asarray(seg["label"])
    free = np.flatnonzero(label < 0)
    if not len(free) or not len(mesh.edges_unique):
        return []
    step = float(np.median(mesh.edges_unique_length))
    limit = max(tol, 2 * noise)
    if min(step, limit) <= 0:
        return []
    neighbors, boundaries = defaultdict(list), defaultdict(list)
    for (i, j), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        neighbors[i].append(j)
        neighbors[j].append(i)
        boundaries[i].append((j, edge))
        boundaries[j].append((i, edge))

    curved = np.zeros(len(label), dtype=bool)
    for ids in _components(free, neighbors):
        if _rolling_component(mesh, ids, noise, step):
            curved[ids] = True
    free = free[~curved[free]]
    directions = []
    seeds = _directions(mesh, seg)
    # Area-ranked samples give oblique strips a starting direction even without anchors.
    order = free[np.argsort(-mesh.area_faces[free], kind="stable")]
    seeds.extend(mesh.face_normals[order[: min(len(order), 128)]])
    for n in seeds:
        if np.linalg.norm(n) > .9 and all(abs(n @ old) < np.cos(np.radians(12.)) for old in directions):
            directions.append(n)
    candidates = []
    triangles = mesh.vertices[mesh.faces[free]]
    for n in directions:
        near = np.abs(mesh.face_normals[free] @ n) > .77
        heights = triangles[near] @ n
        ids = free[near]
        if len(ids) < 4:
            continue
        middle = np.median(heights, axis=1)
        order = np.argsort(middle)
        middle, ids, heights = middle[order], ids[order], heights[order]
        buckets = np.rint(middle / (.75 * limit))
        boundaries_idx = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1, len(buckets)]
        offsets = [np.median(middle[a:b]) for a, b in zip(boundaries_idx[:-1], boundaries_idx[1:])]
        offsets.extend(_thickness_offsets(seg["regions"], n, limit, step))
        # Opposite sides of thin stock are separate offset bands along the same direction.
        for offset in offsets:
            a, b = np.searchsorted(middle, [offset - limit, offset + limit])
            if b - a < 4:
                continue
            mask = np.max(np.abs(heights[a:b] - offset), axis=1) <= limit
            for group in _components(ids[a:b][mask], neighbors):
                if len(group) >= 4:
                    region = _strip(mesh, group, n, limit, noise, step, boundaries)
                    if region:
                        candidates.append(region)
    occupied = label >= 0
    output = []
    for region in sorted(candidates, key=lambda r: (-r["area"], r["median_error_mm"])):
        ids = region["faces"]
        if not occupied[ids].any():
            occupied[ids] = True
            output.append(region)
    return output
