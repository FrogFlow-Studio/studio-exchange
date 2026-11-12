"""Closed, region-aware triangle simplification for the patch layout."""
from __future__ import annotations

import numpy as np
import trimesh


def _topology(faces, label, count):
    """Return edges, incident triangles and the two kinds of vertex adjacency."""
    corners = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    order = np.lexsort((corners[:, 1], corners[:, 0]))
    # The input and all accepted collapses have two faces per edge.
    edges = corners[order[::2]]
    paired = (order // 3).reshape(-1, 2)
    if not np.array_equal(edges, corners[order[1::2]]) or np.any(np.all(edges[1:] == edges[:-1], axis=1)):
        raise ValueError("coarsen_by_regions expects a closed edge-manifold mesh")
    incident = [[] for _ in range(count)]
    for i, face in enumerate(faces.tolist()):
        for v in face:
            incident[v].append(i)
    links = [set() for _ in range(count)]
    for u, v in edges.tolist():
        links[u].add(v)
        links[v].add(u)
    seam = label[paired[:, 0]] != label[paired[:, 1]]
    seam_links = [set() for _ in range(count)]
    for u, v in edges[seam].tolist():
        seam_links[u].add(v)
        seam_links[v].add(u)
    degree = np.bincount(edges[seam].ravel(), minlength=count)
    return edges, paired, seam, degree, incident, links, seam_links


def _regularize_labels(mesh, label, triangles):
    """Remove sub-resolution islands and single-face notches from the seam graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra

    out = label.copy()
    if np.all(out == out[0]):
        return out
    area = mesh.area_faces
    pitch = np.sqrt(4 * float(area.sum()) / (np.sqrt(3) * triangles))
    adj = mesh.face_adjacency
    a, b = adj.T
    edges = mesh.vertices[mesh.face_adjacency_edges]
    vector = edges[:, 1] - edges[:, 0]
    length = np.linalg.norm(vector, axis=1)
    centers = mesh.triangles_center
    eps = np.finfo(float).eps * max(pitch, 1.)
    links = np.maximum(np.linalg.norm(centers[a] - centers[b], axis=1), eps)
    edge_inset = []
    for side in (a, b):
        delta = centers[side] - edges[:, 0]
        t = np.clip(np.einsum("ij,ij->i", delta, vector) / np.maximum(length**2, eps), 0., 1.)
        edge_inset.append(np.maximum(np.linalg.norm(delta - t[:, None] * vector, axis=1), eps))
    for _ in range(3):
        same = out[a] == out[b]
        graph = csr_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(len(out), len(out)))
        count, ids = connected_components(graph, directed=False)
        size = np.bincount(ids, weights=area, minlength=count)
        boundary = ~same
        perimeter = np.bincount(np.r_[ids[a[boundary]], ids[b[boundary]]],
                                weights=np.r_[length[boundary], length[boundary]], minlength=count)
        if not np.any(boundary):
            break
        # Geodesic distance from the actual seam to each face center. A long
        # ribbon can have substantial area but no interior at the target pitch.
        inside = np.r_[a[boundary], b[boundary]]
        inset = np.full(len(out), np.inf)
        np.minimum.at(inset, inside, np.r_[edge_inset[0][boundary], edge_inset[1][boundary]])
        rim = np.flatnonzero(np.isfinite(inset))
        distance_graph = csr_matrix(
            (np.r_[links[same], inset[rim]],
             (np.r_[a[same], np.full(len(rim), len(out))], np.r_[b[same], rim])),
            shape=(len(out) + 1, len(out) + 1),
        )
        depth = dijkstra(distance_graph, directed=False, indices=len(out))[:-1]
        core = np.bincount(ids, weights=area * (depth >= 0.5 * pitch), minlength=count)
        thin = (2 * size < pitch * perimeter) & (core < 0.12 * size)
        narrow = thin | ((size < 5 * pitch**2) & (2 * size < pitch * perimeter))
        targets = {}
        for u, v, weight in zip(np.r_[a[boundary], b[boundary]],
                                np.r_[b[boundary], a[boundary]],
                                np.r_[length[boundary], length[boundary]]):
            region = ids[u]
            if narrow[region] and not narrow[ids[v]]:
                by_label = targets.setdefault(region, {})
                by_label[out[v]] = by_label.get(out[v], 0.) + weight
        rename = np.full(count, -2, dtype=int)
        has_label = np.bincount(ids, weights=out >= 0, minlength=count) > 0
        rename[thin & has_label] = -1
        for region, scores in targets.items():
            best = max(scores, key=scores.get)
            if scores[best] >= 0.62 * perimeter[region]:
                rename[region] = best
        selected = (rename[ids] >= -1) & (rename[ids] != out)
        if not np.any(selected):
            break
        out[selected] = rename[ids[selected]]

    # A single dangling triangle can create two artificial corners. Flip only
    # when two adjacent faces agree and the edge on that side is not a crease.
    neighbors = np.full((len(out), 3), -1, dtype=int)
    positions = np.zeros(len(out), dtype=int)
    smooth = np.cos(np.radians(28))
    for u, v, dot in zip(a, b, np.einsum("ij,ij->i", mesh.face_normals[a], mesh.face_normals[b])):
        if dot >= smooth:
            neighbors[u, positions[u]] = v
            neighbors[v, positions[v]] = u
            positions[u] += 1
            positions[v] += 1
    for _ in range(3):
        choices = np.where(neighbors >= 0, out[neighbors], -2)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            take = (choices[:, i] >= -1) & (choices[:, i] == choices[:, j]) & (area < 0.20 * pitch**2)
            out[take] = choices[take, i]
    return out


def _quadrics(vertices, faces, edges, seam):
    points = vertices[faces]
    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(area[:, None], np.finfo(float).tiny)
    plane = np.c_[normal, -np.einsum("ij,ij->i", normal, points[:, 0])]
    weighted = area[:, None, None] * plane[:, :, None] * plane[:, None, :]
    quadrics = np.zeros((len(vertices), 4, 4))
    np.add.at(quadrics, faces.ravel(), np.repeat(weighted, 3, axis=0))
    # Penalize displacement perpendicular to the original boundary polyline.
    boundary = edges[seam]
    start = vertices[boundary[:, 0]]
    tangent = vertices[boundary[:, 1]] - start
    length2 = np.einsum("ij,ij->i", tangent, tangent)
    unit = tangent / np.sqrt(np.maximum(length2[:, None], np.finfo(float).tiny))
    perpendicular = np.eye(3) - unit[:, :, None] * unit[:, None, :]
    transform = np.concatenate((np.broadcast_to(np.eye(3), (len(start), 3, 3)), -start[..., None]), axis=2)
    line = transform.transpose(0, 2, 1) @ perpendicular @ transform
    weighted_line = 4 * length2[:, None, None] * line
    for side in range(2):
        np.add.at(quadrics, boundary[:, side], weighted_line)
    return quadrics


def _candidates(vertices, quadrics, edges, seam, degree):
    a, b = edges.T
    da, db = degree[a], degree[b]
    # An interior vertex may collapse into a boundary vertex. On the seam only
    # degree-two vertices may disappear. Junctions and corners stay in place.
    allowed = ((da == 0) | (db == 0) | (seam & ((da == 2) | (db == 2))))
    ids = np.flatnonzero(allowed)
    a, b, da, db = a[ids], b[ids], da[ids], db[ids]
    fixed_a = (da != 0) & ((da != 2) | (db == 0))
    fixed_b = (db != 0) & ((db != 2) | (da == 0))
    q = quadrics[a] + quadrics[b]
    p = np.c_[vertices[a], np.ones(len(a))]
    delta = np.c_[vertices[b] - vertices[a], np.zeros(len(a))]
    curvature = np.einsum("ij,ijk,ik->i", delta, q, delta)
    slope = np.einsum("ij,ijk,ik->i", delta, q, p)
    t = np.clip(np.divide(-slope, curvature, out=np.full(len(a), 0.5),
                          where=curvature > 1e-20), 0., 1.)
    t[fixed_a], t[fixed_b] = 0., 1.
    point = p + delta * t[:, None]
    error = np.maximum(0., np.einsum("ij,ijk,ik->i", point, q, point))
    length2 = np.einsum("ij,ij->i", delta, delta)
    # A seam edge has extra line quadrics. Balance that penalty with
    # interior collapses so a fine boundary can use the same triangle budget.
    error *= np.where((da != 0) & (db != 0), 0.2, 1.)
    order = np.lexsort((length2, error))
    # The order matters only for determinism. Vertices on the seam must survive
    # an interior collapse even when they happen to be the second endpoint.
    keep = np.where(fixed_b, b, a)
    drop = np.where(fixed_b, a, b)
    return order, ids, keep, drop, point[:, :3]


def _valid_collapse(vertices, faces, incident, links, u, v, gone, point):
    common = links[u] & links[v]
    opposite = set(faces[gone].ravel()) - {u, v}
    if len(common) != 2 or common != opposite:
        return None
    changed = np.array(sorted((set(incident[u]) | set(incident[v])) - set(gone)))
    before = faces[changed]
    after = np.where(before == v, u, before)
    if len({tuple(sorted(face)) for face in after.tolist()}) != len(after):
        return None
    p = vertices[before]
    new = np.where((after == u)[..., None], point, vertices[after])
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    candidate = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    area2 = np.einsum("ij,ij->i", candidate, candidate)
    old2 = np.einsum("ij,ij->i", normal, normal)
    dot = np.einsum("ij,ij->i", normal, candidate)
    if np.any(area2 <= 1e-12 * old2) or np.any(dot <= 0.15 * np.sqrt(old2 * area2)):
        return None
    return changed, after


def coarsen_by_regions(mesh, label, triangles, *, regularize=True):
    """Return a closed coarse mesh and one inherited integer label per face.

    Only edge collapses satisfying the link condition are used. Collapsing an
    interface edge shortens its polyline without changing the adjacent labels.
    The target is best-effort when the boundary graph or topology requires more
    faces. Regularization can transfer sub-resolution label islands and notches
    to their neighbors. Pass regularize=False to keep the original seam graph.
    Pass all -1 labels for a region-independent closed simplification.
    """
    labels = np.asarray(label)
    if labels.shape != (len(mesh.faces),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label must have one integer per triangle")
    if triangles < 4:
        raise ValueError("triangles must be at least four")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("coarsen_by_regions expects a closed oriented mesh")
    if len(mesh.faces) <= triangles:
        return mesh.copy(), labels.copy()

    # Center the coordinates to avoid cancellation in the quadric evaluation.
    original_labels = labels
    labels = _regularize_labels(mesh, labels, triangles) if regularize else labels
    input_change = float(mesh.area_faces[labels != original_labels].sum() / mesh.area)
    origin = mesh.vertices.mean(axis=0)
    vertices = np.asarray(mesh.vertices) - origin
    faces = np.array(mesh.faces)
    edges, paired, seam, degree, incident, links, seam_links = _topology(faces, labels, len(vertices))
    quadrics = _quadrics(vertices, faces, edges, seam)
    while len(faces) > triangles + (triangles % 2):
        order, ids, keep, drop, points = _candidates(vertices, quadrics, edges, seam, degree)
        used = set()
        dead = np.zeros(len(faces), dtype=bool)
        remaining = len(faces)
        # Independent vertex stars let us update a batch and rebuild adjacency once.
        # Rebuild costs before considering the expensive tail. Otherwise an
        # isolated high-error edge can be chosen while cheaper stars are blocked
        # only for the current batch. Visit the tail if no cheap collapse works.
        cap = max(700, len(order) // 3)
        for rank, i in enumerate(order):
            if rank >= cap and remaining < len(faces):
                break
            u, v = int(keep[i]), int(drop[i])
            if u in used or v in used or seam_links[u] & seam_links[v]:
                continue
            gone = paired[ids[i]]
            result = _valid_collapse(vertices, faces, incident, links, u, v, gone, points[i])
            if result is None:
                continue
            changed, after = result
            faces[changed] = after
            vertices[u] = points[i]
            quadrics[u] += quadrics[v]
            dead[gone] = True
            used.update(links[u] | links[v] | {u, v})
            remaining -= 2
            if remaining <= triangles + (triangles % 2):
                break
        if remaining == len(faces):
            break
        faces, labels = faces[~dead], labels[~dead]
        if remaining <= triangles + (triangles % 2):
            break
        edges, paired, seam, degree, incident, links, seam_links = _topology(faces, labels, len(vertices))
    active, inverse = np.unique(faces, return_inverse=True)
    coarse = trimesh.Trimesh(vertices=vertices[active] + origin,
                            faces=inverse.reshape(-1, 3), process=False)
    if not coarse.is_watertight or not coarse.is_winding_consistent:
        raise RuntimeError("region coarsening produced an invalid topology")
    final_labels = _regularize_labels(coarse, labels, triangles) if regularize else labels
    coarse.metadata["region_layout"] = {
        "source_relabel_area_share": input_change,
        "coarse_relabel_area_share": float(coarse.area_faces[labels != final_labels].sum() / coarse.area),
    }
    return coarse, final_labels.copy()
