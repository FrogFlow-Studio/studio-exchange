"""Resolve narrow free zones without moving vertices owned by other regions."""
from collections import defaultdict, Counter

import numpy as np
import trimesh

from patch_faces import topology
from plane_fillets import fillets


def _plane(region):
    n = np.asarray(region["normal"], float)
    n = n / np.linalg.norm(n)
    return n, float(n @ region["origin"])


def _groups(ids, neighbors):
    pending = set(map(int, ids))
    while pending:
        group = [pending.pop()]
        for i in group:
            for j in neighbors[i]:
                if j in pending:
                    pending.remove(j)
                    group.append(j)
        yield np.asarray(group)


def classify(V, F, label, regions, tol):
    V, F, label = np.asarray(V), np.asarray(F), np.asarray(label)
    mesh = trimesh.Trimesh(V, F, process=False)
    if not len(F) or tol <= 0:
        return []
    step = np.median(mesh.edges_unique_length)
    neighbors, contacts = defaultdict(list), defaultdict(list)
    for (i, j), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        if label[i] < 0 and label[j] < 0:
            neighbors[i].append(j)
            neighbors[j].append(i)
        elif (label[i] < 0) != (label[j] < 0):
            free, support = (i, j) if label[i] < 0 else (j, i)
            contacts[free].append((int(label[support]), edge))
    # Seed local candidates around plane pairs, including strips connected
    # to a wider free region elsewhere on the part.
    from scipy.spatial import cKDTree

    free = np.flatnonzero(label < 0)
    seeds = sorted({(face, i) for face, rows in contacts.items()
                    for i, _ in rows if regions[i]["kind"] == "plane"})
    local = []
    if seeds:
        p = mesh.triangles_center
        matches = cKDTree(p[[f for f, _ in seeds]]).query_pairs(4 * step)
        pairs = {tuple(sorted((seeds[a][1], seeds[b][1]))) for a, b in matches if seeds[a][1] != seeds[b][1]}
        for i, j in sorted(pairs):
            (a, da), (b, db) = _plane(regions[i]), _plane(regions[j])
            axis = np.cross(a, b)
            if np.linalg.norm(axis) < .18:
                continue
            normals = np.array([a, b])
            line = normals.T @ np.linalg.solve(normals @ normals.T, [da, db])
            axis /= np.linalg.norm(axis)
            distance = np.linalg.norm(np.cross(p[free] - line, axis), axis=1)
            local.extend(_groups(free[distance <= 1.1 * tol], neighbors))
    output = []
    checked = set()
    for group in [*_groups(free, neighbors), *local]:
        key = frozenset(group)
        if key in checked:
            continue
        checked.add(key)
        rails = defaultdict(list)
        for i in group:
            for support, edge in contacts[i]:
                rails[support].append(edge)
        if len(rails) < 2:
            continue
        segments = {k: np.linalg.norm(V[e][:, 1] - V[e][:, 0], axis=1) for k, e in rails.items()}
        lengths = {k: e.sum() for k, e in segments.items()}
        pair = tuple(sorted(lengths, key=lengths.get, reverse=True)[:2])
        local_step = min(step, float(np.median(np.concatenate([segments[k] for k in pair]))))
        if min(lengths[k] for k in pair) < 2 * local_step or any(regions[k]["kind"] != "plane" for k in pair):
            continue
        normals, offsets = map(np.array, zip(*(_plane(regions[k]) for k in pair)))
        axis = np.cross(*normals)
        if np.linalg.norm(axis) < .18:
            continue
        axis /= np.linalg.norm(axis)
        vertices = np.unique(F[group])
        p = V[vertices]
        length = np.ptp(p @ axis)
        area = mesh.area_faces[group].sum()
        if length < 3 * local_step or area / length > 3 * local_step:
            continue
        bounds = [np.unique(rails[k]) for k in pair]
        axial = [V[b] @ axis for b in bounds]
        if min(a.max() for a in axial) - max(a.min() for a in axial) < .55 * length:
            continue
        offsets_at_points = p @ normals.T - offsets
        gap = np.min(np.abs(offsets_at_points), axis=1)
        line = normals.T @ np.linalg.solve(normals @ normals.T, offsets)
        # Certify that both observed rails can be joined by the extended planes.
        distances = [np.linalg.norm(np.cross(V[b] - line, axis), axis=1) for b in bounds]
        common = {"faces": group, "planes": pair, "rails": bounds, "origin": line, "axis": axis}
        if max(float(d.max()) for d in distances) <= tol and gap.max() <= tol:
            output.append({**common, "kind": "sharp"})
            continue
        center = p.mean(axis=0)
        normal = np.linalg.svd(p - center, full_matrices=False)[2][-1]
        cross = np.sum(mesh.face_normals[group] * mesh.area_faces[group, None], axis=0)
        if normal @ cross < 0:
            normal = -normal
        # A bevel must be supported at both rails and not be a continuation of either side.
        # Use a stricter planar error budget to avoid explaining shallow circular arcs as bevels.
        residual = np.abs((p - center) @ normal)
        if (np.max(np.abs(normals @ normal)) < np.cos(np.radians(12.))
                and np.percentile(residual, 95) <= .18 * tol and residual.max() <= .35 * tol):
            output.append({**common, "kind": "chamfer", "origin": center, "normal": normal})
    claimed = set(int(i) for row in output if row["kind"] == "sharp" for i in row["faces"])
    blends = fillets(mesh, {"label": label, "regions": regions}, tol, 0.)
    for row in blends:
        if (label[row["faces"]] < 0).all() and not claimed.intersection(row["faces"]):
            accepted = set(map(int, row["faces"]))
            output = [other for other in output if not accepted.intersection(other["faces"])]
            output.append(row)
    chosen, result = set(), []
    for row in sorted(output, key=lambda row: (row["kind"] != "sharp", -len(row["faces"]))):
        if not chosen.intersection(row["faces"]):
            result.append(row)
            chosen.update(row["faces"])
    return result


def _sharp(V, F, label, row, regions, tol):
    """Split each triangle along the plane bisector, keeping all exterior edges."""
    ids = row["faces"]
    chosen = set(map(int, ids))
    pair = row["planes"]
    p0, p1 = (_plane(regions[k]) for k in pair)
    rails = [set(map(int, b)) for b in row["rails"]]
    incidence = defaultdict(set)
    for i, tri in enumerate(F):
        for v in tri:
            incidence[int(v)].add(i)
    local = np.unique(F[ids])
    outside = {v for v in local if not incidence[v] <= chosen}
    a, b = (V[local] @ p[0] - p[1] for p in (p0, p1))
    field = dict(zip(local, np.abs(b) - np.abs(a)))
    points = list(V)
    for v in local:
        side = 0 if field[v] >= 0 else 1
        if v in outside:
            if v not in rails[side] or abs((p0, p1)[side][0] @ V[v] - (p0, p1)[side][1]) > 1e-5:
                return None
            continue
        normal, offset = (p0, p1)[side]
        points[v] = V[v] - (V[v] @ normal - offset) * normal
    cuts = {}
    uses = topology(F[ids])
    made, labels, originals = [], [], []
    for i in ids:
        tri = F[i]
        for side, sign in enumerate((1, -1)):
            poly = []
            for u, v in zip(tri, np.roll(tri, -1)):
                gu, gv = sign * field[u], sign * field[v]
                if gu >= 0:
                    poly.append(u)
                if gu * gv < 0:
                    key = tuple(sorted((u, v)))
                    if key not in cuts:
                        raw = (gv * V[u] - gu * V[v]) / (gv - gu)
                        line, axis = row["origin"], row["axis"]
                        point = line + axis * ((raw - line) @ axis)
                        if np.linalg.norm(raw - point) > tol:
                            return None
                        cuts[key] = len(points)
                        points.append(point)
                    poly.append(cuts[key])
            for j in range(1, len(poly) - 1):
                made.append((poly[0], poly[j], poly[j + 1]))
                labels.append(pair[side])
                originals.append(i)
    if not made:
        return None
    points, made = np.asarray(points), np.asarray(made)
    xyz, old = points[made], V[F[originals]]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    old_cross = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-10) or np.any(np.einsum("ij,ij->i", cross, old_cross) <= 0):
        return None
    before = Counter(k for k, rows in uses.items() if len(rows) == 1)
    fringe = {}
    for (u, v) in tuple(before):
        cut = cuts.get((u, v))
        if cut is None:
            continue
        touching = (incidence[u] & incidence[v]) - chosen
        if len(touching) != 1:
            return None
        i = next(iter(touching))
        support = label[i]
        if i in fringe or support < 0 or regions[support]["kind"] != "plane":
            return None
        n, d = _plane(regions[support])
        if abs(points[cut] @ n - d) > 1e-5:
            return None
        tri = F[i]
        k = next(j for j in range(3) if {tri[j], tri[(j + 1) % 3]} == {u, v})
        a, b, c = np.roll(tri, -k)
        replacement = [(a, cut, c), (cut, b, c)]
        xyz = points[replacement]
        crosses = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        original = np.cross(V[b] - V[a], V[c] - V[a])
        if np.any(crosses @ original <= 1e-12):
            return None
        fringe[i] = replacement
        del before[u, v]
        before[tuple(sorted((u, cut)))] += 1
        before[tuple(sorted((v, cut)))] += 1
    after = Counter(k for k, rows in topology(made).items() if len(rows) == 1)
    if before != after:
        return None
    # Keep cap triangle indices stable for later, disjoint strips.
    outer = F.copy()
    for i, (first, second) in fringe.items():
        outer[i] = first
        made = np.vstack((made, second))
        labels.append(label[i])
    keep = np.ones(len(F), bool)
    keep[ids] = False
    return points, np.vstack((outer[keep], made)), np.r_[label[keep], labels]


def resolve(V, F, label, regions, tol, *, audit=None):
    """Return edited arrays. Append accepted region records to the caller's list."""
    V, F, label = np.array(V, copy=True), np.array(F, copy=True), np.array(label, copy=True)
    rows = classify(V, F, label, regions, tol)
    log = Counter(proposed=len(rows))
    # Process non-topological changes before replacing triangles.
    for row in rows:
        ids = row["faces"]
        if row["kind"] == "chamfer":
            n, o = row["normal"], row["origin"]
            unique = np.unique(F[ids])
            outside = np.setdiff1d(unique, np.unique(F[np.setdiff1d(np.arange(len(F)), ids)]))
            border = np.setdiff1d(unique, outside)
            if np.any(np.abs((V[border] - o) @ n) > 1e-5):
                continue
            candidate = V.copy()
            candidate[outside] -= ((V[outside] - o) @ n)[:, None] * n
            old, new = V[F[ids]], candidate[F[ids]]
            a = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
            b = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
            if np.any(np.einsum("ij,ij->i", a, b) <= 0):
                continue
            V = candidate
            rec = {"kind": "plane", "normal": n, "origin": o}
        elif row["kind"] == "fillet":
            rec = {**row, "kind": "cylinder", "r": row["radius"], "fillet_planes": row["planes"]}
        else:
            continue
        rec["faces"] = ids
        rec["area"] = float(trimesh.triangles.area(V[F[ids]]).sum())
        label[ids] = len(regions)
        regions.append(rec)
        log[row["kind"]] += 1
    remap = np.arange(len(F))
    for row in (r for r in rows if r["kind"] == "sharp"):
        ids = remap[row["faces"]]
        result = _sharp(V, F, label, {**row, "faces": ids}, regions, tol)
        if result is None:
            log["sharp_rejected"] += 1
            continue
        keep = np.ones(len(F), bool)
        keep[ids] = False
        inverse = np.full(len(F), -1, int)
        inverse[keep] = np.arange(np.count_nonzero(keep))
        valid = remap >= 0
        remap[valid] = inverse[remap[valid]]
        V, F, label = result
        log["sharp"] += 1
    areas = trimesh.triangles.area(V[F])
    for i, region in enumerate(regions):
        ids = np.flatnonzero(label == i)
        region.update(faces=ids, area=float(areas[ids].sum()))
    if audit is not None:
        audit.update(log)
    return V, F, label
