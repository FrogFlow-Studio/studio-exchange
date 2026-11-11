"""Разметка поверхности на плоскости, цилиндры, конусы, сферы и свободную форму.

Каждый треугольник чистой поверхности получает номер области или -1 (свободная форма).
Плоскости берутся ростом области от крупных треугольников (walls.planar_patches), цилиндры —
ростом от затравки: по соседям оценивается ось (направление, которому перпендикулярны все
нормали) и окружность в поперечной плоскости, затем область растёт, пока треугольники лежат
на этом цилиндре. Способ общий: размеры и положение берутся только из самой сетки.

python regions.py скан.stl [папка]   — печатает доли площади по видам областей
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import trimesh


def _neighbours(mesh: trimesh.Trimesh) -> list[list[int]]:
    nb = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nb[a].append(b)
        nb[b].append(a)
    return nb


def _circle(q: np.ndarray):
    """Окружность по точкам на плоскости (алгебраическая подгонка): центр, радиус."""
    A = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q**2).sum(1), rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def _cylinder(P: np.ndarray, N: np.ndarray, W: np.ndarray):
    """Цилиндр по точкам и нормалям: (ось, точка на оси, радиус, знак) или None.
    Знак +1 — нормали наружу от оси (вал, скругление снаружи), -1 — к оси (отверстие, галтель)."""
    if len(P) < 6:
        return None
    _, s, vt = np.linalg.svd(N * np.sqrt(W)[:, None], full_matrices=False)
    # нормали цилиндра лежат в одной плоскости: третье сингулярное число мало, второе — нет
    if s[1] < 0.12 * s[0] or s[2] > 0.5 * s[1]:
        return None
    ax = vt[2]
    u = vt[0] - (vt[0] @ ax) * ax
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    q = np.c_[P @ u, P @ v]
    c2, r = _circle(q)
    if not np.isfinite(r) or r < 1e-3:
        return None
    c = c2[0] * u + c2[1] * v
    rad = P - c
    rad -= np.outer(rad @ ax, ax)
    d = np.linalg.norm(rad, axis=1)
    sign = 1.0 if float(np.average(((rad / np.maximum(d, 1e-9)[:, None]) * N).sum(1), weights=W)) > 0 else -1.0
    return ax, c, r, sign


def _on_cylinder(P, N, cyl, tol: float, cos_n: float) -> np.ndarray:
    ax, c, r, sign = cyl
    rad = P - c
    rad -= np.outer(rad @ ax, ax)
    d = np.linalg.norm(rad, axis=1)
    rn = rad / np.maximum(d, 1e-9)[:, None]
    return (np.abs(d - r) < tol) & ((rn * N).sum(1) * sign > cos_n)



def _components(mask, nb):
    unseen = set(np.flatnonzero(mask))
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        members = [seed]
        while queue:
            for g in nb[queue.popleft()]:
                if g in unseen:
                    unseen.remove(g)
                    queue.append(g)
                    members.append(g)
        yield np.array(members)


def _refine_cylinder(vertices, cyl, tol):
    """Fit geometry at vertices. The centres of triangles sit inside a circular chord."""
    from scipy.optimize import least_squares

    axis, origin, radius, sign = cyl
    u, v = _frame(axis)
    center = vertices.mean(axis=0)
    pts = vertices - center
    offset = origin - center

    def parameters(x):
        n = axis + x[2] * u + x[3] * v
        return n / np.linalg.norm(n), x[0] * u + x[1] * v

    def residual(x):
        n, o = parameters(x)
        d = pts - o
        return np.linalg.norm(d - np.outer(d @ n, n), axis=1) - x[4]

    start = [offset @ u, offset @ v, 0, 0, radius]
    fit = least_squares(residual, start, loss="soft_l1", f_scale=tol / 3, max_nfev=20)
    if fit.x[4] <= 0:
        return cyl
    ax, c = parameters(fit.x)
    return ax, center + c, float(fit.x[4]), sign


def _frame(axis):
    u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _axial_drift(h, errors, tol, *, floor=None, trim_share=0.125):
    """A cone or a doubly curved band can fit a cylinder locally but drifts along its axis."""
    order = np.argsort(h)
    trim = int(trim_share * len(order))
    if trim:
        order = order[trim:-trim]
    sections = np.array_split(order, 4)
    centers = np.array([np.median(h[s]) for s in sections])
    if np.ptp(centers) < 3 * tol:
        return False
    systematic = np.array([np.median(errors[s]) for s in sections])
    scatter = 1.4826 * np.median(np.concatenate([
        np.abs(errors[s] - z) for s, z in zip(sections, systematic)
    ]))
    return np.ptp(systematic) > max(tol / 3 if floor is None else floor, 2 * scatter)


def _cylinder_region(mesh, m, cyl, min_area, tol, *, sectional=False):
    ax, c, r, sign = cyl
    fc = mesh.triangles_center[m]
    rad = fc - c
    h = rad @ ax
    rad -= np.outer(h, ax)
    u, v = _frame(ax)
    s = np.sort(np.arctan2(rad @ v, rad @ u))
    gap = np.max(np.diff(np.r_[s, s[0] + 2 * np.pi]))
    span = float(np.degrees(2 * np.pi - gap))
    area = float(mesh.area_faces[m].sum())
    if area < min_area or span < 40:
        return None
    points = mesh.vertices[np.unique(mesh.faces[m])] - c
    z = points @ ax
    residual = np.linalg.norm(points - np.outer(z, ax), axis=1) - r
    core = np.abs(residual) < tol
    if core.mean() < (0.5 if sectional else 0.7):
        return None
    rms = float(np.sqrt(np.mean(residual[core]**2)))
    sag = r * (1 - np.cos(np.radians(min(span, 180.0)) / 2))
    normal_axial = mesh.face_normals[m] @ ax
    # The normals near rounded rims are unreliable. Compare the axial drift inside
    # the wall to distinguish a cylinder from the middle band of a torus.
    if (sag < max(3 * tol, 8 * rms) or rms > tol / 2
            or abs(np.median(normal_axial)) > np.sin(np.radians(6))
            or (sectional and _axial_drift(h, normal_axial, tol, floor=0.09, trim_share=0.25))
            or np.ptp(h) < 4 * tol
            or (not sectional and _axial_drift(z[core], residual[core], tol, floor=0.8 * tol))):
        return None
    return {"kind": "cylinder", "axis": ax, "origin": c + ax * float(h.mean()), "r": float(r),
            "inner": bool(sign < 0), "span_deg": round(span, 1), "length": float(np.ptp(h)),
            "area": area, "faces": m}


def _circle_from_loop(q, tol):
    """Fit the circular arc even if a chip closes part of the loop with a chord."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    t = np.linspace(0, arc[-1], 64, endpoint=False)
    xy = np.column_stack([np.interp(t, arc, q[:, i]) for i in range(2)])
    for sample in (xy, *(xy[(i + np.array([0, 21, 42])) % 64] for i in range(0, 64, 4))):
        c, r = _circle(sample)
        if r < 3 * tol:
            continue
        err = np.abs(np.linalg.norm(xy - c, axis=1) - r)
        good = err < tol
        if good.mean() < 0.6 or np.sqrt(np.mean(err[good]**2)) > tol / 2:
            continue
        v = xy[good] - c
        angle = np.sort(np.arctan2(v[:, 1], v[:, 0]))
        if np.max(np.diff(np.r_[angle, angle[0] + 2 * np.pi])) <= np.pi:
            return _circle(xy[good])
    return None


def _section_regions(mesh, patches, nb, label, tol, min_area):
    """Cross sections seed cylindrical and conical walls through noisy normals."""
    area, fn, fc = mesh.area_faces, mesh.face_normals, mesh.triangles_center
    seen = []
    seen_cones = []
    for p in patches:
        points = mesh.vertices[np.unique(mesh.faces[p["faces"]])]
        origin = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
        n = axes[-1] * np.sign(axes[-1] @ p["normal"])
        if np.quantile(np.abs((points - origin) @ n), 0.9) > tol / 2:
            continue
        depths = np.array([2, 4, 8, 12, 16, 24, 32]) * tol
        sections = []
        paths = mesh.section_multiplane(plane_origin=origin, plane_normal=n, heights=-depths)
        for section in paths:
            circles = []
            for loop in section.discrete if section is not None else ():
                if len(loop) < 7 or np.linalg.norm(loop[0] - loop[-1]) > tol:
                    continue
                circle = _circle_from_loop(loop, tol)
                if circle is not None:
                    c, r = circle
                    transform = section.metadata["to_3D"]
                    circles.append((transform[:3, :2] @ c + transform[:3, 3], r))
            sections.append(circles)
        for j in range(len(depths) - 1, 0, -1):
            for c, radius in sections[j]:
                if any(abs(n @ a) > 0.95 and abs(radius - rr) < tol
                       and np.linalg.norm(np.cross(c - cc, a)) < 2 * tol for a, cc, rr in seen):
                    continue
                pairs = [(depths[j] - depths[i], cp, rp) for i in range(j)
                         for cp, rp in sections[i] if abs(radius - rp) < tol / 3
                         and np.linalg.norm(np.cross(cp - c, n)) < tol / 2]
                if not pairs:
                    continue
                span, cp, rp = max(pairs, key=lambda pair: pair[0])
                if span < 3 * tol:
                    continue
                axis = cp - c
                axis /= np.linalg.norm(axis)
                h = (fc - c) @ axis
                rad = fc - c - np.outer(h, axis)
                d = np.linalg.norm(rad, axis=1)
                dot = np.einsum("ij,ij->i", rad, fn) / np.maximum(d, 1e-9)
                mask = (np.abs(d - radius) < tol) & (np.abs(dot) > 0.15) & (label < 0)
                for m in sorted(_components(mask, nb), key=lambda f: -area[f].sum()):
                    if area[m].sum() < min_area:
                        break
                    if np.min(h[m]) > tol or np.max(h[m]) < span - tol:
                        continue
                    alignment = np.average(dot[m], weights=area[m])
                    if abs(alignment) < 0.55:
                        continue
                    pts = mesh.vertices[np.unique(mesh.faces[m])]
                    z = (pts - c) @ axis
                    core = pts[(z > -tol) & (z < span + tol)]
                    seed = (axis, c, (radius + rp) / 2, np.sign(alignment))
                    cyl = _refine_cylinder(core, seed, tol) if len(core) >= 12 else seed
                    if abs(cyl[0] @ n) < 0.995 or abs(cyl[2] - radius) > tol:
                        cyl = seed
                    reg = _cylinder_region(mesh, m, cyl, min_area, tol, sectional=True)
                    if reg is not None and reg["span_deg"] >= 180:
                        seen.append(cyl[:3])
                        yield reg
                        break

        for j in range(len(depths) - 1, 1, -1):
            for c, radius in sections[j]:
                if any(np.linalg.norm(np.cross(c - apex, axis)) < 2 * tol
                       and abs((c - apex) @ axis * slope - radius) < tol
                       for apex, axis, slope in seen_cones):
                    continue
                for i in range(j - 1):
                    dz = depths[j] - depths[i]
                    for cp, rp in sections[i]:
                        slope = (rp - radius) / dz
                        if abs(rp - radius) < 2 * tol or np.linalg.norm(np.cross(cp - c, n)) > tol:
                            continue
                        if not any(abs(rm - radius - (depths[j] - depths[k]) * slope) < tol / 2
                                   and np.linalg.norm(np.cross(cm - c, n)) < tol
                                   for k in range(i + 1, j) for cm, rm in sections[k]):
                            continue
                        axis = (cp - c) / np.linalg.norm(cp - c) * np.sign(slope)
                        model = (c - axis * radius / abs(slope), axis, abs(slope))
                        err, normal = _curved_distance(fc, model, "cone")
                        dot = np.einsum("ij,ij->i", normal, fn)
                        mask = (np.abs(err) < tol) & (np.abs(dot) > 0.5) & (label < 0)
                        for m in sorted(_components(mask, nb), key=lambda f: -area[f].sum()):
                            if area[m].sum() < min_area:
                                break
                            reg = _curved_region(mesh, m, model, "cone", tol, min_area)
                            if reg is not None:
                                seen_cones.append(model)
                                yield reg
                                break


def _sphere(points):
    if len(points) < 6:
        return None
    mean = points.mean(axis=0)
    p = points - mean
    a = np.c_[2 * p, np.ones(len(p))]
    x, _, rank, _ = np.linalg.lstsq(a, np.einsum("ij,ij->i", p, p), rcond=None)
    if rank < 4 or x[3] + x[:3] @ x[:3] <= 0:
        return None
    return x[:3] + mean, float(np.sqrt(x[3] + x[:3] @ x[:3]))


def _cone(points, normals):
    """Normals of a cone lie on a plane offset from zero on the unit sphere."""
    n = normals - normals.mean(axis=0)
    _, s, vt = np.linalg.svd(n, full_matrices=False)
    if s[1] < 0.15 * s[0] or s[2] > 0.35 * s[1]:
        return None
    ax = vt[2]
    u, v = _frame(ax)
    mean = points.mean(axis=0)
    p = points - mean
    q = np.c_[p @ u, p @ v]
    h = p @ ax
    # Expand ||q - center||^2 = (r0 + slope*h)^2 and solve for center and slope.
    a = np.c_[2 * q, np.ones(len(q)), h, h**2]
    x, _, rank, _ = np.linalg.lstsq(a, np.einsum("ij,ij->i", q, q), rcond=None)
    r2 = x[2] + x[:2] @ x[:2]
    if rank < 5 or r2 <= 0:
        return None
    r = np.sqrt(r2)
    slope = x[3] / (2 * r)
    if abs(slope) < 1e-5 or np.any(r + slope * h <= 0):
        return None
    apex = mean + x[0] * u + x[1] * v - ax * r / slope
    return apex, ax * np.sign(slope), abs(float(slope))


def _curved_distance(p, model, kind):
    if kind == "sphere":
        center, radius = model
        rad = p - center
        d = np.linalg.norm(rad, axis=1)
        return d - radius, rad / np.maximum(d, 1e-9)[:, None]
    apex, axis, slope = model
    rad = p - apex
    h = rad @ axis
    rad -= np.outer(h, axis)
    d = np.linalg.norm(rad, axis=1)
    normal = rad / np.maximum(d, 1e-9)[:, None] - axis * slope
    scale = np.hypot(1, slope)
    return (d - h * slope) / scale, normal / scale


def _refine_cone(points, model, tol):
    from scipy.optimize import least_squares

    apex, axis, slope = model
    center = points.mean(axis=0)
    h = float((center - apex) @ axis)
    origin = apex + axis * h
    radius = h * slope
    u, v = _frame(axis)
    local = points - origin

    def params(x):
        ax = axis + x[2] * u + x[3] * v
        ax /= np.linalg.norm(ax)
        return x[0] * u + x[1] * v, ax, x[4], x[5]

    def residual(x):
        c, ax, r, k = params(x)
        d = local - c
        z = d @ ax
        return (np.linalg.norm(d - np.outer(z, ax), axis=1) - r - k * z) / np.hypot(1, k)

    x = least_squares(residual, [0, 0, 0, 0, radius, slope], loss="soft_l1",
                      f_scale=tol / 3, max_nfev=20).x
    c, ax, r, k = params(x)
    if abs(k) < 1e-5 or r <= 0:
        return None
    return origin + c - ax * r / k, ax * np.sign(k), abs(float(k))


def _curved_region(mesh, m, model, kind, tol, min_area):
    area = float(mesh.area_faces[m].sum())
    if area < min_area or len(m) < 8:
        return None
    points = mesh.vertices[np.unique(mesh.faces[m])]
    if len(points) < 8:
        return None
    model = _sphere(points) if kind == "sphere" else _refine_cone(points, model, tol)
    if model is None:
        return None
    errors, _ = _curved_distance(points, model, kind)
    rms = np.sqrt(np.mean(errors**2))
    if rms > tol / 2 or np.quantile(np.abs(errors), 0.95) > tol:
        return None
    fc = mesh.triangles_center[m]
    fn = mesh.face_normals[m]
    _, normals = _curved_distance(fc, model, kind)
    alignment = np.einsum("ij,ij->i", normals, fn)
    weights = mesh.area_faces[m]
    inner = float(np.average(alignment, weights=weights)) < 0
    if kind == "sphere":
        center, radius = model
        _, s, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        # Detect bending along two independent directions, not a narrow cylindrical strip.
        width = np.ptp((points - center) @ axes[1])
        sag = radius - np.sqrt(max(0., radius**2 - (width / 2)**2))
        if sag < max(5 * tol, 10 * rms) or s[1] < 0.35 * s[0]:
            return None
        # A sphere has the same normal gradient along both tangent directions.
        # A toroidal band can have low distance error yet disagree on this gradient.
        x = (fc - np.average(fc, axis=0, weights=weights)) @ axes[:2].T
        measured = fn @ axes[:2].T * (-1 if inner else 1)
        y = measured - np.average(measured, axis=0, weights=weights)
        w = np.sqrt(weights)[:, None]
        grad = np.linalg.lstsq(x * w, y * w, rcond=None)[0]
        if np.linalg.norm(radius * grad - np.eye(2), ord=2) > 0.3:
            return None
        centered = points - center
        if any(_axial_drift(centered @ direction, errors, tol, floor=tol / 5)
               for direction in axes):
            return None
        params = {"center": center, "r": radius}
    else:
        apex, ax, slope = model
        p = points - apex
        h = p @ ax
        u, v = _frame(ax)
        s = np.sort(np.arctan2(p @ v, p @ u))
        span = 2 * np.pi - np.max(np.diff(np.r_[s, s[0] + 2 * np.pi]))
        r = float(np.median(h)) * slope
        sag = r * (1 - np.cos(min(span, np.pi) / 2))
        if (h.min() < -tol or sag / np.hypot(1, slope) < max(3 * tol, 8 * rms)
                or span < np.radians(100) or slope * np.ptp(h) < max(5 * tol, 10 * rms)):
            return None
        # Reject non-linear meridians even if a small residual fits within tolerance.
        if _axial_drift(h, errors, tol):
            return None
        axial_normals = fn @ ax * (-1 if inner else 1)
        order = np.argsort((fc - apex) @ ax)
        bands = np.array_split(order, 5)[1:-1]
        levels = np.array([np.median(axial_normals[b]) for b in bands])
        expected = -slope / np.hypot(1, slope)
        if np.max(np.abs(levels - expected)) > 0.08 or np.ptp(levels) > 0.065:
            return None
        _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        if np.ptp((points - points.mean(axis=0)) @ axes[-1]) < 5 * tol:
            return None
        params = {"apex": apex, "axis": ax, "semi_angle_deg": float(np.degrees(np.arctan(slope)))}
    return {"kind": kind, **params, "inner": inner, "area": area, "faces": m}


def _curved_components(mesh, label, nb, tol, min_area, cos_n):
    area, fc, fn = mesh.area_faces, mesh.triangles_center, mesh.face_normals
    adj = mesh.face_adjacency
    graph = [[] for _ in nb]
    for a, b in adj[mesh.face_adjacency_angles < np.radians(35)]:
        graph[a].append(b)
        graph[b].append(a)

    def candidate(seed, model, kind):
        if model is None:
            return None
        err, normal = _curved_distance(fc, model, kind)
        dot = np.einsum("ij,ij->i", normal, fn)
        direction = np.sign(np.median(dot[seed]))
        mask = (np.abs(err) < tol) & (dot * direction > cos_n) & (label < 0)
        if area[seed[mask[seed]]].sum() < 0.65 * area[seed].sum():
            return None
        for m in sorted(_components(mask, nb), key=lambda f: -area[f].sum()):
            if area[m].sum() < min_area:
                break
            if np.intersect1d(m, seed).size < 0.4 * len(seed):
                continue
            reg = _curved_region(mesh, m, model, kind, tol, min_area)
            if reg is not None:
                return reg
        return None

    def models(f):
        points = mesh.vertices[np.unique(mesh.faces[f])]
        return ("sphere", _sphere(points)), ("cone", _cone(fc[f], fn[f]))

    for m in sorted(_components(label < 0, graph), key=lambda f: -area[f].sum()):
        if len(m) < 12 or area[m].sum() < min_area:
            continue
        for kind, model in models(m):
            r = candidate(m, model, kind)
            if r is not None:
                yield r
                break
        # Components can include a blended rim, a chip or an adjacent tangent surface.
        # Sample spaced geodesic neighborhoods so these do not bias the initial model.
        tried = set()
        for seed in m[np.argsort(-area[m])]:
            if seed in tried or label[seed] >= 0:
                continue
            queue, seen, members = deque([seed]), {seed}, [seed]
            for size in (96, 384, 1200):
                while queue and len(members) < size:
                    for g in graph[queue.popleft()]:
                        if g not in seen and label[g] < 0:
                            seen.add(g)
                            members.append(g)
                            queue.append(g)
                if len(members) < 12:
                    break
                f = np.asarray(members)
                if area[f].sum() >= min_area:
                    for kind, model in models(f):
                        r = candidate(f, model, kind)
                        if r is not None:
                            yield r
                            break
                    if label[seed] >= 0:
                        break
                if not queue:
                    break
            tried.update(members[:96])


def _propose(mesh: trimesh.Trimesh, tol: float = 0.3, min_area: float | None = None, max_angle: float = 12.0) -> dict:
    """Разметка треугольников с полями label, regions и area.

    label содержит индекс области либо -1. У каждой области есть kind, faces
    и area. Ось конуса направлена от apex к растущему радиусу, semi_angle_deg
    задаётся в градусах. Сфера задаётся через center и r.
    """
    import walls

    area = mesh.area_faces
    total = float(area.sum())
    if min_area is None:  # область мельче этого не стоит точной грани: доля всей поверхности
        min_area = max(4.0, 0.0015 * total)
    fn, fc = mesh.face_normals, mesh.triangles_center
    nb = _neighbours(mesh)
    label = -np.ones(len(mesh.faces), int)
    regions: list[dict] = []

    cos_n = float(np.cos(np.radians(2 * max_angle)))
    cos_c = float(np.cos(np.radians(25.0)))
    patches = walls.planar_patches(mesh, max_angle=max_angle, max_dist=2 * tol, min_area=min_area)
    for detect in (_section_regions(mesh, patches, nb, label, tol, min_area),
                   _curved_components(mesh, label, nb, tol, min_area, cos_n)):
        for r in detect:
            label[r["faces"]] = len(regions)
            regions.append(r)
    tried = np.zeros(len(mesh.faces), bool)
    for seed in np.argsort(-area):
        if label[seed] >= 0 or tried[seed]:
            continue
        # затравка: кольца свободных соседей; у крупного цилиндра малое кольцо почти плоское,
        # поэтому кольцо растёт, пока нормали не разойдутся достаточно для оценки оси
        tried[seed] = True
        ring, q, seen, cyl = [seed], deque([seed]), {seed}, None
        for size in (60, 250, 1000, 4000):
            while q and len(ring) < size:
                f = q.popleft()
                for g in nb[f]:
                    # через излом кольцо не идёт: за ребром уже другая поверхность
                    if g not in seen and label[g] < 0 and fn[f] @ fn[g] > cos_c:
                        seen.add(g)
                        ring.append(g)
                        q.append(g)
            if len(ring) < 6:
                break
            r_ = np.asarray(ring)
            cyl = _cylinder(fc[r_], fn[r_], area[r_])
            if cyl is not None and _on_cylinder(fc[r_], fn[r_], cyl, tol, cos_n).mean() >= 0.8:
                break
            cyl = None
            if not q:
                break
        if cyl is None:
            tried[np.asarray(ring[:60])] = True  # рядом с неудачной затравкой заново не пробуем
            continue
        members: list[int] = []
        for _ in range(4):  # рост и уточнение цилиндра по набранной области
            ok = _on_cylinder(fc, fn, cyl, tol, cos_n) & (label < 0)
            if not ok[seed]:
                break
            members, q, got = [seed], deque([seed]), {seed}
            while q:
                f = q.popleft()
                for g in nb[f]:
                    if g not in got and ok[g]:
                        got.add(g)
                        members.append(g)
                        q.append(g)
            m = np.asarray(members)
            new = _cylinder(fc[m], fn[m], area[m])
            if new is None:
                members = []
                break
            if abs(new[2] - cyl[2]) < 0.02 * cyl[2] and abs(float(new[0] @ cyl[0])) > 0.9995:
                cyl = new
                break
            cyl = new
        if not members:
            continue
        m = np.asarray(members)
        points = mesh.vertices[np.unique(mesh.faces[m])]
        cyl = _refine_cylinder(points, cyl, tol)
        r = _cylinder_region(mesh, m, cyl, min_area, tol)
        if r is None:
            tried[m] = True
            continue
        label[m] = len(regions)
        regions.append(r)
    for p in patches:
        f = np.asarray(p["faces"])
        # в плоскость идут только треугольники, действительно лежащие на ней: рост области допускает больше
        n, o = np.asarray(p["normal"]), np.asarray(p["point"])
        f = f[(np.abs((fc[f] - o) @ n) < tol) & (label[f] < 0)]  # цилиндры уже размечены: узкие полосы на них плоскостями не считаются
        if area[f].sum() < min_area:
            continue
        # плоскость на кривой оболочке — кусок в пределах допуска, но не плоскость: у настоящей
        # вершины лежат заметно ближе допуска
        v = mesh.vertices[np.unique(mesh.faces[f])]
        if float(np.sqrt((((v - o) @ n) ** 2).mean())) > 0.5 * tol:
            continue
        label[f] = len(regions)
        regions.append({"kind": "plane", "normal": n, "origin": o, "area": float(area[f].sum()), "faces": f})

    return {"label": label, "regions": regions, "area": total}


def _refit_scan_cylinder(r, f, evidence, tol):
    raw = evidence.scan_points(f)
    if raw is not None and len(raw) >= 30:
        h = (raw - r["origin"]) @ r["axis"]
        lower, upper = np.quantile(h, [0.15, 0.85])
        core = raw[(h > lower) & (h < upper)]
        seed = r["axis"], r["origin"], r["r"], -1. if r["inner"] else 1.
        ax, o, rad, _ = (_refine_cylinder(core, seed, max(tol, evidence.noise))
                          if len(core) >= 12 else seed)
        axis_shift = np.linalg.norm(np.cross(o - r["origin"], r["axis"]))
        if (abs(rad - r["r"]) < tol and axis_shift < tol
                and abs(ax @ r["axis"]) >= np.cos(np.radians(3))):
            points = evidence.fc[f] - o
            z = points @ ax
            u, v = _frame(ax)
            azimuth = np.sort(np.arctan2(points @ v, points @ u))
            gap = np.max(np.diff(np.r_[azimuth, azimuth[0] + 2 * np.pi]))
            r = {**r, "axis": ax, "origin": o + ax * float(z.mean()), "r": rad,
                 "length": float(np.ptp(z)), "span_deg": round(float(np.degrees(2 * np.pi - gap)), 1)}
    return r


def segment(mesh: trimesh.Trimesh, tol: float = 0.3, min_area: float | None = None,
            max_angle: float = 12.0, *, scan: trimesh.Trimesh | None = None) -> dict:
    """Label clean mesh faces, using optional raw scan observations for the fits.

    boundary_support is the share of a proposal's shared-edge perimeter near an
    observed change in normals. A cone can also have plane_boundary_support.
    """
    from region_support import Evidence
    import walls

    proposed = _propose(mesh, tol, min_area, max_angle)
    if len(mesh.faces) == 0:
        return proposed
    evidence = Evidence(mesh, scan, tol)
    area = mesh.area_faces
    if min_area is None:
        min_area = max(4., 0.0015 * proposed["area"])
    label = np.full(len(area), -1, dtype=int)
    out = []

    def accept(r, support):
        faces = r["faces"]
        if r["area"] < min_area:
            return
        label[faces] = len(out)
        out.append({**r, "boundary_support": support})

    nb = _neighbours(mesh)
    guide_centers = evidence.guide.triangles_center
    band = max(tol, 2.5 * evidence.noise)
    support_min = evidence.boundary_minimum
    fit_tol = min(2 * tol, max(tol, 2 * evidence.noise))
    for r in sorted(proposed["regions"], key=lambda p: -p["area"]):
        f = r["faces"]
        free = f[label[f] < 0]
        if area[free].sum() < min_area:
            continue
        support = evidence.boundary_share(f)
        kind = r["kind"]
        if kind in ("sphere", "cone") and support < support_min:
            if kind == "sphere":
                model = r["center"], r["r"]
            else:
                model = r["apex"], r["axis"], np.tan(np.radians(r["semi_angle_deg"]))
            distance, normals = _curved_distance(guide_centers, model, kind)
            sign = -1 if r["inner"] else 1
            dot = np.einsum("ij,ij->i", normals, evidence.normals) * sign
            mask = (np.abs(distance) < 1.5 * band) & (dot > np.cos(np.radians(2.5 * max_angle)))
            for m in _components(mask & (label < 0), nb):
                if area[np.intersect1d(m, free)].sum() < 0.6 * area[free].sum():
                    continue
                witness = evidence.boundary_share(m)
                if witness < support_min:
                    continue
                extended = _curved_region(mesh, m, model, kind, fit_tol, min_area)
                if extended is not None:
                    r, support, free = extended, witness, m
                    break
        if support < support_min:
            continue
        if r["kind"] == "plane" and not evidence.stable_plane(f, r["normal"], r["origin"]):
            n, o = evidence.fit_plane(f, r["normal"], r["origin"])
            if not evidence.stable_plane(f, n, o):
                continue
            r = {**r, "normal": n, "origin": o}
        if r["kind"] == "cylinder":
            r = _refit_scan_cylinder(r, free, evidence, tol)
        accept({**r, "faces": free, "area": float(area[free].sum())}, support)

    # Propose from a smoothed normal field, then re-grow from each seed using the
    # signed distance and the normal field. A fragmented seed does not cap coverage.
    normal_angle = max_angle + np.degrees(np.arctan2(evidence.noise, max(tol, evidence.pitch)))
    normal_angle = float(np.clip(normal_angle, max_angle, max_angle * 2))
    patches = walls.planar_patches(evidence.guide, max_angle=normal_angle,
                                  max_dist=max(2 * tol, 3 * evidence.noise), min_area=min_area)
    cos_n = np.cos(np.radians(min(3 * max_angle, 1.8 * normal_angle)))
    confirmed = list(out)
    plane_candidates = []
    for p in sorted(patches, key=lambda p: -p["area"]):
        seeds = p["faces"]
        if area[seeds[label[seeds] < 0]].sum() < min_area:
            continue
        n, o = evidence.fit_plane(seeds, p["normal"], p["point"])
        close = (np.abs((guide_centers - o) @ n) < band) & (evidence.normals @ n > cos_n)
        for faces in _components(close, nb):
            overlap = np.intersect1d(seeds, faces, assume_unique=True)
            added = faces[label[faces] < 0]
            if area[overlap].sum() < min_area or area[added].sum() < min_area:
                continue
            nn, oo = evidence.fit_plane(faces, n, o)
            if not evidence.stable_plane(faces, nn, oo):
                continue
            support = evidence.boundary_share(faces)
            if support >= max(0.25, support_min - 0.08):
                plane_candidates.append({"kind": "plane", "normal": nn, "origin": oo,
                                         "faces": faces, "area": float(area[faces].sum()),
                                         "boundary_support": support})
    all_planes = [r for r in confirmed if r["kind"] == "plane"] + plane_candidates
    plane_candidates.sort(key=lambda r: -r["area"])
    paired_ids = {i for i, p in enumerate(plane_candidates)
                  if any(evidence.opposing_plane(p, q) for q in all_planes if q is not p)}
    paired_planes = [p for i, p in enumerate(plane_candidates) if i in paired_ids]
    for i, p in enumerate(plane_candidates):
        f = p["faces"]
        added = f[label[f] < 0]
        if area[added].sum() < min_area:
            continue
        if i not in paired_ids:
            n = p["normal"]
            witnesses = [r["faces"] for r in (*confirmed, *paired_planes)
                         if r["kind"] != "plane" or n @ r["normal"] < np.cos(np.radians(max_angle))]
            if (p["boundary_support"] < support_min or not witnesses
                    or not evidence.neighboring_support(f, np.concatenate(witnesses))):
                continue
        accept({**p, "faces": added, "area": float(area[added].sum())}, p["boundary_support"])

    # A tapered wall can reach a verified plane even when triangulation
    # and rounding leave a gap between its sampled patch and the sharp edge.
    caps = [p for p in out if p["kind"] == "plane"]
    for r in proposed["regions"]:
        if r["kind"] != "cone":
            continue
        f = r["faces"]
        added = f[label[f] < 0]
        if area[added].sum() < min_area:
            continue
        if area[added].sum() < 0.7 * r["area"]:
            continue
        axis = r["axis"]
        slope = np.tan(np.radians(r["semi_angle_deg"]))
        model = r["apex"], axis, slope
        rim_tol = max(2 * tol, 3 * evidence.noise)
        for p in caps:
            n, o = p["normal"], p["origin"]
            if p["area"] < 0.15 * r["area"]:
                continue
            if np.min(np.abs((evidence.fc[f] - o) @ n)) > 2 * evidence.seam_band:
                continue
            if (evidence.neighboring_support(f, p["faces"], fraction=0.12)
                    or evidence.conical_rim(p["faces"], model, rim_tol)):
                accept({**r, "faces": added, "area": float(area[added].sum()),
                        "plane_boundary_support": True}, evidence.boundary_share(f))
                break

    if evidence.noise >= tol / 2:
        # Reuse the verified planes as section anchors on the normal guide. The
        # original mesh still owns the labels and the independent fit points.
        anchors = [{"normal": p["normal"], "point": p["origin"], "faces": p["faces"]}
                   for p in out if p["kind"] == "plane" and p["area"] >= 3 * min_area]
        for r in _section_regions(evidence.guide, anchors, nb, label, fit_tol, min_area):
            if r["kind"] != "cylinder":
                continue
            f = r["faces"]
            support = evidence.boundary_share(f)
            if support >= support_min:
                r = _refit_scan_cylinder(r, f, evidence, fit_tol)
                accept({**r, "area": float(area[f].sum())}, support)
    return {"label": label, "regions": out, "area": proposed["area"]}


def summary(seg: dict) -> dict:
    out = dict.fromkeys(("plane", "cylinder", "cone", "sphere"), 0.0)
    for r in seg["regions"]:
        out[r["kind"]] += r["area"]
    share = {k: round(v / seg["area"], 4) if seg["area"] else 0.0 for k, v in out.items()}
    share["free"] = round(1 - sum(share.values()), 4)
    share["planes"] = sum(r["kind"] == "plane" for r in seg["regions"])
    share["cylinders"] = sum(r["kind"] == "cylinder" for r in seg["regions"])
    share["cones"] = sum(r["kind"] == "cone" for r in seg["regions"])
    share["spheres"] = sum(r["kind"] == "sphere" for r in seg["regions"])
    return share


if __name__ == "__main__":
    import time

    import surface

    scan = trimesh.load(sys.argv[1], force="mesh")
    t = time.time()
    print("regions: cleaning scan", file=sys.stderr, flush=True)
    clean = surface.surface_mesh(scan)
    print(f"regions: labeling {len(clean.faces)} triangles", file=sys.stderr, flush=True)
    seg = segment(clean, scan=scan)
    print(summary(seg), "с", round(time.time() - t, 1))
    for r in sorted(seg["regions"], key=lambda r: -r["area"]):
        if r["kind"] == "cone":
            print(f"  конус полуугол {r['semi_angle_deg']:.2f}° площадь {r['area']:.0f}")
        elif r["kind"] == "sphere":
            print(f"  сфера R={r['r']:.2f} площадь {r['area']:.0f}")
        elif r["kind"] == "cylinder":
            print(f"  цилиндр R={r['r']:.2f} {'внутр.' if r['inner'] else 'наруж.'} охват {r['span_deg']}° длина {r['length']:.1f} площадь {r['area']:.0f}")
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        col = np.full((len(clean.faces), 4), (190, 190, 190, 255), np.uint8)
        for r in seg["regions"]:
            palette = {"plane": (70, 130, 220, 255), "cone": (180, 110, 210, 255),
                       "sphere": (220, 180, 60, 255)}
            col[r["faces"]] = palette.get(r["kind"], (230, 90, 60, 255) if r.get("inner") else (60, 180, 90, 255))
        clean.visual.face_colors = col
        clean.export(out / "regions.ply")
