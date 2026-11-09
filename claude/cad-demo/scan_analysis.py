"""Шаг 1: проверка скана и «лист замеров» для ИИ.

Без ИИ, только геометрия: диагностика сетки, выравнивание (основная плоскость
вниз, главные оси по X/Y), уровни горизонтальных плоскостей, сечения на
нескольких высотах, окружности (отверстия, бобышки) и контуры.
"""
from __future__ import annotations

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"не удалось прочитать сетку из {path}")
    return mesh


def diagnostics(mesh: trimesh.Trimesh) -> dict:
    ext = mesh.extents
    parts = mesh.split(only_watertight=False)
    note = []
    if ext.max() < 2:
        note.append("габарит меньше 2 единиц: возможно, файл в метрах, проверьте единицы")
    if ext.max() > 3000:
        note.append("габарит больше 3 м: возможно, файл в микронах или это не одна деталь")
    if not mesh.is_watertight:
        note.append("сетка не замкнута (дыры): объём и знак отклонений приблизительные")
    if len(parts) > 1:
        note.append(f"в файле {len(parts)} отдельных кусков, берётся самый большой")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(parts)),
        "extents_mm": [round(float(v), 2) for v in ext],
        "volume_mm3": round(float(mesh.volume), 1) if mesh.is_watertight else None,
        "warnings": note,
    }


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: m.area)


def align(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Самая большая плоская грань (опорная) смотрит вниз, лежит на Z=0,
    главные оси в плане по X/Y, центр габарита в плане в начале координат."""
    m = mesh.copy()
    normals, areas = m.face_normals, m.area_faces
    # Кластеризуем нормали: ищем направление с наибольшей площадью граней.
    best_n, best_a = None, -1.0
    idx = np.argsort(-areas)[:4000]
    for n in normals[idx[:: max(1, len(idx) // 400)]]:
        a = areas[(normals @ n) > 0.995].sum()
        if a > best_a:
            best_a, best_n = a, n
    T = trimesh.geometry.align_vectors(best_n, [0, 0, -1])
    m.apply_transform(T)
    # Ось в плане: по боковым (вертикальным) граням. Берём углы их нормалей по
    # модулю 90° и ищем пик по площади: прямые стенки задают оси детали.
    # (PCA по точкам не годится: скол перекашивает облако.)
    n2, a2 = m.face_normals, m.area_faces
    side = np.abs(n2[:, 2]) < 0.2
    ang = np.degrees(np.arctan2(n2[side, 1], n2[side, 0])) % 90.0
    hist, edges = np.histogram(ang, bins=180, range=(0, 90), weights=a2[side])
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peak = (edges[np.argmax(hist)] + 0.25)
    sel = np.abs(((ang - peak + 45) % 90) - 45) < 1.5
    peak = peak + np.average(((ang[sel] - peak + 45) % 90) - 45, weights=a2[side][sel]) if sel.any() else peak
    if peak > 45:
        peak -= 90
    R = trimesh.transformations.rotation_matrix(np.radians(-peak), [0, 0, 1])
    m.apply_transform(R)
    # Z=0 по опорной плоскости (средняя высота её граней, а не по шумовому минимуму).
    bottom = (m.face_normals[:, 2] < -0.985)
    z0 = float(np.average(m.triangles_center[bottom, 2], weights=m.area_faces[bottom])) if bottom.any() else m.bounds[0][2]
    lo, hi = m.bounds
    Tt = trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -z0])
    m.apply_transform(Tt)
    total = Tt @ R @ T
    return m, total


def plane_levels(mesh: trimesh.Trimesh, min_share: float = 0.01) -> list[dict]:
    n, a, c = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    out = []
    for sign, name in ((1, "смотрит вверх"), (-1, "смотрит вниз")):
        sel = (n[:, 2] * sign) > 0.985
        if not sel.any():
            continue
        z, w = c[sel, 2], a[sel]
        order = np.argsort(z)
        z, w = z[order], w[order]
        groups, cur = [], [0]
        for i in range(1, len(z)):
            if z[i] - z[i - 1] > 0.4:
                groups.append(cur)
                cur = []
            cur.append(i)
        groups.append(cur)
        for g in groups:
            area = w[g].sum()
            if area < min_share * a.sum():
                continue
            out.append({"z_mm": round(float(np.average(z[g], weights=w[g])), 2),
                        "area_mm2": round(float(area), 1), "facing": name})
    return sorted(out, key=lambda d: d["z_mm"])


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = sol
    r = float(np.sqrt(k + cx ** 2 + cy ** 2))
    resid = float(np.abs(np.linalg.norm(pts - [cx, cy], axis=1) - r).max())
    return np.array([cx, cy]), r, resid


def describe_ring(pts: np.ndarray, tol_round: float = 0.25) -> dict:
    c, r, resid = fit_circle(pts)
    poly = trimesh.path.polygons.Polygon(pts)
    d = {"area_mm2": round(float(poly.area), 1)}
    if resid < max(tol_round, 0.03 * r) and len(pts) >= 8:
        d.update({"shape": "окружность", "center_xy": [round(float(c[0]), 2), round(float(c[1]), 2)],
                  "diameter": round(2 * r, 2), "roundness_err": round(resid, 3)})
    else:
        simp = poly.simplify(0.4)
        coords = np.asarray(simp.exterior.coords)[:-1]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        d.update({"shape": "контур", "bbox_xy": [[round(float(lo[0]), 2), round(float(lo[1]), 2)],
                                                    [round(float(hi[0]), 2), round(float(hi[1]), 2)]],
                  "vertices_simplified": [[round(float(x), 1), round(float(y), 1)] for x, y in coords[:60]],
                  "vertices_total": int(len(coords))})
    return d


def sections(mesh: trimesh.Trimesh, levels: list[dict]) -> list[dict]:
    zs = sorted({lv["z_mm"] for lv in levels} | {float(mesh.bounds[0][2]), float(mesh.bounds[1][2])})
    heights = []
    for a, b in zip(zs, zs[1:]):
        if b - a > 6:
            heights += [a + (b - a) * f for f in (0.25, 0.5, 0.75)]
        elif b - a > 0.8:
            heights.append((a + b) / 2)
    out = []
    for h in heights:
        sec = mesh.section(plane_origin=[0, 0, h], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        planar, _ = sec.to_2D(to_2D=np.eye(4))
        regions = []
        for poly in planar.polygons_full:
            if poly.area < 1.0:
                continue
            outer = np.asarray(poly.exterior.coords)[:-1]
            reg = {"outer": describe_ring(outer), "inner": []}
            for hole in poly.interiors:
                if trimesh.path.polygons.Polygon(hole).area < 0.5:
                    continue
                reg["inner"].append(describe_ring(np.asarray(hole.coords)[:-1]))
            regions.append(reg)
        out.append({"z_mm": round(h, 2), "regions": regions})
    return out


def vertical_sections(mesh: trimesh.Trimesh) -> list[dict]:
    """Профили: разрез плоскостью XZ (Y=0) и YZ (X=0). Координаты (горизонталь, Z)."""
    out = []
    for name, normal, to2d in (
        ("XZ при Y=0 (точки: X, Z)", [0, 1, 0], np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)),
        ("YZ при X=0 (точки: Y, Z)", [1, 0, 0], np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)),
    ):
        sec = mesh.section(plane_origin=[0, 0, 0], plane_normal=normal)
        if sec is None:
            continue
        planar, _ = sec.to_2D(to_2D=to2d)
        regs = [{"outer": describe_ring(np.asarray(p.exterior.coords)[:-1]),
                 "inner": [describe_ring(np.asarray(h.coords)[:-1]) for h in p.interiors
                           if trimesh.path.polygons.Polygon(h).area > 0.5]}
                for p in planar.polygons_full if p.area > 1.0]
        out.append({"plane": name, "regions": regs})
    return out


def measurement_sheet(path: str) -> tuple[dict, trimesh.Trimesh]:
    raw = load_mesh(path)
    diag = diagnostics(raw)
    mesh, T = align(largest_component(raw))
    lv = plane_levels(mesh)
    sheet = {
        "file": str(path),
        "diagnostics": diag,
        "frame": "Деталь выровнена: опорная (самая большая) плоскость на Z=0, "
                 "главная ось в плане по X, центр габарита в плане в (0,0). Все размеры в мм.",
        "bounds_mm": [[round(float(v), 2) for v in mesh.bounds[0]], [round(float(v), 2) for v in mesh.bounds[1]]],
        "horizontal_planes": lv,
        "sections": sections(mesh, lv),
        "vertical_sections": vertical_sections(mesh),
    }
    return sheet, mesh
