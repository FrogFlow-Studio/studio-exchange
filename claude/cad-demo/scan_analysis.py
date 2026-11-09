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
    # Кандидаты в опорную плоскость: направления нормалей с большой площадью.
    # Опорная = на неё деталь можно положить: все точки по одну сторону от неё.
    # Порог по нормали мягкий (0.97), чтобы шумный скан не терял грани.
    V = m.vertices
    idx = np.argsort(-areas)[:6000]
    cands = []
    for n in normals[idx[:: max(1, len(idx) // 500)]]:
        if any(float(n @ c) > 0.97 for c, _, _ in cands):
            continue
        sel = (normals @ n) > 0.97
        nn = (normals[sel] * areas[sel, None]).sum(axis=0)
        nn /= np.linalg.norm(nn)
        for _ in range(3):  # уточнение средней нормалью
            sel = (normals @ nn) > 0.97
            nn = (normals[sel] * areas[sel, None]).sum(axis=0)
            nn /= np.linalg.norm(nn)
        d = float(np.average(m.triangles_center[sel] @ nn, weights=areas[sel]))
        support = float(((V @ nn) <= d + 0.5 + 0.01 * max(m.extents)).mean())
        cands.append((nn, float(areas[sel].sum()), support))
    ok = [c for c in cands if c[2] > 0.995]
    best_n = max(ok or cands, key=lambda c: c[1])[0]
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
    bottom = (m.face_normals[:, 2] < -0.97)
    zb = m.triangles_center[bottom, 2] if bottom.any() else m.vertices[:, 2]
    zb = zb[zb < np.percentile(zb, 2) + 3.0]  # только нижняя плоскость, не другие грани «вниз»
    z0 = float(np.median(zb))
    lo, hi = m.bounds
    Tt = trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -z0])
    m.apply_transform(Tt)
    total = Tt @ R @ T
    return m, total


def noise_estimate(mesh: trimesh.Trimesh) -> float:
    """Шум скана: разброс точек опорной плоскости (Z≈0) по высоте, мм."""
    c, n = mesh.triangles_center, mesh.face_normals
    sel = (n[:, 2] < -0.97) & (np.abs(c[:, 2]) < 2.0)
    if sel.sum() < 30:
        return 0.0
    z = c[sel, 2]
    return float(1.4826 * np.median(np.abs(z - np.median(z))))  # устойчивое СКО


def plane_levels(mesh: trimesh.Trimesh, min_share: float = 0.01, noise: float = 0.0) -> list[dict]:
    n, a, c = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    out = []
    for sign, name in ((1, "смотрит вверх"), (-1, "смотрит вниз")):
        sel = (n[:, 2] * sign) > 0.97
        if not sel.any():
            continue
        z, w = c[sel, 2], a[sel]
        order = np.argsort(z)
        z, w = z[order], w[order]
        groups, cur = [], [0]
        for i in range(1, len(z)):
            if z[i] - z[i - 1] > max(0.4, 4 * noise):
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


def partial_circle(pts: np.ndarray, tol: float = 0.3, min_share: float = 0.45) -> dict | None:
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        try:
            c, r, _ = fit_circle(tri)
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(r) or r < 1:
            continue
        inl = np.abs(np.linalg.norm(pts - c, axis=1) - r) < tol
        if best is None or inl.sum() > best[0].sum():
            best = (inl, c, r)
    if best is None or best[0].mean() < min_share:
        return None
    c, r, _ = fit_circle(pts[best[0]])
    # Доля по ДЛИНЕ контура, а не по числу точек: прямой край скола описан
    # редкими точками и иначе почти не влиял бы на долю.
    inl = np.abs(np.linalg.norm(pts - c, axis=1) - r) < tol
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    on = inl & np.roll(inl, -1)
    share = float(seg[on].sum() / seg.sum())
    if share < min_share:
        return None
    return {"center_xy": [round(float(c[0]), 2), round(float(c[1]), 2)], "diameter": round(2 * r, 2),
            "share_of_contour": round(share, 2)}


def hole_patterns(rings: list[dict]) -> list[dict]:
    """Отверстия одного диаметра по окружности: шаг, PCD, где пропущены."""
    circ = [r for r in rings if r.get("shape") == "окружность"]
    out, used = [], set()
    for i, a in enumerate(circ):
        if i in used:
            continue
        grp = [j for j, b in enumerate(circ) if abs(b["diameter"] - a["diameter"]) < 0.3]
        if len(grp) < 3:
            continue
        used |= set(grp)
        P = np.array([circ[j]["center_xy"] for j in grp])
        c, R, resid = fit_circle(P)
        if resid > 0.5:
            continue
        ang = np.sort(np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])) % 360)
        gaps = np.diff(np.r_[ang, ang[0] + 360])
        n = int(round(360 / gaps.min()))
        step = 360 / n
        if n > 24 or np.abs(gaps / step - np.round(gaps / step)).max() > 0.04:
            out.extend(rect_pattern(circ, grp))
            continue
        missing = []
        for k in range(n):
            t = ang[0] + k * 360 / n
            if np.min(np.abs((ang - t + 180) % 360 - 180)) > 360 / n / 3:
                missing.append([round(float(c[0] + R * np.cos(np.radians(t))), 2),
                                round(float(c[1] + R * np.sin(np.radians(t))), 2)])
        out.append({"pattern": "отверстия по окружности", "hole_diameter": round(float(np.mean([circ[j]["diameter"] for j in grp])), 2),
                    "found": len(grp), "likely_total": n, "pcd_diameter": round(2 * R, 2),
                    "center_xy": [round(float(c[0]), 2), round(float(c[1]), 2)],
                    "first_angle_deg": round(float(ang[0]), 1), "missing_positions_xy": missing})
    return out


def rect_pattern(circ: list[dict], grp: list[int]) -> list[dict]:
    """3–4 отверстия по углам прямоугольника; если одного нет — где оно было."""
    P = np.array([circ[j]["center_xy"] for j in grp])
    if len(P) not in (3, 4):
        return []
    from itertools import permutations

    for a, b, c in permutations(range(len(P)), 3):
        u, v = P[b] - P[a], P[c] - P[a]
        if abs(np.dot(u, v)) < 0.02 * np.linalg.norm(u) * np.linalg.norm(v):
            d = P[b] + P[c] - P[a]
            have = len(P) == 4 and np.min(np.linalg.norm(P - d, axis=1)) < 0.5
            if len(P) == 4 and not have:
                continue
            return [{"pattern": "отверстия по углам прямоугольника",
                     "hole_diameter": round(float(np.mean([circ[j]["diameter"] for j in grp])), 2),
                     "found": len(P), "likely_total": 4,
                     "sides_mm": [round(float(np.linalg.norm(u)), 2), round(float(np.linalg.norm(v)), 2)],
                     "missing_positions_xy": [] if have else [[round(float(d[0]), 2), round(float(d[1]), 2)]]}]
    return []


def describe_ring(pts: np.ndarray, tol_round: float = 0.25) -> dict:
    c, r, resid = fit_circle(pts)
    poly = trimesh.path.polygons.Polygon(pts)
    d = {"area_mm2": round(float(poly.area), 1)}
    # Круглость по СКО отклонения, а не по максимуму: шумный скан даёт отдельные
    # выбросы, а форма при этом круглая.
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - r) ** 2)))
    if len(pts) >= 8 and r < 1e4 and rms < max(tol_round * 0.6, 0.006 * r) and resid < max(4 * tol_round, 0.03 * r):
        d.update({"shape": "окружность", "center_xy": [round(float(c[0]), 2), round(float(c[1]), 2)],
                  "diameter": round(2 * r, 2), "roundness_rms": round(rms, 3)})
    else:
        # Частичная дуга: если большая часть контура лежит на одной окружности
        # (край сломан), сообщаем её — это подсказка о целой форме.
        arc = partial_circle(pts)
        if arc:
            d["partial_circle"] = arc
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
            pats = hole_patterns(reg["inner"])
            if pats:
                reg["hole_patterns"] = pats
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


def summary(sheet: dict) -> list[str]:
    """Короткая сводка для человека и ИИ: главное из листа замеров простыми словами."""
    out = []
    lo, hi = sheet["bounds_mm"]
    out.append(f"Габарит после выравнивания: {hi[0]-lo[0]:.1f} × {hi[1]-lo[1]:.1f} × {hi[2]-lo[2]:.1f} мм (X × Y × Z).")
    up = [p["z_mm"] for p in sheet["horizontal_planes"] if p["facing"] == "смотрит вверх"]
    if up:
        out.append("Горизонтальные площадки сверху на высотах Z: " + ", ".join(f"{z:.2f}" for z in up) + " мм.")
    seen = set()
    for sec in sheet["sections"]:
        for i, reg in enumerate(sec["regions"]):
            o = reg["outer"]
            if o.get("shape") == "окружность":
                key = ("o", round(o["diameter"]), round(o["center_xy"][0]), round(o["center_xy"][1]))
                txt = f"круглый контур Ø{o['diameter']} с центром {o['center_xy']}"
            elif o.get("partial_circle"):
                pc = o["partial_circle"]
                key = ("p", round(pc["diameter"]))
                txt = (f"контур — окружность Ø{pc['diameter']} с центром {pc['center_xy']}, но совпадает лишь "
                       f"{int(pc['share_of_contour']*100)}% контура: остальное — выступы (рёбра) или скол")
            else:
                bb = o.get("bbox_xy")
                key = ("c", tuple(round(v) for v in sum(bb, [])))
                txt = f"контур {bb[1][0]-bb[0][0]:.1f} × {bb[1][1]-bb[0][1]:.1f} мм ({o.get('vertices_total')} вершин после упрощения)"
            if key not in seen:
                seen.add(key)
                out.append(f"Сечение Z={sec['z_mm']}: {txt}.")
            for h in reg["inner"]:
                if h.get("shape") == "окружность":
                    k2 = ("h", round(h["diameter"]), round(h["center_xy"][0]), round(h["center_xy"][1]))
                    if k2 not in seen:
                        seen.add(k2)
                        out.append(f"Отверстие Ø{h['diameter']} в {h['center_xy']} (видно на Z={sec['z_mm']}).")
                elif h.get("partial_circle"):
                    pc = h["partial_circle"]
                    k2 = ("hp", round(pc["diameter"]), round(pc["center_xy"][0]), round(pc["center_xy"][1]))
                    if k2 not in seen:
                        seen.add(k2)
                        bb = h["bbox_xy"]
                        out.append(f"Отверстие Ø{pc['diameter']} в {pc['center_xy']} с вырезом сбоку "
                                   f"(паз/шпоночный паз?): отверстие целиком в рамке {bb}, Z={sec['z_mm']}.")
            for pat in reg.get("hole_patterns", []):
                k3 = ("pat", pat["pattern"], round(pat["hole_diameter"]))
                if k3 in seen:
                    continue
                seen.add(k3)
                t = f"{pat['pattern']}: Ø{pat['hole_diameter']}, найдено {pat['found']} из {pat['likely_total']}"
                if pat.get("pcd_diameter"):
                    t += f", диаметр окружности центров {pat['pcd_diameter']}, центр {pat['center_xy']}, первый угол {pat['first_angle_deg']}°"
                if pat.get("sides_mm"):
                    t += f", стороны прямоугольника {pat['sides_mm']}"
                if pat["missing_positions_xy"]:
                    t += f". НЕ ХВАТАЕТ отверстий в {pat['missing_positions_xy']} — вероятно, отломаны"
                out.append(t + ".")
    d = sheet["diagnostics"]
    out.append(f"Шум скана ~{d.get('noise_mm', 0)} мм. " + " ".join(d["warnings"]))
    return out


def measurement_sheet(path: str) -> tuple[dict, trimesh.Trimesh]:
    raw = load_mesh(path)
    diag = diagnostics(raw)
    # Единицы: STL их не хранит. Деталь меньше 2 единиц почти наверняка в метрах.
    if max(raw.extents) < 2:
        raw.apply_scale(1000.0)
        diag["warnings"].append("файл похож на метры: увеличен в 1000 раз (в мм). Проверьте габарит")
        diag["scaled_to_mm"] = 1000.0
    # Дыры сетки (места, которые сканер не увидел): штопаем мелкие, чтобы сечения
    # замыкались. Исходный файл не меняется.
    if not raw.is_watertight:
        before = len(raw.faces)
        trimesh.repair.fill_holes(raw)
        diag["holes_filled"] = bool(len(raw.faces) > before)
        diag["watertight_after_repair"] = bool(raw.is_watertight)
        if raw.is_watertight:
            diag["warnings"] = [w for w in diag["warnings"] if not w.startswith("сетка не замкнута")]
            diag["warnings"].append("в сетке были мелкие дыры, заштопаны автоматически")
    mesh, T = align(largest_component(raw))
    noise = noise_estimate(mesh)
    diag["noise_mm"] = round(noise, 3)
    if noise > 0.1:
        diag["warnings"].append(f"скан шумный (~{noise:.2f} мм): мелкие размеры менее надёжны")
    lv = plane_levels(mesh, noise=noise)
    sheet = {
        "file": str(path),
        "diagnostics": diag,
        "frame": "Деталь выровнена: опорная (самая большая) плоскость на Z=0, "
                 "главная ось в плане по X, центр габарита в плане в (0,0). Все размеры в мм.",
        "transform_from_scan_4x4": [[round(float(v), 6) for v in row] for row in T],
        "bounds_mm": [[round(float(v), 2) for v in mesh.bounds[0]], [round(float(v), 2) for v in mesh.bounds[1]]],
        "horizontal_planes": lv,
        "sections": sections(mesh, lv),
        "vertical_sections": vertical_sections(mesh),
    }
    sheet = {"summary": summary(sheet), **sheet}
    return sheet, mesh
