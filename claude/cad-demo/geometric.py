"""Построение модели БЕЗ ИИ: обычный геометрический алгоритм (страховка).

Деталь режется по высоте на слои (между горизонтальными площадками; высокие
участки — тоньше, чтобы наклонные рёбра шли ступеньками). Каждый слой — контур
сечения, вытянутый по высоте, минус отверстия. Восстановление «по замыслу»
только там, где это проверяемо геометрией:
  * контур почти круглый (дуга совпадает на большей части) → полная окружность;
  * контур почти симметричен относительно оси X или Y → дополняется зеркальной
    копией (так закрывается отколотый угол);
  * отверстия, которые повторяются по окружности или по углам прямоугольника,
    но одного не хватает → недостающее добавляется.
Всё остальное копируется со скана как есть. Результат — такой же код CadQuery,
как у ИИ, поэтому дальше идёт та же проверка и тот же отчёт.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh
from shapely.affinity import scale as sh_scale, translate as sh_translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

import scan_analysis as sa


def _section_polys(mesh: trimesh.Trimesh, z: float) -> list[Polygon]:
    return sa.best_section(mesh, z)


def _slab_bounds(mesh: trimesh.Trimesh, levels: list[float], step: float = 2.0) -> list[tuple[float, float]]:
    zs = sorted(set([0.0, float(mesh.bounds[1][2])] + [z for z in levels if 0.3 < z < mesh.bounds[1][2] - 0.3]))
    out = []
    for a, b in zip(zs, zs[1:]):
        if b - a < 0.3:
            continue
        n = max(1, int(round((b - a) / step))) if b - a > 4 * step else 1
        edges = np.linspace(a, b, n + 1)
        out += list(zip(edges[:-1], edges[1:]))
    return out


def _mirror_axis(poly: Polygon, axis: str) -> tuple[float, float]:
    """Лучшая ось симметрии около центра габарита: (положение, IoU)."""
    lo_x, lo_y, hi_x, hi_y = poly.bounds
    c0 = (lo_x + hi_x) / 2 if axis == "x" else (lo_y + hi_y) / 2
    best = (c0, 0.0)
    for c in np.arange(c0 - 3, c0 + 3.01, 0.25):
        if axis == "x":
            m = sh_translate(sh_scale(poly, -1, 1, origin=(0, 0)), 2 * c, 0)
        else:
            m = sh_translate(sh_scale(poly, 1, -1, origin=(0, 0)), 0, 2 * c)
        iou = poly.intersection(m).area / max(poly.union(m).area, 1e-9)
        if iou > best[1]:
            best = (float(c), float(iou))
    return best


def _restore_outer(outer: Polygon, notes: list[str], z: float) -> Polygon:
    ext = np.asarray(outer.exterior.coords)[:-1]
    pc = sa.partial_circle(ext)
    if pc and pc["share_of_contour"] >= 0.6:
        c, d = pc["center_xy"], pc["diameter"]
        full = Polygon([(c[0] + d / 2 * np.cos(t), c[1] + d / 2 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 180, endpoint=False)])
        if full.area > outer.area * 1.003:
            notes.append(f"Z≈{z:.1f}: контур — дуга Ø{d}, совпадает {int(pc['share_of_contour']*100)}%: достроен до полной окружности")
        return full if pc["share_of_contour"] < 0.995 else outer
    res = outer
    for axis in ("x", "y"):
        c, iou = _mirror_axis(res, axis)
        if 0.88 < iou < 0.98:
            m = (sh_translate(sh_scale(res, -1, 1, origin=(0, 0)), 2 * c, 0) if axis == "x"
                 else sh_translate(sh_scale(res, 1, -1, origin=(0, 0)), 0, 2 * c))
            u = unary_union([res, m]).buffer(0.05).buffer(-0.05)
            if u.geom_type != "Polygon":  # зеркальная копия не срослась — не трогаем контур
                continue
            res = u
            notes.append(f"Z≈{z:.1f}: контур симметричен относительно {'X=' if axis == 'x' else 'Y='}{c:.2f} "
                         f"(совпадение {iou*100:.0f}%): недостающая часть дополнена зеркально")
    return res


def _same(A: list, B: list, thr: float = 0.985) -> bool:
    if len(A) != len(B):
        return False
    ua = unary_union([o.difference(unary_union(c)) if c else o for o, c in A])
    ub = unary_union([o.difference(unary_union(c)) if c else o for o, c in B])
    return ua.intersection(ub).area / max(ua.union(ub).area, 1e-9) > thr


def _poly_code(p: Polygon, tol: float = 0.3) -> str:
    q = p.simplify(tol)
    pts = np.round(np.asarray(q.exterior.coords)[:-1], 2).tolist()
    return json.dumps(pts)


def build(mesh: trimesh.Trimesh, sheet: dict) -> tuple[str, dict]:
    notes: list[str] = []
    levels = [p["z_mm"] for p in sheet["horizontal_planes"]]
    slabs = _slab_bounds(mesh, levels)
    # Отверстия и узоры — из листа замеров (они уже посчитаны и проверены).
    holes, patterns = [], []
    for sec in sheet["sections"]:
        for reg in sec["regions"]:
            for h in reg["inner"]:
                if h.get("shape") == "окружность":
                    holes.append((tuple(h["center_xy"]), h["diameter"], sec["z_mm"]))
            patterns += reg.get("hole_patterns", [])
    # Сгруппировать одинаковые отверстия, взять высоты, где они видны.
    groups: list[dict] = []
    for c, d, z in holes:
        for g in groups:
            if np.hypot(g["c"][0] - c[0], g["c"][1] - c[1]) < 0.8 and abs(g["d"] - d) < 0.4:
                g["zs"].append(z)
                g["ds"].append(d)
                break
        else:
            groups.append({"c": c, "d": d, "zs": [z], "ds": [d]})
    for g in groups:
        g["d"] = float(np.median(g["ds"]))
    for pat in patterns:
        sib = [g for g in groups if abs(g["d"] - pat["hole_diameter"]) < 0.4]
        if not sib:
            continue
        zs = sum((g["zs"] for g in sib), [])
        for m in pat["missing_positions_xy"]:
            if all(np.hypot(g["c"][0] - m[0], g["c"][1] - m[1]) > 1.0 for g in groups):
                groups.append({"c": tuple(m), "d": pat["hole_diameter"], "zs": list(zs), "ds": [], "restored": True})
                notes.append(f"Отверстие Ø{pat['hole_diameter']} в {m} добавлено: {pat['pattern']}, найдено "
                             f"{pat['found']} из {pat['likely_total']}")

    # Слои: внешний контур (с восстановлением), минус некруглые вырезы (пазы).
    layers = []
    prev = None
    for a, b in slabs:
        polys = _section_polys(mesh, (a + b) / 2)
        if not polys:
            continue
        shapes = []
        for p in polys:
            outer = _restore_outer(Polygon(p.exterior), notes, (a + b) / 2)
            cuts = []
            for ring in p.interiors:
                pts = np.asarray(ring.coords)[:-1]
                d = sa.describe_ring(pts)
                if d.get("shape") != "окружность":
                    cuts.append(Polygon(pts))  # паз или вырез — как в скане
            shapes.append((outer, cuts))
        if prev is not None and _same(prev["shapes"], shapes):
            prev["b"] = b  # тот же контур — продлеваем слой, а не плодим ступеньки
            continue
        prev = {"a": a, "b": b, "shapes": shapes}
        layers.append(prev)
    # убрать дубли заметок
    seen, uniq = set(), []
    for n in notes:
        key = n.split(": ", 1)[-1].split("(")[0] if n.startswith("Z≈") else n
        key = "".join(ch for ch in key if not ch.isdigit())  # «Ø89.98» и «Ø89.99» — одно и то же
        if key not in seen:
            seen.add(key)
            uniq.append(n)
    notes = uniq

    lines = ["import cadquery as cq", "", "# Построено геометрическим алгоритмом без ИИ: слои по сечениям скана.",
             "parts = []"]
    for L in layers:
        for outer, cuts in L["shapes"]:
            lines.append(f"s = cq.Workplane('XY').workplane(offset={L['a']:.3f}).polyline({_poly_code(outer)}).close()"
                         f".extrude({L['b'] - L['a']:.3f})")
            for c in cuts:
                lines.append(f"s = s.cut(cq.Workplane('XY').workplane(offset={L['a'] - 0.01:.3f}).polyline({_poly_code(c, 0.15)})"
                             f".close().extrude({L['b'] - L['a'] + 0.02:.3f}))")
            lines.append("parts.append(s)")
    lines += ["result = parts[0]", "for p in parts[1:]:", "    result = result.union(p)", "", "# Отверстия"]
    for g in groups:
        z0, z1 = 0.0, float(mesh.bounds[1][2])
        # отверстие режем на высоту слоёв, где оно видно (с запасом до границ слоёв)
        zs = g["zs"]
        # высоты — до ближайших горизонтальных площадок вокруг тех мест, где отверстие видно
        bands = sorted(set([0.0, float(mesh.bounds[1][2])] + [z for z in levels if 0.3 < z < mesh.bounds[1][2] - 0.3]))
        lo_b = [z for z in bands if z <= min(zs) + 0.01]
        hi_b = [z for z in bands if z >= max(zs) - 0.01]
        if lo_b and hi_b:
            z0, z1 = max(lo_b), min(hi_b)
        mark = "  # восстановлено" if g.get("restored") else ""
        lines.append(f"result = result.cut(cq.Workplane('XY').workplane(offset={z0 - 0.05:.3f}).center({g['c'][0]:.3f}, {g['c'][1]:.3f})"
                     f".circle({g['d'] / 2:.3f}).extrude({z1 - z0 + 0.1:.3f})){mark}")
    code = "\n".join(lines) + "\n"
    meta = {
        "plan": f"Без ИИ: {len(layers)} слоёв по сечениям скана, {len(groups)} отверстий (круглые подогнаны по окружности).",
        "assumptions": ["наклонные поверхности — ступеньками по ~2 мм", "некруглые вырезы скопированы со скана как есть"],
        "restored": notes or ["ничего не восстановлено: явной симметрии или повторов не найдено"],
        "check_by_engineer": ["это черновик по сечениям, не замысел конструктора: наклоны и скругления приблизительные",
                              "восстановленные места (список выше)"],
    }
    return code, meta
