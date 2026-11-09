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
    # скол сдвигает максимум IoU в сторону уцелевшей части; середина габарита
    # (крайние кромки обычно уцелели) надёжнее, если она почти так же хороша
    iou0, _ = _iou_mirror(poly, axis, c0)
    if iou0 >= best[1] - 0.01:
        return (float(c0), float(iou0))
    return best


def _iou_mirror(poly: Polygon, axis: str, c: float) -> tuple[float, Polygon]:
    m = (sh_translate(sh_scale(poly, -1, 1, origin=(0, 0)), 2 * c, 0) if axis == "x"
         else sh_translate(sh_scale(poly, 1, -1, origin=(0, 0)), 0, 2 * c))
    return poly.intersection(m).area / max(poly.union(m).area, 1e-9), m


def _restore_outer(outer: Polygon, notes: list[str], z: float, axes: dict | None = None,
                   min_iou: float = 0.88) -> Polygon:
    ext = np.asarray(outer.exterior.coords)[:-1]
    pc = sa.partial_circle(ext)
    if pc and pc["share_of_contour"] >= 0.6:
        c, d = pc["center_xy"], pc["diameter"]
        full = Polygon([(c[0] + d / 2 * np.cos(t), c[1] + d / 2 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 180, endpoint=False)])
        if full.area > outer.area * 1.003:
            notes.append(f"Z≈{z:.1f}: контур — дуга Ø{d}, совпадает {int(pc['share_of_contour']*100)}%: достроен до полной окружности")
        if pc["share_of_contour"] >= 0.995:
            return outer
        # окружность ДОБАВЛЯЕТСЯ к контуру, а не заменяет его: выступы (ребро,
        # прилив) на остальной части контура остаются
        u = unary_union([outer, full]).buffer(0.05).buffer(-0.05)
        return Polygon(u.exterior) if u.geom_type == "Polygon" else full
    res = outer
    # Зеркалим только относительно ОБЩИХ осей детали (найдены один раз по основанию
    # или по узору отверстий). Своя ось у каждого слоя давала ложную «симметрию».
    for axis, c in (axes or {}).items():
        iou, m = _iou_mirror(res, axis, c)
        if min_iou < iou < 0.98:
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
    # закрываем зубцы шума глубже 0.3 мм (острые углы сохраняются), потом упрощаем:
    # почти самопересекающийся зубчатый контур ядро CAD сливает часами и без памяти
    q = p.buffer(0.15, join_style=2).buffer(-0.15, join_style=2)
    if q.geom_type != "Polygon" or q.area < 0.5 * p.area:
        q = p
    q = q.simplify(tol)
    raw = np.round(np.asarray(q.exterior.coords)[:-1], 2).tolist()
    pts = [pt for i, pt in enumerate(raw) if pt != raw[i - 1]]  # повтор точки = ребро нулевой длины
    return json.dumps(pts)


def side_holes(mesh: trimesh.Trimesh, step: float = 1.5) -> list[dict]:
    """Отверстия с горизонтальной осью (в стенках): ищем окружности в вертикальных
    сечениях по X и по Y на разных расстояниях и собираем их по оси."""
    out = []
    lo, hi = mesh.bounds
    for ax, (normal, to2d) in {
        "X": ([1, 0, 0], np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)),
        "Y": ([0, 1, 0], np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)),
    }.items():
        i = 0 if ax == "X" else 1
        found = []
        for off in np.arange(lo[i] + step / 2, hi[i], step):
            o = [0.0, 0.0, 0.0]
            o[i] = float(off)
            sec = mesh.section(plane_origin=o, plane_normal=normal)
            if sec is None:
                continue
            planar, _ = sec.to_2D(to_2D=to2d)
            for p in planar.polygons_full:
                for ring in p.interiors:
                    d = sa.describe_ring(np.asarray(ring.coords)[:-1])
                    if d.get("shape") == "окружность" and d["diameter"] > 1.5:
                        found.append((d["center_xy"], d["diameter"], float(off)))
        groups: list[dict] = []
        for c, dia, off in found:
            for g in groups:
                if np.hypot(g["c"][0] - c[0], g["c"][1] - c[1]) < 0.8 and abs(g["d"] - dia) < 0.5:
                    g["offs"].append(off)
                    g["ds"].append(dia)
                    break
            else:
                groups.append({"c": c, "d": dia, "offs": [off], "ds": [dia]})
        for g in groups:
            if len(g["offs"]) < 2:
                continue
            out.append({"axis": ax, "c": [round(float(v), 3) for v in g["c"]], "d": round(float(np.median(g["ds"])), 3),
                        "from": round(min(g["offs"]) - step / 2, 3), "to": round(max(g["offs"]) + step / 2, 3)})
    return out



def _side_footprint(h: dict, za: float, zb: float):
    """След отверстия в стенке на плане слоя za..zb (или None, если слой его не задевает)."""
    from shapely.geometry import box
    u, v = h["c"]
    r = h["d"] / 2 + 1.3
    if zb < v - r or za > v + r:
        return None
    if h["axis"] == "X":
        return box(h["from"] - 0.5, u - r, h["to"] + 0.5, u + r)
    return box(u - r, h["from"] - 0.5, u + r, h["to"] + 0.5)


def _fill_between(layers: list[dict], notes: list[str], sides: list[dict] | None = None) -> None:
    """Скол посреди высоты: если материал есть и ниже, и выше слоя в одном и том же
    месте плана, а в самом слое его нет — это откол, а не замысел. Дополняем слой
    пересечением контуров нижнего и верхнего слоёв.

    Не трогаем проточки и канавки: если недостача идёт по всему контуру слоя (больше
    60% его границы), это скорее канавка на валу, чем скол.
    Отверстия с горизонтальной осью при этом тоже заполняются, но их потом
    вырезает точный цилиндр (side_holes)."""
    U = [unary_union([o for o, _ in L["shapes"]]) for L in layers]
    for k in range(1, len(layers) - 1):
        cand = []
        for i in range(k):
            for j in range(k + 1, len(layers)):
                c = U[i].intersection(U[j])
                if not c.is_empty and c.area > 1.0:
                    cand.append(c)
        if not cand:
            continue
        deficit = unary_union(cand).difference(U[k]).buffer(-0.3).buffer(0.3)
        if deficit.is_empty or deficit.area < max(4.0, 0.01 * U[k].area):
            continue
        edge = U[k].boundary
        touched = edge.intersection(deficit.buffer(0.4)).length / max(edge.length, 1e-6)
        if touched > 0.6:
            continue
        filled = unary_union([U[k], deficit])
        cuts = [c for _, cs in layers[k]["shapes"] for c in cs]
        polys = list(filled.geoms) if filled.geom_type == "MultiPolygon" else [filled]
        shapes = []
        for p in polys:
            if p.geom_type != "Polygon" or p.area < 1.0:
                continue
            outer = Polygon(p.exterior)
            shapes.append((outer, [c for c in cuts if outer.contains(c.representative_point())]))
        if not shapes:
            continue
        layers[k]["shapes"] = shapes
        U[k] = unary_union([o for o, _ in shapes])
        L = layers[k]
        fp = [f for f in (_side_footprint(h, L["a"], L["b"]) for h in sides or []) if f is not None]
        if fp and deficit.difference(unary_union(fp)).area < 4.0:
            continue  # это отверстие в стенке, его вырежет точный цилиндр — не скол
        notes.append(f"Z {L['a']:.1f}–{L['b']:.1f}: материал есть ниже и выше, а в слое нет "
                     f"({deficit.area:.0f} мм²): похоже на скол, достроено по соседним слоям")



def _snap_faces(mesh: trimesh.Trimesh, h: dict) -> dict:
    """Концы отверстия — по граням стенки вокруг него, а не по шагу сечений
    (иначе кольцо-заглушка торчит из стенки на доли миллиметра)."""
    i = 0 if h["axis"] == "X" else 1
    V, N = mesh.vertices, mesh.vertex_normals
    u, v = h["c"]
    rr = np.hypot(V[:, 1 - i] - u, V[:, 2] - v)
    ring = (rr > h["d"] / 2 + 0.3) & (rr < h["d"] / 2 + 2.0) & (np.abs(N[:, i]) > 0.9)
    out = dict(h)
    for key, sign, ref in (("from", -1, h["from"]), ("to", 1, h["to"])):
        near = ring & (np.abs(V[:, i] - ref) < 1.0) & (np.sign(N[:, i]) == sign)
        if near.sum() >= 5:
            out[key] = round(float(np.median(V[near, i])), 3)
    return out

def _arc_share(mesh: trimesh.Trimesh, h: dict, u: float, v: float) -> float:
    """Какая доля окружности отверстия (по углу) видна на скане: точки на расстоянии
    радиуса от оси между гранями стенки. Сплошная стенка даёт 0, целое отверстие ~1,
    отверстие, задетое сколом, — частичную дугу."""
    i = 0 if h["axis"] == "X" else 1
    V = mesh.vertices
    inside = (V[:, i] > h["from"] + 0.5) & (V[:, i] < h["to"] - 0.5)
    du, dv = V[inside, 1 - i] - u, V[inside, 2] - v
    rr = np.hypot(du, dv)
    # точка стенки отверстия: на радиусе И нормаль смотрит на ось (иначе засчитаются
    # чужие поверхности, которые просто проходят на этом расстоянии, например
    # вертикальное отверстие поперёк)
    N = mesh.vertex_normals[inside]
    radial = (N[:, 1 - i] * du + N[:, 2] * dv) / np.maximum(rr, 1e-9)
    on = (np.abs(rr - h["d"] / 2) < 0.3) & (np.abs(radial) > 0.8) & (np.abs(N[:, i]) < 0.4)
    if on.sum() < 10:
        return 0.0
    bins = np.unique(((np.arctan2(dv[on], du[on]) + np.pi) / (2 * np.pi) * 24).astype(int) % 24)
    return len(bins) / 24


def _mirror_side_holes(mesh: trimesh.Trimesh, sides: list[dict], axes: dict, notes: list[str]) -> list[dict]:
    """Отверстие в стенке, задетое сколом: в сечениях оно уже не замкнутая окружность
    и не находится. Если на зеркальном месте (по общей оси детали) у найденного
    отверстия скан показывает часть его стенки (дугу), — достраиваем целым."""
    out = []
    for h in sides:
        ax_key = "y" if h["axis"] == "X" else "x"  # отверстие вдоль X лежит в плоскости YZ: зеркалим по Y
        if ax_key not in axes:
            continue
        u, v = h["c"]
        u2 = 2 * axes[ax_key] - u
        if abs(u2 - u) < h["d"]:
            continue
        if any(o["axis"] == h["axis"] and np.hypot(o["c"][0] - u2, o["c"][1] - v) < h["d"] / 2 for o in sides + out):
            continue
        share = _arc_share(mesh, h, u2, v)
        if share < 0.3:
            continue  # на месте пары нет и следа отверстия: стенка сплошная, не выдумываем
        g = dict(h, c=[round(float(u2), 3), v], restored=True)
        out.append(g)
        notes.append(f"Отверстие Ø{h['d']:.2f} в стенке: парное к найденному, на скане видна только часть "
                     f"его окружности ({share:.0%}) — восстановлено целым")
    return out


def sloped_planes(mesh: trimesh.Trimesh, noise: float = 0.05) -> list[dict]:
    """Наклонные плоские грани (не горизонтальные и не вертикальные): рёбра жёсткости,
    фаски, скосы. Слои дают на них ступеньки; по найденной плоскости ступеньки
    срезаются, а впадины между ними заполняются."""
    # нормали мелких треугольников шумного скана гуляют на десятки градусов:
    # группируем по нормалям сглаженной копии, а плоскость подгоняем по исходным точкам
    sm = mesh.copy()
    trimesh.smoothing.filter_taubin(sm, iterations=10)
    fn, fa, fc = sm.face_normals, mesh.area_faces, mesh.triangles_center
    total = fa.sum()
    cand = (np.abs(fn[:, 2]) > 0.1) & (np.abs(fn[:, 2]) < 0.97)
    used = np.zeros(len(fn), bool)
    adj = mesh.face_adjacency
    out = []
    while True:
        free = cand & ~used
        if fa[free].sum() < 0.005 * total:
            break
        seed = np.flatnonzero(free)[np.argmax(fa[free])]
        sel = free & (fn @ fn[seed] > 0.97)
        n = (fn[sel] * fa[sel, None]).sum(0)
        n /= np.linalg.norm(n)
        sel = free & (fn @ n > 0.97)
        used |= sel
        used[seed] = True
        e = adj[sel[adj[:, 0]] & sel[adj[:, 1]]]
        for comp in trimesh.graph.connected_components(e, nodes=np.flatnonzero(sel), min_len=20):
            comp = np.asarray(comp)
            area = float(fa[comp].sum())
            if area < max(30.0, 0.005 * total):
                continue
            pts, w = fc[comp], fa[comp]
            c = np.average(pts, axis=0, weights=w)
            nn = np.linalg.svd((pts - c) * np.sqrt(w)[:, None], full_matrices=False)[2][-1]
            if nn @ n < 0:
                nn = -nn
            res = (pts - c) @ nn
            if np.sqrt(np.average(res ** 2, weights=w)) > max(0.12, 2.5 * noise):
                continue  # не плоскость (скругление, конус) — оставляем слоям
            # контур грани в её плоскости: объединение треугольников
            ex = np.cross(nn, [0, 0, 1.0])
            ex = ex / np.linalg.norm(ex) if np.linalg.norm(ex) > 1e-6 else np.array([1.0, 0, 0])
            ey = np.cross(nn, ex)
            tri = mesh.triangles[comp]
            uv = np.stack([(tri - c) @ ex, (tri - c) @ ey], axis=-1)
            poly = unary_union([Polygon(t) for t in uv if Polygon(t).area > 1e-6]).buffer(0.2).buffer(-0.2)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.is_empty or poly.area < 20:
                continue
            # насколько можно заходить внутрь и наружу, не задев другую стенку
            smp = pts[np.linspace(0, len(pts) - 1, min(60, len(pts))).astype(int)]
            reach = []
            for sgn in (-1, 1):
                o = smp + sgn * 0.3 * nn
                loc, ri, _ = mesh.ray.intersects_location(o, np.tile(sgn * nn, (len(o), 1)), multiple_hits=False)
                dist = np.full(len(o), 99.0)
                if len(ri):
                    dist[ri] = np.linalg.norm(loc - o[ri], axis=1) + 0.3
                reach.append(float(np.percentile(dist, 5)))
            out.append({"origin": c.tolist(), "normal": nn.tolist(), "xdir": ex.tolist(),
                        "in": round(min(2.5, 0.8 * reach[0]), 2), "out": round(min(2.5, 0.8 * reach[1]), 2),
                        "outline": Polygon(poly.exterior).simplify(0.3), "area": round(area, 1)})
    return out


def _slope_code(planes: list[dict], depth: float) -> list[str]:
    lines = []
    if planes:
        lines += ["", "# Наклонные плоские грани: срезаем ступеньки слоёв по плоскости и заполняем впадины"]
    for k, p in enumerate(planes):
        pts = json.dumps(np.round(np.asarray(p["outline"].exterior.coords)[:-1], 2).tolist())
        pl = (f"cq.Plane(origin={tuple(round(v, 3) for v in p['origin'])}, xDir={tuple(round(v, 5) for v in p['xdir'])}, "
              f"normal={tuple(round(v, 5) for v in p['normal'])})")
        di, do = min(depth, p["in"]), min(depth, p["out"])
        if di < 0.3 or do < 0.3:
            continue
        lines.append(f"pl = {pl}")
        lines.append(f"fill = cq.Workplane(pl).workplane(offset={-di:.2f}).polyline({pts}).close().extrude({di:.2f})")
        lines.append(f"trim = cq.Workplane(pl).polyline({pts}).close().extrude({do:.2f})")
        lines.append("try:")
        lines.append("    nxt = result.union(fill).cut(trim)")
        lines.append("    assert nxt.val().isValid() and len(nxt.solids().vals()) == len(result.solids().vals())")
        lines.append("    result = nxt")
        lines.append("except Exception:")
        lines.append(f"    pass  # грань {k + 1}: ядро не справилось — оставляем ступеньки")
    return lines

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

    # Общие оси симметрии детали: центр узора отверстий, иначе — ось основания.
    axes: dict = {}
    axes_strong = False
    for pat in patterns:
        if pat.get("pcd_diameter"):
            continue  # круговой узор: контур и так восстанавливается дугой
        pts = [list(g["c"]) for g in groups if abs(g["d"] - pat["hole_diameter"]) < 0.4]
        if len(pts) == 4:
            cx, cy = np.mean(pts, axis=0)
            axes = {"x": float(cx), "y": float(cy)}
            axes_strong = True  # ось задана четырьмя отверстиями — сильное основание
            break
    if not axes and slabs:
        base = unary_union(_section_polys(mesh, (slabs[0][0] + slabs[0][1]) / 2))
        if base.geom_type == "Polygon":
            for ax in ("x", "y"):
                c, iou = _mirror_axis(Polygon(base.exterior), ax)
                if iou > 0.85:
                    axes[ax] = c

    # Слои: внешний контур (с восстановлением), минус некруглые вырезы (пазы).
    layers = []
    prev = None
    for a, b in slabs:
        polys = _section_polys(mesh, (a + b) / 2)
        if not polys:
            continue
        shapes = []
        for p in polys:
            # ось по четырём отверстиям надёжнее: тогда зеркалим и при большом сколе
            outer = _restore_outer(Polygon(p.exterior), notes, (a + b) / 2, axes, 0.75 if axes_strong else 0.88)
            cuts = []
            for ring in p.interiors:
                pts = np.asarray(ring.coords)[:-1]
                d = sa.describe_ring(pts)
                c = Polygon(pts)
                # щель тоньше ~0.5 мм или меньше 1.5 мм² — это шум сечения, а не вырез
                if d.get("shape") != "окружность" and c.is_valid and c.area >= 1.5 and 2 * c.area / c.length >= 0.5:
                    cuts.append(c)  # паз или вырез — как в скане
            shapes.append((outer, cuts))
        if prev is not None and _same(prev["shapes"], shapes):
            prev["b"] = b  # тот же контур — продлеваем слой, а не плодим ступеньки
            continue
        prev = {"a": a, "b": b, "shapes": shapes}
        layers.append(prev)
    # отверстия в стенках ищем до достройки слоёв: их место — не скол
    sides = [_snap_faces(mesh, h) for h in side_holes(mesh)]
    sides = [h for h in sides if _arc_share(mesh, h, *h["c"]) >= 0.7]  # стенка отверстия видна по кругу
    sides += _mirror_side_holes(mesh, sides, axes, notes)
    _fill_between(layers, notes, sides)
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
            # слой заходит в нижний на 0.05 мм: касание гранями ядро иногда оставляет
            # отдельными телами, а перекрытие сливается надёжно
            lap = 0.05 if L is not layers[0] else 0.0
            lines.append(f"s = cq.Workplane('XY').workplane(offset={L['a'] - lap:.3f}).polyline({_poly_code(outer)}).close()"
                         f".extrude({L['b'] - L['a'] + lap:.3f})")
            for c in cuts:
                if len(json.loads(_poly_code(c, 0.15))) < 3:
                    continue
                lines.append(f"s = s.cut(cq.Workplane('XY').workplane(offset={L['a'] - lap - 0.01:.3f}).polyline({_poly_code(c, 0.15)})"
                             f".close().extrude({L['b'] - L['a'] + lap + 0.02:.3f}))")
            lines.append("parts.append(s)")
    lines += ["# Слои сливаем по одному; если ядро не справилось — повтор с допуском 0.01 мм",
              "result = parts[0]",
              "for p in parts[1:]:",
              "    try:",
              "        nxt = result.union(p)",
              "        assert nxt.val().isValid() and nxt.val().Volume() > result.val().Volume() * 0.999",
              "    except Exception:",
              "        nxt = result.union(p, tol=0.01)",
              "    result = nxt"]
    slopes = [p for p in sloped_planes(mesh, float(sheet.get("diagnostics", {}).get("noise_mm", 0.05) or 0.05))
              if min(p["in"], p["out"]) >= 0.3]
    lines += _slope_code(slopes, depth=2.5)
    for p in slopes:
        nz = abs(p["normal"][2])
        notes.append(f"Наклонная грань {p['area']:.0f} мм² (наклон {np.degrees(np.arccos(nz)):.0f}° к горизонту): "
                     f"построена плоскостью, а не ступеньками")
    lines += ["", "# Отверстия"]
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
    if sides:
        lines.append("")
        lines.append("# Отверстия с горизонтальной осью: слои дают ступеньки, поэтому сначала")
        lines.append("# заполняем место вокруг (кольцо +1.2 мм), потом режем точный цилиндр")
    for h in sides:
        u, v = h["c"]
        L = h["to"] - h["from"]
        def wp(m):  # m — запас за грань: заглушка ровно по стенке, вырез насквозь с запасом
            if h["axis"] == "X":
                return f"cq.Workplane('YZ').workplane(offset={h['from'] - m:.3f}).center({u:.3f}, {v:.3f})"
            # плоскость XZ смотрит в −Y: смещение и ось берём со знаком
            return f"cq.Workplane('XZ').workplane(offset={-h['to'] - m:.3f}).center({u:.3f}, {v:.3f})"
        lines.append(f"plug = {wp(0)}.circle({h['d'] / 2 + 1.2:.3f}).extrude({L:.3f})")
        lines.append(f"result = result.union(plug).cut({wp(0.5)}.circle({h['d'] / 2:.3f}).extrude({L + 1:.3f}))")
        notes.append(f"Отверстие Ø{h['d']:.2f} с осью вдоль {h['axis']} (в стенке): вырезано точным цилиндром, а не ступеньками")
    code = "\n".join(lines) + "\n"
    meta = {
        "plan": f"Без ИИ: {len(layers)} слоёв по сечениям скана, {len(groups)} отверстий (круглые подогнаны по окружности).",
        "assumptions": [("плоские наклонные грани построены плоскостью; " if slopes else "")
                        + "скругления и конусы — ступеньками по ~2 мм", "некруглые вырезы скопированы со скана как есть"],
        "restored": notes or ["ничего не восстановлено: явной симметрии или повторов не найдено"],
        "check_by_engineer": ["это черновик по сечениям, не замысел конструктора: наклоны и скругления приблизительные",
                              "восстановленные места (список выше)"],
    }
    return code, meta
