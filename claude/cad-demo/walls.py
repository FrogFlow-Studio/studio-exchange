"""Разбор скана на стенки: плоские участки и пары параллельных граней (тонкостенные детали).

Тонкостенная деталь (пластик, листовой металл) собрана из пластин. По отдельным сечениям
её форму приходится угадывать, а здесь каждая пластина находится целиком:
1) сетка делится на плоские участки (рост области по соседним треугольникам);
2) участки с противоположными нормалями на небольшом расстоянии, между которыми есть
   материал, — две стороны одной стенки: плоскость, толщина, контур;
3) черновик: каждая стенка выдавливается по контуру, всё объединяется.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _frame(n: np.ndarray):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def planar_patches(mesh: trimesh.Trimesh, max_angle: float = 10.0, max_dist: float = 0.6,
                   min_area: float = 25.0) -> list[dict]:
    """Плоские участки: рост области от крупных треугольников, сравнение с плоскостью области."""
    # Сглаживание нормалей нужно только шумному скану: на сетке из CAD (длинные узкие
    # треугольники) оно, наоборот, портит нормали.
    from scan_analysis import noise_estimate

    fn = mesh.face_normals
    if noise_estimate(mesh) > 0.05:
        sm = mesh.copy()
        trimesh.smoothing.filter_taubin(sm, iterations=8)
        fn = sm.face_normals
    fc = mesh.triangles_center
    area = mesh.area_faces
    nbrs = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nbrs[a].append(b)
        nbrs[b].append(a)
    label = -np.ones(len(mesh.faces), int)
    cos_t = np.cos(np.radians(max_angle))
    out = []
    for seed in np.argsort(-area):
        if label[seed] >= 0:
            continue
        n0, p0 = fn[seed], fc[seed]
        q, members = deque([seed]), [seed]
        label[seed] = len(out)
        while q:
            f = q.popleft()
            for g in nbrs[f]:
                if label[g] < 0 and fn[g] @ n0 > cos_t and abs((fc[g] - p0) @ n0) < max_dist:
                    label[g] = len(out)
                    members.append(g)
                    q.append(g)
            if len(members) in (20, 200, 2000):  # уточнить плоскость по набранным точкам
                w = area[members]
                p0 = np.average(fc[members], axis=0, weights=w)
                n1 = np.average(fn[members], axis=0, weights=w)
                n0 = n1 / np.linalg.norm(n1)
        A = float(area[members].sum())
        if A < min_area:
            label[members] = -2  # мелочь: скругления, шум
            continue
        w = area[members]
        p0 = np.average(fc[members], axis=0, weights=w)
        n1 = np.average(fn[members], axis=0, weights=w)
        n0 = n1 / np.linalg.norm(n1)
        out.append({"faces": np.array(members), "normal": n0, "point": p0, "area": A})
    return out


def _outline(mesh, faces, origin, u, v) -> Polygon:
    tri = mesh.vertices[mesh.faces[faces]] - origin
    polys = [Polygon(np.c_[t @ u, t @ v]).buffer(0.05) for t in tri]
    return unary_union(polys).buffer(0.3).buffer(-0.3)


def walls(mesh: trimesh.Trimesh, max_thick: float = 25.0) -> list[dict]:
    """Все пары «лицевая — обратная сторона». Один большой участок может быть стороной
    нескольких стенок (обшивка, к которой с другой стороны подходят разные рёбра)."""
    pats = planar_patches(mesh)
    pats.sort(key=lambda p: -p["area"])
    frames = [_frame(p["normal"]) for p in pats]
    outl: dict = {}

    def ol(k, ref):
        key = (k, ref)
        if key not in outl:
            u, v = frames[ref]
            outl[key] = _outline(mesh, pats[k]["faces"], pats[ref]["point"], u, v)
        return outl[key]

    seen, out = set(), []
    for i, a in enumerate(pats):
        for j, b in enumerate(pats):
            if j == i or (min(i, j), max(i, j)) in seen or a["normal"] @ b["normal"] > -0.97:
                continue
            t = float((a["point"] - b["point"]) @ a["normal"])  # b лежит на t позади a
            if not 0.6 < t < max_thick:
                continue
            pa, pb = ol(i, i), ol(j, i)
            ov = pa.intersection(pb)
            if ov.area < 15:
                continue
            u, v = frames[i]
            o = a["point"]
            pts = [ov.representative_point()]
            if ov.geom_type != "Polygon":
                pts = [g.representative_point() for g in ov.geoms if g.area > 5][:6] or pts
            mids = [o + u * c.x + v * c.y - a["normal"] * t / 2 for c in pts]
            if mesh.is_watertight and not mesh.contains(mids).all():
                continue
            seen.add((min(i, j), max(i, j)))
            full = _mid_section(mesh, o, a["normal"], u, v, t, ov)
            out.append({"normal": a["normal"], "origin": a["point"], "u": u, "v": v, "thickness": t,
                        "outline": full, "area": float(full.area)})
    out += _single_sided(mesh, pats, frames, out, max_thick)
    if mesh.is_watertight:
        out = _check_material(mesh, out)
    return _dedupe(out)


def _mid_section(mesh, origin, n, u, v, t, base):
    """Точный контур стенки — сечение тела по её средней плоскости (там видны и стыки
    с соседними стенками, и настоящий свободный край), но только рядом с base: так стенка
    доходит до соседней через скругление, а не обрывается перед ним щелью."""
    mid_o = origin - n * t / 2
    sec = mesh.section(plane_origin=mid_o, plane_normal=n)
    if sec is None:
        return base
    T = np.eye(4)
    T[:3, :3] = np.vstack([u, v, n])
    T[:3, 3] = -T[:3, :3] @ mid_o
    try:
        pl, _ = sec.to_2D(to_2D=T)
        body = unary_union([p.buffer(0) for p in pl.polygons_full])
        cand = body.intersection(base.buffer(min(3.0, 1.5 * t)))
        if cand.area > 0.7 * base.area:
            return cand
    except Exception:  # noqa: BLE001
        pass
    return base


def _check_material(mesh, ws: list[dict]) -> list[dict]:
    """Стенка должна лежать в материале. Если толщина завышена (луч ушёл в ребро), пробуем
    обычную толщину детали; если и так большая часть вне тела — стенку выбрасываем.
    Точки всех стенок проверяются вместе, кусками по 1000 (по одной стенке — в разы дольше,
    одним куском — не хватает памяти)."""
    from shapely import contains_xy
    paired = [w["thickness"] for w in ws if not w.get("single")]
    typical = float(np.median(paired)) if paired else None
    samples = []
    for i, w in enumerate(ws):
        g = w["outline"]
        x0, y0, x1, y1 = g.bounds
        xy = np.random.default_rng(i).uniform([x0, y0], [x1, y1], (3000, 2))
        samples.append(xy[contains_xy(g, xy[:, 0], xy[:, 1])][:40])

    def share(ts):
        pts, owner = [], []
        for i, (w, xy) in enumerate(zip(ws, samples)):
            if ts[i] is None or len(xy) == 0:
                continue
            for f in (0.25, 0.5, 0.75):
                pts.append(w["origin"] + xy[:, :1] * w["u"] + xy[:, 1:2] * w["v"] - w["normal"] * ts[i] * f)
                owner.append(np.full(len(xy), i))
        res = np.zeros(len(ws))
        if pts:
            P = np.vstack(pts)
            ins = np.concatenate([mesh.contains(P[k:k + 1000]) for k in range(0, len(P), 1000)]).astype(float)
            own = np.concatenate(owner)
            res = np.bincount(own, weights=ins, minlength=len(ws)) / np.maximum(np.bincount(own, minlength=len(ws)), 1)
        return res

    first = share([w["thickness"] for w in ws])
    retry = [typical if (typical and first[i] < 0.8 and w["thickness"] > typical * 1.5) else None
             for i, w in enumerate(ws)]
    second = share(retry) if any(r is not None for r in retry) else np.zeros(len(ws))
    keep = []
    for i, w in enumerate(ws):
        if first[i] >= 0.8:
            keep.append(w)
        elif retry[i] is not None and second[i] >= 0.8:
            keep.append(dict(w, thickness=typical))
    return keep


def _project(w, origin, u, v):
    """Контур стенки w в координатах другой плоскости (origin, u, v)."""
    g = w["outline"]
    res = []
    for gg in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
        r = np.asarray(gg.exterior.coords)
        P = w["origin"] + r[:, :1] * w["u"] + r[:, 1:2] * w["v"] - origin
        res.append(Polygon(np.c_[P @ u, P @ v]).buffer(0))
    return unary_union(res)


def _single_sided(mesh, pats, frames, paired, max_thick) -> list[dict]:
    """Плоский участок (или его часть), не закрытый парными стенками: обратная сторона
    бывает кривой, разбитой рёбрами на куски или мелкой. Толщину меряем лучами внутрь из
    точек участка: первое попадание в поверхность, смотрящую навстречу, — другая сторона."""
    out = []
    fc, fn = mesh.triangles_center, mesh.face_normals
    for k, p in enumerate(pats):
        n = p["normal"]
        u, v = frames[k]
        faces = p["faces"]
        cover = []
        for w in paired:
            if abs(w["normal"] @ n) < 0.97:
                continue
            mid = w["origin"] - w["normal"] * w["thickness"] / 2
            if abs((p["point"] - mid) @ w["normal"]) > w["thickness"] / 2 + 0.8:
                continue
            cover.append(_project(w, p["point"], u, v))
        if cover:
            cov = unary_union(cover).buffer(0.5)
            q = fc[faces] - p["point"]
            from shapely import contains_xy
            faces = faces[~contains_xy(cov, q @ u, q @ v)]
        if len(faces) < 5 or mesh.area_faces[faces].sum() < 15:
            continue
        orig = fc[faces] - n * 0.05
        dirs = np.repeat(-n[None], len(faces), axis=0)
        try:
            loc, ri, ti = mesh.ray.intersects_location(orig, dirs, multiple_hits=False)
        except Exception:  # noqa: BLE001
            continue
        d = np.full(len(faces), np.inf)
        d[ri] = np.einsum("ij,j->i", orig[ri] - loc, n)
        d[ri[fn[ti] @ n > -0.5]] = np.inf
        ok = (d > 0.6) & (d < max_thick)
        if ok.sum() < 0.5 * len(faces):
            continue
        # лучи, попавшие в рёбра и стыки, дают толщину больше настоящей: берём самую
        # частую толщину (пик гистограммы с шагом 0,25 мм), а не среднюю
        h, e = np.histogram(d[ok], bins=np.arange(0.6, max_thick + 0.25, 0.25))
        t = float(np.median(d[ok & (np.abs(d - e[h.argmax()] - 0.125) < 0.5)]))
        # только те треугольники, за которыми эта же стенка (а не стык с другой)
        faces = faces[ok]
        if len(faces) < 5:
            continue
        g = _outline(mesh, faces, p["point"], u, v)
        if g.area < 15:
            continue
        out.append({"normal": n, "origin": p["point"], "u": u, "v": v, "thickness": t,
                    "outline": g, "area": float(g.area), "single": True})
    return out


def _dedupe(ws: list[dict]) -> list[dict]:
    """Одна и та же стенка находится несколько раз (большая сторона в паре с кусками другой).
    Почти совпадающие пластины CAD-ядро сливает плохо, поэтому оставляем одну, а от
    остальных — только то, что она не покрывает."""
    keep: list[dict] = []
    for w in sorted(ws, key=lambda w: -w["area"]):
        g = w["outline"]
        for k in keep:
            if abs(k["normal"] @ w["normal"]) < 0.99:
                continue
            mid_w = w["origin"] - w["normal"] * w["thickness"] / 2
            mid_k = k["origin"] - k["normal"] * k["thickness"] / 2
            if abs((mid_w - mid_k) @ k["normal"]) > 0.8:
                continue
            # контур k в координатах w
            d = k["origin"] - w["origin"]
            ring = np.asarray(k["outline"].exterior.coords) if k["outline"].geom_type == "Polygon" else None
            if ring is None:
                continue
            pts3 = d + ring[:, :1] * k["u"] + ring[:, 1:2] * k["v"]
            kp = Polygon(np.c_[pts3 @ w["u"], pts3 @ w["v"]]).buffer(0)
            g = g.difference(kp.buffer(0.5))
            if g.is_empty or g.area < 15:
                break
        if g.is_empty or g.area < 15:
            continue
        keep.append(dict(w, outline=g, area=float(g.area)))
    return keep


def describe(ws: list[dict], top: int = 25) -> list[dict]:
    """Короткое описание стенок для ИИ."""
    res = []
    for k, w in enumerate(sorted(ws, key=lambda w: -w["area"])[:top]):
        poly = w["outline"]
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        s = poly.simplify(0.8)
        o, u, v = w["origin"], w["u"], w["v"]
        pts3 = [np.round(o + u * x + v * y, 1).tolist() for x, y in list(s.exterior.coords)[:-1][:40]]
        res.append({"wall": k + 1, "normal": np.round(w["normal"], 3).tolist(),
                    "thickness_mm": round(w["thickness"], 2), "area_mm2": round(w["area"], 0),
                    "outline_3d_on_outer_face": pts3, "holes_in_outline": len(s.interiors)})
    return res


def draft_code(ws: list[dict], buf: float = 0.2) -> str:
    """CadQuery: каждая стенка — контур на своей плоскости, выдавленный на толщину внутрь."""
    lines = ["import cadquery as cq", "", "parts = []"]
    for w in ws:
        poly = w["outline"]
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            if g.area < 15:
                continue
            # контур уже доходит до стыков (сечение по средней плоскости); чуть-чуть нахлёста
            g = g.buffer(buf, join_style=2).simplify(0.3)
            if g.geom_type != "Polygon":
                continue
            ext = [(round(x, 2), round(-y, 2)) for x, y in list(g.exterior.coords)[:-1]]  # ось y эскиза = -v
            if len(ext) < 3:
                continue
            o, u, n = w["origin"], w["u"], w["normal"]
            pl = (f"cq.Plane(origin={tuple(np.round(o, 3).tolist())}, xDir={tuple(np.round(u, 5).tolist())}, "
                  f"normal={tuple(np.round(-n, 5).tolist())})")
            lines.append(f"wp = cq.Workplane({pl}).polyline({ext}).close()")
            for r in g.interiors:
                hole = [(round(x, 2), round(-y, 2)) for x, y in list(r.coords)[:-1]]
                if len(hole) >= 3 and Polygon(hole).area > 4:
                    lines.append(f"wp = wp.polyline({hole}).close()")
            # нормаль плоскости эскиза смотрит внутрь детали; xDir=u, ось y эскиза = n×u = v
            lines.append(f"parts.append(wp.extrude({round(w['thickness'], 3)}))")
    lines += ["",
              "# попарно, деревом: так объединения короче и меньше копят ошибки OCC",
              "shapes = [p.val() for p in parts]",
              "while len(shapes) > 1:",
              "    nxt = []",
              "    for k in range(0, len(shapes) - 1, 2):",
              "        try:",
              "            r = shapes[k].fuse(shapes[k + 1])",
              "            ok = r.isValid() and r.Volume() > 0.9 * max(shapes[k].Volume(), shapes[k + 1].Volume())",
              "            nxt.append(r if ok else cq.Compound.makeCompound([shapes[k], shapes[k + 1]]))",
              "        except Exception:",
              "            nxt.append(cq.Compound.makeCompound([shapes[k], shapes[k + 1]]))",
              "    if len(shapes) % 2:",
              "        nxt.append(shapes[-1])",
              "    shapes = nxt",
              "result = shapes[0].clean()",
              "# мелкие оторванные куски (шум на краях) отбрасываем: деталь — одно тело",
              "sol = sorted(result.Solids(), key=lambda x: -x.Volume())",
              "if len(sol) > 1 and sum(x.Volume() for x in sol[1:]) < 0.03 * sol[0].Volume():",
              "    result = sol[0]"]
    return "\n".join(lines)
