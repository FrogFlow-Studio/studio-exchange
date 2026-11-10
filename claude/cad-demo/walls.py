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
from shapely.geometry import Point, Polygon
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
    return _merge_coplanar(out, area) if MERGE_COPLANAR else out


def _merge_coplanar(pats: list[dict], area: np.ndarray) -> list[dict]:
    """Шумный скан режет одну плоскую грань на много участков (рост области упирается в шум).
    Участки в одной плоскости (нормали почти совпадают, плоскости ближе 0,4 мм) сливаем в один:
    иначе одна пластина детали собиралась из десятка обрезков, а круглые отверстия в ней
    терялись между кусками."""
    order = sorted(range(len(pats)), key=lambda k: -pats[k]["area"])
    groups: list[list[int]] = []
    for k in order:
        p = pats[k]
        for g in groups:
            q = pats[g[0]]
            if p["normal"] @ q["normal"] > 0.995 and abs((p["point"] - q["point"]) @ q["normal"]) < 0.4:
                g.append(k)
                break
        else:
            groups.append([k])
    out = []
    for g in groups:
        faces = np.concatenate([pats[k]["faces"] for k in g])
        w = area[faces]
        n = np.sum([pats[k]["normal"] * pats[k]["area"] for k in g], axis=0)
        n /= np.linalg.norm(n)
        p0 = np.sum([pats[k]["point"] * pats[k]["area"] for k in g], axis=0) / sum(pats[k]["area"] for k in g)
        out.append({"faces": faces, "normal": n, "point": p0, "area": float(w.sum())})
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
            # середина пары должна быть в материале (иначе это две стенки по разные стороны щели);
            # у незамкнутого скана mesh.contains не работает — тогда проверяем по ближайшим точкам
            inside = mesh.contains(mids) if mesh.is_watertight else _inside(mesh, np.asarray(mids))
            if not inside.all():
                continue
            seen.add((min(i, j), max(i, j)))
            full = _mid_section(mesh, o, a["normal"], u, v, t, ov)
            out.append({"normal": a["normal"], "origin": a["point"], "u": u, "v": v, "thickness": t,
                        "outline": full, "area": float(full.area)})
    out += _single_sided(mesh, pats, frames, out, max_thick)
    out = [_clip_depth(mesh, w) for w in out]
    out = [w for w in out if w["outline"].area >= 15]
    out = _check_material(mesh, out)
    out = [_grow(mesh, w) for w in out]
    # после вычитания соседних стенок отверстие, открытое в стык, становится замкнутым — скругляем ещё раз
    out = [dict(w, outline=_carve_holes(_round_holes(w["outline"], w.get("body")), w.get("body")))
           for w in _dedupe(out)]
    return out + (_hole_walls(mesh) if HOLES else [])


HOLES = True  # отверстия скана переносим в модель точно: кольцо материала вокруг и цилиндр насквозь


def _hole_walls(mesh) -> list[dict]:
    """Посадочные отверстия — самое важное в детали. Каждое круглое отверстие скана
    переносим в модель отдельно: кольцо материала шириной 2 мм вокруг (только там, где в
    скане есть материал) и в конце вырез цилиндром диаметра из скана на толщину пластины."""
    import holes as H

    out = []
    try:
        found = H.find_holes(mesh)
    except Exception:  # noqa: BLE001
        return out
    for h in found:
        try:
            m = H._measure(mesh, h, 0.6)
            if m is None:
                continue
            r = m["d"] / 2
            n, u, v, c = h["normal"], h["u"], h["v"], h["center"]
            # длина отверстия — пока сечение скана на этой глубине даёт ту же окружность
            t = 0.6
            for dep in np.arange(0.9, 30, 0.3):
                q = H._measure(mesh, h, float(dep))
                if q is None or abs(q["d"] - m["d"]) > 0.4 or q["offset"] > 0.5:
                    break
                t = float(dep)
            t += 0.3  # край отверстия: последнее ровное сечение чуть не доходит до грани
            if not 0.6 < t < 30:
                continue
            wf = {"origin": c, "normal": n, "u": u, "v": v}
            secs = [b for b in (_body_at(mesh, wf, dd) for dd in sorted({0.6, t / 2})) if b is not None and not b.is_empty]
            if not secs:
                continue
            body = unary_union(secs)
            ring = Point(0, 0).buffer(r + 2.0, 64).difference(Point(0, 0).buffer(r, 64))
            g = ring.intersection(body.buffer(0.1))
            if g.is_empty or g.area < 0.5 * ring.area:
                continue
            out.append(dict(wf, thickness=t, outline=g, area=float(g.area), single=True,
                            hole={"r": r, "t": t}))
        except Exception:  # noqa: BLE001
            continue
    return out


GROW_MM = 0.5
BAND_MM = 0.7
CLIP_AT = (0.25, 0.5, 0.75, 0.95)
CLIP_BUF = 0.1
CHECK_PTS = 100
CHECK_MIN = 0.9
MERGE_COPLANAR = True
MERGE_SAME = 0.0  # сливаем кусок пластины не больше этой площади, мм² (0 — не сливать)
DEDUPE_GAP = 0.5
WEDGE = False  # «выдавить до грани»: на кронштейне студии давал лишнее, ступени по глубине точнее


def _grow(mesh, w):
    """Стенка доходит до соседней: к контуру добавляется материал её среднего сечения в
    полосе GROW_MM вокруг. Без этого на стыках и в скруглениях остаются щели."""
    if GROW_MM <= 0 or w.get("until"):
        return w
    g = w["outline"]
    mid_o = w["origin"] - w["normal"] * w["thickness"] / 2
    sec = mesh.section(plane_origin=mid_o, plane_normal=w["normal"])
    if sec is None:
        return w
    T = np.eye(4)
    T[:3, :3] = np.vstack([w["u"], w["v"], w["normal"]])
    T[:3, 3] = -T[:3, :3] @ w["origin"]
    try:
        pl, _ = sec.to_2D(to_2D=T)
        body = unary_union([p.buffer(0) for p in pl.polygons_full])
        g2 = unary_union([g, body.intersection(g.buffer(GROW_MM))])
        g2 = _round_holes(g2, body)
    except Exception:  # noqa: BLE001
        return w
    return dict(w, outline=g2, area=float(g2.area), body=body)


def _circle(pts: np.ndarray):
    """Окружность по точкам (алгебраическая подгонка): центр, радиус, СКО."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 0)))
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - r) ** 2)))
    return c, r, rms


def _round_holes(g, body):
    """Круглые отверстия делаем круглыми: контур отверстия в скане — ломаная с шумом, а по
    замыслу это окружность. Окружность подгоняем по сечению тела посередине стенки (там нет
    фаски и скругления кромки)."""
    if g.is_empty or body is None or body.is_empty:
        return g
    from shapely.geometry import Point

    rings = [r for b in ([body] if body.geom_type == "Polygon" else list(body.geoms)) for r in b.interiors]
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    res = []
    for poly in polys:
        new_int = []
        for ring in poly.interiors:
            pts = np.asarray(ring.coords)[:-1]
            c0, r0, rms0 = _circle(pts) if len(pts) >= 6 else (None, 0, 1e9)
            if c0 is None or not (0.75 <= r0 <= 20) or rms0 > 0.15 * r0:
                new_int.append(ring)
                continue
            cand = [r for r in rings if Point(c0).distance(Polygon(r).centroid) < max(1.0, 0.5 * r0)]
            if not cand:  # в теле на этом месте отверстия нет — это не отверстие, а стык
                new_int.append(ring)
                continue
            c, r, rms = _circle(np.asarray(cand[0].coords)[:-1])
            if rms > 0.15 * r or abs(r - r0) > max(0.5, 0.3 * r0):
                c, r = c0, r0
            ang = np.linspace(0, 2 * np.pi, 49)[:-1]
            new_int.append(np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
        try:
            res.append(Polygon(poly.exterior, new_int).buffer(0))
        except Exception:  # noqa: BLE001
            res.append(poly)
    return unary_union(res)


def _carve_holes(g, body):
    """Круглое отверстие тела вырезаем в контуре стенки точной окружностью, даже если после
    вычитания соседних стенок оно открылось в стык: вокруг отверстия возвращаем материал тела
    (кольцо 1,5 мм), саму окружность вырезаем."""
    if g.is_empty or body is None or body.is_empty:
        return g
    try:
        for b in ([body] if body.geom_type == "Polygon" else list(body.geoms)):
            for ring in b.interiors:
                pts = np.asarray(ring.coords)[:-1]
                if len(pts) < 8:
                    continue
                c, r, rms = _circle(pts)
                if not (0.75 <= r <= 12) or rms > 0.1 * r:
                    continue
                disk = Point(c).buffer(r, 48)
                ann = disk.buffer(1.5).difference(disk)
                if body.intersection(ann).area < 0.9 * ann.area:
                    continue  # вокруг нет сплошного материала — это не отверстие в пластине
                if any(Polygon(q).contains(Point(c)) and abs(Polygon(q).area - disk.area) < 0.3 * disk.area
                       for p in ([g] if g.geom_type == "Polygon" else list(g.geoms)) for q in p.interiors):
                    continue  # отверстие в стенке уже есть
                if g.buffer(0.05).intersection(disk.buffer(1.0)).area < 0.5 * disk.buffer(1.0).difference(disk).area:
                    continue  # отверстие не в этой стенке
                g2 = unary_union([g, body.intersection(disk.buffer(r * 0.3 + 1.5))]).difference(disk)
                if not g2.is_empty and g2.is_valid and g2.area - g.area < 0.5 * ann.area:
                    g = g2
    except Exception:  # noqa: BLE001
        return g
    return g


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


def _body_at(mesh, w, depth):
    """Сечение тела плоскостью стенки на глубине depth, в координатах стенки."""
    o = w["origin"] - w["normal"] * depth
    sec = mesh.section(plane_origin=o, plane_normal=w["normal"])
    if sec is None:
        return None
    T = np.eye(4)
    T[:3, :3] = np.vstack([w["u"], w["v"], w["normal"]])
    T[:3, 3] = -T[:3, :3] @ w["origin"]
    pl, _ = sec.to_2D(to_2D=T)
    return unary_union([p.buffer(0) for p in pl.polygons_full])


def _clip_depth(mesh, w):
    """Глубокая стенка (толстое место) не должна выходить из тела у дна: боковые грани
    толстых мест бывают с уклоном, и прямая призма по контуру верха вылезает наружу.
    Обрезаем контур по сечениям тела на половине и у дна стенки."""
    if not w.get("single") or w["thickness"] < 3:
        return w
    g = w["outline"]
    try:
        for f in CLIP_AT:
            b = _body_at(mesh, w, w["thickness"] * f)
            if b is None:
                return w
            g = g.intersection(b.buffer(CLIP_BUF))
    except Exception:  # noqa: BLE001
        return w
    return dict(w, outline=g, area=float(g.area))


_KD: dict = {}


def _inside(mesh, P: np.ndarray) -> np.ndarray:
    """Точка внутри тела? По знаку нормали у ближайших точек поверхности (плотная выборка
    + KD-дерево): с mesh.contains совпадает на 99,7%, но в сотни раз быстрее и без
    всплесков памяти (contains на нескольких тысячах точек съедал гигабайты)."""
    from scipy.spatial import cKDTree

    key = id(mesh)
    if key not in _KD:
        n = int(np.clip(mesh.area / 0.25, 20000, 400000))  # точка на ~0,25 мм²
        pts, fi = trimesh.sample.sample_surface_even(mesh, n, seed=0)
        _KD.clear()
        _KD[key] = (cKDTree(pts), pts, mesh.face_normals[fi])
    tree, pts, nrm = _KD[key]
    _, ii = tree.query(P, k=3)
    s = np.einsum("ijk,ijk->ij", P[:, None, :] - pts[ii], nrm[ii]) < 0
    return s.sum(axis=1) >= 2


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
        samples.append(xy[contains_xy(g, xy[:, 0], xy[:, 1])][:CHECK_PTS])

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
            ins = _inside(mesh, np.vstack(pts)).astype(float)
            own = np.concatenate(owner)
            res = np.bincount(own, weights=ins, minlength=len(ws)) / np.maximum(np.bincount(own, minlength=len(ws)), 1)
        return res

    first = share([None if w.get("until") else w["thickness"] for w in ws])
    for i, w in enumerate(ws):
        if w.get("until"):
            first[i] = 1.0  # клин ограничен гранями скана с обеих сторон
    retry = [typical if (typical and first[i] < CHECK_MIN and w["thickness"] > typical * 1.5) else None
             for i, w in enumerate(ws)]
    second = share(retry) if any(r is not None for r in retry) else np.zeros(len(ws))
    keep = []
    for i, w in enumerate(ws):
        if first[i] >= CHECK_MIN:
            keep.append(w)
        elif retry[i] is not None and second[i] >= CHECK_MIN:
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
    lab = -np.ones(len(mesh.faces), int)
    for k, p in enumerate(pats):
        lab[p["faces"]] = k
    for k, p in enumerate(pats):
        n = p["normal"]
        u, v = frames[k]
        faces = p["faces"]
        orig = fc[faces] - n * 0.05
        dirs = np.repeat(-n[None], len(faces), axis=0)
        try:
            loc, ri, ti_ = mesh.ray.intersects_location(orig, dirs, multiple_hits=False)
        except Exception:  # noqa: BLE001
            continue
        d = np.full(len(faces), np.inf)
        d[ri] = np.einsum("ij,j->i", orig[ri] - loc, n)
        d[ri[fn[ti_] @ n > -0.5]] = np.inf
        hit_face = np.full(len(faces), -1)
        hit_face[ri] = ti_
        # Закрыт парной стенкой — только если под треугольником и глубина та же: за тонкой
        # пластиной бывает толстый прилив, и его надо добрать, а не считать готовым.
        covered = np.zeros(len(faces), bool)
        q = fc[faces] - p["point"]
        from shapely import contains_xy
        for w in paired:
            if abs(w["normal"] @ n) < 0.97:
                continue
            mid = w["origin"] - w["normal"] * w["thickness"] / 2
            if abs((p["point"] - mid) @ w["normal"]) > w["thickness"] / 2 + 0.8:
                continue
            inside = contains_xy(_project(w, p["point"], u, v).buffer(0.5), q @ u, q @ v)
            same = ~np.isfinite(d) | (np.abs(d - w["thickness"]) < max(0.7, 0.3 * w["thickness"]))
            covered |= inside & same
        faces, d, hit_face = faces[~covered], d[~covered], hit_face[~covered]
        if len(faces) < 5 or mesh.area_faces[faces].sum() < 15:
            continue
        ok = (d > 0.6) & (d < max_thick)
        if ok.sum() < 0.5 * len(faces):
            continue
        # Клин: лучи упираются в одну и ту же плоскую грань, но на разной глубине (грань не
        # параллельна). Такой кусок — не пластина, а тело «от грани до грани»: выдавливаем
        # до противоположной плоскости, а не на одну толщину (иначе внутри остаются карманы).
        hit = np.where(hit_face >= 0, lab[np.maximum(hit_face, 0)], -1)
        hb = hit[ok & (hit >= 0) & (hit != k)]
        if WEDGE and len(hb):
            b = int(np.bincount(hb).argmax())
            on_b = ok & (hit == b)
            nb = pats[b]["normal"]
            if (on_b.sum() >= 0.6 * ok.sum() and n @ nb > -0.97 and n @ nb < -0.2
                    and np.ptp(d[on_b]) > 1.5 and on_b.sum() >= 5):
                g = _outline(mesh, faces[on_b], p["point"], u, v)
                if g.area >= 15:
                    out.append({"normal": n, "origin": p["point"], "u": u, "v": v,
                                "thickness": float(d[on_b].max()) + 1.0, "outline": g,
                                "area": float(g.area), "single": True,
                                "until": {"point": pats[b]["point"], "normal": nb}})
                faces, d = faces[~on_b], d[~on_b]
                ok = (d > 0.6) & (d < max_thick)
                if ok.sum() < 5:
                    continue
        # Глубина под участком бывает разной (тонкая пластина переходит в толстый прилив).
        # Ступени по глубине с шагом BAND_MM: ступень j — все треугольники глубже её нижней
        # границы, выдавленные на её толщину. Ступени вложены, поэтому грань закрыта целиком
        # (первой ступенью), а толстые места добираются следующими. Одна «типичная» толщина на
        # весь участок оставляла в толстых местах карманы.
        band = np.floor(d / BAND_MM)
        for bv in np.unique(band[ok]):
            sel = ok & (band == bv)
            if sel.sum() < 5:
                continue
            deeper = ok & (band >= bv)
            g = _outline(mesh, faces[deeper], p["point"], u, v)
            if g.area < 15:
                continue
            out.append({"normal": n, "origin": p["point"], "u": u, "v": v,
                        "thickness": float(np.median(d[sel])), "outline": g, "area": float(g.area),
                        "single": True})
    return out


def _to_frame(geom, src: dict, dst: dict):
    """Контур стенки src (вместе с отверстиями и всеми кусками) в координатах стенки dst."""
    import shapely

    def f(xy):
        P = src["origin"] + xy[:, :1] * src["u"] + xy[:, 1:2] * src["v"] - dst["origin"]
        return np.c_[P @ dst["u"], P @ dst["v"]]

    return shapely.transform(geom, f).buffer(0)


def _dedupe(ws: list[dict]) -> list[dict]:
    """Одна и та же стенка находится несколько раз (большая сторона в паре с кусками другой,
    шумная грань разбита на участки). Пластины в одной плоскости и одной толщины сливаем в
    одну до выдавливания: так нет ни совпадающих граней (их CAD-ядро сливает долго и плохо),
    ни щелей между кусками. От пластин другой толщины остаётся только то, что не покрыто."""
    keep: list[dict] = []
    for w in sorted(ws, key=lambda w: -w["area"]):
        g = w["outline"]
        merged = False
        for i, k in enumerate(keep):
            if abs(k["normal"] @ w["normal"]) < 0.99:
                continue
            mid_w = w["origin"] - w["normal"] * w["thickness"] / 2
            mid_k = k["origin"] - k["normal"] * k["thickness"] / 2
            if abs((mid_w - mid_k) @ k["normal"]) > 0.8:
                continue
            if w["area"] <= MERGE_SAME and abs(w["thickness"] - k["thickness"]) < 0.4 and not w.get("until") and not k.get("until"):
                u = unary_union([k["outline"], _to_frame(g, w, k)])
                keep[i] = dict(k, outline=u, area=float(u.area))
                merged = True
                break
            g = g.difference(_to_frame(k["outline"], k, w).buffer(DEDUPE_GAP))
            if g.is_empty or g.area < 15:
                break
        if merged or g.is_empty or g.area < 15:
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
            # нахлёст наружу — только по внешнему краю: отверстия от него сужались бы
            ext_g = Polygon(g.exterior).buffer(buf, join_style=2).simplify(0.3)
            if ext_g.geom_type != "Polygon":
                continue
            inner = ext_g.buffer(-0.3)
            rings = [Polygon(r).simplify(0.1) for r in g.interiors]
            rings = [r for r in rings if r.is_valid and r.area > 4 and inner.contains(r)]
            g2 = Polygon(ext_g.exterior, [r.exterior for r in rings])
            g = g2 if g2.is_valid else ext_g
            ext = [(round(x, 2), round(-y, 2)) for x, y in list(g.exterior.coords)[:-1]]  # ось y эскиза = -v
            if len(ext) < 3:
                continue
            o, u, n = w["origin"], w["u"], w["normal"]
            pl = (f"cq.Plane(origin={tuple(np.round(o, 3).tolist())}, xDir={tuple(np.round(u, 5).tolist())}, "
                  f"normal={tuple(np.round(-n, 5).tolist())})")
            lines.append("try:")
            lines.append(f"    wp = cq.Workplane({pl}).polyline({ext}).close()")
            for r in g.interiors:
                hole = [(round(x, 2), round(-y, 2)) for x, y in list(r.coords)[:-1]]
                if len(hole) >= 3 and Polygon(hole).area > 4:
                    lines.append(f"    wp = wp.polyline({hole}).close()")
            # нормаль плоскости эскиза смотрит внутрь детали; xDir=u, ось y эскиза = n×u = v
            if w.get("until"):
                pb, nb = w["until"]["point"], w["until"]["normal"]
                xb = _frame(nb)[0]
                hs = (f"cq.Workplane(cq.Plane(origin={tuple(np.round(pb, 3).tolist())}, "
                      f"xDir={tuple(np.round(xb, 5).tolist())}, normal={tuple(np.round(-nb, 5).tolist())}))"
                      ".rect(2000, 2000).extrude(1000)")
                lines.append(f"    parts.append(wp.extrude({round(w['thickness'], 3)}).intersect({hs}))  # клин до грани")
            else:
                lines.append(f"    parts.append(wp.extrude({round(w['thickness'], 3)}))")
            lines.append("except Exception:  # одна неудачная пластина не роняет весь черновик")
            lines.append("    pass")
    cuts = [w for w in ws if w.get("hole")]
    lines += ["", "# посадочные отверстия: цилиндры диаметра из скана, вырезаются в самом конце",
              "hole_cuts = [" + ", ".join(
                  f"({tuple(np.round(w['origin'] + w['normal'] * 0.3, 3).tolist())}, "
                  f"{tuple(np.round(-w['normal'], 5).tolist())}, {round(w['hole']['r'], 3)}, "
                  f"{round(w['hole']['t'] + 0.3, 3)})" for w in cuts) + "]"]
    lines += ["",
              "# попарно, деревом: так объединения короче и меньше копят ошибки OCC",
              "shapes = [p.val() for p in parts]",
              "while len(shapes) > 1:",
              "    nxt = []",
              "    for k in range(0, len(shapes) - 1, 2):",
              "        try:",
              "            r = shapes[k].fuse(shapes[k + 1])",
              "            # isValid на каждом шаге стоил ~30 с из 2,5 мин; итоговое тело всё равно проверяется",
              "            ok = len(r.Solids()) > 0 and r.Volume() > 0.9 * max(shapes[k].Volume(), shapes[k + 1].Volume())",
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
              "    result = sol[0]",
              "# пустоты внутри (стенки сошлись не вплотную и замкнули полость) заполняем: оставляем",
              "# только наружную оболочку — у литой детали внутренних полостей нет",
              "from OCP.BRepClass3d import BRepClass3d",
              "if not isinstance(result, cq.Solid) and len(result.Solids()) == 1:",
              "    result = result.Solids()[0]",
              "if isinstance(result, cq.Solid) and len(result.Shells()) > 1:",
              "    outer = cq.Shell(BRepClass3d.OuterShell_s(result.wrapped))",
              "    filled = cq.Solid.makeSolid(outer)",
              "    # наружная оболочка не меняется, поэтому отверстия (они выходят наружу) остаются;",
              "    # прибавиться может только объём закрытых полостей — иначе заливку отменяем",
              "    voids = sum(abs(cq.Solid.makeSolid(sh).Volume()) for sh in result.Shells() if not sh.isSame(outer))",
              "    if filled.isValid() and abs(filled.Volume() - result.Volume() - voids) < 0.01 * result.Volume() + 1:",
              "        result = filled",
              "import math",
              "for c_, d_, r_, h_ in hole_cuts:",
              "    try:",
              "        cyl = cq.Solid.makeCylinder(r_, h_, cq.Vector(*c_), cq.Vector(*d_))",
              "        cut = result.cut(cyl)",
              "        removed = result.Volume() - cut.Volume()",
              "        # вырез не должен задеть ничего, кроме самой пластины",
              "        if len(cut.Solids()) == len(result.Solids()) and -1 < removed <= 1.05 * math.pi * r_ * r_ * h_ + 1:",
              "            result = cut.Solids()[0] if len(cut.Solids()) == 1 else cut",
              "    except Exception:",
              "        pass"]
    return "\n".join(lines)
