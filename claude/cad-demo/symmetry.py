"""Зеркальная симметрия скана и кандидаты в сколы по ней (этап Д: достройка на любой форме).

Идея. У целой детали с плоскостью симметрии отражение поверхности ложится на саму поверхность.
Скол нарушает это с одной стороны: отражённая целая половина проходит там, где у скана материала
нет. Такие места — кандидаты в сколы; настоящая несимметричная деталь (бобышка с одной стороны)
даёт то же самое, поэтому кандидат подтверждается видом поверхности скана под ним: излом шероховат,
конструктивная поверхность гладкая.

python symmetry.py скан.stl
"""
from __future__ import annotations

import sys

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _reflect(P, n, d):
    return P - 2 * ((P @ n) - d)[:, None] * n


class _Near:
    """Ближайшая точка поверхности без взрыва памяти: trimesh.closest_point на точке, далёкой от
    поверхности, перебирает почти все треугольники (rav4: 10,8 ГБ). Сначала — дерево по плотной выборке,
    точный поиск — только для близких точек; для далёких хватает приближения (они всё равно дальше tol)."""

    def __init__(self, mesh: trimesh.Trimesh, n: int = 200_000):
        self.mesh = mesh
        self.P, self.F = trimesh.sample.sample_surface(mesh, n, seed=7)
        self.tree = cKDTree(self.P)
        self.step = 2.0 * float(np.sqrt(mesh.area / n))

    def __call__(self, pts, tol):
        dist, j = self.tree.query(pts)
        near, face = self.P[j].copy(), self.F[j].copy()
        close = dist < tol + self.step
        if close.any():
            near[close], dist[close], face[close] = trimesh.proximity.closest_point(self.mesh, pts[close])
        return near, dist, face


def find_planes(mesh: trimesh.Trimesh, tol: float = 0.3, starts: int = 60, keep: int = 4) -> list[dict]:
    """Плоскости зеркальной симметрии, лучшие первыми: normal, d (n·x = d), share — доля поверхности,
    чьё отражение лежит на поверхности в пределах tol. Поиск: направления по сфере и главные оси,
    грубая оценка по облаку точек, уточнение лучших, итоговая мера — расстояние точка–поверхность."""
    P = trimesh.sample.sample_surface(mesh, 120_000, seed=1)[0]
    tree = cKDTree(P)
    c = P.mean(0)
    size = float(np.linalg.norm(P - c, axis=1).max())
    rng = np.random.default_rng(0)
    Ps = P[rng.choice(len(P), 2000, replace=False)]

    def cost(n, d, cap):
        # верхняя граница поиска: далёкая от облака точка иначе обходит почти всё дерево (Hyundai: 83 с из 99)
        return float(np.minimum(tree.query(_reflect(Ps, n, d), distance_upper_bound=cap)[0], cap).mean())

    k = np.arange(starts) + 0.5   # полусфера: решётка Фибоначчи
    z = k / starts
    phi = np.pi * (1 + 5 ** 0.5) * k
    dirs = np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    dirs = np.vstack([np.linalg.svd(P - c, full_matrices=False)[2], np.eye(3), dirs])
    cap0 = 0.05 * size
    coarse = []
    for n in dirs:
        d = float(c @ n)
        for _ in range(4):  # смещение: медиана середин пар «точка — ближайшая к отражению»
            dist, j = tree.query(_reflect(Ps, n, d), distance_upper_bound=cap0)
            dist = np.minimum(dist, cap0)
            ok = (dist < max(np.percentile(dist, 50), tol)) & (j < len(P))
            d = float(np.median(((Ps[ok] + P[j[ok]]) / 2) @ n))
        coarse.append((cost(n, d, cap0), n, d))
    coarse.sort(key=lambda t: t[0])
    out = []
    for _, n0, d0 in coarse[: 2 * keep]:
        e1 = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n0, e1)

        def plane(x):
            n = n0 + x[0] * e1 + x[1] * e2
            n = n / np.linalg.norm(n)
            return n, d0 + x[2]

        x = np.zeros(3)
        for cap in (cap0, 0.01 * size, max(1.0, 3 * tol), tol):  # грубо → точно
            simplex = np.array([x, x + [0.03, 0, 0], x + [0, 0.03, 0], x + [0, 0, 0.01 * size]])
            x = minimize(lambda v: cost(*plane(v), cap), x, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 250, "initial_simplex": simplex}).x
        n, d = plane(x)
        if any(abs(n @ o["normal"]) > np.cos(np.radians(2)) and abs(abs(d) - abs(o["d"])) < tol for o in out):
            continue
        out.append({"normal": n, "d": float(d), "cost": cost(n, d, tol)})
    out.sort(key=lambda o: o["cost"])
    out = out[:keep]
    Q = trimesh.sample.sample_surface(mesh, 15_000, seed=5)[0]
    near = _Near(mesh)
    for o in out:
        dist = near(_reflect(Q, o["normal"], o["d"]), tol)[1]
        o["share"] = float((dist < tol).mean())
    out.sort(key=lambda o: -o["share"])
    return out


def _mirror(mesh: trimesh.Trimesh, plane: dict) -> trimesh.Trimesh:
    B = mesh.copy()
    B.vertices = _reflect(B.vertices, plane["normal"], plane["d"])
    B.invert()
    return B


def chips(mesh: trimesh.Trimesh, plane: dict, tol: float = 0.3, min_share: float = 0.002,
          source: trimesh.Trimesh | None = None) -> list[dict]:
    """Места, где отражению скана не хватает материала скана: кандидаты в сколы.
    Возвращает куски отражённой поверхности: faces (грани отражения), area, depth, rough_deg
    (разброс нормалей скана под куском) и smooth_deg (то же у отражённой, целой стороны)."""
    n, d = plane["normal"], plane["d"]
    B = _mirror(mesh if source is None else source, plane)
    cen = B.triangles_center
    near, dist, fa = _Near(mesh)(cen, tol)
    outside = (dist > tol) & (np.einsum("ij,ij->i", cen - near, mesh.face_normals[fa]) > 0)
    adj = B.face_adjacency
    adj = adj[outside[adj[:, 0]] & outside[adj[:, 1]]]
    tc = cKDTree(mesh.triangles_center)

    def spread(m, faces, tree, r):
        v = []
        for f in faces[:: max(1, len(faces) // 150)]:
            nb = tree.query_ball_point(m.triangles_center[f], r)
            v.append(np.degrees(np.arccos(np.clip(m.face_normals[nb] @ m.face_normals[f], -1, 1))).mean())
        return float(np.median(v)) if v else 0.0

    tb = cKDTree(cen)
    r = 3 * float(np.sqrt(mesh.area_faces.mean()))
    out = []
    for comp in trimesh.graph.connected_components(adj, nodes=np.where(outside)[0], min_len=3):
        comp = np.asarray(comp)
        area = float(B.area_faces[comp].sum())
        if area < min_share * mesh.area:
            continue
        out.append({"faces": comp, "scan_faces": np.unique(fa[comp]), "area": area, "depth": float(dist[comp].max()),
                    "center": cen[comp].mean(0), "rough_deg": spread(mesh, np.unique(fa[comp]), tc, r),
                    "smooth_deg": spread(B, comp, tb, r)})
    out.sort(key=lambda o: -o["area"])
    return out, B


if __name__ == "__main__":
    m = trimesh.load(sys.argv[1], force="mesh")
    pl = find_planes(m)
    for p in pl:
        print(f"плоскость n={np.round(p['normal'], 3)} d={p['d']:.3f}: совпало {p['share'] * 100:.1f}%")
    if pl and pl[0]["share"] > 0.6:
        for p in pl[:2]:
            cs, _ = chips(m, p)
            print(f"кандидаты по плоскости {np.round(p['normal'], 2)}:")
            for c in cs[:8]:
                print(f"  {c['area']:.0f} мм², центр {np.round(c['center'], 1)}, глубина {c['depth']:.2f} мм, "
                      f"шероховатость скана {c['rough_deg']:.1f}°, целой стороны {c['smooth_deg']:.1f}°")


def restore(mesh: trimesh.Trimesh, plane: dict, chip: dict, B: trimesh.Trimesh, grow: float = 1.0):
    """Достройка скола отражением целой стороны: скан ∪ (отражение ∩ область скола).
    Область — выпуклая оболочка куска отражённой поверхности, раздутая на grow мм: отражение берётся
    только у скола, чужие несимметричные детали не переносятся. None, если сетка не замкнута."""
    import manifold3d as m3

    def man(t):
        return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))

    def tri(m):
        o = m.to_mesh()
        return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)

    hull = chip["hull"] if "hull" in chip else _region(mesh, chip, B, grow)
    try:
        a, b, h = man(mesh), man(B), man(hull)
        if a.is_empty() or b.is_empty() or h.is_empty():
            return None
        u = a + (b ^ h)
        parts = u.decompose()
        if len(parts) > 1:  # тонкие отщепы от пересечения оболочки с поверхностью
            u = max(parts, key=lambda p: p.volume())
        if u.volume() < a.volume():
            return None
        out = tri(u)
        shells = out.split(only_watertight=False)
        if len(shells) > 1:  # замкнутая полость внутри тела: у скана их не бывает (сканер видит только снаружи)
            keep = [q for q in shells if q.volume > 0]
            if keep and len(keep) < len(shells):
                out = trimesh.util.concatenate(keep)
        return out
    except Exception:  # noqa: BLE001 — сетка не замкнута или булева операция не удалась
        return None


def _region(mesh, chip, B, grow):
    """Область скола: выпуклая оболочка куска отражённой поверхности и дна излома в скане, раздутая на grow."""
    V = B.vertices[np.unique(B.faces[chip["faces"]])]
    if len(chip.get("scan_faces", ())):  # дно излома в скане: иначе под достроенной коркой остаётся полость
        V = np.vstack([V, mesh.vertices[np.unique(mesh.faces[chip["scan_faces"]])]])
    hull = trimesh.convex.convex_hull(V)
    c = hull.centroid
    hull.vertices = c + (hull.vertices - c) * (1 + grow / max(1e-6, np.linalg.norm(hull.vertices - c, axis=1).mean()))
    return hull


def merge_chips(mesh, cs, B, grow: float = 1.0):
    """Куски одного скола (сквозной излом выходит на две стороны детали) — один кандидат:
    сливаются те, чьи области пересекаются."""
    import manifold3d as m3

    if len(cs) < 2:
        return cs
    hulls = []
    for c in cs:
        h = _region(mesh, c, B, grow)
        hulls.append(m3.Manifold(m3.Mesh(np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.uint32))))
    group = list(range(len(cs)))
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if group[i] != group[j] and not (hulls[i] ^ hulls[j]).is_empty():
                gi, gj = group[i], group[j]
                group = [gi if g == gj else g for g in group]
    out = []
    for g in sorted(set(group)):
        m = [c for c, q in zip(cs, group) if q == g]
        if len(m) == 1:
            out.append(m[0])
            continue
        w = np.array([c["area"] for c in m])
        out.append({"faces": np.concatenate([c["faces"] for c in m]),
                    "scan_faces": np.unique(np.concatenate([c["scan_faces"] for c in m])),
                    "area": float(w.sum()), "depth": max(c["depth"] for c in m),
                    "center": (np.array([c["center"] for c in m]) * w[:, None]).sum(0) / w.sum(),
                    "rough_deg": max(c["rough_deg"] for c in m), "smooth_deg": max(c["smooth_deg"] for c in m)})
    out.sort(key=lambda o: -o["area"])
    return out


def restore_all(mesh: trimesh.Trimesh, tol: float = 0.3, min_share: float = 0.8):
    """Достройка всех сколов по найденным плоскостям симметрии. Скол, пересекающий одну плоскость,
    ею не достраивается целиком (отражается сам в себя) — остаток закрывает следующая плоскость.
    Возвращает (сетка или исходная, список достроенного: plane, area, depth, center, volume)."""
    planes = [p for p in find_planes(mesh, tol) if p["share"] >= min_share]
    if planes:
        planes = [p for p in planes if p["share"] >= planes[0]["share"] - 0.07]
    N = 15_000

    def shares(m):
        Q = trimesh.sample.sample_surface(m, N, seed=5)[0]
        near = _Near(m)
        return [float((near(_reflect(Q, q["normal"], q["d"]), tol)[1] < tol).mean()) for q in planes]

    cur, log = mesh, []
    base = shares(cur) if len(planes) > 1 else None
    for _ in range(6 * max(1, len(planes))):
        # за шаг — одна достройка: та, что добавляет больше всего. Скол, пересекающий плоскость, ею
        # достраивается лишь частично (отражается сам в себя); другая плоскость даёт его целиком.
        best = None
        for p in planes:
            # отражается только измеренное (исходная сетка): достроенное второй раз не отражается,
            # иначе неверная плоскость размножает копии выступов по кругу
            cs, B = chips(cur, p, tol, source=mesh)
            for c in merge_chips(cur, cs, B):
                r = restore(cur, p, c, B)
                if r is None:
                    continue
                gain = float(r.volume - cur.volume)
                # полнота: у скола объём сравним с площадью × глубину (образцы: 0,13–0,32); тонкая
                # плёнка от неточного совпадения отражения (над отверстием: 0,035) — не скол
                if gain < 0.08 * c["area"] * c["depth"]:
                    continue
                if base is not None:
                    # согласие плоскостей: настоящая достройка не делает деталь менее симметричной
                    # относительно других найденных плоскостей; копия выступа по неверной плоскости
                    # делает (крышка: диагональ размножала ушки). Допуск — разброс выборки, 3σ.
                    now = shares(r)
                    if any(q is not p and v < b0 - 3 * np.sqrt(max(b0 * (1 - b0), 1e-4) / N)
                           for q, v, b0 in zip(planes, now, base)):
                        continue
                if best is None or gain > best[0]:
                    V = B.vertices[np.unique(B.faces[c["faces"]])]
                    c = dict(c, radius=float(np.linalg.norm(V - c["center"], axis=1).max()) + 1.0,
                             hull=_region(cur, c, B, 1.0))
                    best = (gain, r, p, c)
                break
        if best is None:
            break
        gain, cur, p, c = best
        if base is not None:
            base = shares(cur)
        log.append({"normal": p["normal"], "d": p["d"], "area": c["area"], "depth": c["depth"],
                    "center": c["center"], "radius": c["radius"], "hull": c["hull"], "volume": gain})
    return cur, log, planes


def describe(entry: dict) -> str:
    """Строка для отчёта «Достроено (этого нет в скане)»."""
    c, n = np.round(entry["center"], 1), np.round(entry["normal"], 2)
    return (f"скол у точки ({c[0]:g}, {c[1]:g}, {c[2]:g}) мм достроен отражением целой стороны по плоскости "
            f"симметрии (нормаль {n[0]:g}, {n[1]:g}, {n[2]:g}): площадь {entry['area']:.0f} мм², глубина "
            f"{entry['depth']:.1f} мм, добавлено {entry['volume']:.0f} мм³")


def zone(entry: dict) -> dict:
    """Шар, внутри которого точки скана на изломе не считаются ошибкой модели (там скол, модель достроена)."""
    return {"center": [float(x) for x in entry["center"]], "radius": float(entry["radius"])}


def apply_to(mesh: trimesh.Trimesh, log: list[dict]) -> trimesh.Trimesh:
    """Те же достройки (плоскость и область из restore_all по сырому скану) на другой сетке той же
    детали — на очищенной поверхности. Отражается сама сетка, в пределах области. Не вышло — как было."""
    src = mesh
    for e in log:
        p = {"normal": e["normal"], "d": e["d"]}
        r = restore(mesh, p, {"hull": e["hull"]}, _mirror(src, p))
        if r is not None and r.volume > mesh.volume:
            mesh = r
    return mesh
