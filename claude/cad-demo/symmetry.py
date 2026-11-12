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
        return float(np.minimum(tree.query(_reflect(Ps, n, d))[0], cap).mean())

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
            dist, j = tree.query(_reflect(Ps, n, d))
            ok = dist < max(np.percentile(dist, 50), tol)
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
    for o in out:
        dist = trimesh.proximity.closest_point(mesh, _reflect(Q, o["normal"], o["d"]))[1]
        o["share"] = float((dist < tol).mean())
    out.sort(key=lambda o: -o["share"])
    return out


def chips(mesh: trimesh.Trimesh, plane: dict, tol: float = 0.3, min_share: float = 0.002) -> list[dict]:
    """Места, где отражению скана не хватает материала скана: кандидаты в сколы.
    Возвращает куски отражённой поверхности: faces (грани отражения), area, depth, rough_deg
    (разброс нормалей скана под куском) и smooth_deg (то же у отражённой, целой стороны)."""
    n, d = plane["normal"], plane["d"]
    B = mesh.copy()
    B.vertices = _reflect(B.vertices, n, d)
    B.invert()
    cen = B.triangles_center
    near, dist, fa = trimesh.proximity.closest_point(mesh, cen)
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

    hull = _region(mesh, chip, B, grow)
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
    cur, log = mesh, []
    for _ in range(6 * max(1, len(planes))):
        # за шаг — одна достройка: та, что добавляет больше всего. Скол, пересекающий плоскость, ею
        # достраивается лишь частично (отражается сам в себя); другая плоскость даёт его целиком.
        best = None
        for p in planes:
            cs, B = chips(cur, p, tol)
            for c in merge_chips(cur, cs, B):
                r = restore(cur, p, c, B)
                if r is None:
                    continue
                gain = float(r.volume - cur.volume)
                # полнота: у скола объём сравним с площадью × глубину (образцы: 0,13–0,32); тонкая
                # плёнка от неточного совпадения отражения (над отверстием: 0,035) — не скол
                if gain < 0.08 * c["area"] * c["depth"]:
                    continue
                if best is None or gain > best[0]:
                    best = (gain, r, p, c)
                break
        if best is None:
            break
        gain, cur, p, c = best
        log.append({"normal": p["normal"], "d": p["d"], "area": c["area"], "depth": c["depth"],
                    "center": c["center"], "volume": gain})
    return cur, log, planes
