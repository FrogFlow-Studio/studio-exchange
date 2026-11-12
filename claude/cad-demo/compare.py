"""Шаг 3: сравнение модели со сканом. Без ИИ, только геометрия.

Для точек скана считаем расстояние со знаком до поверхности модели:
  «снаружи модели» — в скане материал есть, в модели нет: модель МАЛА, надо чинить;
  «внутри модели»  — модель больше скана: это либо скол (материал восстановлен
                     сознательно), либо модель ВЕЛИКА.
Для точек модели считаем расстояние до скана: где модель далеко от скана,
это достроенная часть (её нет в скане).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from cad_checks import measured


def clusters(points: np.ndarray, values: np.ndarray, link: float = 2.0, min_pts: int = 15) -> list[dict]:
    if len(points) == 0:
        return []
    tree = cKDTree(points)
    pairs = tree.query_pairs(link, output_type="ndarray")
    import scipy.sparse as sp
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    n, lab = connected_components(g, directed=False)
    out = []
    for k in range(n):
        sel = lab == k
        if sel.sum() < min_pts:
            continue
        p, v = points[sel], values[sel]
        out.append({
            "points": int(sel.sum()),
            "center_mm": [round(float(x), 1) for x in p.mean(axis=0)],
            "bbox_mm": [[round(float(x), 1) for x in p.min(axis=0)], [round(float(x), 1) for x in p.max(axis=0)]],
            "mean_dev_mm": round(float(np.abs(v).mean()), 2),
            "max_dev_mm": round(float(np.abs(v).max()), 2),
        })
    return sorted(out, key=lambda c: -c["points"])


class ScanComparison:
    """Reuse samples and spatial indexes for one immutable scan."""

    def __init__(self, scan):
        self.scan = scan
        self.samples = {}
        self.vertices = None

    def sample(self, n, seed):
        key = n, seed
        if key not in self.samples:
            self.samples[key] = trimesh.sample.sample_surface_even(self.scan, n, seed=seed)[0]
        return self.samples[key]

    def distances(self, points):
        # Same candidate AABBs and tie-breaking as trimesh 5.2 closest_point.
        # Only the distances are needed in the model-to-scan direction.
        from trimesh.constants import tol as mesh_tol
        from trimesh.triangles import closest_point
        from trimesh.util import diagonal_dot

        scan = self.scan
        if self.vertices is None:
            self.vertices = cKDTree(scan.vertices[scan.referenced_vertices])
        radii = self.vertices.query(points)[0][:, None] + mesh_tol.merge
        bounds = np.column_stack((points - radii, points + radii))
        tree = scan.triangles_tree
        try:
            triangles, counts = tree.intersection_v(bounds[:, :3], bounds[:, 3:])
        except Exception:
            candidates = [list(tree.intersection(box)) for box in bounds]
            triangles = np.concatenate(candidates).astype(int)
            counts = np.array(list(map(len, candidates)))
        counts = np.asarray(counts, dtype=np.intp)
        if not len(points) or np.any(counts == 0):
            return trimesh.proximity.closest_point(scan, points)[1]
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        owners = np.repeat(np.arange(len(points)), counts)
        query = points[owners]
        vectors = query - closest_point(scan.triangles.view(np.ndarray)[triangles], query)
        squared = diagonal_dot(vectors, vectors)
        nearest = np.minimum.reduceat(squared, starts)
        # Legacy considers the normal of the two closest triangles on near ties.
        # Sorting only these groups keeps both the old tie rule and its order.
        tied = np.add.reduceat((squared - nearest[owners] < mesh_tol.merge).astype(int), starts)
        ambiguous = np.flatnonzero((tied >= 2) & (nearest > mesh_tol.merge))
        if len(ambiguous):
            best = np.array([starts[i] + squared[starts[i]:starts[i] + counts[i]].argsort()[:2]
                             for i in ambiguous])
            directions = vectors[best] / squared[best, None] ** 0.5
            normals = scan.face_normals[triangles[best]]
            choice = (normals * directions).sum(axis=2).argmax(axis=1)
            nearest[ambiguous] = squared[best[np.arange(len(best)), choice]]
        return nearest ** 0.5


def signed_distance(model: trimesh.Trimesh, pts: np.ndarray, tol: float) -> np.ndarray:
    """>0 внутри модели, <0 снаружи — как trimesh.proximity.signed_distance.

    На сетке модели в сотни тысяч треугольников (гладкое тело) проверка «внутри» лучами у trimesh
    набирает кандидатов на каждый луч: 1,4 МБ на точку, на 12 000 точках 17 ГБ — соковыжималку
    студии 12.11 убивало по памяти. Поэтому на крупной сетке лучами проверяются только точки дальше
    допуска (знак решает, «не хватает» или «лишнее»), кусками по 500; у ближних точек знак берётся
    по нормали ближайшего треугольника — он влияет только на оттенок цвета внутри допуска."""
    if len(model.faces) <= 300_000:
        return trimesh.proximity.signed_distance(model, pts)
    closest, dist, tri = trimesh.proximity.closest_point(model, pts)
    sign = np.where(((pts - closest) * model.face_normals[tri]).sum(axis=1) > 0, -1.0, 1.0)
    far = np.flatnonzero(dist > tol)
    for i in range(0, len(far), 500):
        part = far[i:i + 500]
        sign[part] = np.where(model.contains(pts[part]), 1.0, -1.0)
    return sign * dist


def compare(scan: trimesh.Trimesh, model: trimesh.Trimesh, tol: float = 0.3, n: int = 12000, seed: int = 0,
            *, prepared=None, timings=None, zones=None) -> dict:
    """zones — места, достроенные по симметрии (шары center, radius): точки скана на изломе там лежат
    внутри модели и в допуск не попадают, но это не ошибка модели; поверхность модели там не «лишняя»."""
    rng = np.random.default_rng(seed)
    scan_seed, model_seed = (int(rng.integers(1 << 30)) for _ in range(2))
    with measured(timings, "compare_sample"):
        sp_pts = (prepared.sample(n, scan_seed) if prepared is not None
                  else trimesh.sample.sample_surface_even(scan, n, seed=scan_seed)[0])
        mp_pts, _ = trimesh.sample.sample_surface_even(model, n, seed=model_seed)
    # signed_distance: >0 внутри модели, <0 снаружи.
    with measured(timings, "compare_signed"):
        sd = signed_distance(model, sp_pts, tol)
    with measured(timings, "compare_reverse"):
        d_model = (prepared.distances(mp_pts) if prepared is not None
                   else trimesh.proximity.closest_point(scan, mp_pts)[1])
    with measured(timings, "compare_metrics"):
        ad = np.abs(sd)
        outside = sd < -tol
        inside = sd > tol
        added = d_model > tol
        ok = ad <= tol
        restored = np.zeros(len(sp_pts), bool)
        for z in zones or ():
            zc, zr = np.asarray(z["center"], float), float(z["radius"])
            restored |= inside & (np.linalg.norm(sp_pts - zc, axis=1) <= zr)
            added &= np.linalg.norm(mp_pts - zc, axis=1) > zr
        inside = inside & ~restored
        within = float((ok | restored).mean())
        if restored.any():   # отклонения — по измеренному: излом под достройкой не «ошибка 2 мм»
            ad = ad[~restored]
        # Шаг связи областей остаётся привязан к площади и прежнему размеру выборки.
        link = max(2.0, 2.5 * float(np.sqrt(scan.area / n)))
        return {
            "tolerance_mm": tol,
            "scan_points": int(len(sp_pts)),
            "within_tol_share": round(within, 4),
            "mean_abs_dev_mm": round(float(ad.mean()), 3),
            "median_abs_dev_mm": round(float(np.median(ad)), 3),
            "p95_abs_dev_mm": round(float(np.percentile(ad, 95)), 3),
            "max_abs_dev_mm": round(float(ad.max()), 3),
            "outside_share": round(float(outside.mean()), 4),
            "inside_share": round(float(inside.mean()), 4),
            "restored_share": round(float(restored.mean()), 4),
            "added_surface_share": round(float(added.mean()), 4),
            "clusters_outside": clusters(sp_pts[outside], sd[outside], link=link, min_pts=8),
            "clusters_inside": clusters(sp_pts[inside], sd[inside], link=link, min_pts=8),
            "clusters_added": clusters(mp_pts[added], d_model[added], link=link, min_pts=8),
            # для картинок
            "_scan_pts": sp_pts, "_scan_dev": sd, "_model_pts": mp_pts, "_model_dev": d_model,
        }


def public(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}
