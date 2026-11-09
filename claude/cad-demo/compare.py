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


def compare(scan: trimesh.Trimesh, model: trimesh.Trimesh, tol: float = 0.3, n: int = 12000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sp_pts, _ = trimesh.sample.sample_surface_even(scan, n, seed=int(rng.integers(1 << 30)))
    # signed_distance: >0 внутри модели, <0 снаружи.
    sd = trimesh.proximity.signed_distance(model, sp_pts)
    ad = np.abs(sd)
    outside = sd < -tol
    inside = sd > tol
    mp_pts, _ = trimesh.sample.sample_surface_even(model, n, seed=int(rng.integers(1 << 30)))
    _, d_model, _ = trimesh.proximity.closest_point(scan, mp_pts)
    added = d_model > tol
    within = float((ad <= tol).mean())
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
        "added_surface_share": round(float(added.mean()), 4),
        "clusters_outside": clusters(sp_pts[outside], sd[outside]),
        "clusters_inside": clusters(sp_pts[inside], sd[inside]),
        "clusters_added": clusters(mp_pts[added], d_model[added]),
        # для картинок
        "_scan_pts": sp_pts, "_scan_dev": sd, "_model_pts": mp_pts, "_model_dev": d_model,
    }


def public(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}
