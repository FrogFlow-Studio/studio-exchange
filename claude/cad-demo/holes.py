"""Отверстия: найти по скану и сверить с моделью (пункт «посадочные места» в «Готово к печати?»).

Отверстие ищем как круглую дыру в плоском участке скана. Диаметр меряем по сечению на
небольшой глубине под поверхностью (там нет фаски и скругления кромки) — и у скана, и у
модели в той же плоскости, поэтому сравнение честное: одинаковым способом.
"""
from __future__ import annotations

import numpy as np
import trimesh

import walls as W


_circle = W._circle


def _circle_robust(pts: np.ndarray):
    """Окружность без пазов и выступов: подгоняем, отбрасываем точки, далеко ушедшие от
    окружности (паз под ключ, скол кромки), и подгоняем снова по оставшимся."""
    c, r, rms = _circle(pts)
    for _ in range(3):
        res = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        keep = res < max(0.3, 2.5 * np.median(res))
        if keep.sum() < max(8, 0.6 * len(pts)) or keep.all():
            break
        pts = pts[keep]
        c, r, rms = _circle(pts)
    return c, r, rms


def find_holes(scan: trimesh.Trimesh, d_min: float = 1.5, d_max: float = 40.0, depth: float = 0.8) -> list[dict]:
    """Отверстия ищем в сечении тела чуть под каждой плоской гранью: шумный скан режет саму
    грань на куски, и по её контуру круглая дыра не видна, а в сечении материала — видна."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    out: list[dict] = []
    for p in W.planar_patches(scan):
        if p["area"] < 50:
            continue
        n = p["normal"]
        u, v = W._frame(n)
        o = p["point"] - n * depth
        sec = scan.section(plane_origin=o, plane_normal=n)
        if sec is None:
            continue
        T = np.eye(4)
        T[:3, :3] = np.vstack([u, v, n])
        T[:3, 3] = -T[:3, :3] @ o
        try:
            pl, _ = sec.to_2D(to_2D=T)
            body = unary_union([q.buffer(0) for q in pl.polygons_full])
        except Exception:  # noqa: BLE001
            continue
        face = W._outline(scan, p["faces"], p["point"], u, v).convex_hull
        for poly in ([body] if body.geom_type == "Polygon" else list(body.geoms)):
            for ring in poly.interiors:
                pts = np.asarray(ring.coords)[:-1]
                if len(pts) < 8:
                    continue
                c, r, rms = _circle(pts)
                if not (d_min <= 2 * r <= d_max) or rms > 0.12 * r or not face.contains(Point(c)):
                    continue
                center = p["point"] + c[0] * u + c[1] * v
                if any(abs(h["normal"] @ n) > 0.97 and
                       np.linalg.norm(np.cross(center - h["center"], h["normal"])) < max(1.0, 0.3 * r)
                       for h in out):
                    continue  # то же отверстие с другой стороны пластины
                out.append({"center": center, "normal": n, "u": u, "v": v, "d_outline": 2 * r})
    return out


def _measure(mesh: trimesh.Trimesh, h: dict, depth: float):
    """Диаметр и центр отверстия в сечении на глубине depth под поверхностью."""
    o = h["center"] - h["normal"] * depth
    sec = mesh.section(plane_origin=o, plane_normal=h["normal"])
    if sec is None:
        return None
    best = None
    r0 = h["d_outline"] / 2
    for e in sec.discrete:
        q = e - o
        pts = np.c_[q @ h["u"], q @ h["v"]]
        if len(pts) < 8:
            continue
        c, r, rms = _circle_robust(pts)
        if np.linalg.norm(c) < max(1.0, 0.5 * r0) and 0.5 * r0 < r < 1.5 * r0 and rms < 0.2 * r:
            if best is None or np.linalg.norm(c) < np.linalg.norm(best[0]):
                best = (c, r)
    if best is None:
        return None
    return {"d": 2 * best[1], "offset": float(np.linalg.norm(best[0]))}



def _coarse(scan: trimesh.Trimesh, faces: int = 300_000) -> trimesh.Trimesh:
    """Для поиска отверстий хватает 300 тыс. треугольников: на миллионе поиск плоских участков
    идёт минуты. Диаметр потом меряется по полному скану."""
    if len(scan.faces) <= faces:
        return scan
    import fast_simplification

    v, f = fast_simplification.simplify(scan.vertices, scan.faces, 1 - faces / len(scan.faces))
    return trimesh.Trimesh(v, f, process=True)

def check(scan: trimesh.Trimesh, model: trimesh.Trimesh, tol: float = 0.2, depth: float = 0.6) -> list[dict]:
    rows = []
    for h in find_holes(_coarse(scan)):
        s = _measure(scan, h, depth)
        if s is None:
            continue
        m = _measure(model, h, depth)
        row = {"center": np.round(h["center"], 1).tolist(), "d_scan": round(s["d"], 2),
               "d_model": round(m["d"], 2) if m else None,
               "offset": round(abs(m["offset"] - s["offset"]), 2) if m else None}
        row["ok"] = bool(m and abs(m["d"] - s["d"]) <= tol and row["offset"] <= tol)
        rows.append(row)
    return rows
