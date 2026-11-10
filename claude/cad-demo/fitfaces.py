"""Уточнение граней без внешних моделей: каждая грань нашей модели размечает свои точки скана,
и по ним заново подгоняется та же поверхность (окружность отверстия или вала, высота плоскости).

То же, что Point2STEP делает для простых поверхностей, но на нашей разметке, без видеокарты, без
сшивки граней и без GPL-зависимостей. На выходе — список поверхностей в формате merge.faces():
цилиндры с осью вдоль Z (x, y, r, площадь) и плоскости, перпендикулярные Z (z, площадь).

python fitfaces.py model.step скан.stl  — печатает найденное
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _circle(q: np.ndarray):
    """Окружность по точкам на плоскости (наименьшие квадраты): центр x, y и радиус."""
    A = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    return sol[0], sol[1], float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))


def _coverage(q: np.ndarray, cx: float, cy: float) -> float:
    """Сколько градусов окружности покрыто точками (по наибольшему разрыву)."""
    ang = np.sort(np.degrees(np.arctan2(q[:, 1] - cy, q[:, 0] - cx)))
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    return float(360 - gaps.max())


def fit(model_step: Path, pts: np.ndarray, band: float = 0.3, min_pts: int = 60):
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    import trimesh
    from scipy.spatial import cKDTree

    model = cq.importers.importStep(str(model_step)).val()
    faces = model.Faces()
    # разметка: точка скана принадлежит ближайшей грани модели
    mp, ml = [], []
    total = model.Area()
    for i, f in enumerate(faces):
        v, t = f.tessellate(0.02, 0.1)
        if not t:
            continue
        m = trimesh.Trimesh([(p.x, p.y, p.z) for p in v], t)
        s, _ = trimesh.sample.sample_surface(m, max(200, int(200000 * m.area / total)), seed=0)
        mp.append(s)
        ml.append(np.full(len(s), i))
    mp, ml = np.vstack(mp), np.concatenate(ml)
    dist, j = cKDTree(mp).query(pts)
    lab = ml[j]

    cyl, pl = [], []
    for i, f in enumerate(faces):
        sel = (lab == i) & (dist < 1.0)
        if sel.sum() < min_pts:
            continue
        g = BRepAdaptor_Surface(f.wrapped)
        p = pts[sel]
        if f.geomType() == "CYLINDER":
            c = g.Cylinder()
            ax = c.Axis()
            if abs(abs(ax.Direction().Z()) - 1) > 1e-3:
                continue
            x0, y0, r0 = ax.Location().X(), ax.Location().Y(), c.Radius()
            q = p[:, :2]
            x, y, r = x0, y0, r0
            k = np.zeros(len(q), bool)
            for _ in range(3):  # точки с соседних граней отсекаются полосой вокруг текущей окружности
                d = np.hypot(q[:, 0] - x, q[:, 1] - y)
                k = np.abs(d - r) < band
                if k.sum() < min_pts:
                    break
                if _coverage(q[k], x, y) >= 200:
                    x, y, r = _circle(q[k])
                else:  # дуга: центр ненадёжен, уточняем только радиус
                    r = float(np.median(np.hypot(q[k, 0] - x, q[k, 1] - y)))
            # стенка должна быть настоящей: точки ложатся на окружность плотно (а не обломок рядом)
            # и покрывают заметную её часть
            res = np.abs(np.hypot(q[k, 0] - x, q[k, 1] - y) - r) if k.sum() else np.array([1.0])
            cover = _coverage(q[k], x, y) if k.sum() >= min_pts else 0.0
            if cover < 90 or float(np.median(res)) > 0.06:
                continue  # обломок или кусочек дуги: размер по нему не уточнить
            if abs(r - r0) < 0.5 and np.hypot(x - x0, y - y0) < 0.5:
                cyl.append((float(x), float(y), float(r), f.Area()))
        elif f.geomType() == "PLANE":
            n = g.Plane().Axis().Direction()
            if abs(abs(n.Z()) - 1) > 1e-3:
                continue
            z0 = g.Plane().Location().Z()
            k = np.abs(p[:, 2] - z0) < band
            if k.sum() >= min_pts:
                pl.append((float(np.median(p[k, 2])), f.Area()))
    return cyl, pl


def main() -> None:
    import trimesh

    m = trimesh.load(sys.argv[2], force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, 200000, seed=2)
    cyl, pl = fit(Path(sys.argv[1]), pts)
    for c in cyl:
        print(f"цилиндр [{c[0]:.3f}, {c[1]:.3f}] Ø{2 * c[2]:.3f}")
    for p in pl:
        print(f"плоскость z={p[0]:.3f}")


if __name__ == "__main__":
    main()
