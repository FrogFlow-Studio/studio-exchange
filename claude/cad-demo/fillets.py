"""Скругления между плоскими гранями: поиск по скану и радиус (без построения грани).

Две плоскости с нормалями n1, n2 наружу пересекаются по прямой через точку p. Скругление радиуса R,
касательное к обеим, — цилиндр с осью вдоль n1 × n2 через c = p − R·m, m = (n1 + n2) / (1 + n1·n2):
расстояние от оси до каждой плоскости тогда ровно R. Неизвестное одно. Для каждой точки поверхности
у линии пересечения, не лежащей ни на одной из плоскостей, R находится из |x + R·m|² = R² (x — точка
в сечении, перпендикулярном оси, от p). Скругление принимается, если радиусы по точкам согласны.
У вогнутого угла то же с обратным знаком. Способ и границы применимости: FEWER_FACES_PLAN.md, шаг 4.

python fillets.py скан.stl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def find_fillets(mesh: trimesh.Trimesh, reach: float = 12.0, min_len: float = 3.0, samples: int = 300_000) -> list[dict]:
    """mesh — чистая поверхность (surface.surface_mesh). Возвращает список скруглений:
    radius_mm, kind (выпуклое/вогнутое), length_mm, spread (разброс радиуса, доля), points."""
    import walls

    pats = walls.planar_patches(mesh)
    if len(pats) < 2:
        return []
    label = -np.ones(len(mesh.faces), int)
    for k, p in enumerate(pats):
        label[p["faces"]] = k
    pts, fi = trimesh.sample.sample_surface(mesh, samples, seed=7)
    nrm, lab = mesh.face_normals[fi], label[fi]
    planes = []
    for k, p in enumerate(pats):  # плоскость участка по его точкам; нормаль наружу
        X = pts[lab == k]
        if len(X) < 50:
            planes.append(None)
            continue
        c0 = X.mean(0)
        n = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
        if n @ p["normal"] < 0:
            n = -n
        res = float(np.median(np.abs((X - c0) @ n)))
        planes.append((n, float(c0 @ n), res))
    cos20 = np.cos(np.radians(20))
    out = []
    for i in range(len(pats)):
        for j in range(i + 1, len(pats)):
            if planes[i] is None or planes[j] is None:
                continue
            (n1, d1, r1), (n2, d2, r2) = planes[i], planes[j]
            c = float(n1 @ n2)
            if abs(c) > cos20:
                continue
            ax = np.cross(n1, n2)
            ax /= np.linalg.norm(ax)
            p = np.linalg.solve(np.array([n1, n2, ax]), np.array([d1, d2, 0.0]))
            rel = pts - p
            t = rel @ ax
            xs = rel - t[:, None] * ax
            rad = np.linalg.norm(xs, axis=1)
            near = rad < reach
            # участок линии, у которого есть обе плоскости
            ti, tj = t[near & (lab == i)], t[near & (lab == j)]
            if len(ti) < 20 or len(tj) < 20:
                continue
            lo, hi = max(np.percentile(ti, 2), np.percentile(tj, 2)), min(np.percentile(ti, 98), np.percentile(tj, 98))
            if hi - lo < min_len:
                continue
            delta = max(0.06, 3 * max(r1, r2))
            e1, e2 = pts @ n1 - d1, pts @ n2 - d2
            cand = (near & (t > lo) & (t < hi) & (np.abs(e1) > delta) & (np.abs(e2) > delta)
                    & (nrm @ n1 > c - 0.1) & (nrm @ n2 > c - 0.1))
            if cand.sum() < 40:
                continue
            x = xs[cand]
            m = (n1 + n2) / (1 + c)
            side = np.median(x @ m)  # выпуклый угол: поверхность по ту же сторону, что и ось (x·m < 0)
            if side > 0:
                x = -x
            a = (1 - c) / (1 + c)
            b = x @ m
            disc = b * b - a * (x * x).sum(1)
            ok = disc >= 0
            if ok.sum() < 40 or ok.mean() < 0.6:
                continue
            R = (-b[ok] + np.sqrt(disc[ok])) / a
            Rm = float(np.median(R))
            spread = float(np.median(np.abs(R - Rm)) / max(Rm, 1e-9))
            fit = float(np.median(np.abs(np.linalg.norm(x[ok] + Rm * m, axis=1) - Rm)))
            if not (0.4 < Rm < reach / 2) or spread > 0.15 or fit > max(0.1, 0.1 * Rm):
                continue
            tt = t[cand][ok]
            out.append({"radius_mm": round(Rm, 1), "kind": "вогнутое" if side > 0 else "выпуклое",
                        "length_mm": round(float(np.percentile(tt, 98) - np.percentile(tt, 2)), 1),
                        "spread": round(spread, 3), "points": int(ok.sum()), "planes": (i, j)})
    return sorted(out, key=lambda f: -f["length_mm"])


def summary(fl: list[dict]) -> str:
    """Строка для отчёта: скругления, сгруппированные по радиусу."""
    if not fl:
        return ""
    groups: dict[float, list[dict]] = {}
    for f in fl:
        groups.setdefault(round(f["radius_mm"] * 2) / 2, []).append(f)
    parts = [f"R ≈ {str(r).replace('.', ',')} мм × {len(g)}" for r, g in sorted(groups.items())]
    return "скругления между плоскими гранями (по скану, грани пока сплайнами): " + ", ".join(parts)


if __name__ == "__main__":
    import surface

    scan = trimesh.load(sys.argv[1], force="mesh")
    fl = find_fillets(surface.surface_mesh(scan) if "--raw" not in sys.argv else scan)
    for f in fl:
        print(f)
    print(summary(fl) or "скруглений не найдено")
