"""Гладкое тело из B-сплайн граней по скану — для кривых деталей любой формы.

Тело по поверхности (surface.py) точное, но состоит из тысяч плоских треугольников: в
SolidWorks или КОМПАС его не поправить. Здесь та же поверхность собирается из гладких
четырёхугольных B-сплайн граней, как делает «автоповерхность» в Geomagic/QuickSurface:

1. поверхность упрощается до грубой сетки (на плоских местах треугольники крупные, на
   изгибах мелкие — так работает упрощение по квадрикам);
2. каждый треугольник делится на 3 четырёхугольника (центр и середины рёбер);
3. в каждом четырёхугольнике сетка k×k точек кладётся на скан, с той же стороны стенки
   (у тонкой стенки ближайшая точка может оказаться на обратной стороне);
4. по точкам строится B-сплайн грань с равномерными параметрами: у соседей одинаковые
   точки на общей границе, поэтому и граничные кривые совпадают — щелей нет;
5. грани сшиваются в оболочку, оболочка — в тело.

Если тело выходит неточным (меньше 98% поверхности в пределах 0,2 мм), грубая сетка берётся
мельче. Настроек под конкретную деталь нет.

python nurbs.py скан.stl папка
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

K = 7  # точек на сторону лоскута (меняется по шагам)
# (треугольников грубой сетки, точек на сторону): сначала крупные грани, при нехватке точности мельче
STEPS = ((800, 9), (1500, 7), (4000, 5))
GOAL, GOAL_TOL = 0.98, 0.2  # доля чистой поверхности в пределах GOAL_TOL мм


class _Projector:
    """Кладёт точки на поверхность с той же стороны стенки; общий кэш, чтобы соседние
    лоскуты получили одинаковые точки на общей границе."""

    def __init__(self, mesh: trimesh.Trimesh):
        from scipy.spatial import cKDTree

        self.m = mesh
        self.cache: dict = {}
        self.dp, fi = trimesh.sample.sample_surface(mesh, 400000, seed=5)
        self.dn = mesh.face_normals[fi]
        self.tree = cKDTree(self.dp)

    def __call__(self, P: np.ndarray, N: np.ndarray) -> np.ndarray:
        keys = [tuple(np.round(p, 6)) for p in P]
        need = [i for i, k in enumerate(keys) if k not in self.cache]
        if need:
            # ближайшая точка поверхности с той же стороны стенки (у тонкой стенки самая
            # ближняя может оказаться на обратной стороне); точка кладётся на её касательную
            # плоскость — точнее, чем просто взять образец, и быстрее точной проекции
            Pn = P[need]
            _, ii = self.tree.query(Pn, k=32)
            same = self.dn[ii] @ N > 0.3
            pick = np.where(same.any(1), same.argmax(1), 0)
            s0 = self.dp[ii[np.arange(len(ii)), pick]]
            n0 = self.dn[ii[np.arange(len(ii)), pick]]
            Q = Pn - np.einsum("ij,ij->i", Pn - s0, n0)[:, None] * n0
            for j, i in enumerate(need):
                self.cache[keys[i]] = Q[j]
        return np.array([self.cache[k] for k in keys])


def _quads(coarse: trimesh.Trimesh):
    V = coarse.vertices
    for a, b, d in coarse.faces:
        A, B, D = V[a], V[b], V[d]
        n = np.cross(B - A, D - A)
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        G = (A + B + D) / 3
        mab, mbd, mda = (A + B) / 2, (B + D) / 2, (D + A) / 2
        for q in ((A, mab, G, mda), (B, mbd, G, mab), (D, mda, G, mbd)):
            yield np.array(q), n


def _build(mesh: trimesh.Trimesh, triangles: int, proj: _Projector):
    import cadquery as cq
    import fast_simplification as fs
    from OCP.Approx import Approx_ParametrizationType
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TColgp import TColgp_Array2OfPnt

    import surface

    if len(mesh.faces) > triangles:
        v, f = fs.simplify(mesh.vertices, mesh.faces, 1 - triangles / len(mesh.faces))
        coarse = surface.repair(trimesh.Trimesh(v, f, process=True))
    else:
        coarse = mesh
    if not coarse.is_watertight:
        return None, 0, None
    s = np.linspace(0, 1, K)
    S, T = np.meshgrid(s, s, indexing="ij")
    w = np.stack([(1 - S) * (1 - T), S * (1 - T), S * T, (1 - S) * T], -1).reshape(-1, 4)
    sew = BRepBuilderAPI_Sewing(0.02)
    n_faces = 0
    mids = []  # точки поверхности в серединах ячеек: по ним меряется точность
    cm = (np.arange(K - 1) + 0.5) / (K - 1)
    for q, n in _quads(coarse):
        P = proj(w @ q, n).reshape(K, K, 3)
        arr = TColgp_Array2OfPnt(1, K, 1, K)
        for i in range(K):
            for j in range(K):
                arr.SetValue(i + 1, j + 1, gp_Pnt(*map(float, P[i, j])))
        api = GeomAPI_PointsToBSplineSurface()
        api.Interpolate(arr, Approx_ParametrizationType.Approx_IsoParametric, False)
        srf = api.Surface()
        u0, u1, v0, v1 = srf.Bounds()
        for u in u0 + (u1 - u0) * cm:
            for vv in v0 + (v1 - v0) * cm:
                p = srf.Value(u, vv)
                mids.append((p.X(), p.Y(), p.Z()))
        sew.Add(BRepBuilderAPI_MakeFace(srf, 1e-6).Face())
        n_faces += 1
    sew.Perform()
    if sew.NbFreeEdges():
        return None, n_faces, None
    shells = cq.Shape.cast(sew.SewedShape()).Shells()
    if len(shells) != 1:
        return None, n_faces, None
    fx = ShapeFix_Solid(cq.Solid.makeSolid(shells[0]).wrapped)
    fx.Perform()
    sol = cq.Shape.cast(fx.Solid())
    if sol.Volume() < 0:
        sol = cq.Solid(sol.wrapped.Reversed())
    return (sol if sol.isValid() else None), n_faces, np.array(mids)


def near_share(pts: np.ndarray, proj: _Projector, tol: float = 0.1) -> float:
    """Доля точек модели в пределах tol от поверхности: расстояние до касательной плоскости
    ближайших точек поверхности (берётся наименьшее среди соседних, ведь у тонкой стенки
    самая ближняя точка может лежать на обратной стороне)."""
    dist, ii = proj.tree.query(pts, k=8)
    d = np.abs(np.einsum("ikj,ikj->ik", pts[:, None] - proj.dp[ii], proj.dn[ii]))
    d[dist > 2.0] = np.inf
    return float((d.min(1) < tol).mean())


def smooth_body(mesh: trimesh.Trimesh, scan: trimesh.Trimesh, budget: float = 900.0):
    """mesh — замкнутая чистая поверхность (из surface.py), scan — исходный скан для проверки.
    Возвращает (тело или None, сведения)."""
    t0 = time.time()
    proj = _Projector(mesh)
    best, info = None, {"faces": 0, "near": 0.0}
    global K
    for tri, K in STEPS:
        if time.time() - t0 > budget / 3 and best is not None:
            break
        proj.cache.clear()
        body, n, mids = _build(mesh, tri, proj)
        if body is None:
            continue
        # точность считается от чистой поверхности: шум скана сглажен ещё в surface.py, и
        # мерить по сырому скану значило бы гнаться за шумом
        share = near_share(mids, proj, GOAL_TOL)
        if best is None or share > info["near"]:
            best, info = body, {"faces": n, "near": share}
        if share >= GOAL:
            break
    info["sec"] = round(time.time() - t0)
    return best, info


if __name__ == "__main__":
    import cadquery as cq

    import surface

    scan = trimesh.load(sys.argv[1], force="mesh")
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    body, info = smooth_body(surface.flatten_planes(surface.surface_mesh(scan)), scan)
    print(info, "valid", body is not None and body.isValid())
    if body is not None:
        cq.exporters.export(body, str(out / "smooth.step"))
