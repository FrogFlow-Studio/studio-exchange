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
5. грани сшиваются в оболочку, оболочка — в тело;
6. смешанная модель: лоскуты на плоских участках кладутся точно на плоскость участка (на
   стыке двух плоскостей — на линию их пересечения) и сливаются в одну плоскую грань на
   участок. Гладкими остаются только кривые места. Если слияние портит тело, берётся тело
   без слияния.

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
# (треугольников грубой сетки, точек на сторону): сначала крупные грани, при нехватке точности мельче.
# Густая сетка точек на крупной грани (13–21 точка) даёт вдвое меньше граней при той же точности, но
# OpenCascade режет такую грань на треугольники по каждой её точке: STL Hyundai вырос до 1,5 млн
# треугольников, и сравнению со сканом не хватило памяти (11.11, EXPERIMENTS.md, Э3). Пока так.
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



def _plane_tags(mesh, quads, raw, proj, w):
    """Лоскуты на плоских участках: номер участка (или -1) и уточнённые плоскости участков.
    Точки таких лоскутов кладутся точно на плоскость, а на стыке двух плоскостей — на линию их
    пересечения, чтобы ребро между ними было точным. Допуск — по шуму самых ровных лоскутов."""
    import walls

    pats = walls.planar_patches(mesh)
    if not pats:
        return -np.ones(len(quads), int), {}
    P_n = np.array([p["normal"] for p in pats])
    P_d = np.array([p["point"] @ p["normal"] for p in pats])
    rms = np.array([np.linalg.svd(P - P.mean(0), compute_uv=False)[-1] / np.sqrt(len(P)) for P in raw])
    tol = max(0.1, 3 * float(np.percentile(rms, 25)))
    cand = -np.ones(len(quads), int)
    for i, ((q, n), P) in enumerate(zip(quads, raw)):
        d = np.abs(P @ P_n.T - P_d).max(0)
        d[P_n @ n < 0.97] = 1e9
        j = int(d.argmin())
        if d[j] < 4 * tol:
            cand[i] = j
    tag = -np.ones(len(quads), int)
    fit = {}
    for k in np.unique(cand[cand >= 0]):
        idx = np.where(cand == k)[0]
        for _ in range(3):  # плоскость участка по точкам его лоскутов, дальние лоскуты отпадают
            X = np.vstack([raw[i] for i in idx])
            c0 = X.mean(0)
            nn = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
            keep = np.array([np.abs((raw[i] - c0) @ nn).max() for i in idx]) < tol
            if keep.all() or not keep.any():
                break
            idx = idx[keep]
        if len(idx) >= 2 and keep.all():
            tag[idx] = k
            fit[int(k)] = (c0, nn)

    def owners():
        o = {}
        for i in np.where(tag >= 0)[0]:
            for key in (tuple(np.round(p, 6)) for p in (w @ quads[i][0])):
                o.setdefault(key, {}).setdefault(int(tag[i]), []).append(i)
        return o

    def onto(p, ks, rounds):
        for _ in range(rounds):
            for k in ks:
                c0, nn = fit[k]
                p = p - ((p - c0) @ nn) * nn
        return p

    # стык двух плоскостей допустим только под заметным углом и рядом с линией пересечения;
    # иначе лоскуты меньшего участка остаются гладкими
    cos20 = np.cos(np.radians(20))
    for _ in range(3):
        bad = set()
        for key, ks in owners().items():
            if len(ks) < 2:
                continue
            p0 = proj.cache[key]
            ns = [fit[k][1] for k in ks]
            steep = all(abs(a @ b) < cos20 for i_, a in enumerate(ns) for b in ns[i_ + 1:])
            if not steep or np.linalg.norm(onto(p0.copy(), ks, 30) - p0) > 3 * tol:
                big = max(ks, key=lambda k: (cand == k).sum())
                for k, qs in ks.items():
                    if k != big:
                        bad.update(qs)
        if not bad:
            break
        tag[list(bad)] = -1
    for key, ks in owners().items():
        proj.cache[key] = onto(proj.cache[key].copy(), ks, 30 if len(ks) > 1 else 1)
    return tag, {k: v for k, v in fit.items() if (tag == k).any()}


def _merge_planes(sewn, faces, tag, fit):
    """Связные группы лоскутов одной плоскости → одна плоская грань (контур из свободных
    границ группы). Группа, где грань выходит невалидной, остаётся как есть."""
    import cadquery as cq
    import networkx as nx
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools_ReShape
    from OCP.Geom import Geom_Plane
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS, TopoDS_Compound
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(sewn, TopAbs_FACE, fmap)
    idx_of = {fmap.FindIndex(f): i for i, f in enumerate(faces)}
    e2f = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(sewn, TopAbs_EDGE, TopAbs_FACE, e2f)
    g = nx.Graph()
    for ei in range(1, e2f.Extent() + 1):
        fl = [idx_of.get(fmap.FindIndex(x)) for x in e2f.FindFromIndex(ei)]
        fl = [x for x in fl if x is not None]
        if len(fl) == 2 and tag[fl[0]] >= 0 and tag[fl[0]] == tag[fl[1]]:
            g.add_edge(fl[0], fl[1])
    rs = BRepTools_ReShape()
    merged = 0
    for grp in (sorted(c) for c in nx.connected_components(g) if len(c) >= 2):
        c0, nn = fit[int(tag[grp[0]])]
        comp = TopoDS_Compound()
        bb = BRep_Builder()
        bb.MakeCompound(comp)
        for i in grp:
            bb.Add(comp, faces[i])
        fb = ShapeAnalysis_FreeBounds(comp, 1e-3, False, False)
        wires = cq.Shape.cast(fb.GetClosedWires()).Wires()
        if not wires or cq.Shape.cast(fb.GetOpenWires()).Wires():
            continue
        wires.sort(key=lambda x: -x.Length())  # внешний контур самый длинный, остальные — отверстия
        pl = Geom_Plane(gp_Pln(gp_Pnt(*map(float, c0)), gp_Dir(*map(float, nn))))
        mk = BRepBuilderAPI_MakeFace(pl, wires[0].wrapped, True)
        for hole in wires[1:]:
            mk.Add(hole.wrapped)
        if not mk.IsDone():
            continue
        sff = ShapeFix_Face(mk.Face())
        sff.Perform()
        pf = TopoDS.Face_s(sff.Face())
        if not BRepCheck_Analyzer(pf).IsValid():
            continue
        ref = cq.Face(faces[grp[0]])
        c_ = ref.Center()
        if ref.normalAt(c_).dot(cq.Face(pf).normalAt(c_)) < 0:
            pf = TopoDS.Face_s(pf.Reversed())
        rs.Replace(faces[grp[0]], pf)
        for i in grp[1:]:
            rs.Remove(faces[i])
        merged += 1
    return (rs.Apply(sewn) if merged else sewn), merged


def _solid(shell_shape):
    import cadquery as cq
    from OCP.ShapeFix import ShapeFix_Solid

    shells = cq.Shape.cast(shell_shape).Shells()
    if len(shells) != 1:
        return None
    fx = ShapeFix_Solid(cq.Solid.makeSolid(shells[0]).wrapped)
    fx.Perform()
    sol = cq.Shape.cast(fx.Solid())
    if sol.Volume() < 0:
        sol = cq.Solid(sol.wrapped.Reversed())
    return sol if sol.isValid() else None

def _build(mesh: trimesh.Trimesh, triangles: int, proj: _Projector, planes: bool = True):
    import fast_simplification as fs
    from OCP.Approx import Approx_ParametrizationType
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt
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
    quads = list(_quads(coarse))
    tag, fit = -np.ones(len(quads), int), {}
    if planes:
        try:
            tag, fit = _plane_tags(mesh, quads, [proj(w @ q, n) for q, n in quads], proj, w)
        except Exception:  # noqa: BLE001  плоскости — улучшение, без них тело то же
            tag, fit = -np.ones(len(quads), int), {}
    faces = []
    for q, n in quads:
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
        fc = BRepBuilderAPI_MakeFace(srf, 1e-6).Face()
        sew.Add(fc)
        faces.append(fc)
        n_faces += 1
    sew.Perform()
    if sew.NbFreeEdges():
        return None, n_faces, None
    sewn = sew.SewedShape()
    sol = None
    if fit:
        # лоскуты одной плоскости сливаются в одну точную плоскую грань; если тело после этого
        # невалидно, берётся тело без слияния (форма та же)
        try:
            from OCP.TopoDS import TopoDS

            merged_shape, merged = _merge_planes(sewn, [TopoDS.Face_s(sew.Modified(f)) for f in faces], tag, fit)
            if merged:
                sol = _solid(merged_shape)
        except Exception:  # noqa: BLE001
            sol = None
    if sol is None:
        sol = _solid(sewn)
    if sol is not None:
        n_faces = len(sol.Faces())
    return sol, n_faces, np.array(mids)


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
