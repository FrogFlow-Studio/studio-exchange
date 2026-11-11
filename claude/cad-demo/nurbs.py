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

import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from cad_checks import measured

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

K = 7  # точек на сторону лоскута (меняется по шагам)
# Шаги: (треугольников грубой сетки, точек на сторону, полюсов на сторону; 0 — интерполяция, склейка парами).
# Сначала самые крупные грани, при нехватке точности мельче:
# 1) треугольники склеены парами в четырёхугольники, сглаживающая подгонка — граней в 5–6 раз меньше,
#    но мелкие детали и острые кромки такая грань сглаживает, поэтому проходит только на плавных деталях;
# 2) треугольник делится на три лоскута, сглаживающая подгонка 7×7 полюсов по 13×13 точкам;
# 3) интерполяция через все точки — запасной путь (EXPERIMENTS.md, Э3, Э5, Э6).
STEPS = ((1500, 17, 10, True), (800, 13, 7, False), (1500, 13, 7, False), (1500, 7, 0, False), (4000, 5, 0, False))
SPLIT = True  # крупная грань (пара), до которой не дотянулась часть поверхности, заменяется шестью мелкими
FACE_TOL = 0.3  # мм: точка поверхности дальше этого от модели считается потерянной (меняется по допуску сравнения)
LOSS_KEEP = 0.004  # доля точек поверхности, которую разрешено терять в оставшихся крупных гранях
PAIR = False  # склейка треугольников парами (меняется по шагам)
COVER = 0.99  # для сглаживающей подгонки: доля чистой поверхности, покрытая моделью в допуске сравнения
NP = None  # полюсов на сторону грани: None — интерполяция через все точки, иначе сглаживающая подгонка
GOAL, GOAL_TOL = 0.98, 0.2  # доля чистой поверхности в пределах GOAL_TOL мм


class _Projector:
    """Кладёт точки на поверхность с той же стороны стенки; общий кэш, чтобы соседние
    лоскуты получили одинаковые точки на общей границе."""

    def __init__(self, mesh: trimesh.Trimesh):
        from scipy.spatial import cKDTree

        self.fast = os.environ.get("CAD_NURBS_FAST", "1") != "0"
        self.m = mesh
        self.cache: dict = {}
        self.dp, fi = trimesh.sample.sample_surface(mesh, 400000, seed=5)
        self.dn = mesh.face_normals[fi]
        self.tree = cKDTree(self.dp)

    def __call__(self, P: np.ndarray, N: np.ndarray) -> np.ndarray:
        keys = list(map(tuple, np.round(P, 6))) if self.fast else [tuple(np.round(p, 6)) for p in P]
        need = [i for i, k in enumerate(keys) if k not in self.cache]
        if need:
            # ближайшая точка поверхности с той же стороны стенки (у тонкой стенки самая
            # ближняя может оказаться на обратной стороне); точка кладётся на её касательную
            # плоскость — точнее, чем просто взять образец, и быстрее точной проекции
            Pn = P[need]
            _, ii = self.tree.query(Pn, k=32)
            Nn = np.broadcast_to(N, P.shape)[need]
            same = np.einsum("ikj,ij->ik", self.dn[ii], Nn) > 0.3
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



def _paired_quads(coarse: trimesh.Trimesh):
    """Треугольники грубой сетки склеиваются парами в четырёхугольники: у замкнутой поверхности
    граф соседства треугольников кубический и без мостов, поэтому разбиение на пары без остатка
    существует всегда (теорема Петерсена). Пары выбираются так, чтобы четырёхугольники были как
    можно ровнее и площе. Граней выходит в 6 раз меньше, чем при делении треугольника на три."""
    import networkx as nx

    V, F, fn = coarse.vertices, coarse.faces, coarse.face_normals
    g = nx.Graph()
    info = {}
    for (a, b), (u, v) in zip(coarse.face_adjacency, coarse.face_adjacency_edges):
        fa = list(F[a])
        oa = next(x for x in fa if x != u and x != v)
        i = fa.index(oa)
        x, y = fa[(i + 1) % 3], fa[(i + 2) % 3]
        ob = next(z for z in F[b] if z != u and z != v)
        q = V[[oa, x, ob, y]]
        n = fn[a] + fn[b]
        ln = np.linalg.norm(n)
        # сложенные треугольники (нормали навстречу) тоже остаются в графе, с нулевым весом: без
        # этих рёбер разбиение на пары без остатка может не существовать (линза: 3 ребра из 2247)
        n = n / ln if ln > 1e-9 else fn[a]
        e = np.roll(q, -1, 0) - q
        turn = np.array([np.cross(e[k - 1], e[k]) @ n for k in range(4)])
        le = np.linalg.norm(e, axis=1)
        sin = turn / np.maximum(np.roll(le, 1) * le, 1e-12)
        shape = float(sin.min())  # 1 у прямоугольника, ≤ 0 у невыпуклого
        wgt = max(shape, 0.0) * max(float(fn[a] @ fn[b]), 0.0) + (1e-3 if shape > 0 else 0.0)
        # вес целым числом: с дробными весами поиск пар в networkx из-за округлений может не завершиться
        # (тестовый фланец считался дольше 15 минут)
        g.add_edge(int(a), int(b), weight=int(round(wgt * 1_000_000)))
        info[(int(a), int(b))] = (q, n, shape)
    g.add_nodes_from(range(len(F)))
    m = nx.max_weight_matching(g, maxcardinality=True)
    if 2 * len(m) != len(F):
        return None
    out = []
    for a, b in m:
        q, n, shape = info.get((a, b)) or info[(b, a)]
        if (b, a) in info and (a, b) not in info:
            a, b = b, a
        out.append((q, n, fn[a], fn[b], shape))
    return out


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


_SCAN = None  # исходный скан текущей детали (ставит smooth_body): по нему разметка уточняет области
PLN0, CYL0 = 50000, 100000  # номера плоских и цилиндрических областей разметки в метках лоскутов


def _onto(p, entry):
    """Точка на точной поверхности области: плоскость (c0, n) или цилиндр ("cyl", c, ось, r)."""
    if len(entry) == 4:
        _, c, ax, r = entry
        d = p - c
        h = d @ ax
        rad = d - h * ax
        ln = np.linalg.norm(rad)
        return p if ln < 1e-9 else c + h * ax + rad * (r / ln)
    c0, nn = entry
    return p - ((p - c0) @ nn) * nn


def _cyl_tags(mesh, quads, raw, proj, w, tag, fit):
    """Лоскуты на цилиндрических областях (regions.segment): точки таких лоскутов кладутся точно
    на цилиндр, а общие с плоскостью или другим цилиндром — на линию их пересечения. Лоскут
    берётся, только если почти все его точки принадлежат одной области и лежат на её цилиндре."""
    from scipy.spatial import cKDTree

    import regions

    seg = regions.segment(mesh, scan=_SCAN)  # параметры областей уточняются по точкам исходного скана
    prim = {j: r for j, r in enumerate(seg["regions"]) if r["kind"] in ("plane", "cylinder")}
    if not prim:
        return tag, fit
    rms = np.array([np.linalg.svd(P - P.mean(0), compute_uv=False)[-1] / np.sqrt(len(P)) for P in raw])
    tol = max(0.1, 3 * float(np.percentile(rms, 25)))
    tree = cKDTree(mesh.triangles_center)
    label = seg["label"]
    tag = tag.copy()
    fit = dict(fit)
    for i, P in enumerate(raw):
        if tag[i] >= 0:
            continue
        lab = label[tree.query(P)[1]]
        j = int(np.bincount(lab[lab >= 0]).argmax()) if (lab >= 0).any() else -1
        if j not in prim or (lab == j).mean() < 0.8:
            continue
        r = prim[j]
        if r["kind"] == "plane":
            key, entry = PLN0 + j, (r["origin"], r["normal"])
            for k, e in fit.items():  # та же плоскость уже найдена прежним способом: берётся её запись
                if len(e) == 2 and abs(float(e[1] @ r["normal"])) > 0.999 and abs(float((r["origin"] - e[0]) @ e[1])) < tol:
                    key, entry = k, e
                    break
            dev = np.abs((P - entry[0]) @ entry[1])
        else:
            key, entry = CYL0 + j, ("cyl", r["origin"], r["axis"], r["r"])
            d = P - r["origin"]
            d = d - np.outer(d @ r["axis"], r["axis"])
            dev = np.abs(np.linalg.norm(d, axis=1) - r["r"])
        if dev.max() < 4 * tol and float(np.sqrt((dev**2).mean())) < tol:
            tag[i] = key
            fit[key] = entry

    def owners():
        o = {}
        for i in np.where(tag >= 0)[0]:
            for key in (tuple(np.round(p, 6)) for p in (w @ quads[i][0])):
                o.setdefault(key, {}).setdefault(int(tag[i]), []).append(i)
        return {k: v for k, v in o.items() if any(t >= PLN0 for t in v)}

    def settle(p, ks):
        for _ in range(40 if len(ks) > 1 else 1):
            for k in ks:
                p = _onto(p, fit[k])
        return p

    for _ in range(3):  # точка, которую нельзя посадить сразу на все свои области, снимает цилиндр с лоскутов
        bad = set()
        for key, ks in owners().items():
            p0 = proj.cache[key]
            p = settle(p0.copy(), ks)
            off = max(np.linalg.norm(_onto(p, fit[k]) - p) for k in ks)
            if np.linalg.norm(p - p0) > 3 * tol or off > 1e-3:
                for k, qs in ks.items():
                    if k >= PLN0:
                        bad.update(qs)
        if not bad:
            break
        tag[list(bad)] = -1
    for key, ks in owners().items():
        proj.cache[key] = settle(proj.cache[key].copy(), ks)
    return tag, {k: v for k, v in fit.items() if (tag == k).any()}


def _cyl_face(entry, wires, group):
    """Одна точная цилиндрическая грань вместо группы лоскутов: поверхность — цилиндр области,
    границы — прежние рёбра группы (общие с соседями). Возвращает грань или None, если она не
    получилась валидной или её площадь не совпала с площадью группы."""
    import cadquery as cq
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopoDS import TopoDS, TopoDS_Face

    _, c, ax, r = entry
    pts = np.array([v.toTuple() for f in group for v in cq.Face(f).Vertices()])
    rad = pts - c
    rad = rad - np.outer(rad @ ax, ax)
    e1 = rad[0] / np.linalg.norm(rad[0])
    e2 = np.cross(ax, e1)
    ang = np.sort(np.arctan2(rad @ e2, rad @ e1))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    k = int(gaps.argmax())
    seam = ang[k] + gaps[k] / 2  # шов цилиндра — в середине самого большого промежутка между точками
    xdir = np.cos(seam) * e1 + np.sin(seam) * e2
    surf = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(*map(float, c)), gp_Dir(*map(float, ax)), gp_Dir(*map(float, xdir))), float(r))
    face = TopoDS_Face()
    bb = BRep_Builder()
    bb.MakeFace(face, surf, 1e-6)
    for wr in wires:
        bb.Add(face, wr.wrapped)
    sff = ShapeFix_Face(face)
    sff.SetPrecision(0.01)
    sff.SetMaxTolerance(0.05)
    sff.Perform()
    try:
        pf = TopoDS.Face_s(sff.Face())
    except Exception:  # noqa: BLE001
        return None
    if not BRepCheck_Analyzer(pf).IsValid():
        return None
    area = sum(cq.Face(f).Area() for f in group)
    if abs(cq.Face(pf).Area() - area) > 0.03 * area:
        return None
    return pf


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
        entry = fit[int(tag[grp[0]])]
        if len(entry) != 4:
            c0, nn = entry
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
        if len(entry) == 4:
            pf = _cyl_face(entry, wires, [faces[i] for i in grp])
            if pf is None:
                continue
        else:
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


def _solid(shell_shape, *, profile=None, parallel=False):
    import cadquery as cq
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Solid

    with measured(profile, "solid_fix"):
        shells = cq.Shape.cast(shell_shape).Shells()
        if len(shells) != 1:
            return None
        fx = ShapeFix_Solid(cq.Solid.makeSolid(shells[0]).wrapped)
        fx.Perform()
        sol = cq.Shape.cast(fx.Solid())
    with measured(profile, "solid_volume"):
        if sol.Volume() < 0:
            sol = cq.Solid(sol.wrapped.Reversed())
    with measured(profile, "solid_validity"):
        valid = BRepCheck_Analyzer(sol.wrapped, True, parallel).IsValid()
    return sol if valid else None


def _basis(K: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Кубический B-сплайн с n полюсами и равномерными узлами: матрица значений в K равномерных точках."""
    from scipy.interpolate import BSpline

    inner = np.linspace(0, 1, n - 2)
    t = np.r_[[0.0] * 3, inner, [1.0] * 3]
    B = BSpline.design_matrix(np.linspace(0, 1, K), t, 3).toarray()
    return B, inner


def _even(P: np.ndarray, ratio: float = 4.0) -> np.ndarray:
    """Ряды сетки, где точки сбились в кучку, раскладываются заново равномерно по длине ломаной.

    На ребре с изломом проекция стягивает точки к одной стороне (одна длинная ступень и десяток
    коротких). При равномерных параметрах кривая по таким точкам петляет, и грань не сшивается с
    соседней. Ряд перекладывается только по собственным точкам и одинаково в обе стороны, поэтому
    общая граница у соседних граней остаётся общей."""
    P = P.copy()
    for r in range(P.shape[0]):
        seg = np.linalg.norm(np.diff(P[r], axis=0), axis=1)
        if seg.max() < 1e-9 or seg.max() <= ratio * max(np.median(seg), 1e-9):
            continue
        L = np.r_[0, np.cumsum(seg)]
        t = np.linspace(0, L[-1], len(L))
        P[r] = np.column_stack([np.interp(t, L, P[r][:, k]) for k in range(3)])
    return P


def _half_basis(K: int, n: int, upper: int) -> tuple[np.ndarray, np.ndarray]:
    """Тот же сплайн, что в `_basis`, но только на половине отрезка (нижней или верхней) и с её
    собственным параметром 0..1. Узлы половины — узлы целого, попавшие в неё, поэтому половина
    граничной кривой целой грани представляется в этом базисе точно, без подгонки."""
    from scipy.interpolate import BSpline

    full = np.linspace(0, 1, n - 2)
    lo = 0.5 * upper
    kn = full[(full > lo + 1e-9) & (full < lo + 0.5 - 1e-9)]
    inner = (np.r_[lo, kn, lo + 0.5] - lo) * 2
    t = np.r_[[0.0] * 3, inner, [1.0] * 3]
    return BSpline.design_matrix(np.linspace(0, 1, K), t, 3).toarray(), inner


def _lsq_fit(P: np.ndarray, Bu: np.ndarray, Bv: np.ndarray):
    """Полюса грани по точкам P (K×K) и остатки в этих точках. Граничные кривые — только по точкам
    своей границы (концы точно в углах), внутренние полюса — по всем точкам при закреплённой границе."""
    nu, nv = Bu.shape[1], Bv.shape[1]
    C = np.zeros((nu, nv, 3))

    def edge(pts, B):
        c = np.zeros((B.shape[1], 3))
        c[0], c[-1] = pts[0], pts[-1]
        rhs = pts - B[:, :1] @ c[:1] - B[:, -1:] @ c[-1:]
        c[1:-1] = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)[0]
        return c

    C[0, :], C[-1, :] = edge(P[0, :], Bv), edge(P[-1, :], Bv)
    C[:, 0], C[:, -1] = edge(P[:, 0], Bu), edge(P[:, -1], Bu)
    A = np.kron(Bu, Bv)  # (K*K, nu*nv): точка (i, j) ↔ полюс (a, b)
    free = np.zeros((nu, nv), bool)
    free[1:-1, 1:-1] = True
    free = free.ravel()
    Cf = C.reshape(-1, 3)
    rhs = P.reshape(-1, 3) - A[:, ~free] @ Cf[~free]
    Cf[free] = np.linalg.lstsq(A[:, free], rhs, rcond=None)[0]
    resid = np.linalg.norm(A @ Cf - P.reshape(-1, 3), axis=1)
    return Cf.reshape(nu, nv, 3), resid


def _bspline(C: np.ndarray, inner_u: np.ndarray, inner_v: np.ndarray):
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

    nu, nv = C.shape[:2]
    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    for a in range(nu):
        for b in range(nv):
            poles.SetValue(a + 1, b + 1, gp_Pnt(*map(float, C[a, b])))

    def knots(inner):
        kn = TColStd_Array1OfReal(1, len(inner))
        mu = TColStd_Array1OfInteger(1, len(inner))
        for i, x in enumerate(inner):
            kn.SetValue(i + 1, float(x))
            mu.SetValue(i + 1, 4 if i in (0, len(inner) - 1) else 1)
        return kn, mu

    ku, mu_ = knots(inner_u)
    kv, mv_ = knots(inner_v)
    return Geom_BSplineSurface(poles, ku, kv, mu_, mv_, 3, 3)


def _evened(P: np.ndarray) -> np.ndarray:
    return _even(_even(P).transpose(1, 0, 2)).transpose(1, 0, 2)


def _lsq_surface(P: np.ndarray, n: int):
    """Сглаживающая грань: n×n полюсов по K×K точкам методом наименьших квадратов.

    Граничные кривые подгоняются только по точкам своей границы, поэтому у соседних лоскутов с
    общей границей они совпадают и грани сшиваются без щелей, как при интерполяции."""
    B, inner = _basis(P.shape[0], n)
    C, _ = _lsq_fit(_evened(P), B, B)
    return _bspline(C, inner, inner)


def _fit_patch(args):
    points, basis = args
    return _lsq_fit(_evened(points), basis, basis)[0]


def _watch_parent():
    """Exit if the demo kills its smooth-body worker on timeout."""
    import multiprocessing
    import threading

    parent = multiprocessing.parent_process()
    if parent is not None:
        def watch():
            parent.join()
            os._exit(1)

        threading.Thread(target=watch, daemon=True).start()


def _fit_patches(grids, basis):
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    import multiprocessing

    cpus = os.cpu_count() or 1
    try:
        jobs = max(1, min(int(os.environ.get("CAD_JOBS", "1")), cpus, 61))
    except ValueError:
        jobs = 1
    args = [(p, basis) for p in grids]
    if jobs > 1 and len(args) >= 64:
        # Each patch is a small least-squares solve. Prevent nested BLAS teams
        # from oversubscribing the same CPU cores in each child.
        limits = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS")
        saved = {key: os.environ.get(key) for key in limits}
        try:
            for key in limits:
                os.environ[key] = "1"
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_watch_parent) as pool:
                return list(pool.map(_fit_patch, args, chunksize=16))
        except (BrokenProcessPool, OSError):
            print("подгонка граней: пул недоступен, считаю последовательно", flush=True)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return [_fit_patch(task) for task in args]


def _vertex_near(mesh, points, threshold):
    """Certify hits using actual mesh vertices; leave borderline points to closest_point."""
    from scipy.spatial import cKDTree
    from trimesh.constants import tol as mesh_tol

    # closest_point can prefer the second triangle on a tie within tol.merge
    # in squared distance. Leave room for that correction and roundoff.
    cutoff_sq = threshold ** 2 - 2 * mesh_tol.merge
    if cutoff_sq <= 0:
        return np.zeros(len(points), dtype=bool)
    tree = cKDTree(mesh.vertices[mesh.referenced_vertices])
    distance, _ = tree.query(points, distance_upper_bound=np.sqrt(cutoff_sq))
    return np.isfinite(distance)


def _lossy(fitted: list, proj: _Projector, tol: float, n_pts: int = 30000) -> set:
    """Номера граней, до которых не дотянулась часть поверхности: у точки чистой поверхности
    ближайшая точка модели дальше tol, и эта ближайшая точка принадлежит грани. Мера двусторонняя:
    остаток подгонки в собственных точках грани таких потерь не видит (Э9)."""
    per = len(fitted[0])
    k = int(round(np.sqrt(per)))
    # грубая сетка модели прямо по точкам граней: расстояние до неё — то же, чем меряется покрытие
    idx = np.arange(k * k).reshape(k, k)
    cell = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
                           np.stack([idx[:-1, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)])
    tris = np.concatenate([cell + j * per for j in range(len(fitted))])
    model = trimesh.Trimesh(np.vstack(fitted), tris, process=False)
    sel = np.random.default_rng(3).choice(len(proj.dp), min(n_pts, len(proj.dp)), replace=False)
    pts = proj.dp[sel]
    total = len(pts)
    if proj.fast:
        pts = pts[~_vertex_near(model, pts, tol)]
    if not len(pts):
        return set()
    _, d, tid = trimesh.proximity.closest_point(model, pts)
    i = (tid // len(cell)) * per   # номер грани × per: дальше делится на per
    cnt = np.bincount(i[d > tol] // per, minlength=len(fitted))
    # заменяются грани с наибольшими потерями, пока в оставшихся крупных гранях не останется
    # не больше LOSS_KEEP всех точек: порог приёмки (COVER) тогда проходится с запасом
    order = np.argsort(-cnt)
    left = cnt.sum() - np.cumsum(cnt[order])
    k = int(np.argmax(left <= LOSS_KEEP * total)) + 1 if cnt.sum() > LOSS_KEEP * total else 0
    return set(order[:k].tolist())


def _edge_key(p0, p1):
    return frozenset((tuple(np.round(p0, 6)), tuple(np.round(p1, 6))))


def _tri_faces(q, na, nb, C, n, proj, coarse_side):
    """Пара с потерями заменяется шестью лоскутами: каждый из двух её треугольников делится на три
    через центр, как в схеме без склейки. Половинка стороны, за которой сосед остался крупной
    гранью, берётся точно с его граничной кривой (coarse_side — такие стороны), остальные границы —
    по точкам поверхности. Возвращает список поверхностей."""
    from scipy.interpolate import BSpline

    tk = np.r_[[0.0] * 3, np.linspace(0, 1, n - 2), [1.0] * 3]
    s = np.linspace(0, 1, K)
    E = BSpline.design_matrix(0.5 * s, tk, 3).toarray()   # первая половина целой кривой
    S, T = np.meshgrid(s, s, indexing="ij")
    w = np.stack([(1 - S) * (1 - T), S * (1 - T), S * T, (1 - S) * T], -1).reshape(-1, 4)
    Bh, ih = _half_basis(K, n, 0)
    # граничные кривые пары от угла к углу: (откуда, куда) → полюса в этом направлении
    curve = {}
    for i0, i1, poles in ((0, 1, C[:, 0]), (1, 2, C[-1, :]), (3, 2, C[:, -1]), (0, 3, C[0, :])):
        curve[(i0, i1)], curve[(i1, i0)] = poles, poles[::-1]
    out = []
    for tri, nn in (((0, 1, 3), na), ((1, 2, 3), nb)):
        G = q[list(tri)].mean(0)
        for k in range(3):
            v, nx, pv = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            quad = np.array([q[v], (q[v] + q[nx]) / 2, G, (q[v] + q[pv]) / 2])
            P = _evened(proj(w @ quad, nn).reshape(K, K, 3))
            if (v, nx) in curve and _edge_key(q[v], q[nx]) in coarse_side:
                P[:, 0] = E @ curve[(v, nx)]
            if (v, pv) in curve and _edge_key(q[v], q[pv]) in coarse_side:
                P[0, :] = E @ curve[(v, pv)]
            Cs, _ = _lsq_fit(P, Bh, Bh)
            out.append(_bspline(Cs, ih, ih))
    return out


def _build(mesh: trimesh.Trimesh, triangles: int, proj: _Projector, planes: bool = True, *,
           profile=None, stage=None, incumbent=None):
    import fast_simplification as fs
    from OCP.Approx import Approx_ParametrizationType
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array2OfPnt

    import surface

    with measured(profile, "coarse"):
        if len(mesh.faces) > triangles:
            coarse = surface.coarsen(mesh, triangles)
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
    S_, T_ = S.ravel(), T.ravel()
    with measured(profile, "layout"):
        pq = _paired_quads(coarse) if PAIR else None
        if PAIR and pq is None:
            return None, 0, None
        if pq is not None:
            quads = [(q, n) for q, n, *_ in pq]
            # у каждой точки нормаль своего треугольника: сторона стенки выбирается точнее
            nrm = [np.where((S_ + T_ <= 1)[:, None], na, nb) for _, _, na, nb, _ in pq]
        else:
            quads = list(_quads(coarse))
            nrm = [n for _, n in quads]
    tag, fit = -np.ones(len(quads), int), {}
    if planes:
        try:
            with measured(profile, "projection"):
                projected = [proj(w @ q, nn_) for (q, _), nn_ in zip(quads, nrm)]
            with measured(profile, "planes"):
                tag, fit = _plane_tags(mesh, quads, projected, proj, w)
            if os.environ.get("CAD_REGIONS", "0") == "1":  # точные цилиндры по разметке областей (этап Б)
                with measured(profile, "cylinders"):
                    tag, fit = _cyl_tags(mesh, quads, projected, proj, w, tag, fit)
        except Exception:  # noqa: BLE001  плоскости — улучшение, без них тело то же
            tag, fit = -np.ones(len(quads), int), {}
    faces, ftag = [], []

    def add(srf, t):
        with measured(profile, "samples"):
            u0, u1, v0, v1 = srf.Bounds()
            for u in u0 + (u1 - u0) * cm:
                for vv in v0 + (v1 - v0) * cm:
                    p = srf.Value(u, vv)
                    mids.append((p.X(), p.Y(), p.Z()))
        with measured(profile, "make_face"):
            fc = BRepBuilderAPI_MakeFace(srf, 1e-6).Face()
            sew.Add(fc)
        faces.append(fc)
        ftag.append(t)

    split, sides = set(), {}
    if NP:
        B, inner = _basis(K, NP)
        A = np.kron(B, B)
        # Project in the original order to preserve the shared boundary cache.
        with measured(profile, "projection"):
            grids = [proj(w @ q, nn_).reshape(K, K, 3) for (q, _), nn_ in zip(quads, nrm)]
        with measured(profile, "fit"):
            pre = _fit_patches(grids, B)
        if SPLIT and pq is not None:
            with measured(profile, "lossy"):
                fitted = [A @ C.reshape(-1, 3) for C in pre]
                split = {i for i in _lossy(fitted, proj, FACE_TOL) if tag[i] < 0}
            for i in split:
                q = quads[i][0]
                for i0, i1 in ((0, 1), (1, 2), (2, 3), (3, 0)):
                    key = _edge_key(q[i0], q[i1])
                    sides[key] = sides.get(key, 0) + 1
    for i, ((q, _), nn_) in enumerate(zip(quads, nrm)):
        if NP:
            C = pre[i]
            if i in split:
                coarse_side = {key for key in (_edge_key(q[i0], q[i1]) for i0, i1 in ((0, 1), (1, 2), (2, 3), (3, 0)))
                               if sides[key] == 1}
                with measured(profile, "split"):
                    surfaces = _tri_faces(q, pq[i][2], pq[i][3], C, NP, proj, coarse_side)
                for srf in surfaces:
                    add(srf, -1)
                continue
            with measured(profile, "fit"):
                srf = _bspline(C, inner, inner)
        else:
            with measured(profile, "projection"):
                P = proj(w @ q, nn_).reshape(K, K, 3)
            with measured(profile, "fit"):
                arr = TColgp_Array2OfPnt(1, K, 1, K)
                for a in range(K):
                    for b in range(K):
                        arr.SetValue(a + 1, b + 1, gp_Pnt(*map(float, P[a, b])))
                api = GeomAPI_PointsToBSplineSurface()
                api.Interpolate(arr, Approx_ParametrizationType.Approx_IsoParametric, False)
                srf = api.Surface()
        add(srf, int(tag[i]))
    tag = np.array(ftag, int)
    n_faces = len(faces)
    mids = np.array(mids)
    if proj.fast and incumbent is not None:
        # This is the same one-sided score the original cascade would use.
        # Below GOAL it cannot pass, and at or below the incumbent it cannot replace it.
        with measured(profile, "near"):
            share = near_share(mids, proj, GOAL_TOL)
        stage["near"] = share
        if share < GOAL and share <= incumbent:
            stage.update(score=share, status="dominated_before_sewing")
            return None, n_faces, None
    with measured(profile, "sewing"):
        sew.Perform()
    if sew.NbFreeEdges():
        return None, n_faces, None
    with measured(profile, "solid"):
        sewn = sew.SewedShape()
        sol = None
        if fit:
            # лоскуты одной плоскости сливаются в одну точную плоскую грань; если тело после этого
            # невалидно, берётся тело без слияния (форма та же)
            try:
                from OCP.TopoDS import TopoDS

                merged_shape, merged = _merge_planes(sewn, [TopoDS.Face_s(sew.Modified(f)) for f in faces], tag, fit)
                if merged:
                    sol = _solid(merged_shape, profile=profile, parallel=proj.fast and n_faces >= 500)
            except Exception:  # noqa: BLE001
                sol = None
        if sol is None:
            sol = _solid(sewn, profile=profile, parallel=proj.fast and n_faces >= 500)
    if sol is not None:
        n_faces = len(sol.Faces())
    return sol, n_faces, mids


def near_share(pts: np.ndarray, proj: _Projector, tol: float = 0.1) -> float:
    """Доля точек модели в пределах tol от поверхности: расстояние до касательной плоскости
    ближайших точек поверхности (берётся наименьшее среди соседних, ведь у тонкой стенки
    самая ближняя точка может лежать на обратной стороне)."""
    dist, ii = proj.tree.query(pts, k=8)
    d = np.abs(np.einsum("ikj,ikj->ik", pts[:, None] - proj.dp[ii], proj.dn[ii]))
    d[dist > 2.0] = np.inf
    return float((d.min(1) < tol).mean())


def cover_share(body, proj: _Projector, tol: float = 0.3, n: int = 30000, *, profile=None) -> float:
    """Доля точек чистой поверхности, у которых в пределах tol мм есть поверхность модели."""
    # 0,01, а не 0,05: допуск разбиения в CadQuery относительный (доля размера грани), и у крупной
    # грани сетка при 0,05 уходила от самой грани до 0,8 мм — «потери» были ошибкой измерения (Э11)
    with measured(profile, "tessellation"):
        v, f = body.tessellate(0.01, 0.5)
        m = trimesh.Trimesh([(q.x, q.y, q.z) for q in v], f, process=False)
    pts = proj.dp[np.random.default_rng(3).choice(len(proj.dp), min(n, len(proj.dp)), replace=False)]
    with measured(profile, "cover_distance"):
        if not proj.fast:
            _, d, _ = trimesh.proximity.closest_point(m, pts)
            return float((d < tol).mean())
        near = _vertex_near(m, pts, tol)
        misses = pts[~near]
        if not len(misses):
            return 1.0
        _, d, _ = trimesh.proximity.closest_point(m, misses)
        return float((np.count_nonzero(near) + np.count_nonzero(d < tol)) / len(pts))


def smooth_body(mesh: trimesh.Trimesh, scan: trimesh.Trimesh, budget: float = 900.0, tol: float = 0.3):
    """mesh — замкнутая чистая поверхность (из surface.py), scan — исходный скан, tol — допуск, с которым
    модель потом сравнивается со сканом. Возвращает (тело или None, сведения)."""
    t0 = time.time()
    proj = _Projector(mesh)
    best, info = None, {"faces": 0, "near": 0.0, "score": -1.0}
    global K, NP, PAIR, FACE_TOL, _SCAN
    _SCAN = scan
    FACE_TOL = min(tol, 0.5)  # потери по граням считаются тем же допуском, что и приёмка
    stages = []
    for step, (tri, K, NP, PAIR) in enumerate(STEPS, 1):
        if time.time() - t0 > budget / 3 and best is not None:
            break
        proj.cache.clear()
        profile = {}
        stage = {"step": step, "times": profile, "status": "invalid"}
        stages.append(stage)
        with measured(profile, "build"):
            body, n, mids = _build(mesh, tri, proj, profile=profile, stage=stage,
                                   incumbent=info["score"] if best is not None else None)
        stage["faces"] = n
        if body is None:
            continue
        # точность считается от чистой поверхности: шум скана сглажен ещё в surface.py, и
        # мерить по сырому скану значило бы гнаться за шумом
        if "near" not in stage:
            with measured(profile, "near"):
                stage["near"] = near_share(mids, proj, GOAL_TOL)
        share = score = stage["near"]
        ok = share >= GOAL
        if NP and ok:
            # Мера выше односторонняя: точки модели лежат у поверхности. Подгонка же не проходит через
            # точки и может срезать ребро или мелкую деталь — тогда куска поверхности в модели нет, и при
            # сравнении со сканом тело проигрывает (линза: 98,9% по этой мере и 96,5% в допуске). Поэтому
            # для подгонки проверяется и обратное, тем же допуском, что и итоговое сравнение.
            # не шире 0,5 мм: у шумного скана допуск сравнения 2 мм, а чистая поверхность уже сглажена
            with measured(profile, "cover"):
                cover = cover_share(body, proj, min(tol, 0.5), profile=profile)
            stage["cover"] = cover
            score = min(share, cover)
            ok = cover >= COVER
        stage.update(near=share, score=score, status="accepted" if ok else "rejected")
        if best is None or score > info["score"]:
            best, info = body, {"faces": n, "near": share, "score": score, "step": step}
        if ok:
            break
    info["stages"] = stages
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
