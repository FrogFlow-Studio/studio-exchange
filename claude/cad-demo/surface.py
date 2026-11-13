"""Тело по поверхности скана — для деталей любой формы (кривые оболочки, линзы, корпуса).

Плоские стенки (walls.py) дают точную «конструкторскую» модель, но только там, где деталь
собрана из пластин. Здесь — общий путь: поверхность скана очищается от шума и мусора,
упрощается, дыры в сетке закрываются, и всё сшивается в одно твёрдое тело CAD-ядра.
Плоские участки выпрямляются в точные плоскости, круглые отверстия вырезаются точно по
диаметру из скана (walls._hole_walls). Достраивание скола — дело ИИ поверх этого тела.

python surface.py скан.stl папка [треугольников]
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


SMOOTH = 10  # шаги сглаживания шума
FLAT_MAX = 0.2  # мм: дальше от подогнанной плоскости вершину не двигаем


def clean(mesh: trimesh.Trimesh, min_share: float = 0.02) -> trimesh.Trimesh:
    """Мусор сканирования (летающие кусочки) выбрасываем: оставляем куски не меньше 2%
    площади самого большого."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    big = max(p.area for p in parts)
    keep = [p for p in parts if p.area >= min_share * big]
    return trimesh.util.concatenate(keep)


def boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Замкнутые контуры дыр в сетке (рёбра, у которых только один треугольник)."""
    import networkx as nx

    e = mesh.edges_sorted[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    g = nx.Graph()
    g.add_edges_from(map(tuple, e))
    loops = []
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        if len(comp) < 3 or any(d % 2 for _, d in sub.degree()):
            continue
        # «восьмёрка» (две дыры касаются в вершине): обходим контур целиком и режем на
        # простые петли в местах, где вершина встречается второй раз
        path = []
        for a, _ in nx.eulerian_circuit(sub):
            if a in path:
                i = path.index(a)
                if len(path) - i >= 3:
                    loops.append(path[i:])
                path = path[:i]
            path.append(a)
        if len(path) >= 3:
            loops.append(path)
    return loops


def fill_loops(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Каждую дыру закрываем веером из её центра. Направление треугольников потом
    согласует fix_normals."""
    V = [mesh.vertices]
    F = [mesh.faces]
    n = len(mesh.vertices)
    # направленные рёбра существующих треугольников: новый треугольник идёт по ребру навстречу
    directed = set(map(tuple, mesh.faces[:, [0, 1]])) | set(map(tuple, mesh.faces[:, [1, 2]])) | set(map(tuple, mesh.faces[:, [2, 0]]))
    for loop in boundary_loops(mesh):
        c = mesh.vertices[loop].mean(axis=0)
        V.append(c[None])
        tri = []
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            tri.append([b, a, n] if (a, b) in directed else [a, b, n])
        F.append(np.array(tri))
        n += 1
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=True)
    if m.volume < 0:
        m.invert()
    return m


def split_pinches(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Вершина, в которой касаются два листа (тонкие стенки слиплись в точке): у каждого
    листа делаем свою копию вершины, иначе поверхность нельзя согласованно ориентировать."""
    import networkx as nx

    adj = nx.Graph()
    adj.add_edges_from(map(tuple, m.face_adjacency))
    V, F = list(m.vertices), m.faces.copy()
    for v, fs in enumerate(m.vertex_faces):
        fs = [x for x in fs if x >= 0]
        if len(fs) < 2:
            continue
        comps = list(nx.connected_components(adj.subgraph(fs).copy())) if all(f in adj for f in fs) else []
        # одиночные треугольники без соседей тоже отдельные листы
        seen = set().union(*comps) if comps else set()
        comps += [{f} for f in fs if f not in seen]
        for comp in comps[1:]:
            V.append(m.vertices[v])
            nv = len(V) - 1
            for f in comp:
                F[f][F[f] == v] = nv
    return trimesh.Trimesh(np.array(V), F, process=False)


def repair(m: trimesh.Trimesh, rounds: int = 12) -> trimesh.Trimesh:
    """После упрощения тонкие стенки иногда слипаются: ребро на 3–4 треугольника. Такие
    треугольники убираем, мелкие оторванные куски выбрасываем, дыры закрываем — пока сетка
    не станет замкнутой."""
    for _ in range(rounds):
        m = clean(m, 0.01)
        e = m.edges_sorted
        bad_edges = trimesh.grouping.group_rows(e, require_count=3)
        rows = trimesh.grouping.group_rows(e)
        over = [g for g in rows if len(g) > 2]
        if over:
            faces = np.unique(np.concatenate([m.edges_face[g] for g in over]))
            keep = np.ones(len(m.faces), bool)
            keep[faces] = False
            m = trimesh.Trimesh(m.vertices, m.faces[keep], process=True)
        del bad_edges
        m.remove_unreferenced_vertices()
        m = fill_loops(m)
        if m.is_watertight and m.is_winding_consistent:
            break
        if m.is_watertight:  # замкнута, но не ориентируется — разводим слипшиеся вершины
            m = split_pinches(m)
            trimesh.repair.fix_winding(m)
            trimesh.repair.fix_normals(m)
            if m.is_winding_consistent:
                break
            # остались места, где соседние треугольники смотрят в разные стороны (кусок
            # поверхности перекручен): убираем их, на следующем круге дыра закроется заново
            bad = flipped_faces(m)
            if len(bad):
                keep = np.ones(len(m.faces), bool)
                keep[bad] = False
                m = trimesh.Trimesh(m.vertices, m.faces[keep], process=True)
    return m


def flipped_faces(m: trimesh.Trimesh) -> np.ndarray:
    """Треугольники по обе стороны ребра, которые проходят его в одну сторону."""
    fa, e = m.face_adjacency, m.face_adjacency_edges

    def forward(f, a, b):
        F = m.faces[f]
        i = np.argmax(F == a[:, None], axis=1)
        return F[np.arange(len(f)), (i + 1) % 3] == b

    same = forward(fa[:, 0], e[:, 0], e[:, 1]) == forward(fa[:, 1], e[:, 0], e[:, 1])
    return np.unique(fa[same])


def meshfix(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Замкнутая правильная сетка через MeshFix (pymeshfix): убирает слипшиеся стенки,
    самопересечения и закрывает дыры. Если его нет или он не справился — свой repair."""
    try:
        import pymeshfix
    except ImportError:
        return repair(m)
    v, f = pymeshfix.clean_from_arrays(np.ascontiguousarray(m.vertices, dtype=np.float64),
                                       np.ascontiguousarray(m.faces, dtype=np.int32))
    r = trimesh.Trimesh(v, f, process=True)
    if len(r.faces) < 0.5 * len(m.faces) or not (r.is_watertight and r.is_winding_consistent):
        return repair(m)
    if r.volume < 0:
        r.invert()
    return r


def flatten_planes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Плоские участки делаем точно плоскими: вершины внутри участка проецируем на
    подогнанную плоскость (вершины на границе двух участков не трогаем)."""
    import walls as W

    pats = W.planar_patches(mesh)
    owner = -np.ones(len(mesh.vertices), int)
    shared = np.zeros(len(mesh.vertices), bool)
    for k, p in enumerate(pats):
        vs = np.unique(mesh.faces[p["faces"]])
        shared[vs[(owner[vs] >= 0) & (owner[vs] != k)]] = True
        owner[vs] = k
    V = mesh.vertices.copy()
    for k, p in enumerate(pats):
        vs = np.where((owner == k) & ~shared)[0]
        n = p["normal"]
        d = (V[vs] - p["point"]) @ n
        # участок, который на деле слегка изогнут, не выпрямляем: двигаем только вершины,
        # лежащие у плоскости в пределах шума скана, иначе форма уходит от скана
        near = np.abs(d) <= FLAT_MAX
        V[vs[near]] -= np.outer(d[near], n)
    return trimesh.Trimesh(V, mesh.faces, process=False)


def solid_from_mesh(mesh: trimesh.Trimesh):
    """Треугольники сшиваются в оболочку и твёрдое тело OCC."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing)
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS

    sew = BRepBuilderAPI_Sewing(1e-3)
    P = [gp_Pnt(*map(float, v)) for v in mesh.vertices]
    for a, b, c in mesh.faces:
        poly = BRepBuilderAPI_MakePolygon(P[a], P[b], P[c], True)
        f = BRepBuilderAPI_MakeFace(poly.Wire(), True)
        if f.IsDone():
            sew.Add(f.Face())
    sew.Perform()
    sh = sew.SewedShape()
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer

    # тонкая деталь после упрощения иногда сшивается не в одну оболочку, а в несколько:
    # каждую делаем телом, самое большое — деталь, остальные (мусор) отбрасываем
    shells = []
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell_s(ex.Current()))
        ex.Next()
    solids = []
    for shell in shells:
        try:
            fix = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(shell).Solid())
            fix.Perform()
            got = cq.Shape.cast(fix.Solid()).Solids()
            if not got:
                continue
            s = got[0]
            if s.Volume() < 0:
                s = cq.Solid(s.wrapped.Reversed())
            solids.append(s)
        except Exception:  # noqa: BLE001
            continue
    if not solids:
        raise RuntimeError("сетку не удалось сшить в тело")
    return max(solids, key=lambda s: abs(s.Volume()))


def coarsen(mesh: trimesh.Trimesh, triangles: int) -> trimesh.Trimesh:
    """Simplify to the layout budget, with closed edge collapses as a fallback."""
    import fast_simplification as fs

    if len(mesh.faces) <= triangles:
        return mesh
    v, f = fs.simplify(mesh.vertices, mesh.faces, 1 - triangles / len(mesh.faces))
    out = repair(trimesh.Trimesh(v, f, process=True))
    usable = out.is_watertight and out.is_winding_consistent
    if usable and len(out.faces) <= 1.3 * triangles:
        return out
    if not (mesh.is_watertight and mesh.is_winding_consistent):
        return out

    from region_layout import coarsen_by_regions

    alt, _ = coarsen_by_regions(mesh, np.full(len(mesh.faces), -1, dtype=int), triangles,
                                regularize=False)
    return alt if not usable or len(alt.faces) < len(out.faces) else out


def surface_mesh(scan: trimesh.Trimesh, faces: int = 20000) -> trimesh.Trimesh:
    """Замкнутая чистая поверхность скана: без мусора и шума, упрощённая, с закрытыми дырами."""
    m = clean(scan.copy())
    trimesh.smoothing.filter_taubin(m, iterations=SMOOTH)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(percent=1 - faces / len(m.faces))
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    m.merge_vertices()
    r = repair(m)
    # свой ремонт сохраняет тонкие оболочки; MeshFix надёжнее, но тонкие стенки может
    # выбросить, поэтому он — только запасной путь, когда сетка так и не стала правильной
    return r if r.is_watertight and r.is_winding_consistent else meshfix(m)


def hole_parameters(scan: trimesh.Trimesh):
    """Cylinder parameters in scan coordinates, independent of the candidate body."""
    import walls

    return [(tuple(w["origin"] + w["normal"] * 0.3), tuple(-w["normal"]),
             float(w["hole"]["r"]), float(w["hole"]["t"] + 0.3))
            for w in walls._hole_walls(scan)]


def _new_walls_off_scan(body, scan: trimesh.Trimesh, spec, away: float = 0.5) -> float | None:
    """Доля площади новых стенок выреза (цилиндр радиуса отверстия и торцы поперёк его оси),
    которая лежит дальше away мм от скана. Настоящее отверстие уже есть в скане, и его стенка ложится
    на скан; «отверстие» на месте неглубокой круглой площадки вырезает дно там, где скан — сплошной
    материал. None — новых стенок не нашлось."""
    c, d, r, h = spec
    c, d = np.asarray(c, float), np.asarray(d, float)
    d = d / np.linalg.norm(d)
    pts, wts = [], []
    for f in body.Faces():
        kind = f.geomType()
        if kind == "CYLINDER":     # стенка: цилиндр того же радиуса и с той же осью, что инструмент
            cyl = f._geomAdaptor().Cylinder()
            ax = cyl.Axis().Direction()
            if abs(cyl.Radius() - r) > 1e-3 or abs(ax.X() * d[0] + ax.Y() * d[1] + ax.Z() * d[2]) < 0.999:
                continue
        elif kind == "PLANE":      # дно: плоскость поперёк оси точно на торце инструмента, внутри его круга
            pln = f._geomAdaptor().Pln()            # дёшево: без расчёта центра и нормали по грани
            ax, loc = pln.Axis().Direction(), pln.Location()
            n = np.array([ax.X(), ax.Y(), ax.Z()])
            if abs(float(n @ d)) < 0.999:
                continue
            along = float((np.array([loc.X(), loc.Y(), loc.Z()]) - c) @ d)
            if min(abs(along), abs(along - h)) > 0.02:
                continue
            rel = np.array(f.Center().toTuple()) - c
            if np.linalg.norm(rel - float(rel @ d) * d) > r:
                continue
        else:
            continue
        vs, tris = f.tessellate(0.05, 0.3)
        if not tris:
            continue
        P = np.array([v.toTuple() for v in vs])
        T = np.array(tris)
        tri = P[T]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        pts.append(tri.mean(axis=1))
        wts.append(area)
    if not pts:
        return None
    P, W = np.vstack(pts), np.concatenate(wts)
    _, dist, _ = trimesh.proximity.closest_point(scan, P)
    return float(W[dist > away].sum() / max(W.sum(), 1e-12))


def cut_holes(s, scan: trimesh.Trimesh, *, volumes=None, timings=None, hole_finder=hole_parameters,
              limit_time: bool = False):
    """Круглые отверстия вырезаются точно по диаметру из скана. Возвращает (тело, сколько).

    limit_time: каждый вырез идёт в отдельном процессе с пределом времени (для тяжёлых тел, где
    безнадёжная булева операция тянется минутами и всё равно откатывается). Предел — 60 с для первого
    выреза, дальше 6× самого долгого удачного, но не меньше 30 с."""
    import cadquery as cq
    from time import perf_counter

    from cad_checks import VolumeMemo, add_count, boolean_cut_limited, boolean_op, measured
    from hole_preflight import HolePreflight

    if volumes is None:
        volumes = VolumeMemo()
    add_count(timings, "hole_ops", 0)
    add_count(timings, "hole_rolled_back_ops", 0)
    add_count(timings, "hole_preflight_rejected", 0)
    add_count(timings, "hole_scan_rejected", 0)
    preflight = HolePreflight() if os.environ.get("CAD_HOLE_PREFLIGHT", "1") != "0" else None
    with measured(timings, "hole_search"):
        specs = hole_finder(scan)

    def note(label, outcome, started, boolean_s=0.0):
        print(f"    {label}: {outcome}; булева {boolean_s:.3f} с, всего {perf_counter() - started:.3f} с",
              flush=True)

    ok_times = []

    def attempt(body, tool, limit, label, *, trial=False):
        add_count(timings, "hole_ops")
        started = perf_counter()
        boolean_s = 0.0
        phase = "булева"
        try:
            t = perf_counter()
            try:
                with measured(timings, "hole_boolean"):
                    if limit_time and not trial:
                        res = boolean_cut_limited(body, tool, max(30.0, 6 * max(ok_times)) if ok_times else 60.0)
                    else:
                        res = boolean_op(body, (tool,), preserve_inputs=trial)
            finally:
                boolean_s = perf_counter() - t
            phase = "проверка"
            with measured(timings, "hole_checks"):
                solids = res.Solids()
                if len(solids) != 1:
                    reason = f"тел {len(solids)}, требуется одно"
                else:
                    before, after = volumes.get(body), res.Volume()
                    delta = before - after
                    solid = solids[0]
                    if not 0 <= delta <= limit:
                        reason = f"изменение объёма {delta:.9g} мм³ вне [0, {limit:.9g}]"
                    elif not solid.isValid():
                        reason = "тело невалидно"
                    else:
                        volumes.remember_cut(res, solid, after)
                        ok_times.append(boolean_s)
                        note(label, f"принято, удалено {delta:.9g} мм³", started, boolean_s)
                        return solid
        except Exception as ex:  # noqa: BLE001
            detail = " ".join(str(ex).split())[:160]
            reason = f"{phase}: исключение {type(ex).__name__}" + (f" ({detail})" if detail else "")
        add_count(timings, "hole_rolled_back_ops")
        note(label, f"откат, {reason}", started, boolean_s)
        return None

    def cylinder(spec):
        c, d, r, h = spec
        return cq.Solid.makeCylinder(r, h, cq.Vector(*c), cq.Vector(*d)), 1.05 * np.pi * r * r * h

    cut = 0
    for i, spec in enumerate(specs):
        label = f"отверстие {i + 1}"
        started = perf_counter()
        try:
            c, d, r, h = spec
            label += f", Ø{2 * r:.6g}, L={h:.17g}, R={r:.17g}, начало {list(map(float, c))}, ось {list(map(float, d))}"
        except (ValueError, TypeError):
            pass
        if preflight is not None:
            try:
                with measured(timings, "hole_preflight"):
                    reason = preflight.reject(s, spec)
            except Exception:  # an inconclusive precheck keeps the original path
                reason = None
            if reason:
                add_count(timings, "hole_preflight_rejected")
                note(label, f"пропуск до булевой, {reason}", started)
                continue
        try:
            tool, limit = cylinder(spec)
        except Exception as ex:  # noqa: BLE001
            note(label, f"инструмент: исключение {type(ex).__name__}", started)
            continue
        new = attempt(s, tool, limit + 1, label)
        if new is not None and os.environ.get("CAD_HOLE_SCAN_CHECK", "1") != "0":
            try:
                off = _new_walls_off_scan(new, scan, spec)
            except Exception:  # noqa: BLE001  проверка не удалась — вырез остаётся
                off = None
            if off is not None and off > 0.3:
                add_count(timings, "hole_scan_rejected")
                note(label, f"откат: {off:.0%} новых стенок дальше 0,5 мм от скана (в скане такого выреза нет)",
                     started)
                new = None
        if new is not None:
            s, cut = new, cut + 1
    return s, cut


def surface_body(scan: trimesh.Trimesh, faces: int = 20000, holes: bool = True, *,
                 prepared: trimesh.Trimesh | None = None, timings=None, hole_finder=hole_parameters):
    m = flatten_planes(surface_mesh(scan, faces) if prepared is None else prepared)
    s = solid_from_mesh(m)
    cut = 0
    if holes:
        s, cut = cut_holes(s, scan, timings=timings, hole_finder=hole_finder)
    return s, {"faces": len(m.faces), "watertight": bool(m.is_watertight), "holes_cut": cut}

if __name__ == "__main__":
    import time

    import cadquery as cq

    t = time.time()
    scan = trimesh.load(sys.argv[1], force="mesh")
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s, info = surface_body(scan, int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
    cq.exporters.export(s, str(out / "model.step"))
    cq.exporters.export(s, str(out / "model.stl"))
    print(info, "valid", s.isValid(), "solids", len(s.Solids()), "sec", round(time.time() - t, 1))


