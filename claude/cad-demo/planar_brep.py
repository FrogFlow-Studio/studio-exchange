"""Тело из выпрямленной сетки: одна плоская область — одна грань.

На входе сетка после idealize.py (вершины лежат на своих плоскостях и линиях их пересечения) и метка
области каждого треугольника. Треугольники одной плоскости сливаются в одну плоскую грань с прямыми
рёбрами; вершины посреди прямого ребра между двумя такими гранями убираются у обеих сразу, поэтому
ребро модели — один отрезок, как в CAD. Всё остальное (свободная форма) пока остаётся треугольниками.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh


DEBUG = False


def collapse_degenerate(V, F, label, eps: float):
    """Вырожденные треугольники (вершины на одной прямой после посадки) убираются. Возвращает (V, F, label).

    «Шапка» — вершина лежит посреди длинной стороны: длинная сторона перекидывается на соседа, оба новых
    треугольника лежат в плоскости соседа. «Игла» — две вершины совпали: короткое ребро стягивается."""
    V = np.asarray(V, float)
    F = np.asarray(F).copy()
    label = np.asarray(label).copy()
    for _ in range(50):
        tri = V[F]
        e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1), np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1)
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        alt = area2 / np.maximum(e.max(1), 1e-300)
        bad = np.flatnonzero(alt < eps)
        if not len(bad):
            break
        owner = {}
        for fi, (a, b, c) in enumerate(F):
            owner[(int(a), int(b))] = fi
            owner[(int(b), int(c))] = fi
            owner[(int(c), int(a))] = fi
        touched_f, touched_v = set(), set()

        def ring(v):
            return {int(x) for fi in np.flatnonzero((F == v).any(1)) for x in F[fi] if x != v}

        merge = {}
        changed = False
        for f in bad:
            f = int(f)
            if f in touched_f:
                continue
            if e[f].min() < eps:                      # игла
                k = int(np.argmin(e[f]))
                a, b = int(F[f, k]), int(F[f, (k + 1) % 3])
                if a in touched_v or b in touched_v or a == b:
                    continue
                if len(ring(a) & ring(b)) != 2:           # стягивание склеило бы поверхность саму с собой
                    continue
                touched_v.update((a, b))
                merge[b] = a
                continue
            k = int(np.argmax(e[f]))                  # шапка
            a, b, c = int(F[f, k]), int(F[f, (k + 1) % 3]), int(F[f, (k + 2) % 3])
            g = owner.get((b, a))
            if g is None or g in touched_f or {a, b, c} & touched_v:
                continue
            d = int(next(v for v in F[g] if v != a and v != b))
            if d == c or (a, d) not in owner or (d, b) not in owner or (c, d) in owner or (d, c) in owner:
                continue
            F[f], F[g] = (a, d, c), (d, b, c)
            label[f] = label[g]
            touched_f.update((f, g))
            touched_v.update((a, b, c, d))
            changed = True
        if merge:
            step = np.arange(len(V))
            for b, a in merge.items():
                step[b] = a
            F = step[F]
            keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
            F, label = F[keep], label[keep]
        elif not changed:
            break
    return V, F, label


def _components(F, label):
    """Связные куски каждой плоской области (связь через общее ребро). -1 — не в куске."""
    edge = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        if label[fi] < 0:
            continue
        for u, v in ((a, b), (b, c), (c, a)):
            edge[(min(u, v), max(u, v))].append(fi)
    parent = np.arange(len(F))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fs in edge.values():
        if len(fs) == 2 and label[fs[0]] == label[fs[1]]:
            a, b = find(fs[0]), find(fs[1])
            if a != b:
                parent[a] = b
    comp = np.array([find(i) if label[i] >= 0 else -1 for i in range(len(F))])
    return comp


def _loops(F, faces):
    """Границы куска: замкнутые цепочки вершин, внутренность слева. None, если кусок касается сам себя в вершине."""
    half = set()
    for fi in faces:
        a, b, c = F[fi]
        half.update(((a, b), (b, c), (c, a)))
    nxt = {}
    for u, v in half:
        if (v, u) not in half:
            if u in nxt:
                return None
            nxt[u] = v
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v)
            if v is None:
                return None
        if v != start:
            return None
        loops.append(loop)
    return loops


def _area(P, n):
    return 0.5 * float(np.cross(P, np.roll(P, -1, axis=0)).sum(0) @ n)


def build(V, F, label, regions, *, sew_tol: float = 1e-3, min_faces: int = 3,
          straight: float = 1e-5, smooth_tol: float | None = None, smooth_budget: float = 300.0,
          smooth_edge_tol: float | None = None):
    """Возвращает (тело CadQuery, сведения). regions — как у regions.segment (нужны только плоскости)."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid,
                                    BRepBuilderAPI_Sewing)
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    V = np.asarray(V, float)
    F = np.asarray(F)
    label = np.array([l if l >= 0 and regions[l]["kind"] == "plane" else -1 for l in label])
    comp = _components(F, label)
    tri_n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])

    for _ in range(6):   # куски, которые не складываются в грань, возвращаются в треугольники; решение повторяется
        ids, counts = np.unique(comp[comp >= 0], return_counts=True)
        for i, c in zip(ids, counts):
            if c < min_faces:
                comp[comp == i] = -1
        groups = defaultdict(list)
        for fi, c in enumerate(comp):
            if c >= 0:
                groups[int(c)].append(fi)
        # вершину посреди ребра можно убрать, если вокруг неё ровно два куска и ни одного свободного треугольника
        around = defaultdict(set)
        for fi, c in enumerate(comp):
            for v in F[fi]:
                around[int(v)].add(int(c) if c >= 0 else -(fi + 2))
        cand = {v for v, g in around.items() if len(g) == 2 and all(x >= 0 for x in g)}
        all_loops = {c: _loops(F, fs) for c, fs in groups.items()}
        # ...и только если она действительно на прямой между соседями по границе (у обоих кусков соседи те же)
        drop = set()
        for loops in all_loops.values():
            for loop in loops or ():
                k = len(loop)
                for i, v in enumerate(loop):
                    if v in cand and v not in drop:
                        a, b = V[loop[i - 1]], V[loop[(i + 1) % k]]
                        ab = b - a
                        ln = np.linalg.norm(ab)
                        if ln > 1e-12 and np.linalg.norm(np.cross(V[v] - a, ab)) / ln <= straight \
                                and 0 < (V[v] - a) @ ab < ln * ln:
                            drop.add(v)
        faces_out, failed = {}, []
        for c, fs in groups.items():
            loops = all_loops[c]
            n = tri_n[fs].sum(0)
            ln = np.linalg.norm(n)
            if loops is None or ln < 1e-12:
                failed.append(c)
                continue
            n = n / ln
            kept = []
            for loop in loops:
                pts = [v for v in loop if v not in drop]
                # убирается только то, что действительно на прямой между соседями
                if len(pts) < 3:
                    kept = None
                    break
                kept.append(pts)
            if kept is None:
                failed.append(c)
                continue
            areas = [_area(V[p], n) for p in kept]
            full = sum(_area(V[p], n) for p in loops)
            if max(areas) <= 0 or abs(sum(areas) - full) > 1e-4 * max(1.0, abs(full)):
                failed.append(c)      # убранные вершины изменили площадь: ребро не прямое
                continue
            faces_out[c] = (n, kept, areas)
        if not failed:
            break
        for c in failed:
            comp[comp == c] = -1

    sew = BRepBuilderAPI_Sewing(sew_tol)
    P = [gp_Pnt(*map(float, v)) for v in V]
    planar, planar_area, rejected = 0, 0.0, 0
    for c, (n, loops, areas) in faces_out.items():
        outer = int(np.argmax(areas))
        o = V[loops[outer][0]]
        if DEBUG:
            print('face', c, 'off-plane', max(float(np.abs((V[l] - o) @ n).max()) for l in loops))
        pln = gp_Pln(gp_Pnt(*map(float, o)), gp_Dir(*map(float, n)))

        def wire(loop):
            poly = BRepBuilderAPI_MakePolygon()
            for v in loop:
                poly.Add(P[v])
            poly.Close()
            return poly.Wire()

        mk = BRepBuilderAPI_MakeFace(pln, wire(loops[outer]), True)
        ok = mk.IsDone()
        for i, loop in enumerate(loops):
            if ok and i != outer:
                mk.Add(wire(loop))
        if ok:
            # грань должна быть правильной и иметь площадь своих треугольников: иначе контур сам себя пересёк
            face = mk.Face()
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            ok = BRepCheck_Analyzer(face).IsValid() and abs(props.Mass() - sum(areas)) <= 1e-3 * max(1.0, sum(areas))
        if ok:
            sew.Add(face)
            planar += 1
            planar_area += sum(areas)
        else:
            comp[comp == c] = -1
            rejected += 1
            if DEBUG:
                print('rej', c, 'done', mk.IsDone(), 'valid', mk.IsDone() and BRepCheck_Analyzer(mk.Face()).IsValid(), 'mass', mk.IsDone() and round(props.Mass(), 2), 'areas', [round(a, 2) for a in areas], [len(l) for l in loops])
    # Остальное (свободная форма, скругления): связные пятна треугольников заменяются одной гладкой гранью
    # по той же прямолинейной границе (patch_faces) — сшивка с соседями не меняется. Пятно, которое не
    # ложится в допуск или не строится, остаётся треугольниками.
    smooth, smooth_area, smooth_tris, smooth_fail = 0, 0.0, 0, 0
    rest = comp < 0
    why = defaultdict(int)        # причины отказа гладких заплаток (пусто, если свободных зон нет)
    if smooth_tol and rest.any():
        import time

        import patch_faces

        started = time.perf_counter()
        own = np.where(label >= 0, label, -1)          # пятна не смешивают разные области (цилиндр и свободу)
        zl = np.asarray([int(l) if l >= 0 and regions[l]["kind"] != "plane" else -1 for l in own])
        zone_comp = _components(F, np.where(rest, np.where(zl >= 0, zl, len(regions)), -1))
        tri_area = 0.5 * np.linalg.norm(tri_n, axis=1)
        parts = []
        for z in np.unique(zone_comp[zone_comp >= 0]):
            ids = np.flatnonzero(zone_comp == z)
            if len(ids) < 2:
                continue
            parts.extend(p for p in patch_faces.split_zones(V, F, ids) if len(p) >= 2)
        for part, face, reason in _patches(V, F, parts, smooth_tol, smooth_edge_tol,
                                           started + smooth_budget):
            if face is None:
                if reason is not None:
                    smooth_fail += 1
                    why[reason] += len(part)
                continue
            sew.Add(face)
            rest[part] = False
            smooth += 1
            smooth_tris += len(part)
            smooth_area += float(tri_area[part].sum())
    tris = 0
    for fi in np.flatnonzero(rest):
        a, b, c = F[fi]
        f = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakePolygon(P[a], P[b], P[c], True).Wire(), True)
        if f.IsDone():
            sew.Add(f.Face())
            tris += 1
    sew.Perform()
    sh = sew.SewedShape()
    solids = []
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    while ex.More():
        try:
            fix = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(ex.Current())).Solid())
            fix.Perform()
            got = cq.Shape.cast(fix.Solid()).Solids()
            if got:
                s = got[0]
                solids.append(cq.Solid(s.wrapped.Reversed()) if s.Volume() < 0 else s)
        except Exception:  # noqa: BLE001
            pass
        ex.Next()
    if not solids:
        raise RuntimeError("сетку не удалось сшить в тело")
    body = max(solids, key=lambda s: abs(s.Volume()))
    return body, {"planar_faces": planar, "triangle_faces": tris, "faces": len(body.Faces()),
                  "valid": bool(body.isValid()), "free_edges": int(sew.NbFreeEdges()),
                  "rejected_faces": rejected, "smooth_faces": smooth, "smooth_triangles": smooth_tris,
                  "smooth_rejected": smooth_fail, "smooth_rejected_triangles": dict(why) if smooth_tol else {},
                  "smooth_area_share": float(smooth_area / max(1e-9, trimesh.Trimesh(V, F, process=False).area)),
                  "planar_area_share": float(planar_area / max(1e-9, trimesh.Trimesh(V, F, process=False).area))}


_PV = _PF = None


def _patch_init(V, F):
    global _PV, _PF
    _PV, _PF = V, F


def _patch_one(part, tol, edge_tol, path):
    """Одна гладкая заплатка в отдельном процессе; грань возвращается файлом BRep."""
    import patch_faces
    from OCP.BRepTools import BRepTools

    try:
        face, rep = patch_faces.patch(_PV, _PF, part, tol, edge_tol)
    except Exception:  # noqa: BLE001
        return None, "error"
    if face is None:
        return None, str((rep or {}).get("reason", "error"))
    BRepTools.Write_s(face, path)
    return path, None


def _patches(V, F, parts, tol, edge_tol, deadline):
    """Заплатки независимы друг от друга, поэтому считаются в нескольких процессах (CAD_PATCH_WORKERS,
    по умолчанию до 4). Порядок и результат те же, что при счёте подряд; что не успело к сроку —
    остаётся треугольниками (reason None). При любой ошибке пула — счёт подряд."""
    import os
    import time

    import patch_faces

    workers = int(os.environ.get("CAD_PATCH_WORKERS", str(min(4, os.cpu_count() or 1))))
    if workers > 1 and len(parts) > 8:
        try:
            yield from _patches_pool(V, F, parts, tol, edge_tol, deadline, workers)
            return
        except Exception as error:  # noqa: BLE001
            print(f"    параллельные заплатки не вышли ({type(error).__name__}): считаю подряд", flush=True)
    for part in parts:
        if time.perf_counter() > deadline:
            yield part, None, None
            continue
        try:
            face, rep = patch_faces.patch(V, F, part, tol, edge_tol)
        except Exception:  # noqa: BLE001
            face, rep = None, None
        yield part, face, (None if face is not None else str((rep or {}).get("reason", "error")))


def _patches_pool(V, F, parts, tol, edge_tol, deadline, workers):
    import tempfile
    import time
    from concurrent.futures import ProcessPoolExecutor, wait

    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS, TopoDS_Shape

    out = []
    with tempfile.TemporaryDirectory() as tmp, \
            ProcessPoolExecutor(workers, initializer=_patch_init, initargs=(V, F)) as pool:
        jobs = [pool.submit(_patch_one, part, tol, edge_tol, f"{tmp}/{i}.brep") for i, part in enumerate(parts)]
        wait(jobs, timeout=max(0.0, deadline - time.perf_counter()))
        for job in jobs:
            job.cancel()
        for part, job in zip(parts, jobs):
            if not job.done() or job.cancelled():
                out.append((part, None, None))
                continue
            path, reason = job.result()
            if path is None:
                out.append((part, None, reason))
                continue
            shape = TopoDS_Shape()
            BRepTools.Read_s(shape, path, BRep_Builder())
            out.append((part, TopoDS.Face_s(shape), None))
    yield from out
