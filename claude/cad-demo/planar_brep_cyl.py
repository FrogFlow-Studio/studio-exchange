"""Точные скругления между двумя плоскостями в прямом теле.

Взято из задачи 31 Севера (chatgpt-1/cad-intelligence/cad-crease-fillet-wall-2026-11-13)."""
from collections import defaultdict

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeWire,
                                BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepLib import BRepLib
from OCP.Geom import Geom_CylindricalSurface, Geom_Plane
from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.ShapeFix import ShapeFix_Edge
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

import planar_brep
from patch_faces import topology


def _point(p):
    return gp_Pnt(*map(float, p))


def _line(a, b):
    return BRepBuilderAPI_MakeEdge(_point(a), _point(b)).Edge()


def _wire(edges):
    maker = BRepBuilderAPI_MakeWire()
    for edge in edges:
        maker.Add(edge)
        if not maker.IsDone():
            raise ValueError("disconnected wire")
    return maker.Wire()


def _plane(row):
    n = np.asarray(row["normal"], float)
    n = n / np.linalg.norm(n)
    return n, float(n @ row["origin"])


def _runs(loop, others):
    segments = list(zip(loop, np.roll(loop, -1)))
    keys = [others.get(tuple(sorted(e))) for e in segments]
    cut = next((i for i in range(len(keys)) if keys[i] != keys[i - 1]), 0)
    segments, keys = segments[cut:] + segments[:cut], keys[cut:] + keys[:cut]
    runs = []
    for (a, b), key in zip(segments, keys):
        if not runs or key != runs[-1][0]:
            runs.append((key, [a, b]))
        else:
            runs[-1][1].append(b)
    return runs


def _section(surface, n, d, start, end, middle):
    pl = Geom_Plane(gp_Pln(_point(n * d), gp_Dir(*n)))
    intersections = GeomAPI_IntSS(surface, pl, 1e-8)
    if not intersections.IsDone() or intersections.NbLines() != 1:
        raise ValueError("torus or non-planar end")
    curve = intersections.Line(1)
    angles = [GeomAPI_ProjectPointOnCurve(_point(p), curve).LowerDistanceParameter() for p in (start, end, middle)]
    a, b, m = angles
    period = 2 * np.pi
    if (m - a) % period < (b - a) % period:
        return BRepBuilderAPI_MakeEdge(curve, a, a + (b - a) % period).Edge()
    return TopoDS.Edge_s(BRepBuilderAPI_MakeEdge(curve, b, b + (a - b) % period).Edge().Reversed())


def _cylinder(V, F, regs, key, own, exterior, tol):
    row = regs[key]
    pair = tuple(row["fillet_planes"])
    n = np.array([_plane(regs[k])[0] for k in pair])
    d = np.array([_plane(regs[k])[1] for k in pair])
    wall = row.get("curved_wall", False)
    axis = np.array(row["axis"], float) if wall else np.cross(*n)
    norm = np.linalg.norm(axis)
    if norm < .15:
        raise ValueError("parallel supports")
    axis /= norm
    radius = row["r"]
    reference = np.asarray(row["origin"])
    if wall:
        center = reference
        tangent = []
        for side, normal, offset in zip(pair, n, d):
            radial_normal = normal - (normal @ axis) * axis
            if abs(normal @ axis) > 1e-5:
                raise ValueError("wall side is not parallel to the cylinder axis")
            unit = radial_normal / np.linalg.norm(radial_normal)
            h = (offset - normal @ center) / np.linalg.norm(radial_normal)
            if abs(h) >= radius:
                raise ValueError("wall cylinder misses side plane")
            along = np.cross(axis, unit) * np.sqrt(1 - (h / radius)**2)
            rail = [v for edge, support in exterior.items() if support == side for v in edge]
            if not rail:
                raise ValueError("wall rail is missing")
            guide = V[rail].mean(axis=0) - center
            tangent.append((h / radius) * unit + np.sign(guide @ along) * along)
        tangent = np.array(tangent)
    else:
        signs = np.sign(n @ reference - d)
        center = reference + n.T @ np.linalg.solve(n @ n.T, d + signs * radius - n @ reference)
        tangent = -signs[:, None] * n
    x = tangent[0]
    y = np.cross(axis, x)
    angle = np.arctan2(tangent[1] @ y, tangent[1] @ x)
    if angle < 0:
        axis, y, angle = -axis, -y, -angle
    surface = Geom_CylindricalSurface(gp_Ax3(_point(center), gp_Dir(*axis), gp_Dir(*x)), radius)
    loops = planar_brep._loops(F, own)
    if loops is None or len(loops) != 1:
        raise ValueError("fillet needs one closed loop")
    runs = _runs(loops[0], exterior)
    if len(runs) != 4 or len({k for k, _ in runs}) != 4 or any(k is None or k < 0 for k, _ in runs):
        raise ValueError("fillet needs four distinct planar rails and ends")
    if any(regs[k]["kind"] != "plane" for k, _ in runs) or not set(pair) <= {k for k, _ in runs}:
        raise ValueError("end must be a plane")
    points = {}
    for i, (support, chain) in enumerate(runs):
        next_support = runs[(i + 1) % 4][0]
        sides = set(pair).intersection((support, next_support))
        if len(sides) != 1:
            raise ValueError("rails and ends must alternate")
        side = next(iter(sides))
        cap = next(k for k in (support, next_support) if k != side)
        normal, offset = _plane(regs[cap])
        axial = normal @ axis
        if abs(axial) < .15:
            raise ValueError("end plane nearly parallel to axis")
        base = center + radius * tangent[pair.index(side)]
        p = base + axis * ((offset - normal @ base) / axial)
        if np.linalg.norm(V[chain[-1]] - p) > tol:
            raise ValueError("corner out of tolerance")
        points[int(chain[-1])] = p
    edges, replacements = [], {}
    for support, chain in runs:
        a, b = chain[0], chain[-1]
        if support in pair:
            edge = _line(points[a], points[b])
        else:
            normal, offset = _plane(regs[support])
            base = center + radius * (np.cos(angle / 2) * x + np.sin(angle / 2) * y)
            middle = base + axis * ((offset - normal @ base) / (normal @ axis))
            edge = _section(surface, normal, offset, points[a], points[b], middle)
        edges.append(edge)
        replacements[support] = (tuple(reversed(chain)), TopoDS.Edge_s(edge.Reversed()))
    carrier = BRepBuilderAPI_MakeFace(surface, 1e-8).Face()
    fix = ShapeFix_Edge()
    for edge in edges:
        fix.FixAddPCurve(edge, carrier, False, 1e-7)
        fix.FixSameParameter(edge)
    maker = BRepBuilderAPI_MakeFace(surface, _wire(edges), True)
    face = maker.Face()
    if not BRepCheck_Analyzer(face).IsValid():
        raise ValueError("invalid cylindrical trim")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex

    maximum = 0.
    for p in V[np.unique(F[own])]:
        vertex = BRepBuilderAPI_MakeVertex(_point(p)).Vertex()
        distance = BRepExtrema_DistShapeShape(vertex, face)
        if not distance.IsDone():
            raise ValueError("fillet distance failed")
        maximum = max(maximum, distance.Value())
    if maximum > tol:
        raise ValueError("trimmed face outside tolerance")
    return face, replacements, points, maximum


def _planar(V, F, ids, replacements, points):
    loops = planar_brep._loops(F, ids)
    if not loops:
        raise ValueError("invalid plane loop")
    tri = V[F[ids]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0)
    n /= np.linalg.norm(n)
    items = []
    for loop in loops:
        loop = list(map(int, loop))
        wires, matched = [], set()
        i = next((loop.index(chain[0]) for chain, _ in replacements if chain[0] in loop), 0)
        ring = loop[i:] + loop[:i]
        i = 0
        while i < len(ring):
            candidates = [(chain, edge) for chain, edge in replacements if chain[0] == ring[i]]
            if candidates:
                chain, edge = candidates[0]
                length = len(chain) - 1
                if tuple((ring + ring)[i:i + length + 1]) != chain:
                    raise ValueError("replacement chain is not a shared contour")
                wires.append(edge)
                matched.add(chain)
                i += length
            else:
                a, b = ring[i], ring[(i + 1) % len(ring)]
                wires.append(_line(points.get(a, V[a]), points.get(b, V[b])))
                i += 1
        loop_keys = {chain for chain, _ in replacements if chain[0] in ring}
        if matched != loop_keys:
            raise ValueError("incomplete contour replacement")
        items.append((planar_brep._area(V[ring], n), _wire(wires)))
    items.sort(key=lambda item: -item[0])
    # Let OCC construct the shared edge pcurves on the plane from its complete wire.
    maker = BRepBuilderAPI_MakeFace(items[0][1], True)
    for _, wire in items[1:]:
        maker.Add(wire)
    if not maker.IsDone() or not BRepCheck_Analyzer(maker.Face()).IsValid():
        raise ValueError("invalid support plane")
    return maker.Face()


def build(V, F, label, regions, *, sew_tol=1e-3, min_faces=3, straight=1e-5,
          smooth_tol=None, smooth_budget=300., smooth_edge_tol=None, tol=.3):
    """Compatible B-rep builder, with a transactional attempt at exact fillet topology."""
    V, F, labels = np.asarray(V, float), np.asarray(F, int), np.asarray(label, int)
    fallback = dict(sew_tol=sew_tol, min_faces=min_faces, straight=straight, smooth_tol=smooth_tol,
                    smooth_budget=smooth_budget, smooth_edge_tol=smooth_edge_tol)
    ids_by_region = {int(k): np.flatnonzero(labels == k) for k in np.unique(labels) if k >= 0}
    uses = topology(F)
    borders = defaultdict(dict)
    for edge, sides in uses.items():
        if len(sides) != 2:
            continue
        a, b = (labels[i] for i, _, _ in sides)
        if a != b:
            borders[a][edge], borders[b][edge] = int(b), int(a)
    exact, replacements, points, rejected = {}, defaultdict(list), {}, {}
    errors = {}
    for key, ids in ids_by_region.items():
        region = regions[key]
        if region["kind"] != "cylinder" or "fillet_planes" not in region:
            continue
        try:
            face, change, positions, error = _cylinder(V, F, regions, key, ids, borders[key], tol)
            if any(i in points and np.linalg.norm(points[i] - p) > sew_tol / 4 for i, p in positions.items()):
                raise ValueError("incompatible shared corner")
        except (RuntimeError, ValueError, IndexError) as exc:
            rejected[key] = str(exc)
            continue
        exact[key], errors[key] = face, error
        points.update(positions)
        for support, edge in change.items():
            replacements[support].append(edge)
    if exact:
        try:
            sewing = BRepBuilderAPI_Sewing(sew_tol)
            for face in exact.values():
                sewing.Add(face)
            covered = set(exact)
            for key, ids in ids_by_region.items():
                if regions[key]["kind"] == "plane":
                    groups = planar_brep._components(F[ids], labels[ids])
                    for component in np.unique(groups):
                        part = ids[groups == component]
                        vertices = set(F[part].ravel())
                        changes = [(chain, e) for chain, e in replacements[key] if set(chain) <= vertices]
                        sewing.Add(_planar(V, F, part, changes, points))
                    covered.add(key)
            triangles = 0
            for a, b, c in F[~np.isin(labels, list(covered))]:
                p = [points.get(int(i), V[i]) for i in (a, b, c)]
                wire = _wire([_line(p[i], p[(i + 1) % 3]) for i in range(3)])
                sewing.Add(BRepBuilderAPI_MakeFace(wire, True).Face())
                triangles += 1
            sewing.Perform()
            shell = sewing.SewedShape()
            if shell.ShapeType() != TopAbs_SHELL or sewing.NbFreeEdges() or sewing.NbMultipleEdges():
                raise ValueError(f"sewing failed: free={sewing.NbFreeEdges()}, multiple={sewing.NbMultipleEdges()}")
            solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell)).Solid()
            if not BRepLib.OrientClosedSolid_s(solid) or not BRepCheck_Analyzer(solid).IsValid():
                raise ValueError("invalid sewn solid")
            body = cq.Solid(solid)
            walls = sum(bool(regions[k].get("curved_wall")) for k in exact)
            return body, {"valid": True, "free_edges": 0, "faces": len(body.Faces()), "triangle_faces": triangles,
                          "planar_faces": len(body.Faces()) - len(exact) - triangles, "exact_fillets": len(exact) - walls,
                          "exact_curved_walls": walls,
                          "fillet_vertex_max_mm": errors, "fillets_rejected": rejected}
        except (RuntimeError, ValueError) as exc:
            rejected["sewing"] = str(exc)
    body, audit = planar_brep.build(V, F, labels, regions, **fallback)
    return body, {**audit, "exact_fillets": 0, "exact_curved_walls": 0, "fillets_rejected": rejected}
