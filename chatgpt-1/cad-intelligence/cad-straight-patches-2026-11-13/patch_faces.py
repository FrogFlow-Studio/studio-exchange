"""Replace a smooth triangular zone with a face sharing its straight boundary."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_C0, GeomAbs_Line
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Edge
from OCP.TopAbs import TopAbs_EDGE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
from scipy.spatial import cKDTree


MAX_BOUNDARY = 64
MAX_TRIANGLES = 240
SEW_TOL = .001


def topology(triangles):
    uses = defaultdict(list)
    for i, tri in enumerate(triangles):
        for a, b in zip(tri, np.roll(tri, -1)):
            uses[tuple(sorted((a, b)))].append((i, a, b))
    return uses


def _loops(uses):
    if any(len(rows) > 2 or (len(rows) == 2 and rows[0][1] == rows[1][1]) for rows in uses.values()):
        return None
    boundary = [(rows[0][1], rows[0][2]) for rows in uses.values() if len(rows) == 1]
    successor = dict(boundary)
    if len(successor) != len(boundary) or set(successor) != set(successor.values()):
        return None
    loops = []
    while successor:
        first = min(successor)
        loop, last = [], first
        while last in successor:
            loop.append(last)
            last = successor.pop(last)
        if first != last or len(loop) < 3:
            return None
        loops.append(np.array(loop))
    return loops


def _pnt(x):
    return gp_Pnt(*map(float, x))


def _edges(points):
    return [BRepBuilderAPI_MakeEdge(_pnt(a), _pnt(b)).Edge() for a, b in zip(points, np.roll(points, -1, axis=0))]


def _samples(points, count):
    if len(points) <= count:
        return points
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    nearest = np.full(len(points), np.inf)
    for _ in range(count - 1):
        nearest = np.minimum(nearest, np.sum((points - points[chosen[-1]]) ** 2, axis=1))
        chosen.append(int(np.argmax(nearest)))
    return points[chosen]


def _fill(boundaries, points):
    builder = BRepOffsetAPI_MakeFilling(3, 8, 2, False, 1e-6, SEW_TOL / 5, .01, .1, 8, 12)
    edges = [_edges(points) for points in boundaries]
    for i, loop in enumerate(edges):
        for edge in loop:
            builder.Add(edge, GeomAbs_C0, i == 0)
    for p in points:
        builder.Add(_pnt(p))
    builder.Build()
    if not builder.IsDone():
        return None
    face = TopoDS.Face_s(builder.Shape())
    if len(edges) == 1:
        return face
    surface = BRep_Tool.Surface_s(face)
    outer = BRepTools.OuterWire_s(face)
    trimmed = BRepBuilderAPI_MakeFace(surface, outer, True)
    fix = ShapeFix_Edge()
    for loop in edges[1:]:
        wire = BRepBuilderAPI_MakeWire()
        for edge in loop:
            fix.FixAddPCurve(edge, face, False, SEW_TOL / 2)
            fix.FixSameParameter(edge)
            wire.Add(edge)
        trimmed.Add(wire.Wire())
    return trimmed.Face()


def _same_boundary(face, vertices, loops):
    ids = np.concatenate(loops)
    tree = cKDTree(vertices[ids])
    expected = {tuple(sorted((a, b))) for loop in loops for a, b in zip(loop, np.roll(loop, -1))}
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    tolerance = 0.
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Line:
            return None
        tolerance = max(tolerance, BRep_Tool.Tolerance_s(edge))
        start, end = curve.FirstParameter(), curve.LastParameter()
        distance, pair = tree.query([curve.Value(start).Coord(), curve.Value(end).Coord()])
        if distance.max() > SEW_TOL / 10:
            return None
        key = tuple(sorted(ids[pair]))
        if key not in expected:
            return None
        expected.remove(key)
        explorer.Next()
    return None if expected else tolerance


def _verify(face, points, xyz, cross, tol):
    if not BRepCheck_Analyzer(face).IsValid():
        return None, {"reason": "invalid_face"}
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    mesh_area = float(np.linalg.norm(cross, axis=1).sum() / 2)
    area_ratio = props.Mass() / mesh_area
    if not .78 <= area_ratio <= 1.22:
        return None, {"reason": "area_changed", "area_ratio": area_ratio}
    surface = BRep_Tool.Surface_s(face)
    uv = BRepTools.UVBounds_s(face)
    project = GeomAPI_ProjectPointOnSurf()
    project.Init(surface, *uv, 1e-5)
    classifier = BRepClass_FaceClassifier()
    distances = []
    for point in points:
        pnt = _pnt(point)
        project.Perform(pnt)
        if project.IsDone() and project.NbPoints():
            u, v = project.LowerDistanceParameters()
            classifier.Perform(face, gp_Pnt2d(u, v), 1e-6)
            if classifier.State() in (TopAbs_IN, TopAbs_ON):
                distances.append(project.LowerDistance())
                continue
        exact = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(pnt).Vertex(), face)
        if not exact.IsDone():
            return None, {"reason": "projection_failed"}
        distances.append(exact.Value())
    maximum = max(distances, default=0.)
    if maximum > tol:
        missed = np.flatnonzero(np.asarray(distances) > tol)
        worst = missed[np.argsort(np.asarray(distances)[missed])[-4:]]
        return None, {"reason": "distance", "vertex_max_mm": maximum, "_retry_points": points[worst]}
    signs = []
    for center, normal in zip(xyz.mean(axis=1), cross):
        project.Perform(_pnt(center))
        if not project.IsDone() or not project.NbPoints():
            return None, {"reason": "normal_projection_failed"}
        u, v = project.LowerDistanceParameters()
        du, dv = gp_Vec(), gp_Vec()
        surface.D1(u, v, gp_Pnt(), du, dv)
        target = du.Crossed(dv)
        length = target.Magnitude()
        if length <= 1e-12:
            return None, {"reason": "singular_normal"}
        signs.append(np.dot(normal, target.Coord()) / (np.linalg.norm(normal) * length))
    aligned = np.array(signs)
    sign = -1 if np.median(aligned) < 0 else 1
    if np.any(sign * aligned <= 0):
        return None, {"reason": "flipped_normal", "min_normal_dot": float(np.min(sign * aligned))}
    if (sign < 0) != (face.Orientation() == TopAbs_REVERSED):
        face = TopoDS.Face_s(face.Reversed())
    return face, {"reason": "accepted", "vertex_max_mm": maximum, "area_ratio": area_ratio,
                  "min_normal_dot": float(np.min(sign * aligned))}


def patch(V, F, faces_of_zone, tol):
    """Return one OCC face or a rejection reason. Original triangle indices are preserved."""
    V, F = np.asarray(V, dtype=float), np.asarray(F, dtype=int)
    ids = np.asarray(faces_of_zone, dtype=int)
    if tol <= 0 or not len(ids):
        return None, {"reason": "empty_zone_or_tolerance"}
    if len(ids) > MAX_TRIANGLES:
        return None, {"reason": "triangle_budget", "triangles": len(ids)}
    triangles = F[ids]
    uses = topology(triangles)
    loops = _loops(uses)
    if not loops:
        return None, {"reason": "invalid_boundary"}
    count = sum(map(len, loops))
    if count > MAX_BOUNDARY:
        return None, {"reason": "boundary_budget", "boundary_edges": count}
    if len(split_zones(V, F, ids, max_boundary=None, max_triangles=None)) != 1:
        return None, {"reason": "split_required", "boundary_edges": count}
    xyz = V[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) < 1e-10):
        return None, {"reason": "degenerate_triangle"}
    loops.sort(key=lambda ring: -np.linalg.norm(np.cross(V[ring], np.roll(V[ring], -1, axis=0)).sum(axis=0)))
    interior = V[np.setdiff1d(np.unique(triangles), np.concatenate(loops))]
    try:
        bounds = [V[loop] for loop in loops]
        seeds = _samples(interior, 16)
        for attempt in range(6):
            face = _fill(bounds, seeds)
            if face is None:
                return None, {"reason": "fill_failed", "boundary_edges": count}
            edge_tolerance = _same_boundary(face, V, loops)
            if edge_tolerance is None:
                return None, {"reason": "boundary_changed", "boundary_edges": count}
            if edge_tolerance > tol / 4:
                return None, {"reason": "boundary_surface_tolerance", "edge_tolerance_mm": edge_tolerance}
            face, report = _verify(face, interior, xyz, cross, tol)
            retry = report.pop("_retry_points", None)
            if retry is None:
                break
            extra = np.unique(np.vstack((seeds, retry)), axis=0)
            if len(extra) == len(seeds):
                break
            seeds = extra
    except (RuntimeError, ValueError) as exc:
        return None, {"reason": "occ_error", "error": str(exc)}
    return face, {**report, "boundary_edges": count, "loops": len(loops), "edge_tolerance_mm": edge_tolerance,
                  "fit_attempts": attempt + 1}


def split_zones(V, F, faces_of_zone, *, max_boundary=MAX_BOUNDARY, max_triangles=MAX_TRIANGLES, crease_deg=50.):
    """Partition into connected patches at normal breaks and by boundary budget."""
    ids = np.asarray(faces_of_zone, dtype=int)
    if not len(ids):
        return []
    if (max_boundary is not None and max_boundary < 3) or (max_triangles is not None and max_triangles < 1):
        raise ValueError("invalid zone budget")
    tris = np.asarray(F)[ids]
    vertices = np.asarray(V)[tris]
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normals /= np.maximum(1e-12, np.linalg.norm(normals, axis=1, keepdims=True))
    centers = vertices.mean(axis=1)
    neighbors = defaultdict(list)
    for rows in topology(tris).values():
        if len(rows) == 2:
            a, b = (row[0] for row in rows)
            if normals[a] @ normals[b] >= np.cos(np.radians(crease_deg)):
                neighbors[a].append(b)
                neighbors[b].append(a)

    def components(indices):
        remaining = set(indices)
        while remaining:
            group = [remaining.pop()]
            for a in group:
                for b in neighbors[a]:
                    if b in remaining:
                        remaining.remove(b)
                        group.append(b)
            yield np.array(sorted(group))

    pending = list(components(range(len(ids))))
    output = []
    while pending:
        group = pending.pop()
        within_faces = max_triangles is None or len(group) <= max_triangles
        within_boundary = max_boundary is None or sum(len(rows) == 1 for rows in topology(tris[group]).values()) <= max_boundary
        if within_faces and within_boundary:
            output.append(ids[group])
        else:
            q = centers[group]
            axis = np.linalg.svd(q - q.mean(axis=0), full_matrices=False)[2][0]
            order = group[np.argsort(q @ axis, kind="stable")]
            mid = len(order) // 2
            pending.extend(components(order[:mid]))
            pending.extend(components(order[mid:]))
    return sorted(output, key=lambda group: int(group.min()))
