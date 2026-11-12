"""Measure the CAD structure of a STEP model for comparison on the same part."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_REVERSED
from OCP.gp import gp_Pnt, gp_Vec


ANALYTIC = ("plane", "cylinder", "cone", "sphere", "torus")
FRACTIONS = (.1, .3, .5, .7, .9)


def _measure(method, shape):
    props = GProp_GProps()
    method(shape, props)
    return props.Mass()


def _normal(surface, pcurve, parameter, sign):
    uv = pcurve.Value(parameter)
    du, dv = gp_Vec(), gp_Vec()
    surface.D1(uv.X(), uv.Y(), gp_Pnt(), du, dv)
    normal = du.Crossed(dv)
    if normal.SquareMagnitude() < 1e-24:
        return None
    normal.Normalize()
    return normal if sign > 0 else normal.Reversed()


def _angle_samples(edge, curve, neighbors):
    if not BRep_Tool.SameParameter_s(edge):
        return []
    data = []
    for face, surface in neighbors:
        pcurve = BRep_Tool.CurveOnSurface_s(edge, face, 0., 0.)
        if pcurve is None:
            return []
        sign = -1 if face.Orientation() == TopAbs_REVERSED else 1
        data.append((surface, pcurve, sign))
    start, end = curve.FirstParameter(), curve.LastParameter()
    if not np.isfinite([start, end]).all():
        return []
    angles = []
    for fraction in FRACTIONS:
        parameter = start + fraction * (end - start)
        normals = [_normal(surface, pcurve, parameter, sign) for surface, pcurve, sign in data]
        if all(n is not None for n in normals):
            dot = normals[0].Dot(normals[1])
            angles.append(float(np.degrees(np.arccos(np.clip(dot, -1., 1.)))))
        else:
            angles.append(None)
    return angles


def _directions(planes):
    if not planes:
        return np.empty((0, 3))
    normals = np.array([normal for normal, _ in planes])
    weights = np.array([area for _, area in planes])
    parallel = np.cos(np.radians(3.))
    order = np.argsort(-weights, kind="stable")
    # Candidate seeds are bounded for STEP models with many planar facets.
    first = max(order[:64], key=lambda i: float(weights[np.abs(normals @ normals[i]) >= parallel].sum()))
    axis = normals[first]
    transverse = order[np.abs(normals[order] @ axis) <= np.sin(np.radians(3.))]
    if not len(transverse):
        return axis[None, :]
    j = max(transverse[:64], key=lambda i: float(weights[np.abs(normals @ normals[i]) >= parallel].sum()))
    second = normals[j] - axis * (normals[j] @ axis)
    second /= np.linalg.norm(second)
    return np.array([axis, second, np.cross(axis, second)])


def quality(shape, *, step=None, tol=.3):
    """Return area, edge and orientation statistics for one imported shape."""
    started = perf_counter()
    faces = shape.Faces()
    if not faces:
        raise ValueError("STEP contains no faces")
    areas = [_measure(BRepGProp.SurfaceProperties_s, face.wrapped) for face in faces]
    total = sum(areas)
    if total <= 0:
        raise ValueError("STEP has zero surface area")
    # Cap the threshold so a coarse input mesh cannot classify a large face as a facet.
    small_threshold = min((2 * step) ** 2, total * .005) if step else total * .001
    incidence = defaultdict(list)
    surface_area = Counter()
    planes, radii = [], []
    small_count = 0
    small_area = 0.
    for face, area in zip(faces, areas):
        support = BRepAdaptor_Surface(face.wrapped)
        kind = face.geomType().lower()
        edges = face.Edges()
        if area < small_threshold:
            small_count += 1
            small_area += area
            if kind == "plane" and len(edges) == 3 and all(e.geomType() == "LINE" for e in edges):
                kind = "small_planar_triangle"
        surface_area[kind] += area
        if kind == "plane":
            planes.append((np.array(support.Plane().Axis().Direction().Coord()), area))
        if kind == "cylinder":
            radii.append(support.Cylinder().Radius())
        for edge in edges:
            incidence[edge].append((face.wrapped, support))

    total_length = straight_length = paired_length = sampled_length = mottled_length = 0.
    bins = Counter()
    for edge, neighbors in incidence.items():
        edge = edge.wrapped
        if BRep_Tool.Degenerated_s(edge):
            continue
        length = _measure(BRepGProp.LinearProperties_s, edge)
        curve = BRepAdaptor_Curve(edge)
        total_length += length
        if curve.GetType().name in ("GeomAbs_Line", "GeomAbs_Circle"):
            straight_length += length
        if len(neighbors) != 2:
            continue
        paired_length += length
        angles = _angle_samples(edge, curve, neighbors)
        for angle in angles:
            if angle is None:
                continue
            portion = length / len(FRACTIONS)
            sampled_length += portion
            if 1. <= angle <= 15.:
                mottled_length += portion
                bins["1_to_15"] += portion
            else:
                bins["below_1" if angle < 1. else "above_15"] += portion

    axes = _directions(planes)
    aligned = 0
    if len(axes):
        dots = np.abs(np.array([normal for normal, _ in planes]) @ axes.T)
        aligned = int(np.count_nonzero(np.any((dots >= np.cos(np.radians(.1))) |
                                               (dots <= np.sin(np.radians(.1))), axis=1)))
    analytical_area = sum(surface_area[kind] for kind in ANALYTIC) / total
    irregular = mottled_length / sampled_length if sampled_length else None
    straight = straight_length / total_length if total_length else None
    components = {"surface": analytical_area, "large_face_area": 1 - small_area / total,
                  "edge_normals": 1 - irregular if irregular is not None else None,
                  "edge_curves": straight}
    weights = {"surface": .50, "large_face_area": .10, "edge_normals": .25, "edge_curves": .15}
    weighted = [(components[k], weight) for k, weight in weights.items() if components[k] is not None]
    score = 100 * sum(v * weight for v, weight in weighted) / sum(weight for _, weight in weighted)
    types = (*ANALYTIC, "bspline", "small_planar_triangle")
    area_shares = {k: surface_area[k] / total for k in dict.fromkeys((*types, *surface_area))}
    return {
        "faces": len(faces), "area_mm2": total, "surface_area_shares": area_shares,
        "small_faces": {"count": small_count, "count_share": small_count / len(faces),
                        "area_share": small_area / total, "threshold_mm2": small_threshold},
        "edges": {"count": len(incidence), "length_mm": total_length, "two_face_length_mm": paired_length,
                  "sampled_length_mm": sampled_length,
                  "normal_angle_length_shares": {k: bins[k] / sampled_length if sampled_length else None
                                                 for k in ("below_1", "1_to_15", "above_15")},
                  "line_or_circle_length_share": straight},
        "dimensions": {"axes": axes.tolist(), "planes": len(planes), "aligned_planes": aligned,
                       "cylinder_faces": len(radii), "round_radius_step_mm": .1,
                       "round_radius_epsilon_mm": .005,
                       "round_radius_faces": sum(abs(r - round(r, 1)) <= .005 for r in radii)},
        "scan_edge_median_mm": step, "tol_mm": tol, "score": round(score, 2),
        "score_components": components, "score_weights": weights,
        "score_scope": "Compare variants of the same part with the same inputs. Coverage is separate.",
        "seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--tol", type=float, default=.3)
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol must be positive and finite")
    started = perf_counter()
    step = None
    if args.scan:
        import trimesh

        lengths = trimesh.load(args.scan, force="mesh").edges_unique_length
        lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        if not len(lengths):
            parser.error("scan contains no positive mesh edges")
        step = float(np.median(lengths))
    shape = cq.importers.importStep(str(args.model)).val()
    result = quality(shape, step=step, tol=args.tol)
    result["total_seconds"] = perf_counter() - started
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
