"""Quality gates and opt-in diagnostics for region-aware smooth bodies."""
from __future__ import annotations

from collections import Counter, defaultdict

import cadquery as cq
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer


def bounded_distances(mesh, points, limit):
    """Distances to triangles, saturated at limit to bound the search."""
    values = np.full(len(points), limit, dtype=float)
    if not len(mesh.faces):
        return values
    tree, triangles = mesh.triangles_tree, mesh.triangles
    for i in range(0, len(points), 256):
        query = points[i:i + 256]
        low, high = query - limit, query + limit
        try:
            ids, counts = tree.intersection_v(low, high)
        except Exception:
            rows = [list(tree.intersection(box)) for box in np.column_stack((low, high))]
            ids, counts = np.concatenate(rows).astype(np.intp), list(map(len, rows))
        if len(ids):
            owner = np.repeat(np.arange(len(query)), np.asarray(counts, dtype=np.intp))
            origins = query[owner]
            nearest = trimesh.triangles.closest_point(triangles[ids], origins)
            np.minimum.at(values[i:i + len(query)], owner, np.linalg.norm(nearest - origins, axis=1))
    return values


def _samples(mesh, count, seed):
    return trimesh.sample.sample_surface(mesh, count, seed=seed)[0]


def _metrics(distances, tol):
    return {"cover": float(np.mean(distances <= tol)), "p95_mm": float(np.quantile(distances, .95))}


class LocalGate:
    def __init__(self, mesh, body_mesh, tol):
        self.mesh = mesh
        self.body_mesh = body_mesh
        self.tol = tol
        self.limit = 2 * tol
        self.probes = _samples(mesh, 30000, 19)
        self.tree = cKDTree(self.probes)

    def __call__(self, old_group, new_face):
        old, new = (self.body_mesh(cq.Shape.cast(shape), self.tol / 6)
                    for shape in (old_group, new_face))
        if not len(old.faces) or not len(new.faces):
            return {"accepted": False, "reason": "unmeshable_group"}
        low, high = old.bounds
        candidates = self.tree.query_ball_point((high + low) / 2, np.linalg.norm(high - low) / 2 + self.limit)
        points = self.probes[candidates]
        points = points[np.all((points >= low - self.limit) & (points <= high + self.limit), axis=1)]
        before = bounded_distances(old, points, self.limit)
        covered = before <= self.tol
        if np.count_nonzero(covered) < 8:
            return {"accepted": False, "reason": "insufficient_local_evidence"}
        lost = int(np.count_nonzero(bounded_distances(new, points[covered], self.limit) > self.tol))
        reverse = [_metrics(bounded_distances(self.mesh, _samples(shape, 256, 20), self.limit), self.tol)
                   for shape in (old, new)]
        accepted = lost == 0 and reverse[1]["cover"] >= reverse[0]["cover"]
        return {"accepted": accepted, "reason": None if accepted else "local_coverage",
                "lost": lost, "old_to_clean": reverse[0], "new_to_clean": reverse[1]}


class WholeBodyGate:
    def __init__(self, clean, scan, body_mesh, tol):
        self.scan = scan
        self.body_mesh = body_mesh
        self.tol = tol
        self.limit = 4 * tol
        self.clean_probes = _samples(clean, 12000, 19)
        self.scan_probes = _samples(scan, 8000, 20)

    def measure(self, body):
        return self.measure_mesh(self.body_mesh(body, min(.05, self.tol / 6)))

    def measure_mesh(self, mesh):
        if not len(mesh.faces):
            raise ValueError("unmeshable body")
        model_probes = _samples(mesh, 8000, 21)
        return {key: _metrics(bounded_distances(target, points, self.limit), self.tol)
                for key, target, points in (("clean_to_body", mesh, self.clean_probes),
                                             ("scan_to_body", mesh, self.scan_probes),
                                             ("body_to_scan", self.scan, model_probes))}

    def regressions(self, before, after):
        return [key for key in before if after[key]["cover"] < before[key]["cover"]
                or after[key]["p95_mm"] > before[key]["p95_mm"]
                or before[key]["p95_mm"] >= self.limit]


def max_edge_tolerance(wrapped):
    return max((BRep_Tool.Tolerance_s(e.wrapped) for e in cq.Shape.cast(wrapped).Edges()), default=0.0)


def edge_record(edge):
    shape = cq.Shape.cast(edge)
    vertices = [v.toTuple() for v in shape.Vertices()]
    return {"tolerance_mm": BRep_Tool.Tolerance_s(edge),
            "xyz_mm": np.mean(vertices, axis=0).tolist() if vertices else None}


def sewing_report(sew, limit=20):
    return {"free_edges": sew.NbFreeEdges(), "multiple_edges": sew.NbMultipleEdges(),
            "free_examples": [edge_record(sew.FreeEdge(i))
                              for i in range(1, min(sew.NbFreeEdges(), limit) + 1)],
            "multiple_examples": [edge_record(sew.MultipleEdge(i))
                                  for i in range(1, min(sew.NbMultipleEdges(), limit) + 1)]}


def _shape_report(wrapped, threshold, source_faces, tags):
    if wrapped.IsNull():
        return {"null_shape": True}
    shape = cq.Shape.cast(wrapped)
    faces = shape.Faces()
    tag_by_face = {cq.Face(f): (i, int(t)) for i, (f, t) in enumerate(zip(source_faces, tags))}
    analyzer = BRepCheck_Analyzer(wrapped, True)
    edges = defaultdict(list)
    bad = []
    for i, face in enumerate(faces):
        context = tag_by_face.get(face)
        label = {"face": i, "kind": face.geomType()}
        if context:
            label.update(patch=context[0], region_tag=context[1])
        for edge in face.Edges():
            edges[edge].append(label)
        if not analyzer.IsValid(face.wrapped):
            status = [x.name for x in analyzer.Result(face.wrapped).Status()]
            subshape_errors = []
            for child in [*face.Wires(), *face.Edges()]:
                check = analyzer.Result(child.wrapped)
                statuses = check.StatusOnShape(face.wrapped) if check.IsStatusOnShape(face.wrapped) else check.Status()
                errors = [x.name for x in statuses if x.name != "BRepCheck_NoError"]
                if errors:
                    subshape_errors.append({"kind": child.ShapeType(), "statuses": errors,
                                            "xyz_mm": child.Center().toTuple()})
            bad.append({**label, "statuses": status, "xyz_mm": face.Center().toTuple(),
                        "subshape_errors": subshape_errors[:12]})
    over = [{"edge": i, **edge_record(edge.wrapped), "owners": owners}
            for i, (edge, owners) in enumerate(edges.items())
            if BRep_Tool.Tolerance_s(edge.wrapped) > threshold]
    over.sort(key=lambda row: -row["tolerance_mm"])
    unshared = [{"edge": i, **edge_record(edge.wrapped), "owners": owners}
                for i, (edge, owners) in enumerate(edges.items())
                if len(owners) == 1 and not BRep_Tool.Degenerated_s(edge.wrapped)
                and not BRep_Tool.IsClosed_s(edge.wrapped, faces[owners[0]["face"]].wrapped)]
    return {"valid": bool(analyzer.IsValid()), "faces": len(faces), "shells": len(shape.Shells()),
            "closed_shells": sum(BRep_Tool.IsClosed_s(s.wrapped) for s in shape.Shells()),
            "face_types": dict(Counter(f.geomType() for f in faces)), "invalid_faces": len(bad),
            "invalid_examples": bad[:20], "over_tolerance_edges": len(over), "over_examples": over[:20],
            "unshared_edges": len(unshared), "unshared_examples": unshared[:20]}


def shape_report(wrapped, threshold, source_faces=(), tags=()):
    try:
        return _shape_report(wrapped, threshold, source_faces, tags)
    except (RuntimeError, TypeError, ValueError) as error:
        return {"diagnostic_error": f"{type(error).__name__}: {error}"}


def guarded_smooth(nurbs, mesh, scan, budget, tol):
    """Use the unlabelled cascade as a reference and cap the regional stage."""
    from time import perf_counter

    start = perf_counter()
    old = nurbs._NO_REGIONS
    nurbs._NO_REGIONS = True
    try:
        base, info = nurbs._smooth_body(mesh, scan, budget, tol)
    finally:
        nurbs._NO_REGIONS = old
    remaining = budget - (perf_counter() - start)
    if remaining <= 0:
        return base, dict(info, region_guard="budget_exhausted")
    if base is None:
        candidate, report = nurbs._smooth_body(mesh, scan, remaining, tol)
        return candidate, dict(report, region_guard="no_unlabelled_body",
                               stages=info["stages"] + report["stages"], sec=round(perf_counter() - start))
    gate = WholeBodyGate(mesh, scan, nurbs.body_mesh, min(tol, .5))
    before = None

    def accept(body, stage):
        nonlocal before
        if stage["faces"] > info["faces"]:
            stage["region_guard_reason"] = "more_faces"
            return False
        try:
            if before is None:
                before = gate.measure(base)
            after = gate.measure(body)
            regressions = gate.regressions(before, after)
        except (RuntimeError, ValueError) as error:
            stage["region_guard_reason"] = f"comparison_failed: {error}"
            return False
        stage["region_guard_comparison"] = {"baseline": before, "candidate": after, "regressions": regressions}
        return not regressions

    try:
        candidate, report = nurbs._smooth_body(mesh, scan, remaining, tol, max_step=info["step"],
                                               accept_region=accept)
    except Exception as error:
        return base, dict(info, sec=round(perf_counter() - start),
                          region_guard="regional_trial_failed", region_guard_error=f"{type(error).__name__}: {error}")
    result = report if candidate is not None else info
    for stage in info["stages"]:
        stage["region_guard_baseline"] = True
    return (candidate if candidate is not None else base), dict(
        result, stages=info["stages"] + report["stages"], sec=round(perf_counter() - start),
        region_guard="accepted" if candidate is not None else "fallback_unlabelled")
