"""Locally undo an unsupported geometric projection around a detected fold."""
from collections import Counter
from functools import partial
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh

from defects import analyze
from nurbs import body_mesh


class ScanProjection:
    """Reuse the triangle locator for region support and all repair candidates."""

    def __init__(self, scan):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator

        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(scan.vertices), deep=True))
        cells = vtkCellArray()
        cells.SetData(3, numpy_to_vtkIdTypeArray(np.ascontiguousarray(scan.faces, dtype=np.int64).ravel(), deep=True))
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(poly)
        self.locator.BuildLocator()

    def __call__(self, points):
        from vtkmodules.vtkCommonCore import reference

        nearest = np.empty_like(points)
        distance = np.empty(len(points))
        target = [0., 0., 0.]
        cell, sub, squared = reference(0), reference(0), reference(0.)
        for i, point in enumerate(points):
            self.locator.FindClosestPoint(point, target, cell, sub, squared)
            nearest[i] = target
            distance[i] = squared.get()
        return nearest, np.sqrt(distance)


def _rings(mask, pairs, count):
    out = mask.copy()
    for _ in range(count):
        selected = out[pairs].any(axis=1)
        out[pairs[selected].ravel()] = True
    return out


def repair(V, F, label, regions, scan, defects, tol, *, rings=3, project=None):
    """Restore a bounded neighborhood. All inputs remain available for rollback."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    label = np.asarray(label, dtype=int).copy()
    points = V.copy()
    regions = [dict(row) for row in regions]
    mesh = trimesh.Trimesh(V, F, process=False)
    stats = Counter()
    if not len(F) or not len(scan.faces) or tol <= 0:
        return points, F, label, regions, dict(stats)
    if project is None:
        project = ScanProjection(scan)
    step = float(np.median(scan.edges_unique_length))
    centers = cKDTree(mesh.triangles_center)
    pairs = mesh.face_adjacency
    area = mesh.area_faces
    counts = np.bincount(label[label >= 0], weights=area[label >= 0], minlength=len(regions))
    supported = np.zeros(len(regions), dtype=bool)
    for tag in np.unique(label[label >= 0]):
        record = regions[tag]
        if record["kind"] not in ("plane", "cylinder"):
            continue
        own = label == tag
        p = V[np.unique(F[own])]
        axes = np.linalg.svd(p-p.mean(axis=0), full_matrices=False)[2]
        width = np.ptp((p-p.mean(axis=0)) @ axes[1])
        # Test evidence on the original scan, before the projection being repaired.
        probes = p[np.linspace(0, len(p)-1, min(128,len(p)), dtype=int)]
        source, _ = project(probes)
        if record["kind"] == "plane":
            residual = np.abs((source-record["origin"]) @ np.asarray(record["normal"]))
            supported[tag] = width > 3.2*step and counts[tag] > 12*step**2 and np.quantile(residual,.9) <= tol
        else:
            relative = source - record["origin"]
            axis = np.asarray(record["axis"])
            radial = relative - (relative @ axis)[:,None]*axis
            supported[tag] = width > 3.2*step and np.quantile(abs(np.linalg.norm(radial,axis=1)-record["r"]),.95) < tol/2
    for defect in defects:
        if defect["type"] != "смятая поверхность":
            continue
        radius = np.sqrt(defect["area_mm2"] / np.pi)
        # The public report can be transferred between tessellations by center/area.
        ids = centers.query_ball_point(defect["center"], max(step, 1.55 * radius))
        if not ids:
            stats["unmapped"] += 1
            continue
        core = np.zeros(len(F), dtype=bool)
        core[ids] = True
        anchored = np.zeros(len(F), dtype=bool)
        assigned = label >= 0
        anchored[assigned] = supported[label[assigned]]
        release = core & ~anchored
        for _ in range(rings):
            grown = _rings(release, pairs, 1) & ~anchored
            if np.array_equal(grown, release):
                break
            release = grown
            stats["expansion_rings"] += 1
        outer = _rings(release, pairs, 1)
        frozen = np.unique(F[anchored | ~outer])
        inner = np.setdiff1d(np.unique(F[release]), frozen)
        if not len(inner):
            stats["no_movable_vertices"] += 1
            continue
        target, dist = project(V[inner])
        delta = target - points[inner]
        allowed = dist <= max(3*tol, step)
        inner, delta = inner[allowed], delta[allowed]
        if not len(inner):
            stats["no_scan_support"] += 1
            continue
        edges = mesh.edges_unique
        edges = edges[np.isin(edges, np.unique(F[outer])).all(axis=1)]
        lengths = np.linalg.norm(V[edges[:,0]]-V[edges[:,1]], axis=1)
        graph = coo_matrix((lengths, edges.T), shape=(len(V),len(V))).tocsr()
        seeds = np.unique(F[core & release])
        distance_from_center = dijkstra(graph,directed=False,indices=seeds,min_only=True)[inner]
        scaled = np.clip(distance_from_center/max(3*step,tol),0.,1.)
        weight = .15+.85*(1.-scaled**2*(3.-2.*scaled))
        changed = np.isin(F, inner).any(axis=1)
        old = points[F[changed]]
        normal = np.cross(old[:, 1]-old[:, 0], old[:, 2]-old[:, 0])
        for scale in (1., .5, .25, .125):
            candidate = points.copy()
            candidate[inner] += delta * (weight * scale)[:, None]
            new = candidate[F[changed]]
            cross = np.cross(new[:, 1]-new[:, 0], new[:, 2]-new[:, 0])
            if np.all(np.einsum("ij,ij->i", cross, normal) > .1 * np.sum(normal**2, axis=1)):
                points = candidate
                label[release | changed] = -1
                stats["repaired_zones"] += 1
                stats["moved_vertices"] += len(inner)
                break
        else:
            stats["orientation_rejected"] += 1
    area = trimesh.triangles.area(points[F])
    for i, record in enumerate(regions):
        own = np.flatnonzero(label == i)
        record.update(faces=own, area=float(area[own].sum()))
    return points, F, label, regions, dict(stats)


def repair_build(V, F, label, regions, scan, tol=.3, *, build=None, overhead_limit=None, progress=None):
    """Run at most two repair attempts, retaining the last accepted body."""
    if build is None:
        import planar_brep

        build = planar_brep.build
    from compare import compare
    from repair import repair as local_repair

    def evaluate(state, budget=None):
        start = perf_counter()
        settings = {} if budget is None else {"smooth_budget": budget}
        try:
            body, audit = build(*state, smooth_tol=tol, smooth_edge_tol=tol/4, **settings)
        except UnboundLocalError as exc:
            if "'why'" not in str(exc):
                raise
            body, audit = build(*state)
        build_seconds = perf_counter()-start
        model = body_mesh(body, min(.05, tol/5))
        model.merge_vertices()
        coverage = compare(scan, model, tol, n=3000, seed=36)
        findings = analyze(model, scan, tol)
        return body, {**audit, "build_seconds":build_seconds, "evaluation_seconds":perf_counter()-start,
                      "coverage": [coverage["within_tol_share"],1-coverage["added_surface_share"]]}, findings, coverage

    def area(findings):
        return sum(row["area_mm2"] for row in findings if row["type"] == "смятая поверхность")

    state = (V, F, label, regions)
    body, audit, findings, _ = evaluate(state)
    baseline_coverage = audit["coverage"]
    baseline_seconds = audit["build_seconds"]
    baseline_fold_area = area(findings)
    baseline_lost_area = sum(row["area_mm2"] for row in findings if row["type"] == "потерянный элемент")
    if progress is not None:
        progress("baseline", audit["build_seconds"], baseline_fold_area)
    started = perf_counter()
    project = ScanProjection(scan)
    geodesic = partial(repair, project=project)
    passes = []
    for _ in range(2):
        if not area(findings):
            break
        accepted = []
        attempts = []
        material = {"выступ над сканом", "вырез в материале", "потерянный элемент"}
        old = [row for row in findings if row["type"] in material]
        methods = (("expanded", geodesic), ("local", local_repair))
        if not passes and overhead_limit is None:
            methods += (("two_rings", partial(geodesic,rings=2)),)
        for name, method in methods:
            if overhead_limit is not None and perf_counter()-started >= baseline_seconds*overhead_limit:
                break
            v, f, l, regs, change = method(*state, scan, findings, tol)
            attempt = {**change, "method": name, "accepted": False}
            attempts.append(attempt)
            if progress is not None:
                progress(name, perf_counter()-started, change.get("repaired_zones", 0))
            if not change.get("repaired_zones"):
                continue
            budget = None if overhead_limit is None else baseline_seconds*overhead_limit-(perf_counter()-started)
            if budget is not None and budget <= 0:
                attempt["reason"] = "budget_exhausted"
                break
            try:
                candidate, check, updated, detail = evaluate((v, f, l, regs), budget)
            except (RuntimeError, ValueError) as error:
                attempt["error"] = str(error)
                continue
            new = [row for row in updated if row["type"] in material]
            material_growth = [row for row in new if not any(row["type"] == previous["type"] and
                                np.linalg.norm(np.subtract(row["center"], previous["center"])) <= 2*tol and
                                row["area_mm2"] <= previous["area_mm2"]+1e-9 for previous in old)]
            unchanged = not material_growth
            coverage_ok = all(a >= b-.002-1e-9 for a, b in zip(check["coverage"], baseline_coverage))
            time_ok = overhead_limit is None or perf_counter()-started <= baseline_seconds*overhead_limit
            valid = (check["valid"] and check["free_edges"] == 0 and area(updated) < area(findings)
                     and unchanged and coverage_ok and time_ok)
            attempt.update(accepted=bool(valid), fold_area_mm2=[area(findings),area(updated)],
                           coverage=check["coverage"],coverage_ok=coverage_ok,
                           material_ok=unchanged, time_ok=time_ok, valid_body=bool(check["valid"]),
                           build_seconds=check["build_seconds"], evaluation_seconds=check["evaluation_seconds"],
                           free_edges=int(check["free_edges"]),
                           material_growth=[{k:r[k] for k in ("type","area_mm2","center")} for r in material_growth])
            if not coverage_ok:
                attempt["deviation_clusters"] = {k:detail[k] for k in
                    ("clusters_outside","clusters_inside","clusters_added")}
            if valid:
                accepted.append((attempt,(v,f,l,regs),candidate,check,updated))
                if overhead_limit is not None:
                    break
        passes.append({"attempts":attempts})
        if not accepted:
            break
        selected, state, body, audit, findings = min(accepted,key=lambda row:area(row[-1]))
        passes[-1]["selected"] = selected["method"]
    return body, {**audit, "repair_passes": passes, "baseline_coverage": baseline_coverage,
                  "baseline_build_seconds":baseline_seconds, "baseline_fold_area_mm2":baseline_fold_area,
                  "baseline_lost_area_mm2":baseline_lost_area,
                  "repair_overhead_seconds":perf_counter()-started,
                  "defects": [{k: value for k, value in row.items() if not k.startswith("_")} for row in findings]}
