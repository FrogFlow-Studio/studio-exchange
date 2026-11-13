"""Locally undo an unsupported geometric projection around a detected fold."""
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from defects import analyze
from nurbs import body_mesh


def _rings(mask, pairs, count):
    out = mask.copy()
    for _ in range(count):
        selected = out[pairs].any(axis=1)
        out[pairs[selected].ravel()] = True
    return out


def repair(V, F, label, regions, scan, defects, tol):
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
    step = float(np.median(scan.edges_unique_length))
    centers = cKDTree(mesh.triangles_center)
    pairs = mesh.face_adjacency
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
        halo = _rings(core, pairs, 2 if radius > 2 * step else 1)
        release = halo & (label < 0)
        for tag in np.unique(label[halo & (label >= 0)]):
            record = regions[tag]
            if record["kind"] not in ("plane", "cylinder"):
                continue
            own = label == tag
            total = mesh.area_faces[own].sum()
            inside = mesh.area_faces[own & halo].sum()
            # A thin strip can run through several separate detected spots.
            # Require both local support and a small transverse width.
            p = V[np.unique(F[own])]
            _, _, axes = np.linalg.svd(p-p.mean(axis=0), full_matrices=False)
            width = np.ptp((p-p.mean(axis=0)) @ axes[1]) if len(axes) > 1 else 0.
            if total and (inside / total >= .55 or (inside > 0 and width <= 2.8 * step)):
                release |= own & halo
        release = _rings(release, pairs, 1) & (halo | (label < 0))
        inner = np.setdiff1d(np.unique(F[release]), np.unique(F[~release]))
        if not len(inner):
            stats["no_movable_vertices"] += 1
            continue
        target, dist, _ = trimesh.proximity.closest_point(scan, V[inner])
        delta = target - points[inner]
        allowed = dist <= max(3*tol, step)
        inner, delta = inner[allowed], delta[allowed]
        if not len(inner):
            stats["no_scan_support"] += 1
            continue
        edge_distance = cKDTree(mesh.triangles_center[~release]).query(V[inner])[0] if (~release).any() else np.full(len(inner), step)
        weight = np.clip(edge_distance / max(step, tol), .35, 1.)
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
                label[release] = -1
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


def repair_build(V, F, label, regions, scan, tol=.3, *, build=None):
    """Run at most two repair attempts, retaining the last accepted body."""
    if build is None:
        import planar_brep

        build = planar_brep.build
    from compare import compare

    def evaluate(state):
        try:
            body, audit = build(*state, smooth_tol=tol, smooth_edge_tol=tol/4)
        except UnboundLocalError as exc:
            if "'why'" not in str(exc):
                raise
            body, audit = build(*state)
        model = body_mesh(body, min(.05, tol/5))
        model.merge_vertices()
        coverage = compare(scan, model, tol, n=3000, seed=36)
        return body, {**audit, "coverage": [coverage["within_tol_share"],
                                          1-coverage["added_surface_share"]]}, analyze(model, scan, tol)

    def area(findings):
        return sum(row["area_mm2"] for row in findings if row["type"] == "смятая поверхность")

    state = (V, F, label, regions)
    body, audit, findings = evaluate(state)
    baseline_coverage = audit["coverage"]
    passes = []
    for _ in range(2):
        if not area(findings):
            break
        v, f, l, regs, change = repair(*state, scan, findings, tol)
        if not change.get("repaired_zones"):
            passes.append({**change, "accepted": False})
            break
        try:
            candidate, check, updated = evaluate((v, f, l, regs))
        except (RuntimeError, ValueError) as error:
            passes.append({**change, "accepted": False, "error": str(error)})
            break
        # A decrease in fold area cannot justify creating a material defect.
        material = {"выступ над сканом", "вырез в материале"}
        old = [row for row in findings if row["type"] in material]
        new = [row for row in updated if row["type"] in material]
        unchanged = all(any(row["type"] == previous["type"] and
                            np.linalg.norm(np.subtract(row["center"], previous["center"])) <= 2*tol and
                            row["area_mm2"] <= previous["area_mm2"] for previous in old) for row in new)
        coverage_ok = all(a >= b-.002-1e-9 for a, b in zip(check["coverage"], baseline_coverage))
        accepted = check["valid"] and check["free_edges"] == 0 and area(updated) < area(findings) and unchanged and coverage_ok
        passes.append({**change, "accepted": bool(accepted), "fold_area_mm2": [area(findings), area(updated)],
                       "coverage": check["coverage"], "coverage_ok": coverage_ok})
        if not accepted:
            break
        state, body, audit, findings = (v, f, l, regs), candidate, check, updated
    return body, {**audit, "repair_passes": passes, "baseline_coverage": baseline_coverage,
                  "defects": [{k: value for k, value in row.items() if not k.startswith("_")} for row in findings]}
