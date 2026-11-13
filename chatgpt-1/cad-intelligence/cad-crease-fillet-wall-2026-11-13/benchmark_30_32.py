"""Reproducible controls for crease strips, exact blends and curved walls."""
import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

import crease_strips
import curved_walls
import idealize
import planar_brep
import planar_brep_cyl
from benchmark_features import meshed
from cad_quality import quality
from compare import compare
from nurbs import body_mesh
from straight_fuzz import fixture, plane_and_edge_errors, scan_mesh


def metrics(body, truth):
    left, right = (body_mesh(shape, .04) for shape in (truth, body))
    left.merge_vertices()
    right.merge_vertices()
    report = compare(left, right, .3, n=2000, seed=31)
    return {"score": quality(body)["score"], "coverage": [report["within_tol_share"], 1 - report["added_surface_share"]],
            **{k: v for k, v in plane_and_edge_errors(truth, body, 1.5).items() if k.startswith("sharp_")}}


def crease(work):
    rows = []
    for kind in ("angle", "ribbed"):
        truth, params = fixture(kind, 27)
        path = work / f"{kind}.step"
        cq.exporters.export(truth, str(path))
        for noise in (.05, .2):
            scan = scan_mesh(path, noise, 27)
            mesh, info = idealize.idealize(scan, noise=noise)
            v, f, label = planar_brep.collapse_degenerate(mesh.vertices, mesh.faces, info["label"], .001)
            regions = info["regions"]
            base, a = planar_brep.build(v, f, label, regions)
            audit = {}
            vv, ff, ll = crease_strips.resolve(v, f, label, regions, .3, audit=audit)
            result, b = planar_brep.build(vv, ff, ll, regions)
            row = {"case": kind, "seed": 27, "noise_mm": noise, "params": params, "classification": audit,
                   "free_area_mm2": [float(trimesh.triangles.area(pts[tri])[lab < 0].sum())
                                     for pts, tri, lab in ((v, f, label), (vv, ff, ll))],
                   "before": {**a, **metrics(base, truth)}, "after": {**b, **metrics(result, truth)}}
            rows.append(row)
            print(kind, noise, audit, row["free_area_mm2"], row["after"]["valid"], flush=True)
    # A box corner with a thin diagonal gap in the labels. The physical
    # gap is beveled by 0.12 mm and both rails lie on the existing supports.
    profile = [(-4, -4), (4, -4), (4, 3.88), (3.88, 4), (-4, 4)]
    rings = np.array([[x, y, z] for z in np.linspace(0, 14, 15) for x, y in profile])
    faces, labels = [], []
    for i in range(14):
        for j in range(5):
            a, b = i * 5 + j, i * 5 + (j + 1) % 5
            faces.extend(((a, b, a + 5), (b, b + 5, a + 5)))
            labels.extend([(-1 if j == 2 else j)] * 2)
    for start, label_value in ((0, 5), (70, 6)):
        for i in range(1, 4):
            faces.append((start, start + i, start + i + 1))
            labels.append(label_value)
    model = trimesh.Trimesh(rings, faces, process=False)
    trimesh.repair.fix_normals(model)
    labels = np.asarray(labels)
    regions = []
    for i in range(7):
        ids = labels == i
        normal = model.face_normals[ids][0] if ids.any() else np.array([1., 0., 0.])
        origin = model.triangles_center[ids][0] if ids.any() else np.zeros(3)
        regions.append({"kind": "plane", "normal": normal, "origin": origin})
    audit = {}
    vv, ff, ll = crease_strips.resolve(model.vertices, model.faces, labels, regions, .3, audit=audit)
    _, built = planar_brep.build(vv, ff, ll, regions)
    result_mesh = trimesh.Trimesh(vv, ff, process=False)
    rows.append({"case": "sharp_control", "noise_mm": 0, "classification": audit,
                 "remaining_free": int(np.count_nonzero(ll < 0)), "watertight": bool(result_mesh.is_watertight),
                 "after": built})
    print("sharp control", audit, built["valid"], result_mesh.is_watertight, flush=True)
    return rows


def records(mesh, label, types, truth_mesh):
    from scipy.optimize import least_squares

    rows = []
    for i, kind in enumerate(types):
        ids = np.flatnonzero(label == i)
        points = mesh.vertices[np.unique(mesh.faces[ids])]
        center = points.mean(axis=0)
        if kind == "Plane":
            n = np.average(truth_mesh.face_normals[ids], weights=truth_mesh.area_faces[ids], axis=0)
            n /= np.linalg.norm(n)
            rows.append({"kind": "plane", "normal": n, "origin": center})
        elif kind == "Cylinder":
            rows.append({"kind": "cylinder"})
        else:
            raise ValueError(kind)
    adjacent = {}
    for i, j in mesh.face_adjacency:
        a, b = int(label[i]), int(label[j])
        if a != b:
            adjacent.setdefault(a, set()).add(b)
            adjacent.setdefault(b, set()).add(a)
    for i, kind in enumerate(types):
        if kind != "Cylinder":
            continue
        candidates = sorted(adjacent[i])
        sides = [k for k in candidates if abs(rows[k]["normal"][2]) < .8]
        n = np.array([rows[k]["normal"] for k in sides])
        axis = np.cross(*n)
        axis /= np.linalg.norm(axis)
        offsets = [q @ rows[k]["origin"] for q, k in zip(n, sides)]
        corner = n.T @ np.linalg.solve(n @ n.T, offsets)
        shift = n.T @ np.linalg.solve(n @ n.T, np.ones(2))
        p = mesh.vertices[np.unique(mesh.faces[label == i])] - corner
        p -= (p @ axis)[:, None] * axis
        radial = lambda radius: np.linalg.norm(p + radius[0] * shift, axis=1) - radius[0]
        radius = float(least_squares(radial, [3.], bounds=(.5, 12.), loss="soft_l1", f_scale=.08).x[0])
        rows[i].update(origin=corner-radius*shift, axis=axis, r=radius, fillet_planes=sides)
    return rows


def blends(work):
    rows = []
    for kind in ("block", "oblique"):
        shape = cq.Workplane("XY").box(32, 24, 12, centered=(True, True, False))
        if kind == "block":
            shape = shape.edges("|Z").fillet(3)
        else:
            shape = shape.edges("|Z and >X").fillet(5).edges("|Z and <X and >Y").fillet(2)
            cutter = cq.Workplane("XZ").polyline([(-25, 10), (25, 7.5), (25, 22), (-25, 22)]).close().extrude(28, both=True)
            shape = shape.cut(cutter)
        truth = shape.val()
        path = work / f"{kind}.step"
        cq.exporters.export(truth, str(path))
        clean, label, kinds, _ = meshed(path, 1.5)
        for noise in (0., .1):
            mesh = clean.copy()
            jitter = np.clip(np.random.default_rng(31).normal(0, noise, len(mesh.vertices)), -2.2*noise, 2.2*noise)
            mesh.vertices += jitter[:, None] * mesh.vertex_normals
            regions = records(mesh, label, kinds, clean)
            # This isolates the exact builder after idealize with independent CAD-labelled boundaries.
            v = mesh.vertices.copy()
            plane_per_vertex = [set() for _ in v]
            for fi, k in enumerate(label):
                if kinds[k] == "Plane":
                    for idx in mesh.faces[fi]:
                        plane_per_vertex[idx].add(k)
            for idx, supports in enumerate(plane_per_vertex):
                if supports:
                    normal = np.array([regions[k]["normal"] for k in supports])
                    d = np.array([regions[k]["normal"] @ regions[k]["origin"] for k in supports])
                    v[idx] -= np.linalg.pinv(normal) @ (normal @ v[idx] - d)
            base, a = planar_brep.build(v, mesh.faces, label, regions)
            result, b = planar_brep_cyl.build(v, mesh.faces, label, regions)
            output = work / f"{kind}-{noise:g}-exact.step"
            cq.exporters.export(result, str(output))
            reread = cq.importers.importStep(str(output)).val()
            b["step_reopened_valid"] = bool(reread.isValid())
            b["step_cylinders"] = sum(face.geomType() == "CYLINDER" for face in reread.Faces())
            row = {"case": kind, "noise_mm": noise, "expected_fillets": kinds.count("Cylinder"),
                   "before": {**a, **metrics(base, truth)}, "after": {**b, **metrics(result, truth)}}
            if kind == "block" and noise == 0.:
                # With cap labels absent there is no supported section curve.
                cap_ids = [i for i, r in enumerate(regions) if r["kind"] == "plane" and abs(r["normal"][2]) > .99]
                bad_labels = np.where(np.isin(label, cap_ids), -1, label)
                _, bad = planar_brep_cyl.build(v, mesh.faces, bad_labels, regions)
                row["unlabelled_cap_control"] = bad
            rows.append(row)
            print(kind, noise, b, flush=True)
    return rows


def wall_fixture(radius, noise, polygon=False):
    width, height, panels, columns, levels = 50., 15., 5, 5, 9
    xs = np.linspace(-width / 2, width / 2, panels * columns + 1)
    sag = np.sqrt(radius**2 - xs**2) - np.sqrt(radius**2 - (width/2)**2)
    if polygon:
        sag = np.interp(xs, xs[::columns], sag[::columns])
    yz = np.column_stack([xs, sag])
    chain = np.r_[yz, [[xs[-1], -12.], [xs[0], -12.]]]
    n = len(chain)
    points = np.array([[x, y, z] for z in np.linspace(0, height, levels) for x, y in chain])
    triangles, label = [], []
    for i in range(levels-1):
        for j in range(n):
            a, b = i*n+j, i*n+(j+1)%n
            c, d = a+n, b+n
            triangles.extend(((a, c, b), (b, c, d)))
            k = j//columns if j < panels*columns else panels+j-panels*columns
            label.extend((k, k))
    # The section is convex. A center fan keeps every original boundary segment.
    for z, start, k in ((0., 0, panels+3), (height, (levels-1)*n, panels+4)):
        center = len(points)
        points = np.vstack((points, [0, -3., z]))
        for j in range(n):
            triangles.append((center, start+j, start+(j+1)%n))
            label.append(k)
    mesh = trimesh.Trimesh(points, triangles, process=False)
    trimesh.repair.fix_normals(mesh)
    clean = mesh.copy()
    label = np.asarray(label)
    ids = np.unique(mesh.faces[label < panels])
    jitter = np.clip(np.random.default_rng(32).normal(0, noise, len(ids)), -2.5 * noise, 2.5 * noise)
    mesh.vertices[ids, 1] += jitter
    rows = []
    for k in range(panels+5):
        q = clean.vertices[np.unique(clean.faces[label == k])]
        center = q.mean(axis=0)
        n = np.average(clean.face_normals[label == k], weights=clean.area_faces[label == k], axis=0)
        n /= np.linalg.norm(n)
        rows.append({"kind": "plane", "normal": n, "origin": center,
                     "area": float(mesh.area_faces[label == k].sum())})
    return mesh, {"label": label, "regions": rows}, panels


def walls(work):
    rows = []
    for radius in (40., 100., 200.):
        for noise in (.05, .2):
            mesh, seg, panels = wall_fixture(radius, noise)
            label, matches = curved_walls.unlabel_curved_walls(mesh, seg, .3, noise)
            polygon, poly_seg, _ = wall_fixture(radius, 0., polygon=True)
            _, false = curved_walls.unlabel_curved_walls(polygon, poly_seg, .3, 0.)
            noisy_poly, noisy_seg, _ = wall_fixture(radius, noise, polygon=True)
            _, false_noisy = curved_walls.unlabel_curved_walls(noisy_poly, noisy_seg, .3, noise)
            # The two variants project only the triangles assigned to a planar region.
            out = {}
            for key, lab in (("before", seg["label"]), ("after", label)):
                active = np.unique(lab[lab >= 0])
                remap = np.full(len(seg["regions"]), -1)
                remap[active] = np.arange(len(active))
                tags = np.where(lab >= 0, remap[lab], -1)
                regions = [seg["regions"][k] for k in active]
                moved, info = idealize.idealize(mesh, noise=noise, seg={"regions": regions, "label": tags}, fair_iters=0)
                v, f, tags = planar_brep.collapse_degenerate(moved.vertices, moved.faces, info["label"], .001)
                regions = info["regions"]
                if key == "after":
                    # Reuse the original group only when idealize preserved face indexing.
                    if len(f) == len(mesh.faces) and np.array_equal(f, mesh.faces):
                        tags, regions = curved_walls.cylindrical_regions(mesh, tags, regions, matches)
                    body, audit = planar_brep_cyl.build(v, f, tags, regions, smooth_tol=.3,
                                                        smooth_budget=5., smooth_edge_tol=.06, tol=max(.3, 2.8*noise))
                else:
                    body, audit = planar_brep.build(v, f, tags, regions, smooth_tol=.3, smooth_budget=5., smooth_edge_tol=.06)
                out[key] = {**audit, "score": quality(body)["score"]}
            row = {"radius_mm": radius, "noise_mm": noise, "false_polygon_groups": len(false),
                   "false_noisy_polygon_groups": len(false_noisy),
                   "curved_faces_unlabelled": int(np.count_nonzero((label < 0) & (seg["label"] < panels))),
                   "curved_triangles": int(np.count_nonzero(seg["label"] < panels)),
                   "true_planes_unlabelled": int(np.count_nonzero((label < 0) & (seg["label"] >= panels))),
                   "groups": [r["radius_mm"] for r in matches], **out}
            rows.append(row)
            print(radius, noise, row["curved_faces_unlabelled"], row["false_polygon_groups"],
                  out["before"]["score"], out["after"]["score"], out["after"]["valid"], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("30", "31", "32"))
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    rows = {"30": crease, "31": blends, "32": walls}[args.task](args.work)
    (args.work / f"task{args.task}.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.task == "30":
        assert rows[-1]["remaining_free"] == 0 and rows[-1]["watertight"]
        assert rows[-1]["classification"].get("sharp") == 1
    elif args.task == "31":
        assert all(row["after"]["step_reopened_valid"] and row["after"]["step_cylinders"] >= 3
                   and row["after"]["score"] > row["before"]["score"] for row in rows)
        control = rows[0]["unlabelled_cap_control"]
        assert control["valid"] and control["free_edges"] == 0 and control["exact_fillets"] == 0
    else:
        assert all(row["after"]["valid"] and row["after"]["free_edges"] == 0
                   and row["after"]["exact_curved_walls"] == 1 and row["true_planes_unlabelled"] == 0
                   and row["after"]["score"] > row["before"]["score"] for row in rows)


if __name__ == "__main__":
    main()
