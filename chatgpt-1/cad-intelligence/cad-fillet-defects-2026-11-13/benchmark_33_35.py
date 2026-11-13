"""Run the CAD and defect controls against the current CAD-demo checkout."""
import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
import trimesh

import crease_strips
import defects
import idealize
import planar_brep
import planar_brep_cyl
from benchmark_30_32 import metrics, records
from benchmark_features import meshed
from straight_fuzz import fixture, scan_mesh


def old_builder():
    path = Path(planar_brep.__file__).with_name("planar_brep_cyl.py")
    spec = importlib.util.spec_from_file_location("compass_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def build(*args, **kwargs):
        try:
            return module.build(*args, **kwargs)
        except UnboundLocalError as exc:
            if "'why'" not in str(exc) or not kwargs.get("smooth_tol"):
                raise
            return module.build(*args, **{**kwargs, "smooth_tol": None})
    return build


def cad_cases(work, task, cases=None):
    from straight_plus import augment

    old = old_builder()
    options = dict(smooth_tol=.3, smooth_edge_tol=.075, smooth_budget=8.)
    rows = []
    labelled = ("block", "free_cap", "oblique", "mixed")
    for kind in cases or ((*labelled, "angle", "ribbed") if task == "33" else ("angle", "ribbed")):
        if kind in labelled:
            part = cq.Workplane("XY").box(24, 18, 10, centered=(True, True, False))
            if kind == "oblique":
                part = part.edges("|Z and >X").fillet(5).edges("|Z and <X and >Y").fillet(2)
                cutter = cq.Workplane("XZ").polyline([(-20, 10), (20, 7), (20, 16), (-20, 16)]).close().extrude(24, both=True)
                part = part.cut(cutter)
            else:
                part = part.edges("|Z").fillet(3)
            truth = part.val()
        else:
            truth, _ = fixture(kind, 27)
        path = work / f"{kind}.step"
        cq.exporters.export(truth, str(path))
        for noise in ((0., .1, .2) if task == "33" else (.1, .2)):
            if kind in labelled:
                clean, labels, kinds, _ = meshed(path, 1.5)
                mesh = clean.copy()
                if noise:
                    offset = np.clip(np.random.default_rng(33).normal(0, noise, len(mesh.vertices)), -.23, .23)
                    mesh.vertices += offset[:, None] * mesh.vertex_normals
                labels = labels.copy()
                regions = records(mesh, labels, kinds, clean)
                V = mesh.vertices.copy()
                F = mesh.faces
                caps = [i for i, row in enumerate(regions) if row["kind"] == "plane" and abs(row["normal"][2]) > .9]
                cap = max(caps, key=lambda i: regions[i]["origin"][2])
                for i in range(len(V)):
                    ids = np.unique(labels[np.any(F == i, axis=1)])
                    planes = [regions[k] for k in ids if regions[k]["kind"] == "plane"]
                    if planes:
                        n = np.array([row["normal"] for row in planes])
                        d = [row["normal"] @ row["origin"] for row in planes]
                        V[i] -= np.linalg.pinv(n) @ (n @ V[i] - d)
                if kind == "free_cap":
                    from patch_faces import topology
                    cap_faces = F[labels == cap]
                    edges = topology(cap_faces)
                    boundary = np.unique([edge for edge, adjacent in edges.items() if len(adjacent) == 1])
                    inner = np.setdiff1d(np.unique(cap_faces), boundary)
                    x, y = V[inner, :2].T
                    V[inner, 2] += .18 * np.maximum(0, 1-(x/12)**2) * np.maximum(0, 1-(y/9)**2)
                    labels[labels == cap] = -1
                if kind == "mixed":
                    # An independent interior free patch with a planar perimeter.
                    local = (labels == cap) & (np.linalg.norm(mesh.triangles_center[:, :2], axis=1) < 4.)
                    labels[local] = -1
                    from patch_faces import topology
                    uses = topology(F[local])
                    boundary = np.unique([edge for edge, sides in uses.items() if len(sides) == 1])
                    ids = np.setdiff1d(np.unique(F[local]), boundary)
                    radius = np.linalg.norm(V[ids, :2], axis=1)
                    V[ids, 2] += .22 * np.maximum(0, 1 - (radius / 4)**2)
            else:
                scan = scan_mesh(path, noise, 27)
                seg = augment(scan, .3, noise)[0] if task == "33" else None
                mesh, info = idealize.idealize(scan, noise=noise, seg=seg)
                V, F, labels = planar_brep.collapse_degenerate(mesh.vertices, mesh.faces, info["label"], .001)
                regions = info["regions"]
                for row in regions:
                    if "fillet_planes" in row:
                        pair = []
                        for i in row["fillet_planes"]:
                            while i >= 0 and regions[i]["kind"] == "merged":
                                i = regions[i]["into"]
                            pair.append(i)
                        if min(pair) < 0 or len(set(pair)) != 2:
                            row.pop("fillet_planes")
                        else:
                            row["fillet_planes"] = pair
            if task == "33":
                configs = [("smooth", planar_brep_cyl.build_stock, V, F, labels),
                           ("old_exact", old, V, F, labels),
                           ("new", planar_brep_cyl.build, V, F, labels)]
                audit = {}
            else:
                audit = {}
                VV, FF, LL = crease_strips.resolve(V, F, labels, regions, .3, audit=audit)
                audit["free_area_mm2"] = [float(trimesh.triangles.area(v[f])[l < 0].sum())
                                          for v, f, l in ((V, F, labels), (VV, FF, LL))]
                configs = [("base", planar_brep.build, V, F, labels), ("new", planar_brep.build, VV, FF, LL)]
            row = {"kind": kind, "noise_mm": noise, "audit": audit}
            for name, build, v, f, l in configs:
                body, report = build(v, f, l, regions, **(options if task == "33" else {}))
                output = work / f"{kind}-{noise:g}-{name}.step"
                cq.exporters.export(body, str(output))
                restored = cq.importers.importStep(str(output)).val()
                faces = restored.Faces()
                curved_area = sum(face.Area() for face in faces if face.geomType() != "PLANE")
                row[name] = {**report, "step_valid": bool(restored.isValid()),
                             "step_cylinders": sum(face.geomType() == "CYLINDER" for face in faces),
                             "curved_step_area_share": curved_area / restored.Area(),
                             **metrics(restored, truth)}
            rows.append(row)
            print(kind, noise, {k: (v.get("exact_fillets"), v.get("smooth_faces"), v.get("coverage")) for k, v in row.items()
                                if isinstance(v, dict) and k != "audit"}, audit, flush=True)
    return rows


def fragmented_control():
    profile = [(-4, -4), (4, -4), (4, 3.88), (3.88, 4), (-4, 4)]
    vertices = np.array([[x, y, z] for z in range(15) for x, y in profile], dtype=float)
    faces, labels = [], []
    for i in range(14):
        for j in range(5):
            a, b = i * 5 + j, i * 5 + (j + 1) % 5
            faces.extend(((a, b, a + 5), (b, b + 5, a + 5)))
            k = -1 if j == 2 else j + (7 if i % 2 and j in (1, 3) else 0)
            labels.extend([k] * 2)
    for start, k in ((0, 5), (70, 6)):
        for i in range(1, 4):
            faces.append((start, start + i, start + i + 1))
            labels.append(k)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(mesh)
    label = np.array(labels)
    regions = [{"kind": "unused"} for _ in range(11)]
    for k in np.unique(label[label >= 0]):
        face = np.flatnonzero(label == k)[0]
        regions[k] = {"kind": "plane", "normal": mesh.face_normals[face], "origin": mesh.triangles_center[face]}
    results = {}
    for mode in (False, True):
        audit = {}
        regs = [dict(r) for r in regions]
        v, f, tags = crease_strips.resolve(mesh.vertices, mesh.faces, label, regs, .3, audit=audit, merge_supports=mode)
        _, built = planar_brep.build(v, f, tags, regs)
        results[str(mode)] = {"free_triangles": int(np.count_nonzero(tags < 0)), **audit, **built}
    return {"kind": "fragmented_control", **results}


def defect_cases(work):
    rows = []
    for name in ("ribbed", "angle"):
        base, _ = fixture(name, 27)
        # Add a flat calibration flange in front, welded to each CAD part.
        flange = cq.Workplane("XY").box(48, 30, 4, centered=(True, True, False)).translate((0, -24, 0)).val()
        base = base.fuse(flange).clean()
        extra = cq.Workplane("XY").box(5, 5, 3, centered=(True, True, False)).translate((17, -31, 4)).val()
        truth = base.fuse(extra).clean()
        pocket = cq.Workplane("XY").workplane(offset=2).center(1, -31).circle(2.8).extrude(3).val()
        model = base.cut(pocket).clean()
        geometries = {}
        for label, shape in (("scan", truth), ("model", model)):
            path = work / f"{name}-{label}.step"
            cq.exporters.export(shape, str(path))
            mesh = scan_mesh(path, 0., 27)
            geometries[label] = mesh
        scan = geometries["scan"]
        model = geometries["model"]
        # Split enough times to sample the local height profiles and sharp folds.
        model = model.subdivide().subdivide()
        p = model.vertices
        top = np.abs(p[:, 2] - 4) < 1e-7
        distance = np.linalg.norm(p[:, :2] - [-15, -28], axis=1)
        p[top, 2] += np.maximum(0, 1 - (distance[top] / 4.2)**2)**2
        q = p[:, :2] - [0, -20]
        band = np.maximum(0., 1 - np.abs(q[:, 0]) / 4) * np.maximum(0., 1 - np.abs(q[:, 1]) / 2.8)
        p[top, 2] += .55 * band[top] * np.cos(9 * q[top, 0])
        started = perf_counter()
        report = defects.analyze(model, scan)
        elapsed = perf_counter() - started
        clean = defects.analyze(scan, scan)
        targets = {"выступ над сканом": [-15,-28,4.8], "вырез в материале": [1,-31,2.5],
                   "смятая поверхность": [0,-20,4.], "потерянный элемент": [17,-31,6.]}
        hits = {k: min([float(np.linalg.norm(np.asarray(r["center"])-p)) for r in report if r["type"] == k], default=None)
                for k,p in targets.items()}
        defects.render(model, scan, report, work / f"{name}.png")
        model.export(work / f"{name}-defects.stl")
        scan.export(work / f"{name}-scan.stl")
        row = {"kind": name, "faces": len(model.faces), "seconds": elapsed,
               "clean_findings": len(clean), "center_errors_mm": hits, "reports": defects._public(report)}
        rows.append(row)
        print(name, row["faces"], round(elapsed,2), hits, len(clean), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("33", "34", "35"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    rows = defect_cases(args.work) if args.task == "35" else cad_cases(args.work, args.task, args.case)
    if args.task == "34":
        rows.append(fragmented_control())
    (args.work / f"task{args.task}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                                     encoding="utf-8")
    if args.task == "35":
        assert all(row["clean_findings"] <= 1 and row["seconds"] < 60
                   and all(error is not None and error < 2 for error in row["center_errors_mm"].values())
                   for row in rows)


if __name__ == "__main__":
    main()
