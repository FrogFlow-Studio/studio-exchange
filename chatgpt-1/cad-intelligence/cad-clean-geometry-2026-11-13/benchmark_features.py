"""Independent CAD-labelled meshes for the narrow-plane and fillet detectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import gmsh
import numpy as np
import trimesh


def _edge(body, x, z):
    return min(body.Edges(), key=lambda e: abs(e.Center().x - x) + abs(e.Center().z - z)
               + (0 if abs(e.startPoint().y - e.endPoint().y) > 10 else 100))


def fixture(which, step):
    if which == "narrow":
        width = 48.
        stock = cq.Workplane("XY").box(width, 40, 6, centered=(True, True, False)).val()
        stock = stock.fillet(3., [_edge(stock, width / 2, 6.)])
        walls = []
        for x, count in zip((-14., 0., 14.), (1, 2, 3)):
            thickness = count * step
            rib = cq.Workplane("XY").box(thickness, 30., 5., centered=(True, True, False)).translate((x, 0, 6)).val()
            stock = stock.fuse(rib)
            walls.extend((x - thickness / 2, x + thickness / 2))
        return stock.clean(), walls

    solids = []
    for i, radius in enumerate((1., 2., 5., None, 3.)):
        if i == 4:
            lower = cq.Workplane("XY").box(26, 26, 9, centered=(True, True, False))
            upper = cq.Workplane("XY").box(13, 26, 9, centered=(False, True, False)).translate((0, 0, 9))
            body = lower.union(upper).val()
            body = body.fillet(radius, [_edge(body, 0., 9.)])
        else:
            body = cq.Workplane("XY").box(26, 26, 18, centered=(True, True, False)).val()
            corner = [_edge(body, 13., 18.)]
            body = body.chamfer(2., None, corner) if radius is None else body.fillet(radius, corner)
        solids.append(body.translate((i * 34., 0, 0)))
    return cq.Compound.makeCompound(solids), []


def meshed(path, step):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", step)
        gmsh.option.setNumber("Mesh.MeshSizeMax", step)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        faces, labels, kinds, centers = [], [], [], []
        for i, (_, tag) in enumerate(gmsh.model.getEntities(2)):
            _, nodes = gmsh.model.mesh.getElementsByType(2, tag)
            tris = order[np.searchsorted(tags[order], nodes)].reshape(-1, 3)
            faces.extend(tris)
            labels.extend([i] * len(tris))
            kinds.append(gmsh.model.getType(2, tag))
            centers.append(gmsh.model.occ.getCenterOfMass(2, tag))
    finally:
        gmsh.finalize()
    mesh = trimesh.Trimesh(xyz.reshape(-1, 3), faces, process=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh, np.array(labels), kinds, centers


def anchors(clean, mesh, truth, kinds, centers, walls):
    label = np.full(len(truth), -1, dtype=int)
    records = []
    ribs = set()
    for i, (kind, center) in enumerate(zip(kinds, centers)):
        ids = np.flatnonzero(truth == i)
        if kind != "Plane" or not len(ids):
            continue
        n = np.average(clean.face_normals[ids], weights=clean.area_faces[ids], axis=0)
        if abs(n[0]) > .99 and any(abs(center[0] - x) < .001 for x in walls):
            ribs.add(i)
            continue
        # Leave the intentionally diagonal chamfer unlabelled.
        if not walls and np.count_nonzero(np.abs(n) > .1) > 1:
            continue
        points = mesh.vertices[np.unique(mesh.faces[ids])]
        center = points.mean(axis=0)
        offset = np.median(points @ n)
        label[ids] = len(records)
        records.append({"kind": "plane", "normal": n, "origin": center + (offset - center @ n) * n,
                        "faces": ids, "area": float(mesh.area_faces[ids].sum())})
    return {"label": label, "regions": records, "area": mesh.area}, ribs


def test_narrow(mesh, seg, truth, kinds, ribs, noise):
    from narrow_planes import narrow_planes

    planes = narrow_planes(mesh, seg, .3, noise)
    covered = set()
    curved = 0
    for p in planes:
        values, counts = np.unique(truth[p["faces"]], return_counts=True)
        for value, count in zip(values, counts):
            if value in ribs and count >= .7 * len(p["faces"]) and count >= .4 * np.count_nonzero(truth == value):
                covered.add(int(value))
        curved += sum(kinds[i] == "Cylinder" for i in truth[p["faces"]])
    return {"matched_rib_sides": len(covered), "expected_rib_sides": len(ribs),
            "candidate_planes": len(planes), "falsely_planar_cylinder_triangles": curved,
            "face_counts": [len(p["faces"]) for p in planes]}


def test_fillets(mesh, seg, truth, kinds, centers, noise):
    from plane_fillets import fillets

    output = fillets(mesh, seg, .3, noise)
    cases = {}
    false = 0
    for f in output:
        ids = f["faces"]
        values, counts = np.unique(truth[ids], return_counts=True)
        tag = int(values[np.argmax(counts)])
        case = int(round(centers[tag][0] / 34.))
        if kinds[tag] != "Cylinder" or counts.max() < .65 * len(ids):
            false += 1
        else:
            cases[str(case)] = {"radius_mm": f["radius"], "residual_mm": f["p95_mm"],
                                "radius_sigma_mm": f["radius_sigma_mm"], "sense": f["sense"],
                                "suggested_radius_mm": f["suggested_radius_mm"],
                                "triangle_precision": float(counts.max() / len(ids)),
                                "triangle_recall": float(counts.max() / np.count_nonzero(truth == tag))}
    return {"cases": cases, "false_fillets": false}


def passed(which, noise, result):
    if which == "narrow":
        return (result["falsely_planar_cylinder_triangles"] == 0 and
                (noise > .05 or result["matched_rib_sides"] >= 5))
    cases = result["cases"]
    return result["false_fillets"] == 0 and all(
        str(case) in cases and abs(cases[str(case)]["radius_mm"] / radius - 1) <= .1
        for case, radius in ((1, 2), (2, 5))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=("narrow", "fillets"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--segmented", action="store_true", help="Also measure with actual regions.segment output.")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    step = 1.5
    shape, walls = fixture(args.which, step)
    path = args.work / f"{args.which}.step"
    cq.exporters.export(shape, str(path))
    clean, truth, kinds, centers = meshed(path, step)
    levels = (0., .05, .2) if walls else (0., .1)
    report = {}
    for noise in levels:
        started = perf_counter()
        mesh = clean.copy()
        mesh.vertices += np.random.default_rng(3311).normal(0., noise, mesh.vertices.shape)
        seg, ribs = anchors(clean, mesh, truth, kinds, centers, walls)
        if walls:
            result = test_narrow(mesh, seg, truth, kinds, ribs, noise)
        else:
            result = test_fillets(mesh, seg, truth, kinds, centers, noise)
        result["seconds"] = perf_counter() - started
        result["median_edge_mm"] = float(np.median(mesh.edges_unique_length))
        result["acceptance_passed"] = passed(args.which, noise, result)
        report[str(noise)] = result
        print(noise, result, flush=True)
        if args.segmented:
            from regions import segment

            raw_seg = segment(mesh, tol=max(.3, 2 * noise), scan=mesh, min_area=12.)
            if walls:
                result["segmented"] = test_narrow(mesh, raw_seg, truth, kinds, ribs, noise)
                result["segmented"]["rib_sides_already_labelled"] = int(sum(
                    np.count_nonzero(raw_seg["label"][truth == i] >= 0) >= .7 * np.count_nonzero(truth == i)
                    for i in ribs
                ))
                row = result["segmented"]
                row["matched_rib_sides"] += row["rib_sides_already_labelled"]
            else:
                result["segmented"] = test_fillets(mesh, raw_seg, truth, kinds, centers, noise)
            result["segmented"]["acceptance_passed"] = passed(args.which, noise, result["segmented"])
            print(noise, "segmented", result["segmented"], flush=True)
    (args.work / f"{args.which}_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["acceptance_passed"] and row.get("segmented", {}).get("acceptance_passed", True)
                    for row in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
