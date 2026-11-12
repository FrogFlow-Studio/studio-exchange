import argparse
from pathlib import Path
import json
import sys
from time import perf_counter

import cadquery as cq
import gmsh
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool

def fixture():
    radii = (17.585, 4.22, 4.30, 3.515, 5.31, 8.735)
    depths = (2.1, 6.3, 6.3, 6.0, 1.8, 4.8)
    spine = cq.Workplane("XY").box(17, 166, 13).val()
    tabs, cutters, truth = [], [], []
    for i, (r, depth) in enumerate(zip(radii, depths)):
        side = -1 if i % 2 else 1
        y = (i // 2 - 1) * 55
        center = cq.Vector(side * 30, y, 0)
        axis = cq.Vector(np.sin(np.radians(side * (6.5 + i % 3))),
                         np.sin(np.radians((i % 2) * 2)), 1).normalized()
        plate = cq.Workplane(cq.Plane(origin=center, normal=axis)).box(50, 47, depth).val()
        tabs.append(plate)
        cutters.append(cq.Solid.makeCylinder(r, depth + 4, center - axis * (depth/2 + 2), axis))
        truth.append((list((center - axis * depth/2).toTuple()), list(axis.toTuple()), r, depth))
    return spine.fuse(*tabs).cut(*cutters).clean(), truth


def mesh_step(path, fine=2., coarse=4.5, noise=.05):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
        gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.add("bracket")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        field = gmsh.model.mesh.field
        box = field.add("Box")
        for key, value in dict(VIn=coarse, VOut=fine, XMin=-57, XMax=-2, YMin=-85,
                               YMax=85, ZMin=-12, ZMax=12, Thickness=5).items():
            field.setNumber(box, key, value)
        field.setAsBackgroundMesh(box)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, nodes = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    order = np.argsort(tags)
    faces = order[np.searchsorted(tags[order], nodes)].reshape(-1, 3)
    scan = trimesh.Trimesh(xyz.reshape(-1, 3), faces)
    trimesh.repair.fix_normals(scan)
    scan.vertices += np.random.default_rng(201112).normal(0, noise, scan.vertices.shape)
    return scan
def sparse_span(scan, bore, tol, noise_mm=0.):
    """Measure inward-facing wall intersections with radial rays."""
    vertices = bore.local(scan.vertices)
    faces = np.asarray(scan.faces)
    triangles = vertices[faces]
    slack = max(tol * .6, 2.5 * noise_mm)
    search_height = max(2 * slack, .25 * bore.length)
    axial = triangles[..., 2]
    relevant = (axial.max(axis=1) > -search_height) & (axial.min(axis=1) < bore.length + search_height)
    relevant &= np.linalg.norm(triangles[..., :2], axis=2).min(axis=1) < bore.radius + slack
    candidates = np.flatnonzero(relevant)
    if not len(candidates):
        return None
    local_tri = triangles[candidates]
    normals = np.asarray(scan.face_normals)[candidates] @ bore.frame.T
    radial = local_tri[..., :2].mean(axis=1)
    walls = np.einsum("ij,ij->i", normals[:, :2], radial) < -.5 * np.linalg.norm(radial, axis=1)
    walls &= np.abs(normals[:, 2]) < .65
    if not walls.any():
        return None
    ids = candidates[walls]
    wall = trimesh.Trimesh(vertices=vertices, faces=faces[ids], process=False)
    az = np.linspace(0., 2*np.pi, 32, endpoint=False)
    rays = np.column_stack((np.cos(az), np.sin(az), np.zeros(len(az))))
    heights = np.linspace(-search_height, bore.length + search_height,
                          max(17, min(128, int((bore.length + 2*search_height) / (tol / 2)) + 1)))
    origins = np.zeros((len(heights)*len(az),3))
    origins[:, 2] = np.repeat(heights,len(az))
    directions = np.tile(rays, (len(heights),1))
    face_idx, ray_idx, points = wall.ray.intersects_id(origins, directions, return_locations=True, multiple_hits=False)
    hits = np.zeros(len(origins),dtype=bool)
    hit_tri = local_tri[walls][face_idx]
    tangent = directions[ray_idx, :2][:, ::-1] * np.array([-1.,1.])
    across = np.ptp(np.einsum("nij,nj->ni", hit_tri[..., :2], tangent), axis=1)
    half_chord = np.minimum(across/2, .65*bore.radius)
    sag = bore.radius - np.sqrt(bore.radius*bore.radius - half_chord*half_chord)
    radius = np.linalg.norm(points[:, :2],axis=1)
    hits[ray_idx] = (radius >= bore.radius - slack - sag) & (radius <= bore.radius + slack)
    rings = hits.reshape(len(heights),len(az))
    # A nearly complete circumference, including every octant, rules out slots and flat caps.
    supported = np.flatnonzero((rings.mean(axis=1) >= .78) &
                               rings.reshape(len(heights),8,4).any(axis=2).all(axis=1))
    runs = np.split(supported, np.flatnonzero(np.diff(supported) != 1) + 1)
    matches = [(float(heights[a[0]]), float(heights[a[-1]])) for a in runs if len(a) >= 3
               and min(bore.length, heights[a[-1]]) - max(0, heights[a[0]]) >= .48*bore.length]
    return matches[0] if len(matches) == 1 else None

def inspect(path, truth, gate):
    from bore_stitch import Bore, exact_wall
    from nurbs import body_mesh

    body = cq.importers.importStep(str(path)).val()
    walls = [exact_wall(body, Bore.from_spec(spec), (0, spec[3]), .3) for spec in truth]
    mesh = body_mesh(body, .03)
    return {"faces": len(body.Faces()), "valid": body.isValid(), "solids": len(body.Solids()),
            "closed": bool(len(body.Shells()) == 1 and BRep_Tool.IsClosed_s(body.Shells()[0].wrapped)),
            "exact_walls": sum(row is not None for row in walls), "walls": walls,
            "coverage_0_3_mm": gate.measure(mesh)}


def run(demo, work, reuse):
    sys.path.insert(0, str(demo.resolve()))
    import bore_stitch as bs
    import nurbs
    import surface

    work.mkdir(parents=True, exist_ok=True)
    if not reuse:
        reference, truth = fixture()
        cq.exporters.export(reference, str(work / "truth.step"))
        (work / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
        scan = mesh_step(work / "truth.step")
        scan.export(work / "scan.ply")
    scan = trimesh.load(work / "scan.ply", force="mesh")
    truth = json.loads((work / "truth.json").read_text(encoding="utf-8"))
    clean = surface.flatten_planes(surface.surface_mesh(scan))
    stage = work / "stage.step"
    if not reuse:
        nurbs._SCAN, nurbs.FACE_TOL = scan, .3
        triangles, nurbs.K, nurbs.NP, nurbs.PAIR = nurbs.STEPS[0]
        body, _, _ = nurbs._build(clean, triangles, nurbs._Projector(clean), profile={}, stage={})
        if body is None:
            raise RuntimeError("the 0k stage did not build")
        cq.exporters.export(body, str(stage))
    body = cq.importers.importStep(str(stage)).val()
    started = perf_counter()
    index = bs.ShellIndex(body)
    rows, plans, used = [], [], set()
    for spec in surface.hole_parameters(scan):
        bore = bs.Bore.from_spec(spec)
        old = bs.observed_span(scan, bore, .3)
        new = sparse_span(scan, bore, .3, .05)
        plan = next((p for p in bs._plans(index, bore, new, .3) if not p.affected & used), None) if new else None
        rows.append({"diameter_mm": 2 * bore.radius, "legacy_span_mm": old, "radial_span_mm": new,
                     "plan": plan.describe() if plan else None})
        if plan:
            plans.append(plan)
            used.update(plan.affected)
    candidate, error = bs._stitch(index, plans, .3) if plans else (None, "no_plan")
    seconds = perf_counter() - started
    gate = bs.Coverage(scan, clean, .3)
    before = inspect(stage, truth, gate)
    after = None
    if candidate is not None:
        trial = work / "candidate.step"
        cq.exporters.export(candidate, str(trial))
        after = inspect(trial, truth, gate)
    valid = bool(after and after["valid"] and after["closed"] and after["solids"] == 1)
    checks = {"at_least_four_walls": bool(after and after["exact_walls"] >= 4),
              "one_valid_closed_solid": valid,
              "face_cap": bool(after and after["faces"] <= int(before["faces"] * 1.1)),
              "coverage_non_decreasing": bool(after and all(
                  after["coverage_0_3_mm"][k] >= value for k, value in before["coverage_0_3_mm"].items()))}
    return {"scan_faces": len(scan.faces),
            "edge_mm_p10_p50_p90": np.quantile(scan.edges_unique_length, [.1, .5, .9]).tolist(),
            "bore_evidence": rows, "baseline": before, "candidate": after, "sewing_error": error,
            "experiment_seconds": seconds, "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 20b coarse-scan bore diagnostics")
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    result = run(args.demo, args.work, args.reuse)
    (args.work / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"]))
    raise SystemExit(0 if all(result["checks"].values()) else 1)
