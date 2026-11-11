"""Score region segmentation on scans with independent, trimmed CAD reference faces."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepAdaptor import BRepAdaptor_Surface


def fixture(name):
    if name == "rounded_shell":
        shell = cq.Workplane("XY").box(58, 40, 17).edges().fillet(4)
        return shell.faces(">Z").shell(-2).val()
    if name == "ribbed_housing":
        body = cq.Workplane("XY").box(63, 44, 5)
        for x, radius in ((-18, 3), (0, 4), (17, 5)):
            boss = cq.Workplane("XY").center(x, 4).circle(radius + 3).extrude(11)
            body = body.union(boss).cut(cq.Workplane("XY").center(x, 4).circle(radius).extrude(22, both=True))
        return body.union(cq.Workplane("XY").center(0, -9).box(40, 3, 12)).val()
    if name == "tapered_bushing":
        outer = cq.Solid.makeCone(22, 13, 28)
        inner = cq.Solid.makeCone(8, 5, 28)
        return outer.cut(inner)
    if name == "spherical_end":
        cap = cq.Solid.makeSphere(21)
        bore = cq.Solid.makeCylinder(4, 46, cq.Vector(0, 0, -23))
        return cap.cut(bore)
    if name == "curved_cover":
        # Two revolved spline skins with continuously varying meridional curvature.
        wp = cq.Workplane("XZ").moveTo(0, 14)
        wp = wp.spline([(5, 13.85), (14, 12.7), (24, 9.8), (34, 4)], includeCurrent=True)
        wp = wp.spline([(34, 3.1), (33, 2.6), (32, 2.8)], includeCurrent=True)
        wp = wp.spline([(24, 7.8), (14, 10.7), (5, 11.85), (0, 12)], includeCurrent=True)
        return wp.close().revolve().val()
    if name == "drilled_plate":
        plate = cq.Workplane("XY").box(91, 54, 9)
        for x, radius in ((-28, 3.4), (-10, 4.7), (12, 3.8), (29, 4.3)):
            plate = plate.cut(cq.Workplane("XY").center(x, 0).circle(radius).extrude(20, both=True))
        return plate.val().rotate((0, 0, 0), (1, 2, 0), 26)
    if name == "toroidal_ring":
        return cq.Solid.makeTorus(28, 6)
    raise ValueError(name)


FIXTURES = ("rounded_shell", "ribbed_housing", "tapered_bushing", "spherical_end",
            "curved_cover", "drilled_plate", "toroidal_ring")
SAMPLES = ("flange", "bushing", "bracket")


def array(xyz):
    return np.array(xyz.Coord())


def describe(face):
    kind = face.geomType().lower()
    s = BRepAdaptor_Surface(face.wrapped)
    if kind == "plane":
        p = s.Plane()
        return dict(kind=kind, origin=array(p.Location()), normal=array(p.Axis().Direction()))
    if kind == "cylinder":
        p = s.Cylinder()
        return dict(kind=kind, origin=array(p.Location()), axis=array(p.Axis().Direction()), r=p.Radius())
    if kind == "cone":
        p = s.Cone()
        direction = array(p.Axis().Direction())
        center = np.array(face.Center().toTuple())
        return dict(kind=kind, apex=array(p.Apex()), axis=direction * np.sign(p.SemiAngle()),
                    semi_angle_deg=abs(np.degrees(p.SemiAngle())), station=center)
    if kind == "sphere":
        p = s.Sphere()
        return dict(kind=kind, center=array(p.Location()), r=p.Radius())
    return dict(kind="free")


def tessellate(shape, tolerance, angular):
    shape.tessellate(tolerance, angular)
    meshes, labels, truth = [], [], []
    for face in shape.Faces():
        points, triangles = face.tessellate(tolerance, angular)
        if not triangles:
            continue
        mesh = trimesh.Trimesh([v.toTuple() for v in points], triangles, process=False)
        meshes.append(mesh)
        labels.extend([len(truth)] * len(mesh.faces))
        truth.append(describe(face))
    return trimesh.util.concatenate(meshes), np.array(labels), truth


def scanned(shape, rng, noise):
    b = shape.BoundingBox()
    # A slanted cut at a seeded, random azimuth, clipping an outer edge.
    center = np.array(b.center.toTuple())
    diagonal = np.linalg.norm([b.xlen, b.ylen, b.zlen])
    angle = rng.uniform(0, 360)
    phi = np.radians(angle)
    normal = np.array([np.cos(phi), np.sin(phi), 0.23])
    mesh, _, _ = tessellate(shape, 0.02, 0.1)
    support = float(np.max((mesh.vertices - center) @ normal))
    offset = support - rng.uniform(0.07, 0.15) * diagonal
    origin = center + normal * (offset / (normal @ normal))
    cutter = cq.Workplane(cq.Plane(tuple(origin), normal=tuple(normal))).box(
        4 * diagonal, 4 * diagonal, 2 * diagonal, centered=(True, True, False))
    broken = shape.cut(cutter.val())
    scan, _, _ = tessellate(broken, 0.02, 0.1)
    scan.merge_vertices()
    scan = refine_edges(scan, 3.0)
    scan.vertices += rng.normal(0, noise, scan.vertices.shape)
    return scan


def refine_edges(mesh, limit):
    """Bisect long edges on both incident triangles to preserve watertight topology."""
    while True:
        edges = mesh.edges_unique
        select = np.flatnonzero(mesh.edges_unique_length > limit)
        if not len(select):
            return mesh
        new = len(mesh.vertices) + np.arange(len(select))
        midpoints = mesh.vertices[edges[select]].mean(axis=1)
        index = np.full(len(edges), -1)
        index[select] = new
        rows = index[mesh.edges_unique_inverse].reshape(-1, 3)
        vertices = np.vstack([mesh.vertices, midpoints])
        faces = []
        count = (rows >= 0).sum(axis=1)
        faces.append(mesh.faces[count == 0])
        for start in range(3):
            a, b, c = start, (start + 1) % 3, (start + 2) % 3
            for n in (1, 2):
                keep = (count == n) & (rows[:, a] >= 0)
                keep &= rows[:, b] < 0 if n == 1 else rows[:, b] >= 0
                if n == 2:
                    keep &= rows[:, c] < 0
                t, m = mesh.faces[keep], rows[keep]
                faces.extend([np.c_[t[:, a], m[:, a], t[:, c]],
                              np.c_[m[:, a], t[:, b], t[:, c]]] if n == 1 else
                             [np.c_[t[:, a], m[:, a], t[:, c]],
                              np.c_[m[:, a], m[:, b], t[:, c]],
                              np.c_[m[:, a], t[:, b], m[:, b]]])
        t, m = mesh.faces[count == 3], rows[count == 3]
        faces.extend([np.c_[t[:, i], m[:, i], m[:, i - 1]] for i in range(3)] + [m])
        mesh = trimesh.Trimesh(vertices, np.vstack(faces), process=False)


def truth_ids(mesh, reference, face_ids, tol):
    # Trimesh projects onto trimmed triangles. Points too far from every intact face
    # belong to the damaged surface. The analytical parameters come from the STEP faces.
    assigned = []
    for chunk in np.array_split(mesh.triangles_center, max(1, (len(mesh.faces) + 255) // 256)):
        _, distances, nearest = trimesh.proximity.closest_point(reference, chunk)
        assigned.extend(np.where(distances <= 2 * tol, face_ids[nearest], -1))
    return np.asarray(assigned)


def matches(pred, ref, tol):
    k = ref["kind"]
    if pred["kind"] != k or k == "free":
        return False
    if k == "plane":
        n = ref["normal"]
        return abs(pred["normal"] @ n) >= np.cos(np.radians(1)) and abs((pred["origin"] - ref["origin"]) @ n) <= 0.1
    if k in ("cylinder", "sphere") and abs(pred["r"] - ref["r"]) > max(0.1, 0.02 * ref["r"]):
        return False
    if k == "sphere":
        return np.linalg.norm(pred["center"] - ref["center"]) <= max(0.1, 0.02 * ref["r"])
    axis = ref["axis"]
    if abs(pred["axis"] @ axis) < np.cos(np.radians(2)):
        return False
    if k == "cone":
        if pred["axis"] @ axis <= 0 or abs(pred["semi_angle_deg"] - ref["semi_angle_deg"]) > 2:
            return False
        # Compare the actual cone around the observed face. For shallow cones,
        # a small error in slope extrapolates to a large error at the distant apex.
        station = ref["apex"] + axis * ((ref["station"] - ref["apex"]) @ axis)
        q = station - pred["apex"]
        h = q @ pred["axis"]
        offset = np.linalg.norm(q - h * pred["axis"])
        r = np.linalg.norm(station - ref["apex"]) * np.tan(np.radians(ref["semi_angle_deg"]))
        fitted_r = h * np.tan(np.radians(pred["semi_angle_deg"]))
        return offset <= tol and abs(fitted_r - r) <= tol
    offset = np.linalg.norm(np.cross(pred["origin"] - ref["origin"], axis))
    return offset <= max(0.1, 0.02 * ref["r"])


def measure(mesh, ids, truth, segment, tol):
    start = time.perf_counter()
    seg = segment(mesh, tol=tol)
    seconds = time.perf_counter() - start
    areas = mesh.area_faces
    truth_area = np.bincount(ids + 1, weights=areas, minlength=len(truth) + 1)[1:]
    correct = np.zeros(len(truth))
    predicted, false_areas, unmatched, regions = {}, {}, {}, []
    assert seg["label"].shape == (len(mesh.faces),)
    assert np.all((seg["label"] >= -1) & (seg["label"] < len(seg["regions"])))
    for i, reg in enumerate(seg["regions"]):
        faces = np.asarray(reg["faces"])
        assert np.all(seg["label"][faces] == i)
        area = float(areas[faces].sum())
        assert abs(area - reg["area"]) <= 1e-8 * max(1, area)
        k = reg["kind"]
        predicted[k] = predicted.get(k, 0.) + area
        overlap = np.bincount(ids[faces] + 1, weights=areas[faces], minlength=len(truth) + 1)[1:]
        matching = [j for j in np.flatnonzero(overlap) if matches(reg, truth[j], tol)]
        good = float(overlap[matching].sum())
        correct[matching] += overlap[matching]
        false_areas[k] = false_areas.get(k, 0.) + area - good
        if good <= 0.5 * area:
            unmatched[k] = unmatched.get(k, 0) + 1
        regions.append({key: (val.tolist() if isinstance(val, np.ndarray) else val)
                        for key, val in reg.items() if key != "faces"} | {
                            "correct_area": good,
                            "dominant_reference_face": int(np.argmax(overlap)) if overlap.sum() else None,
                        })
    return dict(seconds=seconds, mesh_area=float(areas.sum()), triangles=len(areas),
                predicted=predicted, false_area=false_areas, false_count=unmatched, regions=regions,
                truth=[{key: (val.tolist() if isinstance(val, np.ndarray) else val)
                        for key, val in row.items()} | {"reference_face": j, "visible_area": truth_area[j],
                                                        "matched_area": correct[j]}
                       for j, row in enumerate(truth)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--base", default="da7feb1")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--candidate-only", action="store_true")
    mode.add_argument("--base-only", action="store_true")
    p.add_argument("--cases", nargs="+", choices=(*SAMPLES, *FIXTURES), default=(*SAMPLES, *FIXTURES))
    p.add_argument("--noise", nargs="+", type=float, default=(0.05, 0.2))
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--tol", type=float, default=0.3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    demo = args.repo.resolve() / "claude" / "cad-demo"
    sys.path.insert(0, str(demo))
    import regions
    import surface

    methods = {}
    if not args.candidate_only:
        base = types.ModuleType("regions_base")
        source = subprocess.check_output(
            ["git", "-C", str(args.repo), "show", f"{args.base}:claude/cad-demo/regions.py"])
        exec(compile(source, "regions_base.py", "exec"), base.__dict__)
        methods["base"] = base.segment
    if not args.base_only:
        methods["candidate"] = regions.segment

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.cases:
        existing = name in SAMPLES
        shape = (cq.importers.importStep(str(demo / "samples" / f"{name}_truth.step")).val()
                 if existing else fixture(name))
        ref, face_ids, truth = tessellate(shape, 0.015, 0.1)
        for noise in ([None] if existing else args.noise):
            if existing:
                scan = trimesh.load(demo / "samples" / f"{name}_scan.stl", force="mesh")
            else:
                rng = np.random.default_rng(args.seed + FIXTURES.index(name))
                scan = scanned(shape, rng, noise)
            mesh = surface.surface_mesh(scan)
            ids = truth_ids(mesh, ref, face_ids, args.tol)
            for method, segment in methods.items():
                row = measure(mesh, ids, truth, segment, args.tol)
                row.update(case=name, noise=noise, method=method, seed=args.seed, tol=args.tol)
                rows.append(row)
                args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(name, noise, method, row["triangles"], round(row["seconds"], 2), row["false_count"],
                      flush=True)


if __name__ == "__main__":
    main()
