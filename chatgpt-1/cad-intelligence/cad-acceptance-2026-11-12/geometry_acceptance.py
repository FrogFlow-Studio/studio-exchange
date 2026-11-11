"""Independent STEP-to-STEP audit of analytic B-rep features in millimeters."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Trsf
from scipy.optimize import linear_sum_assignment
import trimesh


@dataclass(frozen=True)
class Tolerance:
    radius_mm: float = 0.15
    axis_offset_mm: float = 0.25
    axis_angle_deg: float = 2.0
    plane_angle_deg: float = 1.0
    plane_mm: float = 0.15
    end_mm: float = 0.5
    cone_angle_deg: float = 1.0
    cover_mm: float = 0.3
    chord_mm: float = 0.03
    tess_angle_rad: float = 0.10
    samples: int = 4000
    seed: int = 19213

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "seed":
                if not isinstance(value, int) or value < 0:
                    raise ValueError("seed must be a nonnegative integer")
            elif not np.isfinite(value) or value <= 0 or (key == "samples" and not isinstance(value, int)):
                raise ValueError(f"invalid measurement setting {key}: {value}")


def xyz(v):
    return np.array(v.Coord() if hasattr(v, "Coord") else v.toTuple(), dtype=float)


def norm(v):
    return float(np.linalg.norm(v))


def angle(u, v, signed=False):
    dot = float(u @ v)
    return float(np.degrees(np.arccos(np.clip(dot if signed else abs(dot), -1, 1))))


def load_step(path):
    shapes = cq.importers.importStep(str(path)).vals()
    if not shapes:
        raise ValueError("STEP contains no shapes")
    return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)


def transform_shape(shape, matrix):
    a = np.asarray(matrix, dtype=float)
    if a.shape != (4, 4) or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("transform must be a homogeneous 4x4 matrix")
    rotation = a[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-6) or np.linalg.det(rotation) <= 0:
        raise ValueError("transform must be a proper rigid motion in millimeters")
    motion = gp_Trsf()
    motion.SetValues(*map(float, a[:3].ravel()))
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, motion, True).Shape())


def topology(shape):
    solids = shape.Solids()
    shells = shape.Shells()
    faces = shape.Faces()
    solid_faces = set(face for solid in solids for face in solid.Faces())
    edges, vertices = shape.Edges(), shape.Vertices()
    solid_edges = set(edge for solid in solids for edge in solid.Edges())
    solid_vertices = set(vertex for solid in solids for vertex in solid.Vertices())
    free = sum(face not in solid_faces for face in faces)
    free_edges = sum(edge not in solid_edges for edge in edges)
    free_vertices = sum(vertex not in solid_vertices for vertex in vertices)
    only_solids = bool(solids and not (free or free_edges or free_vertices))
    closed_shells = sum(bool(BRep_Tool.IsClosed_s(shell.wrapped)) for shell in shells)
    return {
        "valid": bool(shape.isValid()),
        "closed": bool(only_solids and shells and closed_shells == len(shells)),
        "connected_solid": bool(len(solids) == 1 and only_solids),
        "solids": len(solids), "shells": len(shells), "closed_shells": closed_shells,
        "faces": len(faces), "edges": len(edges), "faces_outside_solids": free,
        "edges_outside_solids": free_edges, "vertices_outside_solids": free_vertices,
        "volume_mm3": float(sum(solid.Volume() for solid in solids)),
        "surface_types": dict(Counter(face.geomType() for face in faces)),
    }


def face_geometry(face):
    kind = face.geomType().lower()
    s = BRepAdaptor_Surface(face.wrapped)
    if kind == "plane":
        plane = s.Plane()
        return {"kind": kind, "origin": xyz(plane.Location()), "axis": xyz(plane.Axis().Direction())}
    if kind == "cylinder":
        cylinder = s.Cylinder()
        point = xyz(s.Value((s.FirstUParameter() + s.LastUParameter()) / 2,
                            (s.FirstVParameter() + s.LastVParameter()) / 2))
        origin, axis = xyz(cylinder.Location()), xyz(cylinder.Axis().Direction())
        radial = point - origin
        radial -= axis * (radial @ axis)
        inner = bool(xyz(face.normalAt(cq.Vector(*point))) @ radial < 0)
        return {"kind": kind, "origin": origin, "axis": axis, "radius": cylinder.Radius(),
                "inner": inner}
    if kind == "cone":
        cone = s.Cone()
        slope = cone.SemiAngle()
        return {"kind": kind, "origin": xyz(cone.Apex()),
                "axis": xyz(cone.Axis().Direction()) * np.sign(slope),
                "semi_angle_deg": abs(float(np.degrees(slope)))}
    if kind == "sphere":
        sphere = s.Sphere()
        return {"kind": kind, "origin": xyz(sphere.Location()), "radius": sphere.Radius()}
    if kind == "torus":
        torus = s.Torus()
        return {"kind": kind, "origin": xyz(torus.Location()), "axis": xyz(torus.Axis().Direction()),
                "radius": torus.MajorRadius(), "minor_radius": torus.MinorRadius()}
    return None


def same_surface(a, b):
    if a["kind"] != b["kind"]:
        return False
    kind = a["kind"]
    delta = a["origin"] - b["origin"]
    eps = 1e-4  # OCC seam merging only, independent from acceptance tolerances
    if kind == "plane":
        return angle(a["axis"], b["axis"]) < .001 and abs(delta @ a["axis"]) < eps
    if kind == "sphere":
        return norm(delta) < eps and abs(a["radius"] - b["radius"]) < eps
    if angle(a["axis"], b["axis"], signed=kind == "cone") >= .001:
        return False
    if kind == "cylinder":
        return (a["inner"] == b["inner"] and abs(a["radius"] - b["radius"]) < eps
                and norm(np.cross(delta, a["axis"])) < eps)
    if kind == "cone":
        return norm(delta) < eps and abs(a["semi_angle_deg"] - b["semi_angle_deg"]) < .001
    return (norm(delta) < eps and abs(a["radius"] - b["radius"]) < eps
            and abs(a["minor_radius"] - b["minor_radius"]) < eps)


def extract(shape, tol):
    faces = shape.Faces()
    data = [face_geometry(f) for f in faces]
    edge_faces = defaultdict(list)
    for i, face in enumerate(faces):
        for edge in face.Edges():
            edge_faces[edge].append(i)
    links = [[] for _ in faces]
    for edge, incident in edge_faces.items():
        if len(incident) == 2:
            a, b = incident
            links[a].append((b, edge))
            links[b].append((a, edge))
    shape.tessellate(tol.chord_mm, tol.tess_angle_rad)
    features = []
    visited = set()
    for i, feature in enumerate(data):
        if i in visited or feature is None:
            continue
        group = [i]
        visited.add(i)
        for a in group:
            for b, _ in links[a]:
                if b not in visited and data[b] is not None and same_surface(feature, data[b]):
                    visited.add(b)
                    group.append(b)
        pts = np.array([v.toTuple() for a in group for v in
                        faces[a].tessellate(tol.chord_mm, tol.tess_angle_rad)[0]])
        if not len(pts):
            pts = np.array([faces[a].Center().toTuple() for a in group])
        areas = np.array([faces[a].Area() for a in group])
        centers = np.array([faces[a].Center().toTuple() for a in group])
        center = np.average(centers, axis=0, weights=areas) if areas.sum() else centers.mean(axis=0)
        f = {**feature, "id": len(features) + 1, "faces": [a + 1 for a in group],
             "center": center, "area_mm2": float(areas.sum()), "role": feature["kind"], "_points": pts}
        if f["kind"] in ("cylinder", "cone"):
            z = (pts - f["origin"]) @ f["axis"]
            f["length_mm"] = float(np.ptp(z))
            f["axis_center"] = f["origin"] + f["axis"] * float((z.min() + z.max()) / 2)
        if f["kind"] == "cylinder":
            wrap = min(360., np.degrees(f["area_mm2"] / max(f["radius"] * f["length_mm"], 1e-12)))
            f["wrap_deg"] = float(wrap)
            if f["inner"] and wrap >= 330:
                f["role"] = "bore"
            elif wrap < 180:
                # A rolling-cylinder fillet has two distinct tangent supports.
                supports = []
                for a in group:
                    for b, edge in links[a]:
                        if b in group or data[b] is None or edge.Length() < 1e-6:
                            continue
                        try:
                            p = edge.positionAt(.5)
                            na, nb = xyz(faces[a].normalAt(p)), xyz(faces[b].normalAt(p))
                            if angle(na, nb, signed=True) <= 3 and not any(
                                    same_surface(data[b], other) for other in supports):
                                supports.append(data[b])
                        except (ValueError, RuntimeError):
                            continue
                if len(supports) >= 2:
                    f["role"] = "cylindrical_fillet"
                elif f["inner"]:
                    f["role"] = "partial_inner_cylinder"
            elif f["inner"]:
                f["role"] = "partial_inner_cylinder"
        features.append(f)
    return features


def box_overlap(a, b):
    width = np.maximum(np.minimum(a.max(0), b.max(0)) - np.maximum(a.min(0), b.min(0)), 0)
    area = min(float(np.prod(np.ptp(a, axis=0))), float(np.prod(np.ptp(b, axis=0))))
    return float(np.prod(width) / area) if area > 1e-12 else 0.


def compare_feature(a, b, tol):
    if a["kind"] != b["kind"]:
        return None
    k = a["kind"]
    delta = b["origin"] - a["origin"]
    radius_scale = max(a.get("radius", 0), b.get("radius", 0))
    drift = norm(a["center"] - b["center"])
    errors, limits = {}, {}
    if k == "plane":
        axis = a["axis"]
        u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
        u /= norm(u)
        frame = np.stack([u, np.cross(axis, u)], axis=1)
        overlap = box_overlap(a["_points"] @ frame, b["_points"] @ frame)
        dist = np.abs((b["_points"] - a["origin"]) @ axis)
        errors.update(angle_deg=angle(axis, b["axis"]), offset_mm=abs(float((b["center"] - a["origin"]) @ axis)),
                      sampled_p95_mm=float(np.quantile(dist, .95)), sampled_max_mm=float(dist.max()),
                      footprint_overlap=overlap)
        limits.update(angle_deg=tol.plane_angle_deg, offset_mm=tol.plane_mm, sampled_p95_mm=tol.plane_mm)
        if overlap < .15 or errors["angle_deg"] > 15 or errors["offset_mm"] > max(1.5, 6 * tol.plane_mm):
            return None
        association = 1 - overlap
    elif k in ("cylinder", "cone"):
        axis = a["axis"]
        station = a["axis_center"]
        alignment = float(b["axis"] @ axis)
        if abs(alignment) < np.cos(np.radians(18)) or (k == "cone" and alignment <= 0):
            return None
        # Intersect the reconstructed axis with the plane through the reference station.
        reconstructed = b["origin"] + b["axis"] * (((station - b["origin"]) @ axis) / alignment)
        a_height = (a["_points"] - station) @ axis
        b_height = (b["_points"] - station) @ axis
        overlap = min(a_height.max(), b_height.max()) - max(a_height.min(), b_height.min())
        overlap /= max(min(np.ptp(a_height), np.ptp(b_height)), 1e-9)
        errors.update(axis_angle_deg=angle(axis, b["axis"], signed=k == "cone"),
                      axis_offset_mm=norm(reconstructed - station), center_mm=norm(b["axis_center"] - station),
                      end_mm=float(max(abs(a_height.min() - b_height.min()), abs(a_height.max() - b_height.max()))),
                      axial_overlap=float(overlap))
        limits.update(axis_angle_deg=tol.axis_angle_deg, axis_offset_mm=tol.axis_offset_mm, end_mm=tol.end_mm)
        if overlap < .20:
            return None
        if k == "cylinder":
            if a["inner"] != b["inner"]:
                return None
            errors["radius_mm"] = abs(a["radius"] - b["radius"])
            limits["radius_mm"] = tol.radius_mm
            if errors["radius_mm"] > max(1., .35 * radius_scale):
                return None
        else:
            errors["semi_angle_deg"] = abs(a["semi_angle_deg"] - b["semi_angle_deg"])
            errors["apex_mm"] = norm(delta)
            radius_a = float((station - a["origin"]) @ axis) * np.tan(np.radians(a["semi_angle_deg"]))
            radius_b = float((reconstructed - b["origin"]) @ b["axis"]) * np.tan(np.radians(b["semi_angle_deg"]))
            errors["radius_at_station_mm"] = abs(radius_a - radius_b)
            limits.update(semi_angle_deg=tol.cone_angle_deg, radius_at_station_mm=tol.radius_mm)
            radius_scale = max(abs(radius_a), abs(radius_b))
            if errors["semi_angle_deg"] > 12 or errors["radius_at_station_mm"] > max(1., .35 * radius_scale):
                return None
        if errors["axis_offset_mm"] > max(1.5, .18 * radius_scale):
            return None
        association = 1 - min(1., overlap)
    else:
        errors.update(center_mm=norm(delta), radius_mm=abs(a["radius"] - b["radius"]))
        limits.update(center_mm=tol.axis_offset_mm, radius_mm=tol.radius_mm)
        if errors["center_mm"] > max(1.5, .18 * radius_scale) or errors["radius_mm"] > max(1., .35 * radius_scale):
            return None
        if k == "torus":
            errors.update(axis_angle_deg=angle(a["axis"], b["axis"]),
                          minor_radius_mm=abs(a["minor_radius"] - b["minor_radius"]))
            limits.update(axis_angle_deg=tol.axis_angle_deg, minor_radius_mm=tol.radius_mm)
            if errors["axis_angle_deg"] > 18:
                return None
        association = 0.
    failed = [name for name, limit in limits.items() if errors[name] > limit]
    if a["role"] != b["role"]:
        failed.append("role")
    cost = association + sum(min(10., errors[name] / limit) for name, limit in limits.items()) + .01 * drift
    return errors, failed, cost


def public(feature):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in feature.items() if not k.startswith("_")}


def match(reference, model, tol):
    rows = []
    for kind in sorted({f["kind"] for f in (*reference, *model)}):
        a = [f for f in reference if f["kind"] == kind]
        b = [f for f in model if f["kind"] == kind]
        # Dedicated dummy columns let each reference stay unmatched. First
        # maximize passing matches, then plausible pair count, then minimize error.
        weights = np.zeros((len(a), len(b) + len(a)))
        comparisons = {}
        reward = 100. * (len(a) + 1)
        for i, fa in enumerate(a):
            for j, fb in enumerate(b):
                test = compare_feature(fa, fb, tol)
                if test is None:
                    continue
                comparisons[i, j] = test
                _, failed, cost = test
                weights[i, j] = (0 if failed else reward * (len(a) + 1)) + reward - min(cost, 99.)
        ir, ic = linear_sum_assignment(weights, maximize=True)
        used = set()
        for i, j in zip(ir, ic):
            row = {"reference": public(a[i]), "model": None, "status": "missing"}
            if (i, j) in comparisons:
                errors, failed, _ = comparisons[i, j]
                row.update(model=public(b[j]), status="parameter_error" if failed else "ok",
                           errors=errors, failed=failed)
                used.add(j)
            rows.append(row)
        rows.extend({"reference": None, "model": public(f), "status": "extra"}
                    for j, f in enumerate(b) if j not in used)
    return rows


def step_mesh(shape, tol):
    vertices, faces = shape.tessellate(tol.chord_mm, tol.tess_angle_rad)
    return trimesh.Trimesh([p.toTuple() for p in vertices], faces, process=False)


def distances(a, b, tol):
    result = {}
    for k, source, target, seed in (("reference_to_model", a, b, tol.seed),
                                    ("model_to_reference", b, a, tol.seed + 1)):
        if not len(source.faces) or not len(target.faces):
            result[k] = None
            continue
        p, _ = trimesh.sample.sample_surface(source, tol.samples, seed=seed)
        d = np.concatenate([trimesh.proximity.closest_point(target, p[i:i+96])[1]
                            for i in range(0, len(p), 96)])
        result[k] = {"within_share": float(np.mean(d <= tol.cover_mm)),
                     "p95_mm": float(np.quantile(d, .95)), "max_sample_mm": float(d.max())}
    return result


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(model, reference=None, *, transform=None, tol=Tolerance()):
    m = load_step(model)
    out = {"units": "mm", "tolerances": asdict(tol), "model_topology": topology(m),
           "reference_topology": None, "surface": None, "features": None,
           "model_sha256": file_hash(model), "reference_sha256": None,
           "feature_counts": None, "outcomes": None,
           "reference_status": "missing_reference" if reference is None else "evaluated"}
    if reference is None:
        return out
    r = load_step(reference)
    if transform is not None:
        r = transform_shape(r, transform)
    mf, rf = extract(m, tol), extract(r, tol)
    out.update(reference_topology=topology(r), reference_sha256=file_hash(reference),
               feature_counts={"reference": dict(Counter(f["role"] for f in rf)),
                               "model": dict(Counter(f["role"] for f in mf))},
               surface=distances(step_mesh(r, tol), step_mesh(m, tol), tol),
               features=match(rf, mf, tol), reference_to_model_frame=transform)
    out["outcomes"] = dict(Counter(f["status"] for f in out["features"]))
    return out


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def markdown(result):
    topology = result["model_topology"]
    lines = ["# Проверка STEP", "",
             f"Тело валидно — {topology['valid']}. Оболочки замкнуты — {topology['closed']}. "
             f"Связное тело — {topology['connected_solid']}. Тел {topology['solids']}, "
             f"граней {topology['faces']}, рёбер {topology['edges']}.", ""]
    if result["features"] is None:
        return "\n".join(lines + ["Для параметрической проверки нужен согласованный STEP-эталон.", ""])
    lines.extend(["| Эталон | Модель | Статус | Вне допуска |",
                  "| --- | --- | --- | --- |"])
    for row in result["features"]:
        ids = [f"{row[key]['role']} #{row[key]['id']}" if row[key] else "—"
               for key in ("reference", "model")]
        failed = ", ".join(row.get("failed", ())) or "—"
        lines.append(f"| {ids[0]} | {ids[1]} | {row['status']} | {failed} |")
    for direction, values in result["surface"].items():
        if values:
            lines.append(f"\n{direction} — {values['within_share']:.1%} в допуске, "
                         f"P95 {values['p95_mm']:.3f} мм.")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--sheet", type=Path, help="demo sheet.json, for a reference in the original scan frame")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    transform = json.loads(a.sheet.read_text(encoding="utf-8"))["transform_from_scan_4x4"] if a.sheet else None
    result = evaluate(a.model, a.reference, transform=transform)
    write_json(a.out, result)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
