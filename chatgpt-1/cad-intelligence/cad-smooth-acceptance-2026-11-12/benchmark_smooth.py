"""Paired task 15 benchmark for the first 0k stage on intact noisy CAD fixtures."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

HERE = Path(__file__).resolve().parent
for project in ("cad-acceptance", "cad-region-layout", "cad-regions"):
    sys.path.insert(0, str(HERE.parent / f"{project}-2026-11-12"))

from benchmark_layout import EXTRA, extra_fixture
from benchmark_regions import FIXTURES, fixture, refine_edges, tessellate
from geometry_acceptance import Tolerance, extract, load_step, match, topology

CASES = (*FIXTURES, *EXTRA)
NOISE = (0.05, 0.2)
TOL = Tolerance()
DEFLECTION = 0.03
SCAN_FACE_TARGET = 65_000
KINDS = ("plane", "cylinder")
SETTINGS = {"CAD_JOBS": "1", "CAD_NURBS_FAST": "1", "CAD_EDGE_KEEP": "0"}


def generate_input(name, seed, output, face_target=SCAN_FACE_TARGET):
    shape = (fixture if name in FIXTURES else extra_fixture)(name)
    reference = output / f"{name}.step"
    cq.exporters.export(shape, str(reference))
    shape = load_step(reference)
    base, _, _ = tessellate(shape, 0.02, 0.1)
    base.merge_vertices()
    base.update_faces(base.nondegenerate_faces() & base.unique_faces())
    base.remove_unreferenced_vertices()
    base = refine_edges(base, 3.0)
    if len(base.faces) > face_target:
        # Request a smaller synthetic mesh before adding noise. QEM can stop above the target.
        base = base.simplify_quadric_decimation(face_count=face_target)
        if not base.is_watertight:
            raise RuntimeError(f"{name}: decimation opened the scan mesh")
    rng = np.random.default_rng(seed)
    return shape, base, rng.standard_normal(base.vertices.shape)


def mesh_with_face_types(nurbs, body):
    mesh = nurbs.body_mesh(body, DEFLECTION)
    kinds = []
    ex = TopExp_Explorer(body.wrapped, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        tri = BRep_Tool.Triangulation_s(face, TopLoc_Location())
        if tri is not None:
            kinds.extend([cq.Face(face).geomType()] * tri.NbTriangles())
        ex.Next()
    if not len(mesh.faces) or len(kinds) != len(mesh.faces):
        raise RuntimeError("triangulation missing or triangle provenance is inconsistent")
    return mesh, np.array(kinds)


def probes(mesh, seed):
    return trimesh.sample.sample_surface(mesh, TOL.samples, seed=seed)


def closest(mesh, points):
    parts = [trimesh.proximity.closest_point(mesh, points[i:i + 96])
             for i in range(0, len(points), 96)]
    return np.concatenate([x[1] for x in parts]), np.concatenate([x[2] for x in parts])


def distribution(distances):
    return {"within_share": float(np.mean(distances <= TOL.cover_mm)),
            "p95_mm": float(np.quantile(distances, 0.95)),
            "max_sample_mm": float(np.max(distances))}


def fixed_probes(mesh, points):
    return closest(mesh, points)[0]


def analytic_features(shape):
    # Retain task 12's per-face trim sampling while skipping global relative meshing.
    # The body-wide mesh for coverage uses the separate absolute deflection above.
    with patch.object(shape, "tessellate", return_value=([], [])):
        return extract(shape, TOL)


def feature_summary(rows):
    summary = {}
    for kind in KINDS:
        reference = [r for r in rows if r["reference"] and r["reference"]["kind"] == kind]
        extra = [r for r in rows if r["status"] == "extra" and r["model"]["kind"] == kind]
        counts = Counter(r["status"] for r in reference)
        summary[kind] = {"reference": len(reference), "ok": counts["ok"],
                         "parameter_error": counts["parameter_error"], "missing": counts["missing"],
                         "extra": len(extra),
                         "ok_share": counts["ok"] / len(reference) if reference else None}
    return summary


def build(nurbs, clean, scan, noise, enabled, reference, ref_mesh, fixed, output, *,
          replace_analytic=True):
    os.environ["CAD_REGIONS"] = str(int(enabled))
    nurbs._NO_REGIONS = False
    nurbs._SCAN = scan
    nurbs.FACE_TOL = min(0.5, max(0.3, noise))
    triangles, nurbs.K, nurbs.NP, nurbs.PAIR = nurbs.STEPS[0]
    profile = {}
    projector = nurbs._Projector(clean)
    context = (nullcontext() if replace_analytic else
               patch.object(nurbs, "_merge_planes", side_effect=lambda sewn, *_: (sewn, False)))
    start = perf_counter()
    with context:
        body, count, _ = nurbs._build(clean, triangles, projector, profile=profile, stage={})
    seconds = perf_counter() - start
    report = {"build_seconds": seconds, "build_profile_seconds": profile, "reported_faces": count,
              "status": "no_solid" if body is None else "built"}
    if body is None:
        return report, None

    cq.exporters.export(body, str(output))
    shape = load_step(output)
    report["topology"] = topology(shape)
    top = report["topology"]
    report["positive_volume"] = top["volume_mm3"] > 0
    report["status"] = "valid" if top["valid"] and top["closed"] and top["connected_solid"] else "invalid"
    if report["status"] != "valid":
        return report, None

    model, types = mesh_with_face_types(nurbs, shape)
    distances, nearest = closest(model, fixed)
    reverse = fixed_probes(ref_mesh, probes(model, TOL.seed + 1)[0])
    rows = match(reference, analytic_features(shape), TOL)
    report.update(features=feature_summary(rows), feature_rows=rows,
                  surface={"reference_to_model": distribution(distances),
                           "model_to_reference": distribution(reverse)},
                  mesh_triangles=len(model.faces))
    return report, (distances, types[nearest])


def delta(baseline, candidate, points, truth_types):
    if baseline is None or candidate is None:
        return None
    d0, t0 = baseline
    d1, t1 = candidate
    missed = np.flatnonzero((d0 <= TOL.cover_mm) & (d1 > TOL.cover_mm))
    recovered = np.flatnonzero((d0 > TOL.cover_mm) & (d1 <= TOL.cover_mm))
    transitions = Counter(f"{truth_types[i]} / {t0[i]} → {t1[i]}" for i in missed)
    largest = missed[np.argsort((d1 - d0)[missed])[-5:][::-1]]
    return {"new_misses": len(missed), "recovered": len(recovered),
            "new_misses_by_reference_and_nearest_face_types": dict(transitions),
            "largest_new_misses": [
                {"xyz_mm": points[i].tolist(), "ref_type": str(truth_types[i]),
                 "before_mm": float(d0[i]), "after_mm": float(d1[i]),
                 "before_nearest_type": str(t0[i]), "after_nearest_type": str(t1[i])}
                for i in largest
            ]}


def write_checkpoint(path, data):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def describe(report):
    result = [report["status"], round(report["build_seconds"], 1), report["reported_faces"]]
    if report["status"] == "valid":
        result.extend(round(x["within_share"] * 100, 2) for x in report["surface"].values())
        result.extend(f"{x['ok']}/{x['reference']}" for x in report["features"].values())
    return result


def audit_inputs(args, nurbs, surface):
    rows = []
    for name in args.case:
        shape, base, noise_field = generate_input(name, args.seed + CASES.index(name), args.work, args.scan_face_target)
        ref_mesh, _ = mesh_with_face_types(nurbs, shape)
        fixed, _ = probes(ref_mesh, TOL.seed)
        for noise in NOISE:
            scan = base.copy()
            scan.vertices += noise * noise_field
            clean = surface.flatten_planes(surface.surface_mesh(scan.copy()))
            row = {"case": name, "noise_sigma_mm": noise,
                   "raw_triangles": len(scan.faces), "clean_triangles": len(clean.faces),
                   "raw": distribution(fixed_probes(scan, fixed)),
                   "clean": distribution(fixed_probes(clean, fixed))}
            rows.append(row)
            print(row, flush=True)
    write_checkpoint(args.work / "input_audit.json", rows)


def audit_controls(args, nurbs):
    data = json.loads((args.work / "results.json").read_text(encoding="utf-8"))
    references = {}
    rows = []
    for pair in data["pairs"]:
        if pair.get("spline_control", {}).get("status") != "valid":
            continue
        name, noise = pair["case"], pair["noise_sigma_mm"]
        if name not in args.case:
            continue
        if name not in references:
            references[name] = mesh_with_face_types(nurbs, load_step(args.work / f"{name}.step"))
        ref_mesh, ref_types = references[name]
        row = {"case": name, "noise_sigma_mm": noise}
        for variant in ("on", "spline-control"):
            shape = load_step(args.work / f"{name}-{noise:.2f}-{variant}.step")
            model, types = mesh_with_face_types(nurbs, shape)
            points, indices = probes(model, TOL.seed + 1)
            distances, nearest = closest(ref_mesh, points)
            owner = types[indices]
            row[variant] = {}
            for kind in sorted(set(owner)):
                subset = distances[owner == kind]
                row[variant][kind] = {"samples": len(subset),
                                      "outside": int(np.count_nonzero(subset > TOL.cover_mm)),
                                      **distribution(subset)}
            missed = np.flatnonzero((distances > TOL.cover_mm) & np.isin(owner, ("PLANE", "CYLINDER")))
            row[f"{variant}_analytic_misses_by_nearest_reference_type"] = dict(Counter(
                f"{owner[i]} → {ref_types[nearest[i]]}" for i in missed))
            largest = missed[np.argsort(distances[missed])[-5:][::-1]]
            row[f"{variant}_largest_analytic_misses"] = [
                {"xyz_mm": points[i].tolist(), "distance_mm": float(distances[i]), "kind": str(owner[i]),
                 "nearest_reference_type": str(ref_types[nearest[i]])}
                for i in largest
            ]
        rows.append(row)
        print(name, noise, row["on"], flush=True)
    write_checkpoint(args.work / "control_audit.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--case", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--seed", type=int, default=15012)
    parser.add_argument("--scan-face-target", type=int, default=SCAN_FACE_TARGET)
    parser.add_argument("--diagnose", action="store_true",
                        help="repeat ON without analytic replacement when paired coverage drops")
    audits = parser.add_mutually_exclusive_group()
    audits.add_argument("--audit-input", action="store_true", help="measure raw and cleaned input only")
    audits.add_argument("--audit-controls", action="store_true",
                        help="localize reverse coverage misses using the saved control STEP files")
    args = parser.parse_args()
    if args.scan_face_target < 1500:
        parser.error("--scan-face-target must be at least the first-stage triangle budget (1500)")
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ.update(SETTINGS)
    sys.path.insert(0, str(args.demo.resolve()))
    import nurbs
    import surface

    if args.audit_input:
        audit_inputs(args, nurbs, surface)
        return
    if args.audit_controls:
        audit_controls(args, nurbs)
        return

    sha = subprocess.check_output(["git", "-C", str(args.demo), "rev-parse", "HEAD"], text=True).strip()
    result = {"baseline_commit": sha, "seed": args.seed, "stage": list(nurbs.STEPS[0]),
              "measurement": {"tolerance": vars(TOL), "mesh_deflection_mm": DEFLECTION,
                              "common_environment": SETTINGS, "diagnose": args.diagnose},
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "logical_cpus": os.cpu_count(), "packages": {
                              name: metadata.version(name) for name in
                              ("cadquery", "cadquery-ocp", "numpy", "scipy", "trimesh", "fast-simplification")}},
              "scan_face_target": args.scan_face_target, "expected_pairs": 2 * len(args.case), "pairs": []}
    path = args.work / "results.json"
    for name in args.case:
        shape, base, noise_field = generate_input(name, args.seed + CASES.index(name), args.work, args.scan_face_target)
        ref_mesh, types = mesh_with_face_types(nurbs, shape)
        fixed, indices = probes(ref_mesh, TOL.seed)
        reference = analytic_features(shape)
        for noise in NOISE:
            print(f"{name} {noise:.2f} starting", flush=True)
            scan = base.copy()
            scan.vertices += noise * noise_field
            clean = surface.flatten_planes(surface.surface_mesh(scan.copy()))
            key = f"{name}-{noise:.2f}"
            row = {"case": name, "noise_sigma_mm": noise, "scan_triangles": len(scan.faces),
                   "clean_triangles": len(clean.faces), "scan_watertight": bool(scan.is_watertight),
                   "clean_watertight": bool(clean.is_watertight),
                   "scan_sha256": hashlib.sha256(scan.vertices.tobytes() + scan.faces.tobytes()).hexdigest()}
            variants = {}
            order = (False, True) if (CASES.index(name) + NOISE.index(noise)) % 2 == 0 else (True, False)
            for enabled in order:
                variant = "on" if enabled else "off"
                row[variant], variants[variant] = build(nurbs, clean.copy(), scan.copy(), noise, enabled,
                                                       reference, ref_mesh, fixed,
                                                       args.work / f"{key}-{variant}.step")
                print(key, variant, describe(row[variant]), flush=True)
            row["paired_reference_probes"] = delta(variants["off"], variants["on"], fixed, types[indices])
            if args.diagnose and all(row[v]["status"] == "valid" for v in ("off", "on")):
                # Isolate analytic replacement while preserving ON layout, tags and fitted patches.
                before, after = (row[v]["surface"] for v in ("off", "on"))
                loss = max(before[k]["within_share"] - after[k]["within_share"] for k in before)
                if loss >= 0.002 and any(row["on"]["topology"]["surface_types"].get(k, 0)
                                         for k in ("PLANE", "CYLINDER")):
                    control, sample = build(nurbs, clean.copy(), scan.copy(), noise, True, reference, ref_mesh,
                                            fixed, args.work / f"{key}-spline-control.step",
                                            replace_analytic=False)
                    row["spline_control"] = control
                    row["control_to_on_reference_probes"] = delta(sample, variants["on"], fixed, types[indices])
                    print(key, "spline_control", describe(control), flush=True)
            result["pairs"].append(row)
            write_checkpoint(path, result)


if __name__ == "__main__":
    main()
