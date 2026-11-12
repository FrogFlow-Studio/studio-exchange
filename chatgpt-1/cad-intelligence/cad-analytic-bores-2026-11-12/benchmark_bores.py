"""Measure one 0k stage and analytic vs. boolean bore construction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import cadquery as cq
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
for folder in ("cad-smooth-acceptance", "cad-acceptance", "cad-regions", "cad-region-layout"):
    sys.path.insert(0, str(HERE.parent / f"{folder}-2026-11-12"))

from benchmark_smooth import CASES as SYNTHETIC, FIXTURES, extra_fixture, fixture, generate_input
from geometry_acceptance import Tolerance, extract, load_step, match, topology, write_json

SYNTHETIC_CASES = ("drilled_plate", "counterbored_plate", "ribbed_housing", "spherical_end", "slotted_block")
SAMPLES = ("flange", "bushing", "bracket")
TOL = Tolerance(radius_mm=0.1, axis_angle_deg=2.0, cover_mm=0.3)


def uniform_input(name, work):
    import gmsh

    reference = (fixture if name in FIXTURES else extra_fixture)(name)
    step = work / f"{name}.step"
    cq.exporters.export(reference, str(step))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
        gmsh.model.add(name)
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, nodes = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    order = np.argsort(tags)
    triangles = order[np.searchsorted(tags[order], nodes)].reshape(-1, 3)
    scan = trimesh.Trimesh(np.reshape(xyz, (-1, 3)), triangles, process=True)
    trimesh.repair.fix_normals(scan)
    if not scan.is_watertight or not scan.is_winding_consistent:
        raise RuntimeError(f"{name}: Gmsh did not produce a closed surface")
    noise = np.random.default_rng(15012 + SYNTHETIC.index(name)).normal(0, 0.05, scan.vertices.shape)
    scan.vertices += noise
    return scan


def prepare(name, demo, work, nurbs, surface, uniform):
    if name in SAMPLES:
        scan = trimesh.load(demo / "samples" / f"{name}_scan.stl", force="mesh")
        reference = load_step(demo / "samples" / f"{name}_truth.step")
        cq.exporters.export(reference, str(work / f"{name}.step"))
    elif uniform:
        scan = uniform_input(name, work)
    else:
        _, base, noise = generate_input(name, 15012 + SYNTHETIC.index(name), work)
        scan = base.copy()
        scan.vertices += noise * 0.05
    scan.export(work / f"{name}.ply")
    started = perf_counter()
    specs = surface.hole_parameters(scan)
    search = perf_counter() - started
    write_json(work / f"{name}-specs.json", specs)
    clean = surface.flatten_planes(surface.surface_mesh(scan))
    nurbs._SCAN = scan
    nurbs.FACE_TOL = 0.3
    triangles, nurbs.K, nurbs.NP, nurbs.PAIR = nurbs.STEPS[0]
    projector = nurbs._Projector(clean)
    start = perf_counter()
    body, count, _ = nurbs._build(clean, triangles, projector, profile={}, stage={})
    report = {"seconds": perf_counter() - start, "search_seconds": search,
              "faces": count, "built": body is not None, "uniform": uniform,
              "scan_triangles": len(scan.faces),
              "edge_lengths_mm_p10_p50_p90": np.quantile(scan.edges_unique_length, (.1, .5, .9)).tolist()}
    if body is not None:
        cq.exporters.export(body, str(work / f"{name}-stage.step"))
    write_json(work / f"{name}-stage.json", report)


def verify(body, scan, ref_mesh, ref_features, nurbs):
    from analytic_bores import coverage_share

    def coverage(source, target):
        result = {}
        for key, a, b, seed in (("reference_to_model", source, target, TOL.seed),
                                ("model_to_reference", target, source, TOL.seed + 1)):
            points = trimesh.sample.sample_surface(a, TOL.samples, seed=seed)[0]
            result[key] = {"within_share": coverage_share(b, points, TOL.cover_mm)}
        return result

    mesh = nurbs.body_mesh(body, 0.03)
    with patch.object(body, "tessellate", return_value=([], [])):
        rows = match(ref_features, extract(body, TOL), TOL)
    matched = [row for row in rows if row["reference"] and row["reference"]["role"] == "bore"]
    return {"topology": topology(body), "scan_coverage": coverage(scan, mesh),
            "reference_coverage": coverage(ref_mesh, mesh),
            "bores_ok": sum(row["status"] == "ok" for row in matched), "bore_rows": matched}


def run_boolean(body, scan, specs, surface, step, result):
    timings = {}
    start = perf_counter()
    candidate, count = surface.cut_holes(body, scan, hole_finder=lambda _: specs,
                                        timings=timings, limit_time=True)
    timings.update(seconds=perf_counter() - start, accepted=count)
    cq.exporters.export(candidate, str(step))
    write_json(result, timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=(*SYNTHETIC_CASES, *SAMPLES), nargs="+",
                        default=(*SYNTHETIC_CASES, *SAMPLES))
    parser.add_argument("--reuse", action="store_true", help="reuse the local one-stage and boolean files")
    parser.add_argument("--uniform", action="store_true", help="use Gmsh 1 mm meshes for synthetic cases")
    parser.add_argument("--boolean", choices=(*SYNTHETIC_CASES, *SAMPLES), nargs="*",
                        default=("drilled_plate", "counterbored_plate", "ribbed_housing"))
    args = parser.parse_args()
    if args.uniform and any(case in SAMPLES for case in args.case):
        parser.error("--uniform is only available for synthetic cases")
    args.work.mkdir(parents=True, exist_ok=True)
    demo = args.demo.resolve()
    sys.path.insert(0, str(demo))
    os.environ.update(CAD_REGIONS="1", CAD_JOBS="1", CAD_EDGE_KEEP="0", CAD_NURBS_FAST="1", CAD_HOLE_BATCH="0")
    import nurbs
    import surface
    from analytic_bores import try_analytic_bores

    sha = subprocess.check_output(["git", "-C", str(demo), "rev-parse", "HEAD"], text=True).strip()
    result = {"base": sha, "stage": list(nurbs.STEPS[0]), "tolerance": vars(TOL),
              "synthetic_noise_sigma_mm": 0.05, "uniform_mesh_target_mm": 1.0 if args.uniform else None,
              "cases": {}}
    for name in args.case:
        print(name, "start", flush=True)
        def path(suffix):
            return args.work / f"{name}{suffix}"

        if not args.reuse or not path("-stage.json").exists():
            prepare(name, demo, args.work, nurbs, surface, args.uniform)
        scan = trimesh.load(path(".ply"), force="mesh")
        specs = json.loads(path("-specs.json").read_text(encoding="utf-8"))
        stage = json.loads(path("-stage.json").read_text(encoding="utf-8"))
        if bool(stage.get("uniform", False)) != args.uniform:
            raise ValueError(f"{name}: use a separate cache directory for this mesh mode")
        reference = load_step(path(".step"))
        ref_mesh = nurbs.body_mesh(reference, 0.03)
        with patch.object(reference, "tessellate", return_value=([], [])):
            ref_features = extract(reference, TOL)
        row = {"detected": len(specs), "stage": stage,
               "reference_bands": sum(f["role"] == "bore" for f in ref_features)}
        result["cases"][name] = row
        if stage["built"]:
            baseline = load_step(path("-stage.step"))
            start = perf_counter()
            candidate, report = try_analytic_bores(baseline, scan, specs, TOL.cover_mm)
            report["seconds"] = perf_counter() - start
            row["trial"] = report
            row["baseline"] = verify(baseline, scan, ref_mesh, ref_features, nurbs)
            if name in args.boolean:
                step, timings = path("-boolean.step"), path("-boolean.json")
                if not args.reuse or not (step.exists() and timings.exists()):
                    run_boolean(baseline, scan, specs, surface, step, timings)
                row["boolean_timing"] = json.loads(timings.read_text(encoding="utf-8"))
                row["boolean"] = verify(load_step(step), scan, ref_mesh, ref_features, nurbs)
            if report["status"] == "accepted":
                candidate_step = path("-analytic.step")
                cq.exporters.export(candidate, str(candidate_step))
                row["analytic"] = verify(load_step(candidate_step), scan, ref_mesh, ref_features, nurbs)
                top = row["analytic"]["topology"]
                assert top["valid"] and top["closed"] and top["connected_solid"] and top["volume_mm3"] > 0
                for label in ("scan_coverage", "reference_coverage"):
                    for direction, entry in row["analytic"][label].items():
                        assert entry["within_share"] >= row["baseline"][label][direction]["within_share"]
                assert row["analytic"]["bores_ok"] > row["baseline"]["bores_ok"]
        write_json(args.output or args.work / "results.json", result)
        print(name, {v: row[v].get("bores_ok") for v in ("baseline", "boolean", "analytic") if v in row},
              row.get("trial"), flush=True)


if __name__ == "__main__":
    main()
