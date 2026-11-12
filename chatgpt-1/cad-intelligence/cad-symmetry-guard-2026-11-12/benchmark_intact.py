"""Generate ten intact symmetric button-scale scans."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cadquery as cq
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"cad-coarse-bores-2026-11-12"))
from benchmark_coarse_bores import mesh_step


def intact(seed):
    rng=np.random.default_rng(210112+seed)
    width, height = rng.uniform(25,31), rng.uniform(12,18)
    part=cq.Workplane("XY").box(48,width,height).edges("|Z").fillet(rng.uniform(2,4))
    if seed % 2:
        cut_width=rng.uniform(5,7)
        for x in (-12,12):
            cut=cq.Workplane("XY").center(x,0).box(cut_width,7,6).translate((0,0,height/2)).val()
            part=part.cut(cut)
    else:
        r=rng.uniform(2.5,4)
        for x in (-12,12):
            cut=cq.Workplane("XY").center(x,0).circle(r).extrude(height,both=True)
            part=part.cut(cut)
    return part.val()


def generate(out):
    out.mkdir(parents=True,exist_ok=True)
    for seed in range(10):
        path=out/f"intact-{seed}.step"
        cq.exporters.export(intact(seed),str(path))
        mesh=mesh_step(path, fine=1.4, coarse=1.4, noise=0.)
        mesh.vertices+=np.random.default_rng(220112+seed).normal(0,.17,mesh.vertices.shape)
        mesh.export(path.with_suffix(".ply"))
    path = out / "intact-0.step"
    for index, depth in enumerate((.68, .91, 1.12)):
        mesh = mesh_step(path, fine=1.4, coarse=1.4, noise=0.)
        x, y, z = mesh.vertices.T.copy()
        top = z > z.max() - .1
        weight = np.maximum(0, 1-(x/10)**4) * np.maximum(0, 1-(y/8)**4)
        mesh.vertices[top, 2] -= depth * weight[top]
        mesh.vertices += np.random.default_rng(220112).normal(0, .17, mesh.vertices.shape)
        mesh.export(out / f"bias-{index}.ply")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path / "symmetry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(path, before, after):
    mesh = trimesh.load(path, force="mesh")
    planes = before.find_planes(mesh)
    result = {"file": path.name, "edge_mm_p50": float(np.median(mesh.edges_unique_length))}
    for module, name in ((before, "before"), (after, "after")):
        with patch.object(module, "find_planes", return_value=planes):
            out, log, _ = module.restore_all(mesh)
        result[name] = {"restored": len(log), "closed": bool(out.is_watertight),
                        "chips": [{key: row[key] for key in ("depth","area","volume")} for row in log]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--before-demo", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--chips", type=Path)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.reuse:
        generate(args.work)
    before = load_module(args.before_demo, "baseline_symmetry")
    after = load_module(args.demo, "candidate_symmetry")
    rows = []
    paths = sorted(args.work.glob("intact-*.ply")) + sorted(args.work.glob("bias-*.ply"))
    if args.chips:
        paths += [args.chips / f"{name}_{i}.stl" for name in ("cover","hull") for i in range(3)]
        paths.append(args.chips / "hull_boss_0.stl")
    for path in paths:
        row = compare(path, before, after)
        rows.append(row)
        print(path.name, row["before"]["restored"], row["after"]["restored"], flush=True)
    dest = args.output or args.work / "report.json"
    dest.write_text(json.dumps(rows, indent=2)+"\n", encoding="utf8")
    intact = [row for row in rows if row["file"].endswith(".ply")]
    chips = [row for row in rows if row["file"].startswith(("cover_", "hull_"))
             and row["file"] != "hull_boss_0.stl"]
    return 0 if (len(intact) == 13 and all(not row["after"]["restored"] for row in intact)
                 and all(row["after"]["restored"] and row["after"]["closed"] for row in chips)
                 and all(not row["after"]["restored"] for row in rows if row["file"] == "hull_boss_0.stl")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
