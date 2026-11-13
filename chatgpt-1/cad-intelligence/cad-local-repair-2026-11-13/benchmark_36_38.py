"""Run repeatable repair, fillet growth and wall discrimination controls."""
import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np

import curved_walls
import fillet_ends
import idealize
import planar_brep
import repair
import trimesh
from benchmark_30_32 import wall_fixture
from compare import compare
from defects import analyze
from nurbs import body_mesh
from straight_fuzz import scan_mesh
from straight_plus import augment


def fixture(radius, turn):
    a = np.radians(turn)
    start = np.array([0., radius])
    end = radius*np.array([np.sin(a), np.cos(a)])
    mid = radius*np.array([np.sin(a/2), np.cos(a/2)])
    p = start - np.array([15., 0.])
    q = end + np.array([np.cos(a), -np.sin(a)]) * 10
    return (cq.Workplane("XY").moveTo(*p).lineTo(*start).threePointArc(tuple(mid), tuple(end))
            .lineTo(*q).lineTo(q[0], min(q[1], p[1])-10).lineTo(p[0], min(q[1], p[1])-10)
            .close().extrude(14).val())


def stage(scan, noise):
    seg, _ = augment(scan, .3, noise)
    mesh, info = idealize.idealize(scan, noise=noise, seg=seg)
    return (*planar_brep.collapse_degenerate(mesh.vertices, mesh.faces, info["label"], .001), info["regions"])


def measure(scan, body):
    mesh = body_mesh(body, .05)
    mesh.merge_vertices()
    defects = analyze(mesh, scan)
    report = compare(scan, mesh, .3, n=3000, seed=36)
    return {"fold_area_mm2": sum(d["area_mm2"] for d in defects if d["type"] == "смятая поверхность"),
            "coverage": [report["within_tol_share"], 1-report["added_surface_share"]]}


def stock(*args, **kwargs):
    try:
        return planar_brep.build(*args, **kwargs)
    except UnboundLocalError as exc:
        if "'why'" not in str(exc):
            raise
        return planar_brep.build(*args, **{**kwargs, "smooth_tol": None})


def repair_cases(work, radii=None):
    rows = []
    for radius, turn in ((10, 90), (20, 60), (30, 30)):
        if radii and radius not in radii:
            continue
        truth = fixture(radius, turn)
        path = work / f"blend-{radius}.step"
        cq.exporters.export(truth, str(path))
        for noise in (0., .1, .2):
            scan = scan_mesh(path, noise, 36)
            V, F, L, regions = stage(scan, noise)
            body, base = stock(V,F,L,regions,smooth_tol=.3,smooth_edge_tol=.075)
            before = {**base, **measure(scan,body)}
            fixed, after = repair.repair_build(V,F,L,regions,scan,build=stock)
            row = {"radius": radius, "turn_deg": turn, "noise_mm": noise, "before": before,
                   "after": {**after, **measure(scan, fixed)}}
            rows.append(row)
            print(radius, noise, before["fold_area_mm2"], row["after"]["fold_area_mm2"], after["repair_passes"], flush=True)
    # Isolate the repair operation when automatic labelling does not generate
    # a movable interior. This control keeps the rest of the mesh identical.
    from benchmark_patches import fixture as patch_fixture

    v, f, _ = patch_fixture("dome", 0.)
    top = np.abs(v[:, 0]) < 2.7
    top &= np.abs(v[:, 1]) < 2.7
    damaged = v.copy()
    damaged[top, 2] += .48 * np.cos(np.pi * v[top, 0]/1.2) * np.cos(np.pi * v[top, 1])
    scan = trimesh.Trimesh(v, f, process=False)
    model = trimesh.Trimesh(damaged, f, process=False)
    faults = analyze(model, scan)
    tags = np.full(len(f), -1)
    fixed, faces, _, _, audit = repair.repair(damaged, f, tags, [], scan, faults, .3)
    after = analyze(trimesh.Trimesh(fixed, faces, process=False), scan)
    area = lambda findings: sum(row["area_mm2"] for row in findings if row["type"] == "смятая поверхность")
    rows.append({"case": "isolated_fold", "fold_area_mm2": [area(faults), area(after)], "audit": audit})
    print(rows[-1], flush=True)
    return rows


def fillet_cases(work):
    import importlib.util
    module_path = Path(__file__).parent.parent/"cad-fillet-defects-2026-11-13"/"planar_brep_cyl.py"
    spec = importlib.util.spec_from_file_location("north_exact", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for kind in ("block", "angle"):
        wp = cq.Workplane("XY").box(24,18,10,centered=(True,True,False))
        shape = wp.edges("|Z").fillet(3) if kind=="block" else wp.edges("|Z and >X").fillet(5).edges("|Z and <X").fillet(2)
        path = work/f"{kind}.step"
        cq.exporters.export(shape.val(),str(path))
        for noise in (0.,.1,.2):
            scan = scan_mesh(path,noise,37)
            V,F,L,regions = stage(scan,noise)
            labels, regs, growth = fillet_ends.extend(V,F,L,regions,.3,noise)
            row={"case":kind,"noise_mm":noise,"growth":growth}
            for tag, builder, args in (("stock",stock,(V,F,L,regions)),
                                       ("before",module.build,(V,F,L,regions)),
                                       ("after",module.build,(V,F,labels,regs))):
                body, audit=builder(*args,smooth_tol=.3,smooth_edge_tol=.075)
                out=work/f"{kind}-{noise}-{tag}.step"
                cq.exporters.export(body,str(out))
                reopened=cq.importers.importStep(str(out)).val()
                row[tag]={**audit,**measure(scan,reopened),"step_valid":bool(reopened.isValid()),
                          "step_cylinders":sum(f.geomType()=="CYLINDER" for f in reopened.Faces())}
            row["coverage_gate"] = all(a >= b-1e-9 for a,b in zip(row["after"]["coverage"], row["stock"]["coverage"]))
            row["selected"] = "after" if row["coverage_gate"] and row["after"]["step_valid"] else "stock"
            rows.append(row)
            print(kind, noise, row["before"]["step_cylinders"], row["after"]["step_cylinders"], growth, flush=True)
    return rows


def wall_cases(seeds):
    rows=[]
    for seed in seeds:
        for radius in (40.,100.,200.):
            for noise in (.05,.2):
                row={"radius":radius,"noise_mm":noise,"seed":seed}
                for polygon in (False,True):
                    mesh, seg, count=wall_fixture(radius,0.,polygon=polygon)
                    ids = np.unique(mesh.faces[seg["label"] < count])
                    offsets = np.random.default_rng(seed).normal(0, noise, len(ids))
                    mesh.vertices[ids,1] += np.clip(offsets,-2.5*noise,2.5*noise)
                    _, groups=curved_walls.unlabel_curved_walls(mesh,seg,.3,noise)
                    row["polygon" if polygon else "arc"]=len(groups)
                rows.append(row)
                print(row,flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("36","37","38"))
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--radius",type=int,action="append")
    parser.add_argument("--seed",type=int,action="append")
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    if args.task == "38":
        rows = wall_cases(args.seed or [32,39,82,101])
    elif args.task == "36":
        rows = repair_cases(args.work, args.radius)
    else:
        rows = fillet_cases(args.work)
    (args.work/f"task{args.task}.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                                                 encoding="utf8")


if __name__=="__main__":
    main()
