"""Run follow-up checks on the same public synthetic fixtures."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
import trimesh

import repair_adaptive
import wall_decision
from benchmark_36_38 import fixture, measure, stage, stock
from benchmark_30_32 import wall_fixture
from straight_fuzz import scan_mesh


def folds(work):
    rows = []
    cases = [(36,10,90,0.),(36,10,90,.2),(36,20,60,.2),(36,30,30,0.),(36,30,30,.2),
             (49,10,90,.2),(61,20,60,.2),(79,30,30,.2)]
    for seed, radius, turn, noise in cases:
        path = work/f"R{radius}.step"
        cq.exporters.export(fixture(radius,turn), str(path))
        scan = scan_mesh(path,noise,seed)
        state = stage(scan,noise)
        body, _ = stock(*state,smooth_tol=.3,smooth_edge_tol=.075)
        before = measure(scan,body)
        result, report = repair_adaptive.repair_build(*state,scan,build=stock)
        after = measure(scan,result)
        row = dict(seed=seed,radius=radius,noise=noise,before=before,after=after,
                   valid=report["valid"],passes=report["repair_passes"])
        print(row, flush=True)
        rows.append(row)
    rows.append(time_repair())
    return rows


def time_repair():
    # Timing isolates one local repair, without rebuilding the B-rep.
    scan = trimesh.creation.icosphere(subdivisions=6,radius=25.)
    keep = np.arange(50000)
    mesh = trimesh.Trimesh(scan.vertices,scan.faces[keep],process=False)
    mesh.remove_unreferenced_vertices()
    center = mesh.triangles_center[1000].copy()
    distance = np.linalg.norm(mesh.vertices-center,axis=1)
    bump = .3*np.maximum(0.,1.-distance/3)
    mesh.vertices += bump[:,None]*mesh.vertex_normals
    defect = {"type":"смятая поверхность","center":center,"area_mm2":50}
    start=perf_counter()
    *_, audit = repair_adaptive.repair(mesh.vertices, mesh.faces, np.full(len(mesh.faces),-1), [], scan,[defect],.3)
    return {"timing_triangles":len(mesh.faces),"seconds":perf_counter()-start,"audit":audit}


def walls():
    seeds = (32,39,82,101,107,109,113,127,131,137)
    rows=[]
    for radius in (40.,100.,200.):
        for noise in (.05,.1,.2):
            for seed in seeds:
                for polygon in (False,True):
                    mesh,seg,count=wall_fixture(radius,0.,polygon=polygon)
                    ids=np.unique(mesh.faces[seg["label"]<count])
                    offsets=np.random.default_rng(seed).normal(0,noise,len(ids))
                    mesh.vertices[ids,1]+=np.clip(offsets,-2.5*noise,2.5*noise)
                    audit=[]
                    _, groups=wall_decision.unlabel_curved_walls(mesh,seg,.3,noise,decisions=audit)
                    rows.append(dict(radius=radius,noise=noise,seed=seed,polygon=polygon,arc=bool(groups),decisions=audit))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("39","40","41"))
    p.add_argument("--work",type=Path,required=True)
    a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    if a.task=="39":
        rows=folds(a.work)
    elif a.task=="40":
        from fillet_supports import benchmark
        rows=benchmark(a.work)
    else:
        rows=walls()
    (a.work/f"task{a.task}.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
