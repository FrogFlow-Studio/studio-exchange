"""Synthetic rectangular charts, noisy scans, STEP round trip and coverage."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_details import mesh, solidify
from compare import compare
from large_surfaces import build, rectangular_chart, step_audit
from nurbs import body_mesh


def cases():
    for radius in (50.,100.,300.):
        yield f"arc_{radius:g}", lambda x,y,r=radius: r-np.sqrt(r*r-x*x)
    yield "saddle", lambda x,y:(x*x-y*y)/200.
    yield "dome_100", lambda x,y:100-np.sqrt(100**2-x*x-y*y)
    yield "torus", lambda x,y:120-np.sqrt((95+np.sqrt(25**2-y*y))**2-x*x)
    # Local outward button-like chart, with the up direction rotated in world space.
    yield "button_side_control", lambda x,y:35-np.sqrt(35**2-x*x)+y*y/230.


def run(work, selected=None, seed=4701, irregular=False):
    rows=[]
    for name,height in cases():
        if selected and name not in selected:
            continue
        width, depth=48.,24.
        x,y=np.meshgrid(np.linspace(-width/2,width/2,145),np.linspace(-depth/2,depth/2,73),indexing="ij")
        if irregular:
            rng=np.random.default_rng(seed+1)
            x[1:-1,:]+=rng.uniform(-.09,.09,x[1:-1,:].shape)
            y[:,1:-1]+=rng.uniform(-.09,.09,y[:,1:-1].shape)
        z=height(x,y)
        reference=solidify(mesh(x,y,z))
        axes=Rotation.from_euler("xyz",[68,21,-12],degrees=True).as_matrix() if name.startswith("button") else np.eye(3)
        shift=np.array([17.,-7.,14.]) if name.startswith("button") else np.zeros(3)
        reference.vertices=reference.vertices @ axes+shift
        normals=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]) @ axes
        supports=np.array([[width/2,0,0],[-width/2,0,0],[0,depth/2,0],[0,-depth/2,0]]) @ axes+shift
        bottom=np.array([0,0,z.min()-5.]) @ axes+shift
        for noise in (0.,.1,.2):
            rng=np.random.default_rng(seed)
            scan=mesh(x,y,z+rng.normal(0,noise,z.shape))
            scan.vertices=scan.vertices @ axes+shift
            start=perf_counter()
            samples,chart=rectangular_chart(scan,np.arange(len(scan.faces)),normals,supports,bottom)
            body,fit=build(samples,chart)
            fit_seconds=perf_counter()-start
            restored,audit=step_audit(body,work/f"{name}-{noise:g}.step")
            tess=body_mesh(restored,.025)
            tess.merge_vertices()
            result=compare(reference,tess,.3,n=2400,seed=47)
            row={"case":name,"sigma_mm":noise,"seed":seed,"irregular":irregular,**fit,**audit,"fit_seconds":fit_seconds,
                 "total_seconds":perf_counter()-start,"triangle_area_share":0.,
                 "coverage":[result["within_tol_share"],1-result["added_surface_share"]]}
            rows.append(row)
            print(name,noise, row["spline_faces"],row["seam_angle_max_deg"],row["mesh_cover_min"],row["coverage"],flush=True)
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--case",action="append",choices=[name for name,_ in cases()])
    parser.add_argument("--seed",type=int,default=4701)
    parser.add_argument("--irregular",action="store_true")
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    rows=run(args.work,args.case,args.seed,args.irregular)
    (args.work/"results.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
    assert all(r["valid"] and r["mesh_cover_min"]>=.98 and r["seam_angle_max_deg"]<=2
               and r["shared_seams"] and r["spline_faces"]*25<=r["spline_area_mm2"]
               and min(r["coverage"])>=.99 for r in rows)
