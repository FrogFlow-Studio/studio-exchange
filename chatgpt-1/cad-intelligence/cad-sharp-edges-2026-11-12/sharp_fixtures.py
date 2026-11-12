"""Task 21a synthetic STEP fixtures with narrow slots, window, and steps."""
from pathlib import Path
import sys

import cadquery as cq
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"cad-coarse-bores-2026-11-12"))
from benchmark_coarse_bores import mesh_step


def fixture(small):
    width,height=(50,44) if small else (87,62)
    h=8.
    body=cq.Workplane("XY").box(width,height,h)
    corners=[]
    for x,y,a,b in ((-15,-14,12,3),(8,-14,20,4),(5,6,25,25)):
        body=body.cut(cq.Workplane("XY").center(x,y).box(a,b,2*h))
        corners.extend((x+i*a/2,y+j*b/2,k*h/2) for i in (-1,1) for j in (-1,1) for k in (-1,1))
    for x,y,a,b,z in ((-18,8,9,13,1),(-17,16,9,6,2)):
        body=body.union(cq.Workplane("XY").center(x,y).box(a,b,z+1).translate((0,0,(h+z-1)/2)))
    return body.val(),np.asarray(corners)


def generate(root):
    root.mkdir(parents=True,exist_ok=True)
    for small in (False,True):
        body,corners=fixture(small)
        name="button" if small else "plate"
        for step in (1.4,2.3):
            case=root/f"{name}-{step}"
            case.mkdir(exist_ok=True)
            transform=trimesh.transformations.rotation_matrix(np.radians(6 if step==1.4 else 9),(1,2,0))
            axis=cq.Vector(1,2,0)
            posed=body.rotate((0,0,0),axis.toTuple(),6 if step==1.4 else 9)
            cq.exporters.export(posed,str(case/"truth.step"))
            np.save(case/"corners.npy",trimesh.transform_points(corners,transform))
            clean=mesh_step(case/"truth.step",fine=step,coarse=step,noise=0.)
            rng=np.random.default_rng(210121)
            noise=rng.normal(size=clean.vertices.shape)
            for level in (.05,.2):
                scan=clean.copy()
                scan.vertices+=level*noise
                scan.export(case/f"scan-{level}.ply")
            print(case.name,len(clean.faces),round(float(np.median(clean.edges_unique_length)),3),flush=True)


if __name__=="__main__":
    generate(Path(sys.argv[1]))
