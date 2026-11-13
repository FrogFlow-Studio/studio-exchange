"""Exercise a curved annulus with a noncircular outer trim and a cylinder bore."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec

from any_zone import fit_zone_any
from compare import compare
from nurbs import body_mesh
from patch_faces import _mesh_cover


def fixture(noise, seed=48):
    theta = np.linspace(0,2*np.pi,180,endpoint=False)
    width = 1/np.sqrt((np.cos(theta)/24)**2+(np.sin(theta)/16)**2)
    q = np.linspace(0,1,36)[:,None]
    radius = 5+q*(width-5)
    x,y = radius*np.cos(theta),radius*np.sin(theta)
    z = 3*(1-(x/24)**2-(y/16)**2)
    index = np.arange(x.size).reshape(x.shape)
    a,b = index[:-1],index[1:]
    c,d = np.roll(b,-1,axis=1),np.roll(a,-1,axis=1)
    faces = np.vstack((np.column_stack((a.ravel(),b.ravel(),c.ravel())),
                       np.column_stack((a.ravel(),c.ravel(),d.ravel()))))
    points = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    clean = points.copy()
    points[:,2] += np.random.default_rng(seed).normal(0,noise,len(points))
    neighbors = {}
    for row, record in ((index[-1],dict(kind="plane",origin=(0,0,0),normal=(0,0,1))),
                        (index[0],dict(kind="cylinder",origin=(0,0,0),axis=(0,0,1),r=5))):
        for i,j in zip(row,np.roll(row,-1)):
            neighbors[tuple(sorted((int(i),int(j))))] = record
    return points,faces,neighbors,trimesh.Trimesh(clean,faces,process=False)


def run(work):
    results = []
    for noise in (0.,.1,.2):
        points, faces, neighbors, clean = fixture(noise)
        started = perf_counter()
        # Keep the raw-scan guard at 97% for lower noise. The 0.2 mm
        # case also has an independent, stricter clean reference below.
        output = fit_zone_any(points,faces,np.arange(len(faces)),neighbors,
                              min_scan_coverage=.95 if noise==.2 else .97)
        row = dict(noise=noise, seconds=perf_counter()-started, accepted=bool(output))
        if output:
            patch = cq.Face(output[0])
            step = work/f"annulus-{noise}.step"
            cq.exporters.export(patch,str(step))
            reopened, = cq.importers.importStep(str(step)).val().Faces()
            model = body_mesh(reopened,.03)
            model.merge_vertices()
            result = compare(clean,model,.3,n=12000,seed=48)
            # Reuse the actual outer edge on the plane and the actual inner
            # edge on the bore. This checks more than fitting to nearby points.
            outer, inner = sorted(patch.Wires(),key=lambda wire: -wire.Length())
            disk = BRepBuilderAPI_MakeFace(outer.wrapped,True).Face()
            sampled = []
            sewing = BRepBuilderAPI_Sewing(1e-5)
            sewing.Add(patch.wrapped)
            sewing.Add(disk)
            for edge in inner.Edges():
                curve = BRepAdaptor_Curve(edge.wrapped)
                sampled.extend(curve.Value(t).Coord() for t in
                               np.linspace(curve.FirstParameter(),curve.LastParameter(),360))
                sewing.Add(BRepPrimAPI_MakePrism(edge.wrapped,gp_Vec(0,0,-5)).Shape())
            sewing.Perform()
            row.update(valid=bool(reopened.isValid()), wire_count=len(patch.Wires()),
                       mesh_cover=_mesh_cover(reopened.wrapped,reopened.Area()),
                       coverage=[result["within_tol_share"],1-result["added_surface_share"]],
                       cylinder_error=float(np.max(abs(np.linalg.norm(np.asarray(sampled)[:,:2],axis=1)-5))),
                       stitched_edges=sewing.NbContigousEdges())
        results.append(row)
        print(json.dumps(row,allow_nan=False),flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    results = run(args.work)
    (args.work/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf8")
    assert all(row["accepted"] and row["valid"] and row["wire_count"] == 2 and row["mesh_cover"] >= .98
               and min(row["coverage"]) >= .99 and row["cylinder_error"] <= 1e-4
               and row["stitched_edges"] >= 2 for row in results)
