"""Synthetic shallow relief and smoothly bent surfaces with exact defect locations."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from detail_defects import find_details
from preserve import relief, curvature
from nurbs import body_mesh
from compare import compare
import planar_brep


def mesh(x,y,z):
    pts=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    i=np.arange(len(pts)).reshape(x.shape)
    a,b,c,d=(q.ravel() for q in (i[:-1,:-1],i[1:,:-1],i[1:,1:],i[:-1,1:]))
    faces=np.vstack((np.column_stack((a,b,c)),np.column_stack((a,c,d))))
    return trimesh.Trimesh(pts,faces,process=False)


def grid(width, length, nx, ny):
    return np.meshgrid(np.linspace(-width/2,width/2,nx),np.linspace(-length/2,length/2,ny),indexing="ij")


def solidify(surface):
    V,F=surface.vertices,surface.faces
    n=len(V)
    base=V.copy()
    base[:,2]=V[:,2].min()-5.
    border=surface.edges_sorted[np.bincount(surface.edges_unique_inverse)[surface.edges_unique_inverse]==1]
    a,b=border.T
    sides=np.vstack((np.column_stack((a,b,b+n)),np.column_stack((a,b+n,a+n))))
    body=trimesh.Trimesh(np.vstack((V,base)),np.vstack((F,F[:,::-1]+n,sides)),process=False)
    trimesh.repair.fix_normals(body)
    return body


def assemble(surface, labels, regions):
    source=solidify(surface)
    extra=np.full(len(source.faces)-len(surface.faces),-1)
    tags=np.r_[labels,extra]
    # Tag the bottom and four side walls to leave only the intended top patch free.
    for normal in ([0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]):
        ids=np.flatnonzero((source.face_normals @ normal > .9999)&(np.arange(len(tags))>=len(labels)))
        tags[ids]=len(regions)
        regions.append(dict(kind="plane"))
    body, audit=planar_brep.build(source.vertices,source.faces,tags,regions,smooth_tol=.3,
                                  smooth_edge_tol=.075,smooth_budget=5.)
    return source,body,audit


def plate(n=171):
    x,y=grid(42,20,n,81)
    h=np.zeros_like(x)
    centers=[]
    for cx,cy,d,z,ring in ((-15,-4,3,.2,False),(-8,4,4,.3,False),(0,-4,6,.45,False),
                           (9,4,8,.8,True),(16,-4,5,.35,True)):
        radius=np.hypot(x-cx,y-cy)
        profile=np.clip((d/2-radius)/.4,0,1)
        if ring:
            profile*=np.clip((radius-d/4)/.32,0,1)
        h+=z*profile
        centers.append((cx,cy))
    text=((abs(x+1)<.55)&(abs(y-4)<2.2))|((abs(x-2)<.55)&(abs(y-4)<2.2))|((abs(y-4)<.55)&(abs(x-.5)<1.6))
    h+=.4*text
    centers.append((.5,4))
    return mesh(x,y,h),mesh(x,y,0*h),centers


def sheet(radius, strips, kind="arc"):
    # Keep the per-panel angular span inside the detector's 3–20 degree range.
    width=2*radius*np.sin(np.radians(2.1*strips))
    x,y=grid(width,24,1+strips*8,25)
    f=lambda q:radius-np.sqrt(radius**2-q**2)
    z=f(x)
    if kind in ("dome","saddle","torus"):
        z += (1 if kind!="saddle" else -1)*(.007*y*y)*(1+.012*x if kind=="torus" else 1)
    xs=np.linspace(-width/2,width/2,strips+1)
    facets=np.interp(x.ravel(),xs,f(xs)).reshape(x.shape)+(z-f(x))
    labels=np.repeat(np.repeat(np.arange(strips),8),24)
    labels=np.tile(labels,2)
    return mesh(x,y,z),mesh(x,y,facets),labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mesh-only",action="store_true")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--progress",action="store_true")
    a=p.parse_args()
    os.environ["CAD_PATCH_WORKERS"]=str(a.workers)
    rows=[]
    source,flat,centers=plate()
    start=perf_counter()
    findings=find_details(flat,source)
    rows.append(dict(case="relief",seconds=perf_counter()-start,
                     findings=[{k:v for k,v in r.items() if k!="_faces"} for r in findings],
                     errors=[min((float(np.linalg.norm(np.asarray(r["center"][:2])-c)) for r in findings
                                  if r["type"]=="сплющенная мелочь"),default=None) for c in centers],
                     correct=len(find_details(source,source))))
    seg=dict(label=np.zeros(len(source.faces),int),regions=[dict(kind="plane",normal=np.array([0,0,1]),origin=np.zeros(3))])
    for noise in (0,.05,.1):
        scan=source.copy()
        scan.vertices[:,2]+=np.random.default_rng(43).normal(0,noise,len(scan.vertices))
        changed,patches=relief(scan,seg,noise)
        points=flat.vertices.copy()
        ids=np.unique(scan.faces[changed["label"]<0])
        points[ids]=scan.vertices[ids]
        result=trimesh.Trimesh(points,scan.faces,process=False)
        after=find_details(result,scan)
        row=dict(case="preserve_relief",noise=noise,patches=len(patches),area=float(scan.area_faces[changed["label"]<0].sum()),
                 defects=sum(r["type"]=="сплющенная мелочь" for r in after))
        if not a.mesh_only and noise==0:
            if a.progress:
                print("relief B-rep",noise,flush=True)
            _, body, audit=assemble(result,changed["label"],changed["regions"])
            output=body_mesh(body,.04)
            output.merge_vertices()
            row["brep"]=dict(valid=audit["valid"],free_edges=audit["free_edges"],smooth_faces=audit["smooth_faces"],
                             defects=sum(r["type"]=="сплющенная мелочь" for r in find_details(output,solidify(scan))))
        rows.append(row)
    for radius,count,kind in ((50,6,"arc"),(100,12,"arc"),(300,20,"arc"),(100,10,"dome"),(100,10,"saddle"),(100,10,"torus")):
        scan,model,tags=sheet(radius,count,kind)
        for noise in (0.,.1,.2):
            ref=scan.copy()
            ref.vertices[:,2]+=np.random.default_rng(42).normal(0,noise,len(ref.vertices))
            findings=find_details(model,ref)
            regions=[dict(kind="plane") for _ in range(count)]
            _,ids=curvature(model,ref,dict(label=tags,regions=regions),findings)
            row=dict(case=kind,radius=radius,strips=count,noise=noise,
                             area=sum(r["area_mm2"] for r in findings if r["type"]=="гранёная кривизна"),
                             seams=sum(r["type"]=="гранёная кривизна" for r in findings),
                             released_area=float(ref.area_faces[ids].sum()),correct=len(find_details(scan,scan)),
                             true_polygon=len(find_details(model,model)))
            if not a.mesh_only and noise==0 and radius==50:
                if a.progress:
                    print("curvature B-rep",radius,flush=True)
                top=model.copy()
                owned=np.unique(top.faces[ids])
                top.vertices[owned]=ref.vertices[owned]
                source,body,audit=assemble(top,np.where(np.isin(np.arange(len(tags)),ids),-1,tags),regions)
                output=body_mesh(body,.04)
                output.merge_vertices()
                reference=solidify(ref)
                cov=compare(reference,output,.3,n=1500,seed=42)
                row["brep"]=dict(valid=audit["valid"],free_edges=audit["free_edges"],smooth_faces=audit["smooth_faces"],
                                 coverage=cov["within_tol_share"],area=sum(r["area_mm2"] for r in find_details(output,reference)
                                                                          if r["type"]=="гранёная кривизна"))
            rows.append(row)
    mesh_big=plate(321)[0].subdivide()
    start=perf_counter()
    findings=find_details(mesh_big,mesh_big)
    rows.append(dict(case="timing",triangles=len(mesh_big.faces),seconds=perf_counter()-start,correct=len(findings)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
