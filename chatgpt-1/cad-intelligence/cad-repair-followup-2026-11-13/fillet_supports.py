"""Experimental conforming trim of an automatic fillet sector."""
import importlib.util
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.optimize import least_squares
import trimesh

from benchmark_36_38 import measure, stage, stock
from defects import _distance
from fillet_ends import extend
from nurbs import body_mesh
from straight_fuzz import scan_mesh
from straight_plus import _sector


def refit(vertices, faces, tags, regions, tol, scan=None):
    """Refit radius with the tangent planes fixed, using scan points when available."""
    output = [dict(row) for row in regions]
    rows = []
    for i, row in enumerate(output):
        if row["kind"] != "cylinder" or "fillet_planes" not in row:
            continue
        ids = np.flatnonzero(tags == i)
        if len(ids) < 4:
            continue
        pair = []
        for k in row["fillet_planes"]:
            while k >= 0 and regions[k]["kind"] == "merged":
                k = regions[k]["into"]
            pair.append(k)
        if min(pair) < 0 or len(set(pair)) != 2 or any(regions[k]["kind"] != "plane" for k in pair):
            continue
        n = np.array([regions[k]["normal"] for k in pair], dtype=float)
        n /= np.linalg.norm(n, axis=1)[:,None]
        axis = np.cross(*n)
        length = np.linalg.norm(axis)
        if length < .18:
            continue
        axis /= length
        d = np.array([normal @ regions[k]["origin"] for normal,k in zip(n,pair)])
        old = float(row["r"])
        sign = np.sign(n @ row["origin"]-d)
        if old <= 0 or np.any(sign == 0):
            continue
        origin = np.asarray(row["origin"])
        root = origin + n.T @ np.linalg.solve(n@n.T,d-n@origin)
        shift = n.T @ np.linalg.solve(n@n.T,sign)
        points = vertices[np.unique(faces[ids])]
        if scan is not None:
            points = trimesh.proximity.closest_point(scan, points)[0]
        p = points-root
        p -= (p@axis)[:,None]*axis
        residual = lambda x: np.linalg.norm(p-x[0]*shift,axis=1)-x[0]
        span = min(.35*old,2*tol)
        fit = least_squares(residual,[old],bounds=([old-span],[old+span]),loss="soft_l1",f_scale=tol/3)
        radius = float(fit.x[0])
        before = float(np.quantile(abs(residual([old])),.95))
        after = float(np.quantile(abs(fit.fun),.95))
        center = root+radius*shift
        accepted = after <= min(tol,before+1e-9)
        if accepted:
            row.update(r=radius,origin=center,axis=axis,fillet_planes=pair)
        rows.append(dict(region=i,radius_mm=[old,radius],p95_mm=[before,after],accepted=bool(accepted)))
    return output, rows


def _split(V, F, tags, field, eligible, support, rail_point, axis, tol, eps=1e-8):
    """Split rail crossings, adding the identical vertex to every incident face."""
    points = list(V)
    cuts = {}
    for tri in F[eligible]:
        for a,b in zip(tri,np.roll(tri,-1)):
            if field[a]*field[b] < -eps**2 and min(abs(field[a]),abs(field[b])) > eps:
                key = tuple(sorted((a,b)))
                if key not in cuts:
                    raw = (field[a]*V[b]-field[b]*V[a])/(field[a]-field[b])
                    projected = rail_point + axis*((raw-rail_point)@axis)
                    if np.linalg.norm(projected-raw) > tol:
                        return V,F,tags,0
                    cuts[key] = len(points)
                    points.append(projected)
    if not cuts:
        return V,F,tags,0
    out, labels = [], []
    for i,tri in enumerate(F):
        edge_cuts = [cuts.get(tuple(sorted((a,b)))) for a,b in zip(tri,np.roll(tri,-1))]
        if all(j is None for j in edge_cuts):
            out.append(tri)
            labels.append(tags[i])
            continue
        if eligible[i] and sum(j is not None for j in edge_cuts) == 2:
            for sign in (1,-1):
                ring=[]
                for a,k in zip(tri,edge_cuts):
                    if sign*field[a] >= -eps:
                        ring.append(a)
                    if k is not None:
                        ring.append(k)
                for j in range(1,len(ring)-1):
                    out.append((ring[0],ring[j],ring[j+1]))
                    labels.append(tags[i] if sign>0 else support)
        else:
            # Fan from the centroid so no inserted edge vertex is skipped.
            ring = [v for a,k in zip(tri,edge_cuts) for v in ((a,k) if k is not None else (a,))]
            center = len(points)
            points.append(V[tri].mean(axis=0))
            out.extend((a,b,center) for a,b in zip(ring,ring[1:]+ring[:1]))
            labels.extend([tags[i]]*len(ring))
    return np.asarray(points),np.asarray(out),np.asarray(labels),len(cuts)


def retrim(V,F,tags,regions,tol=.3):
    """Cut the labelled band along computed tangent generators, keeping a conforming mesh."""
    V,F,tags = np.asarray(V).copy(),np.asarray(F).copy(),np.asarray(tags).copy()
    rows=[]
    for i,row in enumerate(regions):
        if row["kind"]!="cylinder" or "fillet_planes" not in row or not np.any(tags==i):
            continue
        try:
            center,axis,x,y,sweep,_,_=_sector({"origin":row["origin"],"axis":row["axis"],"radius":row["r"],
                                             "planes":row["fillet_planes"],"axis_bounds_mm":(-1,1)},regions)
        except ValueError:
            continue
        before=(V,F,tags)
        cuts=0
        for k,(rail,plane) in enumerate(zip((x,np.cos(sweep)*x+np.sin(sweep)*y),row["fillet_planes"])):
            direction=np.cross(axis,rail)*(1 if k==0 else -1)
            field=(V-center)@direction
            own=tags==i
            face_points=V[F]-center
            axial=face_points@axis
            low,high=axial[own].min(),axial[own].max()
            radial=np.linalg.norm(face_points-axial[...,None]*axis,axis=2)
            # Reclaim a neighboring side sliver only near the cylinder and within its observed length.
            adjacent=trimesh.Trimesh(V,F,process=False).face_adjacency
            near=np.zeros(len(F),bool)
            near[adjacent[own[adjacent].any(axis=1)].ravel()]=True
            reclaim=near & ((tags==plane)|(tags<0))
            reclaim&=(np.abs(radial-row["r"])<=tol).all(axis=1)
            reclaim&=(axial.min(axis=1)>=low-tol)&(axial.max(axis=1)<=high+tol)
            reclaim&=(face_points@rail>0).all(axis=1)&(field[F].min(axis=1)<0)&(field[F].max(axis=1)>0)
            tags=tags.copy()
            tags[reclaim]=i
            V,F,tags,count=_split(V,F,tags,field,own|reclaim,plane,center+row["r"]*rail,axis,tol)
            cuts+=count
        if cuts:
            mesh=trimesh.Trimesh(V,F,process=False)
            if not mesh.is_watertight or not mesh.is_winding_consistent or np.any(mesh.area_faces<1e-10):
                V,F,tags=before
                rows.append(dict(region=i,cuts=cuts,accepted=False))
            else:
                rows.append(dict(region=i,cuts=cuts,accepted=True))
    return V,F,tags,rows


def benchmark(work):
    src=Path(__file__).parent.parent/"cad-fillet-defects-2026-11-13"/"planar_brep_cyl.py"
    spec=importlib.util.spec_from_file_location("followup_builder",src)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=[]
    for kind in ("block","angle"):
        wp=cq.Workplane("XY").box(24,18,10,centered=(True,True,False))
        shape=wp.edges("|Z").fillet(3) if kind=="block" else wp.edges("|Z and >X").fillet(5).edges("|Z and <X").fillet(2)
        path=work/f"{kind}.step"
        cq.exporters.export(shape.val(),str(path))
        for noise in (0.,.1,.2):
            scan=scan_mesh(path,noise,37)
            V,F,L,R=stage(scan,noise)
            tags,regions,growth=extend(V,F,L,R,.3,noise)
            fitted,fit=refit(V,F,tags,regions,.3,scan)
            vv,ff,ll,trim=retrim(V,F,tags,fitted)
            row=dict(case=kind,noise=noise,growth=growth,refit=fit,trim=trim)
            variants=(("stock",stock,(V,F,L,R)),("37",module.build,(V,F,tags,regions)),
                      ("refit",module.build,(V,F,tags,fitted)),("40",module.build,(vv,ff,ll,fitted)))
            for name,build,args in variants:
                try:
                    body,audit=build(*args,smooth_tol=.3,smooth_edge_tol=.075)
                    out=work/f"{kind}-{noise}-{name}.step"
                    cq.exporters.export(body,str(out))
                    body=cq.importers.importStep(str(out)).val()
                    row[name]={**measure(scan,body),"valid":bool(body.isValid()),
                               "cylinders":sum(f.geomType()=="CYLINDER" for f in body.Faces()),
                               "rejected":audit.get("fillets_rejected",{})}
                    if kind == "angle" and noise == 0. and name in ("stock","40"):
                        mesh = body_mesh(body, .04)
                        mesh.merge_vertices()
                        distances, _, _ = _distance(mesh, scan)
                        far = mesh.triangles_center[distances > .3]
                        row[name]["far_centers"] = {"count":len(far), "mean":far.mean(axis=0).tolist() if len(far) else None}
                        row[name]["cylinder_centers"] = [list(face.Center().toTuple()) for face in body.Faces()
                                                         if face.geomType() == "CYLINDER"]
                except (RuntimeError, ValueError) as error:
                    row[name]={"valid":False,"error":str(error),"cylinders":0}
            baseline=row["stock"]["coverage"]
            available=[name for name in ("stock","37","refit","40") if row[name]["valid"] and
                       all(a>=b-1e-9 for a,b in zip(row[name]["coverage"],baseline))]
            row["selected"]=max(available,key=lambda name:row[name]["cylinders"],default=None)
            rows.append(row)
            print(kind,noise,[(k,row[k]["cylinders"],row[k].get("coverage")) for k,_,_ in variants],fit,trim,flush=True)
    return rows
