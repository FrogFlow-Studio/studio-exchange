"""Detect shallow missing detail and moderate facets in an aligned model."""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree
from defects import _clusters, _distance, _load, _public, analyze


def _rows(mesh, mask, kind, source, values, minimum, *, angles=None):
    for ids in _clusters(mesh, mask):
        area = float(mesh.area_faces[ids].sum())
        if area < minimum:
            continue
        row = dict(type=kind, source=source, area_mm2=area,
                   center=np.average(mesh.triangles_center[ids], axis=0, weights=mesh.area_faces[ids]).tolist(),
                   max_deviation_mm=float(np.max(values[ids])), description=kind, _faces=ids)
        if angles is not None:
            row["mean_seam_angle_deg"] = float(np.mean(angles[ids][angles[ids] > 0]))
        yield row


def find_details(model, scan, tol=.3):
    step = float(np.median(scan.edges_unique_length))
    distances, _, _ = _distance(model, scan)
    reverse, signed, _ = _distance(scan, model)
    # Compare averaged normals on both sides of each seam, at a scale that
    # suppresses single-triangle noise without erasing a true crease.
    a,b = model.face_adjacency.T
    dots = np.einsum("ij,ij->i",model.face_normals[a],model.face_normals[b])
    angle = np.degrees(np.arccos(np.clip(dots,-1,1)))
    supported = (angle>=3.-1e-7) & (angle<=20.)
    chosen = np.flatnonzero(supported)
    if len(chosen):
        edge = model.vertices[model.face_adjacency_edges[chosen]]
        mid = edge.mean(axis=1)
        across = model.triangles_center[b[chosen]]-model.triangles_center[a[chosen]]
        axis = edge[:,1]-edge[:,0]
        axis /= np.linalg.norm(axis,axis=1)[:,None]
        across -= np.einsum("ij,ij->i",across,axis)[:,None]*axis
        across /= np.maximum(1e-12,np.linalg.norm(across,axis=1))[:,None]
        tree = cKDTree(scan.triangles_center)
        radius = max(.75*step,tol)
        means=[]
        for sign in (-1,1):
            _, nearby = tree.query(mid+sign*radius*across,k=min(12,len(scan.faces)))
            normals=scan.face_normals[nearby].mean(axis=1)
            means.append(normals/np.maximum(np.linalg.norm(normals,axis=1),1e-12)[:,None])
        jump=np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i",*means),-1,1)))
        distance,_=tree.query(mid)
        # Coincident analytic seams, including a genuine relief edge, have
        # negligible scan distance on both sides.
        distinct = np.maximum(distances[a[chosen]], distances[b[chosen]]) > max(.002,.004*step)
        supported[chosen] = distinct & (jump<=.72*angle[chosen]) & (distance<=max(2*step,3*tol))
    # Triangles on a long seam may only touch at vertices, so expand one
    # adjacency ring before connected-component aggregation.
    result = []
    if supported.any():
        mask = np.zeros(len(model.faces), bool)
        seam_angles = np.zeros(len(mask))
        for end in (a,b):
            mask[end[supported]] = True
            np.maximum.at(seam_angles,end[supported],angle[supported])
        connected = mask[a] | mask[b]
        mask[a[connected]] = True
        mask[b[connected]] = True
        for row in _rows(model,mask,"гранёная кривизна","model",distances,max(2.,step**2),angles=seam_angles):
            xyz = model.triangles_center[row["_faces"]]
            if np.linalg.norm(np.ptp(xyz,axis=0)) >= 3*step:
                result.append(row)
    # Shallow isolated differences surrounded by scan points in agreement with the model.
    small = (reverse >= .095) & (reverse <= 1.) & (abs(signed) >= .65*reverse)
    tree = cKDTree(scan.triangles_center) if small.any() else None
    for row in _rows(scan,small,"сплющенная мелочь","scan",reverse,max(2.,2*step**2)):
        part = row["_faces"]
        scale = np.sqrt(row["area_mm2"]/np.pi)
        near = np.asarray(tree.query_ball_point(row["center"],max(3*step,2.3*scale)))
        if not len(near) or np.mean(reverse[near] < .075) < .20 or scale > 6.:
            continue
        if np.quantile(reverse[part],.8) < max(.12,.4*tol):
            continue
        row["height_difference_p90_mm"] = float(np.quantile(reverse[part],.9))
        result.append(row)
    return sorted(result,key=lambda row:-row["area_mm2"])


def find(model_step_or_stl, scan_stl, tol=.3):
    model, scan = _load(model_step_or_stl), _load(scan_stl)
    return _public(sorted(analyze(model,scan,tol)+find_details(model,scan,tol),key=lambda row:-row["area_mm2"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("scan")
    parser.add_argument("--tol",type=float,default=.3)
    a=parser.parse_args()
    print(json.dumps(find(a.model,a.scan,a.tol),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
