"""Partition wrapped free zones into graph-like charts for a guarded CAD trial."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.ShapeFix import ShapeFix_Face
from OCP.TopoDS import TopoDS

from large_surfaces import Chart, fit_surface
import patch_faces


def partition(vertices, faces, ids, *, span=15., min_area=25., max_triangles=900, max_boundary=240):
    """Keep components intact whenever a single bounded chart can cover them."""
    result = []
    pending = patch_faces.split_zones(vertices, faces, ids, max_boundary=None, max_triangles=None)
    while pending:
        part = pending.pop()
        xyz = vertices[faces[part]]
        cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        normal = cross.sum(axis=0)
        normal /= max(1e-12, np.linalg.norm(normal))
        centers = xyz.mean(axis=1)
        axis = np.linalg.svd(centers-centers.mean(axis=0), full_matrices=False)[2][0]
        order = centers@axis
        topology = patch_faces.topology(faces[part])
        loops = patch_faces._loops(topology)
        area = lengths.sum()/2
        if (area >= min_area and loops and len(part) <= max_triangles and sum(map(len, loops)) <= max_boundary
                and np.ptp(order) <= 1.5*span and np.all(cross@normal > .35*lengths)):
            result.append((part, True))
            continue
        if len(part) < 4 or area < 2*min_area:
            result.append((part, False))
            continue
        half = np.argsort(order, kind="stable")
        mid = len(half)//2
        for selected in (half[:mid],half[mid:]):
            pending.extend(patch_faces.split_zones(vertices,faces,part[selected],
                                                   max_boundary=None,max_triangles=None))
    return sorted(result, key=lambda item: int(item[0].min()))


def _trimmed_surface(bounds, xyz, cross, tol, span):
    centers = xyz.mean(axis=1)
    vertices = np.unique(xyz.reshape(-1,3), axis=0)
    samples = np.vstack((vertices, centers))
    origin = samples.mean(axis=0)
    normal = cross.sum(axis=0)
    normal /= np.linalg.norm(normal)
    projected = samples-origin - np.outer((samples-origin)@normal, normal)
    axis = np.linalg.svd(projected, full_matrices=False)[2][0]
    frame = np.array([axis,np.cross(normal,axis),normal])
    uv = (samples-origin) @ frame[:2].T
    size = np.ptp(uv, axis=0)
    if np.min(size)<.3 or len(samples)<16:
        return None
    pad = .04*size + .01
    origin += (uv.min(axis=0)-pad) @ frame[:2]
    bottom = float(np.min((samples-origin)@normal)-2*tol-10)
    chart = Chart(origin,frame,tuple(size+2*pad),bottom)
    edge_samples = np.vstack([loop+t*(np.roll(loop,-1,axis=0)-loop)
                              for loop in bounds for t in (0.,.25,.5,.75)])
    weights = np.r_[np.ones(len(samples)),np.full(len(edge_samples),5.)]
    surface, *_ = fit_surface(np.vstack((samples,edge_samples)),chart,span=span,
                              smoothing=.002,tol=tol,sample_weights=weights)
    wires = []
    for loop in bounds:
        wire = BRepBuilderAPI_MakeWire()
        for edge in patch_faces._edges(loop):
            wire.Add(edge)
        if not wire.IsDone():
            return None
        wires.append(wire.Wire())
    maker = BRepBuilderAPI_MakeFace(surface, wires[0], False)
    for wire in wires[1:]:
        maker.Add(wire)
    if not maker.IsDone():
        return None
    fix = ShapeFix_Face(maker.Face())
    fix.SetPrecision(patch_faces.SEW_TOL/2)
    fix.Perform()
    return TopoDS.Face_s(fix.Face())


def make_patch(vertices, faces, part, tol, edge_tol, span):
    """Try a fit with the original boundary, returning a reason on rejection."""
    tris = faces[part]
    loops = patch_faces._loops(patch_faces.topology(tris))
    if not loops:
        return None, "chart_boundary"
    xyz = vertices[tris]
    cross = np.cross(xyz[:, 1]-xyz[:, 0],xyz[:, 2]-xyz[:, 0])
    length = np.linalg.norm(cross,axis=1)
    if np.any(length <= 1e-10):
        return None,"chart_degenerate"
    loops.sort(key=lambda ring: -np.linalg.norm(np.cross(vertices[ring],
              np.roll(vertices[ring], -1, axis=0)).sum(axis=0)))
    bounds = [vertices[loop] for loop in loops]
    interior = vertices[np.setdiff1d(np.unique(tris), np.concatenate(loops))]
    try:
        face = _trimmed_surface(bounds, xyz, cross, tol, span)
        if face is None:
            return None,"chart_fit"
        edge = patch_faces._same_boundary(face,vertices,loops)
        if edge is None or edge > edge_tol:
            return None,"chart_edge"
        face, report = patch_faces._verify(face,interior,xyz,cross,tol)
        return face, None if face is not None else report["reason"]
    except ValueError:
        return None, "chart_fit"
    except RuntimeError:
        return None, "chart_occ"


def trial(vertices, faces, ids, tol=.2, edge_tol=.3, *, span=6.):
    """Report fit yield before sewing, preserving every input triangle in the audit."""
    counts = Counter()
    areas = Counter()
    for part, chart in partition(vertices,faces,ids):
        xyz = vertices[faces[part]]
        area = np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1).sum()/2
        if chart:
            face, reason = make_patch(vertices,faces,part,tol,edge_tol,span)
        else:
            face, reason = None, "no_chart"
        kind = "fit" if face is not None else reason
        counts[kind] += 1
        areas[kind] += float(area)
    return {"counts": dict(counts), "area_mm2": dict(areas),
            "fit_area_share": areas["fit"]/sum(areas.values()) if areas else 0.}


def main():
    import fairing
    import idealize
    import planar_brep
    import regions
    import scan_uncertainty

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--span", type=float, default=6.)
    args = parser.parse_args()
    if not 5 <= args.span <= 15:
        parser.error("span must be between 5 and 15 mm")
    started = perf_counter()
    scan = trimesh.load(args.scan, force="mesh")
    noise = scan_uncertainty.estimate(scan).working_mm
    fit = idealize.fit_tol(.3,noise)
    step = np.median(scan.edges_unique_length)
    segmentation = regions.segment(scan,tol=fit,min_area=max(6.,4*step*step*.43),scan=scan)
    mesh, info = idealize.idealize(scan,noise=noise,seg=segmentation)
    v, f, label = planar_brep.collapse_degenerate(mesh.vertices,mesh.faces,info["label"],.001)
    v, _ = fairing.fair(v,f,label,info["regions"],.8*fit)
    ids = np.flatnonzero([i<0 or info["regions"][i]["kind"]!="plane" for i in label])
    report = trial(v,f,ids,fit*2/3,fit,span=args.span)
    print(json.dumps({**report,"seconds":perf_counter()-started},indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
