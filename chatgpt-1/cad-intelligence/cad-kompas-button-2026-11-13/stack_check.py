"""Task 57. Find an axis with a stable, contiguous stack of outer sections."""
import numpy as np
from shapely.geometry import Polygon

import kompas_build


def stack_check(scan, *, samples=32):
    """Inspect Cartesian axes and principal directions. Does not modify the scan."""
    if samples<12 or len(scan.vertices)<20:
        return {"suitable":False,"axis":None,"axes":[]}
    centered = scan.vertices-scan.bounds.mean(axis=0)
    _,axes = np.linalg.eigh(centered.T@centered)
    results = []
    for axis in np.vstack((np.eye(3),axes.T)):
        if any(abs(axis@row["axis"])>.9995 for row in results):
            continue
        u = np.eye(3)[np.argmin(abs(axis))]
        u = np.cross(u,axis)
        u /= np.linalg.norm(u)
        frame = np.column_stack((u,np.cross(axis,u),axis))
        transformed = scan.copy()
        transformed.vertices = centered@frame
        dz = transformed.extents[2]/samples
        if dz<=0:
            continue
        sections = kompas_build.sections(transformed,dz)
        interior = sections[3:-3]
        dominant = []
        splits = 0
        for _,polys in interior:
            hulls = [Polygon(p.exterior) for p in polys]
            external = [p for p in hulls if not any(q.area>p.area and q.covers(p) for q in hulls)]
            major = max(external,key=lambda p:p.area,default=None)
            dominant.append(major)
            if major is not None:
                splits += sum(p is not major and p.area>.12*major.area for p in external)>0
        continuity = []
        for a,b in zip(dominant,dominant[1:]):
            continuity.append(0. if a is None or b is None else a.intersection(b).area/max(a.area,b.area))
        missing = sum(p is None for p in dominant)
        jumps = sum(x<.62 for x in continuity)
        severe = sum(x<.35 for x in continuity)
        score = 1-(missing+splits+jumps)/max(1,len(interior))
        results.append(dict(axis=axis.tolist(),score=float(score),missing=missing,split_levels=int(splits),
                            contour_jumps=jumps,severe_jumps=severe,sections=len(interior),
                            height_mm=float(transformed.extents[2])))
    best = max(results,key=lambda r:(r["severe_jumps"]==0,r["score"],-r["height_mm"]),default=None)
    suitable = best is not None and best["score"]>=.9 and best["severe_jumps"]==0
    return dict(suitable=suitable,axis=best["axis"] if suitable else None,axes=results)
