"""Rebuild a windowed sleeve and its scan-supported latch protrusions."""
from collections import defaultdict
from itertools import groupby

import cadquery as cq
import numpy as np
import shapely
from OCP.Standard import Standard_Failure
from shapely.errors import GEOSException
from shapely.affinity import translate
from shapely.geometry import Point

from cylindrical_islands import _fit
from windowed_islands import _drilled_body, _layer_radii, _window_polygons


class Rejected(ValueError):
    def __init__(self, reason, level):
        super().__init__(reason)
        self.level = level


def _prism(poly, low, high):
    wire = cq.Wire.makePolygon([cq.Vector(x,y,low) for x,y in list(poly.exterior.coords)[:-1]],close=True)
    return cq.Solid.extrudeLinear(wire,[],cq.Vector(0,0,high-low))


def _polygons(shape):
    return [p for p in getattr(shape,"geoms",[shape]) if p.geom_type=="Polygon" and p.area>.07]


def hooked_island(sections, low, high, dz):
    """Return a candidate and audit, or a single diagnostic with the failing height."""
    try:
        return _build(sections,low,high,dz),None
    except Rejected as error:
        z_mm = error.level
        reason = str(error)
    except (RuntimeError,ValueError,Standard_Failure,GEOSException) as error:
        z_mm = float(low)
        reason = f"CAD operation: {type(error).__name__}"
    heights = sorted({z for z,_ in sections})
    level = min(range(len(heights)),key=lambda i:abs(heights[i]-z_mm),default=0)
    return None,dict(reason=reason,level=level+1,z_mm=z_mm)


def _build(sections,low,high,dz):
    grouped = defaultdict(list)
    for z,poly in sections:
        grouped[z].append(poly)
    rows = sorted(grouped.items())
    if len(rows)<5:
        raise Rejected("too few levels",float(low))
    z = np.array([z for z,_ in rows])
    if np.max(np.diff(z))>1.5*dz:
        raise Rejected("missing section",float(z[np.argmax(np.diff(z))+1]))
    intact = [(i,p[0]) for i,(_,p) in enumerate(rows) if len(p)==1 and len(p[0].interiors)==1]
    if len(intact)<2 or len(intact)==len(rows) or intact[0][0]!=0 or intact[-1][0]!=len(rows)-1:
        raise Rejected("complete rings required at both ends",float(z[0]))
    fit = _fit([p.exterior for _,p in intact],diameter_step=.05,sample_segments=True)
    if fit is None:
        raise Rejected("outer arc fit failed",float(z[0]))
    center,diameter = fit
    outer = diameter/2
    ring_radii = {}
    for i,p in intact:
        inner = _layer_radii([p],center,outer)
        if inner is None:
            raise Rejected("end bore fit failed",float(z[i]))
        ring_radii[i] = inner
    bounds = [z[0]-dz/2,*((z[:-1]+z[1:])/2),high+.02]
    radii = [None]*len(rows)
    cuts, gaps, additions = [], [], []
    for complete,run in groupby(range(len(rows)),key=ring_radii.__contains__):
        indices = list(run)
        if complete:
            for i in indices:
                radii[i] = ring_radii[i]
            continue
        inner,windows,reliefs,extras = _window_series(rows,indices,center,outer,bounds)
        for i in indices:
            radii[i] = inner
        cuts.extend(windows)
        gaps.extend(reliefs)
        additions.extend(extras)
    if not cuts:
        raise Rejected("no supported windows",float(z[0]))
    radii = [radii[0],*[float(np.median(radii[i-1:i+2])) for i in range(1,len(rows)-1)],radii[-1]]
    drilled = _drilled_body(radii,bounds,center,outer,low,high)
    if drilled is None:
        raise Rejected("unsupported bore step",float(z[0]))
    body,bores = drilled
    sleeve = body.cut(*cuts,*gaps).clean()
    if not sleeve.isValid() or len(sleeve.Solids())!=1:
        raise Rejected("window cuts did not leave one solid",float(z[0]))
    volume = sum(tool.Volume() for tool in additions)
    if volume>.3*sleeve.Volume():
        raise Rejected("additions exceed 30% of sleeve volume",float(z[0]))
    result = sleeve.fuse(*additions).clean() if additions else sleeve
    if not result.isValid() or len(result.Solids())!=1:
        raise Rejected("additions did not fuse into one solid",float(z[0]))
    return result,dict(diameter_mm=diameter,bores=bores,windows=len(cuts),gaps=len(gaps),
                       additions=len(additions),addition_volume_mm3=volume,center_mm=center.tolist())


def _window_series(rows,indices,center,outer,bounds):
    plain = []
    for i in indices:
        radius = _layer_radii(rows[i][1],center,outer)
        if radius is not None:
            plain.append((i,radius))
    if len(plain)<3:
        raise Rejected("no clean window reference",float(rows[indices[0]][0]))
    most = max(len(rows[i][1]) for i,_ in plain)
    plain = [(i,r) for i,r in plain if len(rows[i][1])==most]
    if len(plain)<3:
        raise Rejected("insufficient full window profiles",float(rows[indices[0]][0]))
    inner = round(float(np.median([r for _,r in plain]))/.025)*.025
    reference = [rows[i] for i,r in plain if abs(r-inner)<.026]
    windows = _window_polygons(reference,center,inner,outer)
    if windows is None:
        raise Rejected("window flanks have no stable planar fit",float(rows[indices[0]][0]))
    outer_disk = Point(*center).buffer(outer,quad_segs=96)
    bore_disk = Point(*center).buffer(inner,quad_segs=96)
    cutter_xy = [translate(p,*center) for p in windows]
    lobes = _polygons(outer_disk.difference(bore_disk).difference(shapely.union_all(cutter_xy)))
    if len(lobes)!=len(windows):
        raise Rejected("ambiguous sleeve sectors",float(rows[indices[0]][0]))
    absent = []
    extras = []
    for i in indices:
        height,polys = rows[i]
        material = shapely.union_all(polys)
        present = [material.intersection(p).area/p.area for p in lobes]
        if any(.12<x<.87 for x in present):
            raise Rejected("partially supported sleeve sector",float(height))
        predicted = shapely.union_all([p for p,x in zip(lobes,present) if x>=.87])
        missing = material.intersection(outer_disk).difference(bore_disk).difference(predicted.buffer(.075))
        if missing.area>.07*sum(p.area for p in lobes):
            raise Rejected("material inside a fitted window",float(height))
        absent.append([x<.12 for x in present])
        for boundary,excess in ((bore_disk,material.intersection(bore_disk.buffer(-.10))),
                                (outer_disk,material.difference(outer_disk.buffer(.10)))):
            for part in _polygons(excess):
                # Recover the entire protrusion, plus a small overlap into
                # the wall. Disconnected material is rejected by the fuse.
                fringe = (boundary.buffer(.045) if boundary is bore_disk else
                          shapely.difference(material,outer_disk.buffer(-.045)))
                patch = material.intersection(fringe).intersection(part.buffer(.19))
                for poly in _polygons(patch):
                    extras.append((i,poly))
    windows = [_prism(p,bounds[indices[0]],bounds[indices[-1]+1]) for p in cutter_xy]
    gaps = []
    for j,lobe in enumerate(lobes):
        for missing,run in groupby(range(len(indices)),key=lambda k:absent[k][j]):
            group = list(run)
            if missing:
                # Extend this gap to both pre-existing windows so the
                # attached sector is completely removed.
                gaps.append(_prism(lobe.buffer(.12),bounds[indices[group[0]]],bounds[indices[group[-1]]+1]))
    return inner,windows,gaps,_protrusions(extras,bounds)


def _protrusions(extras,bounds):
    tracks = []
    for i,poly in extras:
        for track in tracks:
            j,last = track[-1]
            if i==j+1 and poly.intersection(last).area>.25*min(poly.area,last.area):
                track.append((i,poly))
                break
        else:
            tracks.append([(i,poly)])
    tools = []
    for track in tracks:
        if len(track)<2:
            continue
        representative = track[len(track)//2][1].simplify(.045,preserve_topology=True)
        if (representative.interiors or any(representative.symmetric_difference(p).area>
                                            .16*max(p.area,representative.area) for _,p in track)):
            # Adapt the outline on each supported slice when the protrusion tapers.
            for i,p in track:
                if p.interiors:
                    raise Rejected("protrusion with unsupported hole",float(bounds[i]))
                tools.append(_prism(p.simplify(.045,preserve_topology=True),bounds[i],bounds[i+1]))
        else:
            tools.append(_prism(representative,bounds[track[0][0]],bounds[track[-1][0]+1]))
    return tools
