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
    cuts, gaps, additions, fills = [], [], [], []
    for complete,run in groupby(range(len(rows)),key=ring_radii.__contains__):
        indices = list(run)
        if complete:
            for i in indices:
                radii[i] = ring_radii[i]
            continue
        inner,windows,reliefs,extras,transitions = _window_series(rows,indices,center,outer,bounds)
        for i in indices:
            radii[i] = inner
        cuts.extend(windows)
        gaps.extend(reliefs)
        additions.extend(extras)
        fills.extend(transitions)
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
    fillet = _fit_window_fillet(sleeve,fills)
    if fillet is not None:
        sleeve,fillet_volume,fillet_radius = fillet
    else:
        fillet_volume = 0.
        fillet_radius = None
        additions.extend(patch for _,patch in fills)
    volume = fillet_volume + sum(tool.Volume() for tool in additions)
    if volume>.3*(sleeve.Volume()-fillet_volume):
        raise Rejected("additions exceed 30% of sleeve volume",float(z[0]))
    result = sleeve.fuse(*additions).clean() if additions else sleeve
    if not result.isValid() or len(result.Solids())!=1:
        raise Rejected("additions did not fuse into one solid",float(z[0]))
    return result,dict(diameter_mm=diameter,bores=bores,windows=len(cuts),gaps=len(gaps),
                       additions=len(additions)+len(fills) if fillet is not None else len(additions),
                       window_end_fillet_mm=fillet_radius,addition_volume_mm3=volume,center_mm=center.tolist())


def _stable_window_rows(rows, candidates):
    """Choose a consecutive plateau of similar areas and aligned contours."""
    profiles = [(i,shapely.union_all(rows[i][1])) for i in candidates]
    best = []
    for start,(i,anchor) in enumerate(profiles):
        run = [i]
        for j,current in profiles[start+1:]:
            if (j!=run[-1]+1 or abs(current.area-anchor.area)>.025*anchor.area
                    or current.symmetric_difference(anchor).area>.035*anchor.area):
                break
            run.append(j)
        if len(run)>len(best):
            best = run
    return best


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
    stable = _stable_window_rows(rows,[i for i,r in plain if abs(r-inner)<.026])
    if len(stable)<3:
        raise Rejected("fewer than three consecutive stable window levels",float(rows[indices[0]][0]))
    reference = [rows[i] for i in stable]
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
    extras, transitions = [], []
    for i in indices:
        height,polys = rows[i]
        material = shapely.union_all(polys)
        present = [material.intersection(p).area/p.area for p in lobes]
        if any(.12<x<.87 for x in present):
            raise Rejected("partially supported sleeve sector",float(height))
        predicted = shapely.union_all([p for p,x in zip(lobes,present) if x>=.87])
        missing = material.intersection(outer_disk).difference(bore_disk).difference(predicted.buffer(.075))
        if i<stable[0] or i>stable[-1]:
            for part in _polygons(missing):
                patch = material.intersection(outer_disk).difference(bore_disk).intersection(part.buffer(.20))
                transitions.extend((i,poly) for poly in _polygons(patch))
        elif missing.area>.07*sum(p.area for p in lobes):
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
    patches = _protrusions(transitions,bounds,smooth=True)
    if patches:
        height = bounds[indices[-1]+1]-bounds[indices[0]]+.04
        base = cq.Vector(*center,bounds[indices[0]]-.02)
        annulus = cq.Solid.makeCylinder(outer,height,base).cut(cq.Solid.makeCylinder(inner,height,base))
        patches = [(bounds[indices[0]],patch.intersect(annulus)) for patch in patches]
    return inner,windows,gaps,_protrusions(extras,bounds),patches


def _protrusions(extras,bounds,smooth=False):
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
        if smooth and not any(p.interiors for _,p in track):
            tool = _transition_loft(track,bounds)
            if tool is not None:
                tools.append(tool)
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
            low,high = bounds[track[0][0]],bounds[track[-1][0]+1]
            from kompas_build import prism, ring_pts

            try:
                outline = ring_pts(track[len(track)//2][1].exterior.coords,24,.025)
                tool = prism(outline,low,high)
                if tool.isValid() and tool.Volume()>0:
                    tools.append(tool)
                    continue
            except (ValueError,RuntimeError,Standard_Failure):
                pass
            tools.append(_prism(representative,low,high))
    return tools


def _transition_loft(track,bounds):
    from kompas_build import _common_par, _wire, align, ring_pts

    try:
        rings = [(ring_pts(p.exterior.coords,24,.025),(bounds[i]+bounds[i+1])/2) for i,p in track]
        rings = align([(rings[0][0],bounds[track[0][0]]),*rings,
                   (rings[-1][0],bounds[track[-1][0]+1])])
        params = _common_par(rings)
        tool = cq.Solid.makeLoft([_wire(points,z,params) for points,z in rings])
        if tool.isValid() and tool.Volume()>0:
            return tool
    except (ValueError,RuntimeError,Standard_Failure):
        pass
    return None


def _fit_window_fillet(sleeve,fills):
    """Replace scan patches with an analytic fillet when the shapes agree."""
    if not fills:
        return None
    edges = []
    candidates = [(edge,np.array(edge.Center().toTuple())) for edge in sleeve.Edges()
                  if edge.geomType()=="LINE"]
    radii = []
    for bottom,patch in fills:
        box = patch.BoundingBox()
        if abs(box.zmin-bottom)>.03:
            return None
        # Match the horizontal flank edge that runs through this patch.
        matching = [(edge,center) for edge,center in candidates if abs(center[2]-bottom)<1e-5
                    and len(edge.Vertices())==2 and all(abs(v.Z-bottom)<1e-5 for v in edge.Vertices())
                    and cq.Vertex.makeVertex(*center).distance(patch)<.04]
        if len(matching)!=1:
            return None
        edge,_ = matching[0]
        height = box.zmax-bottom
        radius = round(height/.1)*.1
        if radius<.2 or radius>edge.Length():
            return None
        radii.append(radius)
        edges.append(edge)
    if max(radii)-min(radii)>.21 or len({edge.hashCode() for edge in edges})!=len(edges):
        return None
    observed = cq.Compound.makeCompound([p for _,p in fills]).cut(sleeve)
    volume = observed.Volume()
    if volume<=0:
        return None
    # The last visible patch can lie below the true end of the rounding.
    median = float(np.median(radii))
    solid = sleeve.Solids()[0]
    for increment in (.4,.3,.2,.1,0.):
        radius = round(median+increment,3)
        try:
            tool = solid.fillet(radius,edges).clean()
            if not tool.isValid() or len(tool.Solids())!=1:
                continue
            added = tool.cut(sleeve)
            added_volume = added.Volume()
            if added_volume>0 and added.intersect(observed).Volume()>=.70*max(volume,added_volume):
                return tool,added_volume,radius
        except (ValueError,RuntimeError,Standard_Failure):
            continue
    return None
