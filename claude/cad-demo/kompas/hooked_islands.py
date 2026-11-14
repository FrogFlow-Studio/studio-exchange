"""Rebuild a windowed sleeve and its scan-supported latch protrusions."""
import os
from collections import Counter, defaultdict
from itertools import groupby

import cadquery as cq
import numpy as np
import shapely
from OCP.Standard import Standard_ConstructionError, Standard_Failure
from OCP.StdFail import StdFail_NotDone
from shapely.errors import GEOSException
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon

from cylindrical_islands import _fit
from windowed_islands import _drilled_body, _layer_radii, _window_polygons, _window_edges


CAD_ERRORS = (RuntimeError,ValueError,Standard_Failure,Standard_ConstructionError,StdFail_NotDone)


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
    except CAD_ERRORS+(GEOSException,) as error:
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
    cuts, gaps, hook_tracks, end_tracks = [], [], [], []
    window_references = []
    for complete,run in groupby(range(len(rows)),key=ring_radii.__contains__):
        indices = list(run)
        if complete:
            for i in indices:
                radii[i] = ring_radii[i]
            continue
        inner,windows,reliefs,extras,transitions,reference = _window_series(rows,indices,center,outer,bounds)
        for i in indices:
            radii[i] = inner
        window_references.append(reference)
        cuts.extend(windows)
        gaps.extend(reliefs)
        hook_tracks.extend(extras)
        end_tracks.extend(transitions)
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
    rollbacks = {}
    attempts = []
    for mode in ("smooth","compact_hooks","ruled","legacy"):
        try:
            result,build = _add_features(sleeve,hook_tracks,end_tracks,bounds,mode=mode,attempts=attempts)
            break
        except Rejected:
            raise
        except CAD_ERRORS as error:
            if mode=="legacy":
                reason = str(error).splitlines()
                detail = reason[0][:120] if reason else type(error).__name__
                raise Rejected(f"legacy additions: {detail}",float(z[0])) from error
            rollbacks[mode] = type(error).__name__ + ": " + str(error).split("\n")[0][:120]
    if rollbacks:
        build["addition_rollbacks"] = rollbacks
        build["rolled_back_groups"] = attempts[:-1]
    return result,dict(diameter_mm=diameter,bores=bores,windows=len(cuts),gaps=len(gaps),
                       window_references=window_references,center_mm=center.tolist(),**build)


def _add_features(sleeve,hook_tracks,end_tracks,bounds,*,mode="smooth",attempts=None):
    methods = Counter()
    groups = []
    hook_mode = "smooth" if mode=="compact_hooks" else mode
    additions = []
    for track in hook_tracks:
        group = {}
        tools,method = _track_tools(track,bounds,mode=hook_mode,diagnostics=group)
        if method=="union_spline":
            # одно выдавливание по объединению сечений может не дотянуться до стенки втулки
            # (сплайн срезает узкую кромку прилегания): тогда этот зацеп строим прежним способом
            try:
                attached = _valid_addition(sleeve,_fused(sleeve,*tools),sum(t.Volume() for t in tools))
            except CAD_ERRORS:
                attached = False
            if not attached:
                group = {"union_spline_detached":True}
                tools,method = _track_tools(track,bounds,mode=hook_mode,diagnostics=group,union=False)
                group["union_spline_detached"] = True
        groups.append(dict(feature="hook",method=method,**group))
        additions.extend(tools)
        methods[method] += 1
    fills = []
    end_options = []
    for bottom,track,annulus in end_tracks:
        group = {}
        end_mode = "legacy" if mode=="compact_hooks" else mode
        tools,method = _track_tools(track,bounds,mode=end_mode,diagnostics=group)
        patches = [tool.intersect(annulus) for tool in tools]
        if any(not patch.Solids() or patch.Volume()<=0 for patch in patches):
            group["empty_after_clipping"] = method
            tools,method = _track_tools(track,bounds,mode="legacy")
            patches = [tool.intersect(annulus) for tool in tools]
            if any(not patch.Solids() or patch.Volume()<=0 for patch in patches):
                raise ValueError("window transition vanished during clipping")
        # заплатка конца окна — одна на окно: слои лесенки сливаются, иначе верхние слои
        # не находят своего ребра дна и скругление отвергается целиком
        merged = patches[0].fuse(*patches[1:]).clean() if len(patches)>1 else patches[0]
        fills.append((bottom,merged if merged.isValid() and len(merged.Solids())==1 else patches[0]))
        end_options.append((track,annulus,patches,dict(feature="window_end",method=method,**group)))
    fillet = _fit_window_fillet(sleeve,fills)
    if fillet is not None:
        sleeve,fillet_volume,fillet_radius = fillet
        methods["analytic_fillet"] += len(end_tracks)
        groups.extend(dict(group,method="analytic_fillet") for _,_,_,group in end_options)
    else:
        fillet_volume,fillet_radius = 0.,None
        for track,annulus,patches,group in end_options:
            if group["method"] in ("smooth_loft","ruled_loft","radial_smooth_loft","radial_ruled_loft","union_prism","union_spline"):
                try:
                    trial = _fused(sleeve,*patches)
                    attached = _valid_addition(sleeve,trial,sum(patch.Volume() for patch in patches))
                except CAD_ERRORS:
                    attached = False
                if not attached:
                    group["attachment_failed"] = group["method"]
                    tools,group["method"] = _track_tools(track,bounds,mode="legacy")
                    patches = [tool.intersect(annulus) for tool in tools]
            additions.extend(patches)
            groups.append(group)
            methods[group["method"]] += 1
    if attempts is not None:
        attempts.append(dict(mode=mode,groups=groups))
    volume = fillet_volume+sum(tool.Volume() for tool in additions)
    if volume>.3*(sleeve.Volume()-fillet_volume):
        raise ValueError("additions exceed 30% of sleeve volume")
    if additions:
        tol = .0001 if mode in ("smooth","compact_hooks") else None
        result = _fused(sleeve,*additions,tol=tol)
        if tol and not _valid_addition(sleeve,result,volume):
            result = sleeve
            for tool in additions:
                next_shape = _fused(result,tool,tol=tol)
                if not _valid_addition(result,next_shape,tool.Volume()):
                    raise ValueError("addition fusion changed sleeve outside additive bounds")
                result = next_shape
        if not _valid_addition(sleeve,result,volume):
            raise ValueError("addition fusion changed sleeve outside additive bounds "
                             f"(solids {len(result.Solids())}, valid {result.isValid()}, "
                             f"delta {result.Volume()-sleeve.Volume():.4f}, max {volume:.4f})")
    else:
        result = sleeve
    return result,dict(additions=len(additions)+(len(fills) if fillet is not None else 0),
                       addition_methods=dict(methods),addition_groups=groups,window_end_fillet_mm=fillet_radius,
                       addition_volume_mm3=volume)


def _fused(base,*tools,tol=None):
    """Слияние с починкой: ядро иногда отдаёт одно тело с мелким дефектом рёбер (isValid False),
    ShapeFix его исправляет, объём при этом не меняется."""
    result = base.fuse(*tools,tol=tol).clean()
    if not result.isValid():
        try:
            fixed = result.fix()
            if fixed.isValid() and abs(fixed.Volume()-result.Volume())<1e-3:
                result = fixed
        except Exception:  # noqa: BLE001  починка не удалась: остаётся исходное тело, его отвергнет проверка
            pass
    return result


def _valid_addition(base,result,max_added):
    if not result.isValid() or len(result.Solids())!=1:
        return False
    delta = result.Volume()-base.Volume()
    return -1e-4<=delta<=max_added+1e-4


def _stable_window_rows(rows,candidates):
    """Rank contiguous plateaus using the area and contour of each individual petal."""
    profiles = [(i,rows[i][1]) for i in candidates]
    runs = set()
    for start,(i,anchor) in enumerate(profiles):
        run = [i]
        for j,polys in profiles[start+1:]:
            if j!=run[-1]+1:
                break
            paired = [(p,min(polys,key=lambda q:p.centroid.distance(q.centroid))) for p in anchor]
            if len({id(q) for _,q in paired})!=len(anchor) or any(
                    abs(p.area-q.area)>.07*p.area or p.symmetric_difference(q).area>.10*p.area
                    for p,q in paired):
                break
            run.append(j)
            if len(run)>=3:
                runs.add(tuple(run))
    return sorted(runs,key=lambda run:(-len(run),run))


def _sampled_flanks(reference,center,inner,outer):
    """Fit flank lines to angular crossings instead of mesh vertices."""
    radii = np.linspace(inner+.15,outer-.15,7)
    mid = (inner+outer)/2
    nominal = _window_edges(reference[len(reference)//2][1],center,mid)
    if nominal is None:
        return None
    contours = []
    for _,polys in reference:
        samples = [_window_edges(polys,center,r) for r in radii]
        if any(row is None or len(row)!=len(nominal) for row in samples):
            continue
        contours.append(np.asarray(samples))
    if len(contours)<3:
        return None

    def flank(edge,side):
        angles = np.stack([values[:,:,side] for values in contours])
        diffs = (angles-edge+np.pi)%(2*np.pi)-np.pi
        nearest = np.take_along_axis(diffs,np.argmin(abs(diffs),axis=2)[...,None],axis=2)[...,0]
        if np.max(abs(nearest))>.3:
            return None
        measured = edge+nearest
        points = np.column_stack(((np.cos(measured)*radii).ravel(),(np.sin(measured)*radii).ravel()))
        origin = np.median(points,axis=0)
        _,_,basis = np.linalg.svd(points-origin,full_matrices=False)
        direction,normal = basis
        if np.quantile(abs((points-origin)@normal),.95)>.055:
            return None
        if direction@origin<0:
            direction = -direction
        return origin-(mid-inner+.5)*direction,origin+2*outer*direction

    windows = []
    for start,end in nominal:
        first,last = flank(start,0),flank(end,1)
        if first is None or last is None:
            return None
        poly = Polygon([first[0],first[1],last[1],last[0]])
        if not poly.is_valid:
            return None
        windows.append(poly)
    return windows


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
    runs = _stable_window_rows(rows,[i for i,r in plain if abs(r-inner)<.026])
    if not runs:
        raise Rejected("fewer than three consecutive stable window levels",float(rows[indices[0]][0]))
    outer_disk = Point(*center).buffer(outer,quad_segs=96)
    bore_disk = Point(*center).buffer(inner,quad_segs=96)
    annulus = outer_disk.difference(bore_disk)
    for stable in runs:
        reference = [rows[i] for i in stable]
        windows = _window_polygons(reference,center,inner,outer)
        fitted = "vertices"
        if windows is None:
            windows = _sampled_flanks(reference,center,inner,outer)
            fitted = "angular_crossings"
        if windows is None:
            continue
        cutter_xy = [translate(p,*center) for p in windows]
        predicted = annulus.difference(shapely.union_all(cutter_xy))
        if all(predicted.symmetric_difference(shapely.union_all(polys)).area<=.065*annulus.area
               for _,polys in reference):
            break
    else:
        first,last = (rows[i][0] for i in (runs[0][0],runs[0][-1]))
        raise Rejected(f"window flanks have no stable planar fit (plateau {first:.3f}–{last:.3f} mm, "
                       f"{len(runs[0])} levels)",float(first))
    lobes = _polygons(predicted)
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
    patches = []
    tracks = _protrusion_tracks(transitions)
    if tracks:
        height = bounds[indices[-1]+1]-bounds[indices[0]]+.04
        base = cq.Vector(*center,bounds[indices[0]]-.02)
        annulus = cq.Solid.makeCylinder(outer,height,base).cut(cq.Solid.makeCylinder(inner,height,base))
        patches = [(bounds[indices[0]],track,annulus) for track in tracks]
    reference = dict(levels=len(stable),z_mm=[float(rows[stable[0]][0]),float(rows[stable[-1]][0])],
                     fit=fitted)
    return inner,windows,gaps,_protrusion_tracks(extras),patches,reference


def _protrusion_tracks(extras):
    tracks = []
    for i,poly in extras:
        for track in tracks:
            j,last = track[-1]
            if i==j+1 and poly.intersection(last).area>.25*min(poly.area,last.area):
                track.append((i,poly))
                break
        else:
            tracks.append([(i,poly)])
    return [track for track in tracks if len(track)>=2]


def _track_tools(track,bounds,*,mode,diagnostics=None,union=True):
    if any(p.interiors for _,p in track):
        raise Rejected("protrusion with unsupported hole",float(bounds[track[0][0]]))
    attempts = {}
    if diagnostics is not None:
        diagnostics.update(sections=len(track),z_mm=round(float(bounds[track[0][0]]),3),
                           areas_mm2=[round(p.area,3) for _,p in track],lofts=attempts)
    if union and os.environ.get("HOOKED_UNION_SPLINE","1")=="1":
        tool,status = _union_spline(track,bounds)
        attempts["union_spline"] = {"status":status}
        if tool is not None:
            return [tool],"union_spline"
    if mode!="legacy":
        for ruled in ((False,True) if mode=="smooth" else (True,)):
            detail = {}
            tool = _transition_loft(track,bounds,ruled,detail)
            attempts["ruled" if ruled else "smooth"] = detail
            if tool is not None:
                return [tool],"ruled_loft" if ruled else "smooth_loft"
    if mode=="smooth":
        for ruled in (False,True):
            tool,status = _radial_loft(track,bounds,ruled)
            attempts["radial_ruled" if ruled else "radial_smooth"] = {"status":status}
            if tool is not None:
                return [tool],"radial_ruled_loft" if ruled else "radial_smooth_loft"
    if mode=="smooth":
        tool = _union_prism(track,bounds)
        if tool is not None:
            return [tool],"union_prism"
    representative = track[len(track)//2][1].simplify(.045,preserve_topology=True)
    if any(representative.symmetric_difference(p).area>.16*max(p.area,representative.area) for _,p in track):
        tools = [_prism(p.simplify(.045,preserve_topology=True),bounds[i],bounds[i+1]) for i,p in track]
        return tools,"stacked_prisms"
    low,high = bounds[track[0][0]],bounds[track[-1][0]+1]
    from kompas_build import prism, ring_pts

    try:
        outline = ring_pts(track[len(track)//2][1].exterior.coords,24,.025)
        tool = prism(outline,low,high)
        if tool.isValid() and tool.Volume()>0:
            return [tool],"spline_prism"
    except CAD_ERRORS:
        pass
    return [_prism(representative,low,high)],"polygon_prism"


def _union_spline(track,bounds):
    """Одно выдавливание сплайнового контура объединения сечений, как ребро в thin_rib_prisms:
    у конструктора зацеп втулки — один эскиз и одно выдавливание, а не стопка призм."""
    from kompas_build import prism, ring_pts

    union = shapely.union_all([poly for _,poly in track]).buffer(.01).buffer(-.01)
    if union.geom_type!="Polygon":
        parts = sorted(union.geoms,key=lambda q:-q.area)
        if parts[0].area<.97*union.area:
            return None,"disconnected"
        union = parts[0]
    if union.interiors:
        return None,"hole"
    low,high = bounds[track[0][0]],bounds[track[-1][0]+1]
    try:
        points = ring_pts(union.exterior.coords,24,.01)
        drawn = shapely.geometry.Polygon(points)
        if not drawn.is_valid or drawn.symmetric_difference(union).area>.15*union.area:
            return None,"spline_deviates"
        tool = prism(points,low-.01,high+.01)
        tool = tool if isinstance(tool,cq.Shape) else cq.Workplane().add(tool).val()
        if not tool.isValid() or tool.Volume()<=0:
            return None,"invalid_solid"
        return tool,"ok"
    except CAD_ERRORS as error:
        return None,type(error).__name__
    except Exception as error:  # noqa: BLE001  OCP StdFail_NotDone не наследует Standard_Failure
        return None,type(error).__name__


def _union_prism(track,bounds):
    union = shapely.union_all([poly for _,poly in track])
    if union.geom_type!="Polygon" or union.interiors:
        return None
    low,high = bounds[track[0][0]],bounds[track[-1][0]+1]
    try:
        tool = _prism(union.simplify(.025,preserve_topology=True),low,high)
        return tool if tool.isValid() and tool.Volume()>0 else None
    except CAD_ERRORS:
        return None


def _transition_loft(track,bounds,ruled,detail=None):
    from kompas_build import _common_par, _wire, align, ring_pts

    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    detail = {} if detail is None else detail
    boxes = np.array([poly.bounds for _,poly in track])
    low,high = boxes[:,:2].min(axis=0),boxes[:,2:].max(axis=0)
    for smoothing in (.025,.008):
        detail.clear()
        try:
            rings = [(ring_pts(p.exterior.coords,32,smoothing),(bounds[i]+bounds[i+1])/2) for i,p in track]
            rings = align([(rings[0][0],bounds[track[0][0]]),*rings,
                           (rings[-1][0],bounds[track[-1][0]+1])])
            params = _common_par(rings)
            tool = cq.Solid.makeLoft([_wire(points,z,params) for points,z in rings],ruled=ruled)
            if not tool.isValid():
                detail.update(status="invalid_solid")
                continue
            if tool.Volume()<=0:
                detail.update(status="nonpositive_volume")
                continue
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(tool.wrapped,box,False,False)
            xmin,ymin,_,xmax,ymax,_ = box.Get()
            excess = max(*(low-[xmin,ymin]),*([xmax,ymax]-high))
            detail.update(status="bounds",excess_mm=round(float(excess),3))
            if excess<=.08:
                detail.update(status="accepted")
                return tool
        except CAD_ERRORS as error:
            detail.update(status=type(error).__name__)
    return None


def _radial_loft(track,bounds,ruled):
    """Align profiles by rays from a point inside every section."""
    from kompas_build import _common_par, _wire

    polygons = [p for _,p in track]
    common = shapely.intersection_all(polygons)
    if common.area<.015:
        return None,"no_common_interior"
    center = np.array(common.representative_point().coords[0])
    if not common.contains(Point(*center)):
        return None,"no_common_interior"
    theta = np.arange(64)*2*np.pi/64
    directions = np.column_stack((np.cos(theta),np.sin(theta)))
    length = max(np.hypot(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1]) for p in polygons)
    rays = [LineString([center,center+2*length*d]) for d in directions]
    rings = []
    for i,poly in track:
        clipped = [poly.intersection(ray) for ray in rays]
        if any(segment.geom_type!="LineString" or segment.is_empty for segment in clipped):
            return None,"multiple_crossings"
        points = np.array([segment.coords[-1] for segment in clipped])
        rings.append((points,(bounds[i]+bounds[i+1])/2))
    rings = [(rings[0][0],bounds[track[0][0]]),*rings,(rings[-1][0],bounds[track[-1][0]+1])]
    try:
        params = _common_par(rings)
        solid = cq.Solid.makeLoft([_wire(points,z,params) for points,z in rings],ruled=ruled)
        if not solid.isValid() or solid.Volume()<=0:
            return None,"invalid_solid"
        # Check actual surface samples. The default box can include spline control-point overshoot.
        vertices,_ = solid.tessellate(.008,.13)
        if not vertices:
            return None,"empty_mesh"
        points = np.array([vertex.toTuple() for vertex in vertices])
        envelopes = np.array([p.bounds for p in polygons])
        if (np.any(points[:,:2].min(axis=0)<envelopes[:,:2].min(axis=0)-.08)
                or np.any(points[:,:2].max(axis=0)>envelopes[:,2:].max(axis=0)+.08)):
            return None,"bounds"
        return solid,"accepted"
    except CAD_ERRORS as error:
        return None,type(error).__name__


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
        # первое сечение заплатки может начаться на слой (0,2 мм) выше дна окна: скругление
        # у самого дна тоньше погрешности сечения, ребро дна при этом то же
        if not -.03<=box.zmin-bottom<=.23:
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
            overlap = added.intersect(observed).Volume()
            # лесенка призм по слоям толще настоящего скругления (каждый слой выдавлен по нижнему,
            # самому широкому сечению), поэтому скругление должно почти целиком лежать внутри
            # лесенки и занимать хотя бы половину её объёма; точность потом проверяет сверка со сканом
            if added_volume>0 and overlap>=.90*added_volume and added_volume>=.50*volume:
                return tool,added_volume,radius
        except CAD_ERRORS:
            continue
    return None
