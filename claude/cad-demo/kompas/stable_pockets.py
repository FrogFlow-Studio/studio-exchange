"""Experimental reconstruction of nearly prismatic pockets from stable sections."""
import os
from time import perf_counter

import cadquery as cq
import numpy as np
from OCP.Standard import Standard_Failure
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import kompas_build as stock


def _wire(ring, z):
    xy = list(ring.coords)[:-1]
    return cq.Wire.makePolygon([cq.Vector(float(x),float(y),float(z)) for x,y in xy],close=True)


def _extrude(poly, low, high, simplify=.045):
    poly = orient(poly.simplify(simplify,preserve_topology=True))
    return cq.Solid.extrudeLinear(_wire(poly.exterior,low),[_wire(h,low) for h in poly.interiors],
                                 cq.Vector(0,0,float(high-low)))


def _boss_changes(piece, supported, top_z):
    """Group adjacent, similar changes into one cutter or addition."""
    tracks = []
    for z,section in supported:
        for adding,patch in ((False,piece.difference(section)),(True,section.difference(piece))):
            for region in getattr(patch,"geoms",[patch]):
                if region.geom_type!="Polygon" or region.area<.7:
                    continue
                for track in tracks:
                    last = track["profiles"][-1]
                    if (track["adding"]==adding and .01<z-track["hi"]<.21
                            and last.intersection(region).area>.65*min(last.area,region.area)):
                        track["hi"] = z
                        track["profiles"].append(region)
                        break
                else:
                    tracks.append(dict(lo=z,hi=z,adding=adding,profiles=[region]))
    for track in tracks:
        if len(track["profiles"])<2:
            continue
        region = track["profiles"][len(track["profiles"])//2]
        tool = _extrude(region.buffer(.025),track["lo"]-.1,min(top_z+.02,track["hi"]+.1))
        yield track["adding"],tool


def _cut(body, tool):
    candidate = body.cut(tool)
    if candidate.isValid() and len(candidate.Solids())==1 and candidate.Volume()>0:
        return candidate
    raise ValueError("boolean failed topology check")


def _largest(polys):
    return max(polys,key=lambda p:Polygon(p.exterior).area)


def _rounded_outer(outer, low, high):
    """Use exact planar/cylindrical faces when the outline supports them."""
    from scipy.optimize import least_squares

    line = outer.exterior
    xy = np.array([line.interpolate(t,normalized=True).coords[0]
                   for t in np.linspace(0,1,240,endpoint=False)])
    x0,y0,x1,y1 = outer.bounds
    half = np.array([(x1-x0)/2,(y1-y0)/2])
    center = np.array([(x0+x1)/2,(y0+y1)/2])

    def residual(params):
        delta = abs(xy-params[:2])-params[2:4]+params[4]
        return np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)-params[4]

    maximum = min(half)*.66
    if maximum<=.51 or min(half)<=.8:
        return None
    estimate = least_squares(residual,np.r_[center,half,min(4.,maximum-.01)],
                             bounds=(np.r_[center-.8,half-.8,.5],np.r_[center+.8,half+.8,maximum]),
                             loss="soft_l1",f_scale=.08)
    if np.quantile(abs(estimate.fun),.98)>.23:
        return None
    x,y,w,h,radius = estimate.x
    return (cq.Workplane().box(2*w,2*h,high-low).edges("|Z").fillet(float(radius)).val()
            .translate((x,y,(low+high)/2)))


def _fillet_outer(body, bounds, radius, *, top):
    x0,y0,x1,y1 = bounds
    height = body.BoundingBox()
    edges = []
    for edge in body.Edges():
        vertices = edge.Vertices()
        if not vertices:
            continue
        lo,hi = min(v.Z for v in vertices),max(v.Z for v in vertices)
        if (top and lo<height.zmax-1.6) or (not top and hi>height.zmin+.45):
            continue
        center = edge.Center()
        if min(abs(center.x-x0),abs(center.x-x1),abs(center.y-y0),abs(center.y-y1))<1.5:
            edges.append(edge)
    if not edges:
        return body,False
    try:
        candidate = body.fillet(radius,edges)
        if candidate.isValid() and len(candidate.Solids())==1 and .95*body.Volume()<candidate.Volume()<body.Volume():
            return candidate,True
    except (RuntimeError,ValueError,Standard_Failure):
        pass
    return body,False


def _profiles(scan):
    data = [(z,_largest(polys),polys) for z,polys in stock.sections(scan,.2) if polys]
    if len(data)<20:
        return None
    outer_area = np.array([Polygon(p.exterior).area for _,p,_ in data])
    full = outer_area>.94*np.quantile(outer_area,.9)
    middle = [(z,p,ps) for (z,p,ps),ok in zip(data,full)
              if ok and sum(Polygon(r).area>8 for r in p.interiors)>=2]
    if len(middle)<12:
        return None
    start = middle[0][0]
    core = [row for row in middle if start+.4<=row[0]<=start+3]
    counts = [sum(Polygon(r).area>8 for r in row[1].interiors) for row in core]
    if len(core)<5:
        return None
    mode = max(set(counts),key=lambda count: (counts.count(count),count))
    core = [row for row,count in zip(core,counts) if count==mode]
    if mode<2 or len(core)<5:
        return None
    # A medoid keeps the measured corners and partitions while avoiding a
    # spurious contour from any one noisy horizontal section.
    outline = min(core,key=lambda a:sum(a[1].symmetric_difference(b[1]).area for b in core))
    if np.median([outline[1].symmetric_difference(b[1]).area for b in core])>.1*outline[1].area:
        return None
    return data,middle,outline


def assemble_stable(scan, log=print):
    """Return a single validated body on supported sections, otherwise defer to 0с."""
    started = perf_counter()
    try:
        profiles = _profiles(scan)
        if profiles is None:
            return None
        data,middle,(_,plane,parts) = profiles
        outer = Polygon(plane.exterior)
        bounds = outer.bounds
        zmin,zmax = scan.bounds[:,2]
        floor = middle[0][0]-.1
        ring = np.asarray(outer.exterior.coords)[:-1]
        top = stock.top_surface(scan,ring,zmax,band=1.65)
        bottom = stock.top_surface(scan,ring,zmin+.35,down=True)
        if top is None or bottom is None:
            return None
        round_edges = os.environ.get("CAD_KOMPAS_STABLE_ROUND", "0") == "1"
        body = _rounded_outer(outer,zmin-.5,zmax+1.) if round_edges else None
        analytic_outline = body is not None
        if body is None:
            body = _extrude(outer,zmin-.5,zmax+1.)
        for below,coef in ((False,top),(True,bottom)):
            clipped = stock.cut_above(body,coef,bounds,below=below)
            if not clipped.isValid() or len(clipped.Solids())!=1 or not 0<clipped.Volume()<body.Volume():
                return None
            body = clipped
        bottom_rounded = top_rounded = False
        if round_edges and analytic_outline:
            body,bottom_rounded = _fillet_outer(body,bounds,.48,top=False)
            body,top_rounded = _fillet_outer(body,bounds,.3,top=True)
        holes = [hole for hole in plane.interiors if Polygon(hole).area>8]
        for hole in holes:
            body = _cut(body,_extrude(Polygon(hole),floor,zmax+1.))

        bosses = 0
        for piece in parts:
            if piece is plane or not outer.contains(piece):
                continue
            contained = [(z,[p for p in ps if p is not main and p.intersection(piece).area>.6*piece.area])
                         for z,main,ps in data if z>=floor]
            supported = [(z,max(ps,key=lambda p:p.intersection(piece).area)) for z,ps in contained if ps]
            if len(supported)<6:
                continue
            top_z = supported[-1][0]+.1
            boss = _extrude(piece,floor-.2,top_z)
            # Use section changes to recover recesses and retained latch material.
            for adding,tool in _boss_changes(piece,supported,top_z):
                updated = boss.fuse(tool) if adding else boss.cut(tool)
                if updated.isValid() and len(updated.Solids())==1:
                    boss = updated
            added = body.fuse(boss)
            if added.isValid() and len(added.Solids())==1:
                body = added
                bosses += 1
        body = body.clean()
        info = {"stable_pockets": True, "voids": len(holes), "voids_skipped": 0,
                "bosses": bosses, "faces": len(body.Faces()), "seconds": perf_counter()-started}
        if round_edges:
            info.update(analytic_outline=analytic_outline,bottom_rounded=bottom_rounded,top_rounded=top_rounded)
        log(f"    stable pockets: {info}")
        return body,info
    except (RuntimeError, ValueError, Standard_Failure) as error:
        log(f"    stable pockets rollback: {error}")
        return None
