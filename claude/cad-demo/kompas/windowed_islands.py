"""Build a concentric sleeve with cylindrical bores and planar window cuts."""
from collections import defaultdict

import cadquery as cq
import numpy as np
import shapely
from OCP.Standard import Standard_Failure
from shapely.geometry import Point, Polygon
from shapely.affinity import translate

from cylindrical_islands import _fit
from partial_islands import _runs


def _rounded_radius(radius):
    return round(2*radius/.05)*.025


def _layer_radii(polys, center, outer):
    if len(polys)==1 and len(polys[0].interiors)==1:
        fit = _fit([polys[0].interiors[0]],diameter_step=.05,sample_segments=True)
        if fit is None or np.linalg.norm(fit[0]-center)>=.05:
            return None
        return fit[1]/2
    if len(polys)<2 or any(p.interiors for p in polys):
        return None
    rings = [np.asarray(p.exterior.coords)[:-1]-center for p in polys]
    # Quantiles are taken per fragment so a small fragment cannot be hidden
    # behind a densely sampled large fragment.
    radii = [np.linalg.norm(ring,axis=1) for ring in rings]
    def inner_peak(r):
        start = np.quantile(r,.015)
        choices = np.arange(start,start+.13,.005)
        scores = np.sum(np.exp(-.5*((r[:,None]-choices)/.022)**2),axis=0)
        return choices[np.argmax(scores)]

    inner = _rounded_radius(np.median([inner_peak(r) for r in radii]))
    if not .25<inner<outer-.2:
        return None
    for r in radii:
        if min(r)<inner-.10 or max(r)>outer+.10:
            return None
        if min(abs(r-inner))>.045 or min(abs(r-outer))>.045:
            return None
        for radius in (inner,outer):
            near = abs(r-radius)
            arc = near[near<.085]
            if len(arc)<3 or np.quantile(arc,.9)>=.05:
                return None
    return inner


def _window_edges(polys, center, radius):
    theta = np.arange(1440)*2*np.pi/1440
    rays = np.column_stack((np.cos(theta),np.sin(theta)))
    points = center+radius*rays
    solid = shapely.union_all(polys)
    missing = ~shapely.contains_xy(solid,points[:,0],points[:,1])
    start = np.flatnonzero(missing & ~np.roll(missing,1))
    end = np.flatnonzero(~missing & np.roll(missing,1))
    pairs = []
    for i in start:
        j = min((int(k) for k in end),key=lambda k:(k-i)%len(theta))
        length = (j-i)%len(theta)
        if length<25 or length>480:
            return None
        pairs.append((theta[i],theta[i]+length*2*np.pi/len(theta)))
    return pairs if len(pairs)==len(polys) else None


def _window_polygons(series,center,inner,outer):
    mid = (inner+outer)/2
    reference = _window_edges(series[len(series)//2][1],center,mid)
    if reference is None:
        return None
    border = []
    for _,polys in series:
        pairs = _window_edges(polys,center,mid)
        if pairs is None or len(pairs)!=len(reference):
            return None
        border.extend(np.asarray(p.exterior.coords)[:-1]-center for p in polys)
    points = np.vstack(border)
    radius = np.linalg.norm(points,axis=1)
    polar = np.arctan2(points[:,1],points[:,0])
    cutters = []
    for start,end in reference:
        cuts = []
        for edge in (start,end):
            angular_distance = abs(np.arctan2(np.sin(polar-edge),np.cos(polar-edge)))
            selected = points[(radius>inner+.10)&(radius<outer-.10)&(angular_distance<.14)]
            if len(selected)<6:
                return None
            origin = np.median(selected,axis=0)
            _,_,basis = np.linalg.svd(selected-origin,full_matrices=False)
            direction, normal = basis
            if np.quantile(abs((selected-origin)@normal),.95)>=.05 or np.ptp(selected@direction)<.35:
                return None
            if direction@origin<0:
                direction = -direction
            # Continue the two measured flank lines past the circular faces.
            cuts.append((origin-(mid-inner+.25)*direction,origin+2*outer*direction))
        poly = Polygon([cuts[0][0],cuts[0][1],cuts[1][1],cuts[1][0]])
        if not poly.is_valid:
            return None
        cutters.append(poly)
    annulus = Point(0,0).buffer(outer,quad_segs=64).difference(Point(0,0).buffer(inner,quad_segs=64))
    predicted = translate(annulus.difference(shapely.union_all(cutters)),*center)
    if any(predicted.symmetric_difference(shapely.union_all(polys)).area>.065*annulus.area
           for _,polys in series):
        return None
    return cutters


def windowed_island(sections, low, high, dz):
    """Recognize complete end rings and interrupted concentric middle slices."""
    grouped = defaultdict(list)
    for z,poly in sections:
        grouped[z].append(poly)
    rows = sorted(grouped.items())
    if len(rows)<5 or len(list(_runs(rows,dz)))!=1:
        return None
    intact = [p[0] for _,p in rows if len(p)==1 and len(p[0].interiors)==1]
    if len(intact)<2 or len(intact)==len(rows):
        return None
    try:
        fit = _fit([p.exterior for p in intact],diameter_step=.05,sample_segments=True)
        if fit is None:
            return None
        center,diameter = fit
        outer = diameter/2
        radii = [_layer_radii(polys,center,outer) for _,polys in rows]
        if any(r is None for r in radii):
            return None
        # Suppress a single quantization flip at a noisy level. A real
        # diameter step must be corroborated by consecutive slices.
        radii = [radii[0],*[float(np.median(radii[i-1:i+2])) for i in range(1,len(radii)-1)],radii[-1]]
        for interrupted,group in _layer_groups(rows):
            if not interrupted:
                continue
            median = _rounded_radius(np.median([radii[i] for i,_ in group]))
            if any(abs(radii[i]-median)>.026 for i,_ in group):
                return None
            for i,_ in group:
                radii[i] = median
        return _build(rows,radii,center,outer,low,high,dz)
    except (ValueError,RuntimeError,Standard_Failure):
        return None


def _layer_groups(rows):
    from itertools import groupby

    for interrupted,group in groupby(enumerate(rows),key=lambda row:len(row[1][1])>1):
        yield interrupted,list(group)


def _build(rows,radii,center,outer,low,high,dz):
    from itertools import groupby

    def cylinder(r,a,b):
        return cq.Solid.makeCylinder(r,b-a,cq.Vector(*center,a))

    body = cylinder(outer,low,high)
    boundaries = [rows[0][0]-dz/2,
                  *[(a[0]+b[0])/2 for a,b in zip(rows,rows[1:])],
                  high+.02]
    bores = []
    drill = []
    for radius,group in groupby(enumerate(radii),key=lambda row:row[1]):
        indices = [i for i,_ in group]
        a,b = boundaries[indices[0]],boundaries[indices[-1]+1]
        if len(indices)<2 or radius>=outer-.2:
            return None
        bores.append(dict(diameter_mm=round(2*radius,2),z_mm=[float(a),float(b)]))
        drill.append(cylinder(radius,a,b))
    body = body.cut(drill[0].fuse(*drill[1:]).clean() if len(drill)>1 else drill[0])
    tools = []
    for interrupted,series in _layer_groups(rows):
        if not interrupted:
            continue
        inner = radii[series[0][0]]
        if len(series)<3 or any(radii[i]!=inner for i,_ in series):
            return None
        layers = [row for _,row in series]
        polygons = _window_polygons(layers,center,inner,outer)
        if polygons is None:
            return None
        a,b = boundaries[series[0][0]],boundaries[series[-1][0]+1]
        for poly in polygons:
            xy = np.asarray(poly.exterior.coords)[:-1]+center
            wire = cq.Wire.makePolygon([cq.Vector(*point,a) for point in xy],close=True)
            tools.append(cq.Solid.extrudeLinear(wire,[],cq.Vector(0,0,b-a)))
    result = body.cut(*tools).clean()
    if result.isValid() and len(result.Solids())==1 and 0<result.Volume()<body.Volume():
        return result,dict(diameter_mm=round(2*outer,2),bores=bores,windows=len(tools),
                           center_mm=center.tolist())
    return None
