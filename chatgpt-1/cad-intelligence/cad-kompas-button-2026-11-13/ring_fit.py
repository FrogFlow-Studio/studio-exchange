"""Task 51. Section correspondence from one fixed set of angular rays."""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from shapely.geometry import Polygon


def correspond(contours, levels, count=64, smooth=.002):
    """Return aligned rings or None when a contour is not star-shaped."""
    polygons = [Polygon(p) for p in contours]
    if len(polygons)<4 or count<8 or any(not p.is_valid for p in polygons):
        return None
    center = np.median([p.centroid.coords[0] for p in polygons],axis=0)
    angle = np.arange(count)*2*np.pi/count
    rays = np.column_stack((np.cos(angle),np.sin(angle)))
    radii = []
    for poly in polygons:
        points = np.asarray(poly.exterior.coords)-center
        a, b = points[:-1],np.diff(points,axis=0)
        det = rays[:,1,None]*b[:,0]-rays[:,0,None]*b[:,1]
        parallel = abs(det)<1e-12
        safe = np.where(parallel,1.,det)
        r = (a[:,1]*b[:,0]-a[:,0]*b[:,1])/safe
        t = (rays[:,0,None]*a[:,1]-rays[:,1,None]*a[:,0])/safe
        valid = ~parallel & (r>0) & (t>=-1e-9) & (t<1-1e-9)
        outer = np.max(np.where(valid,r,-np.inf),axis=1)
        inner = np.min(np.where(valid,r,np.inf),axis=1)
        if not np.isfinite(outer).all() or np.any(outer-inner>.65):
            return None
        radii.append(outer)
    values = np.asarray(radii)
    z = np.asarray(levels,dtype=float)
    if np.any(np.diff(z)<=0):
        return None
    if len(z)>=5 and smooth>0:
        values = np.column_stack([make_smoothing_spline(z,column,lam=smooth)(z) for column in values.T])
    return center+values[:,:,None]*rays
