"""Fit periodic outer profiles over all supplied heights."""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from shapely.geometry import Polygon


def fit_profiles(contours, z, *, count=None, modes=None, axial_penalty=.008):
    import os
    count = count or int(os.environ.get("WALL_PROFILE_COUNT", "96"))  # точек на кольцо; меньше — проще поверхность стенки
    modes = modes or int(os.environ.get("WALL_PROFILE_MODES", "28"))  # гармоник по кругу; меньше — глаже
    z = np.asarray(z,dtype=float)
    if len(z)<5 or len(z)!=len(contours) or np.any(np.diff(z)<=0):
        return None
    polygons = [Polygon(ring) for ring in contours]
    if any(not poly.is_valid or poly.is_empty for poly in polygons):
        return None
    center = np.median([poly.centroid.coords[0] for poly in polygons],axis=0)
    theta = np.arange(count)*2*np.pi/count
    rays = np.column_stack((np.cos(theta),np.sin(theta)))
    sections = []
    for poly in polygons:
        p = np.asarray(poly.exterior.coords)-center
        a,b = p[:-1],np.diff(p,axis=0)
        determinant = rays[:,1,None]*b[:,0]-rays[:,0,None]*b[:,1]
        parallel = abs(determinant)<1e-12
        safe = np.where(parallel,1.,determinant)
        r = (a[:,1]*b[:,0]-a[:,0]*b[:,1])/safe
        t = (rays[:,0,None]*a[:,1]-rays[:,1,None]*a[:,0])/safe
        valid = ~parallel & (r>0) & (t>=-1e-9) & (t<1-1e-9)
        outer = np.max(np.where(valid,r,-np.inf),axis=1)
        inner = np.min(np.where(valid,r,np.inf),axis=1)
        if not np.isfinite(outer).all() or np.any(outer-inner>.65):
            return None
        sections.append(outer)
    coefficients = np.fft.rfft(sections,axis=1)
    frequencies = np.arange(coefficients.shape[1])
    coefficients /= 1+(frequencies/max(1,modes))**8
    filtered = np.fft.irfft(coefficients,n=count,axis=1)
    filtered = np.column_stack([make_smoothing_spline(z,col,lam=axial_penalty)(z) for col in filtered.T])
    if np.any(filtered<=0):
        return None
    return center+filtered[:,:,None]*rays
