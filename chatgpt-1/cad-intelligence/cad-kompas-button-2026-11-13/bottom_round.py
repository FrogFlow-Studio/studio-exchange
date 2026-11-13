"""Task 49. Estimate an inset radius and round the lower outer edge."""
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon


def estimate_radius(contours, z, bottom_z, *, bounds=(.2,.85)):
    """Fit the quarter-circle inset profile to section areas."""
    z = np.asarray(z)
    areas = np.asarray([Polygon(c).area for c in contours])
    height = z-bottom_z
    near = (height>0) & (height<1.1)
    ref = (height>.9) & (height<2.5)
    if near.sum()<3 or ref.sum()<3:
        return None
    area = np.median(areas[ref])
    perim = np.median([Polygon(c).length for c,use in zip(contours,ref) if use])
    inset = (area-areas[near])/perim
    if np.max(inset) < .04:
        return None

    def residual(parameters):
        radius,offset = parameters
        h = np.clip(height[near]+offset,0.,radius)
        predicted = radius-np.sqrt(np.maximum(0.,2*radius*h-h*h))
        return predicted-inset

    result = least_squares(residual,[.45,0.],bounds=([bounds[0],-.15],[bounds[1],.15]),
                           loss="soft_l1",f_scale=.06)
    return float(result.x[0]) if result.success and np.median(abs(result.fun))<.09 else None


def round_bottom(body, radius, *, selection_band=.25):
    """Round the selected lower contour, rolling back on invalid topology."""
    if radius is None or not np.isfinite(radius) or radius<=0:
        return body,False
    extent = body.BoundingBox()
    candidates = [(edge,max(v.Z for v in edge.Vertices())) for edge in body.Edges() if edge.Vertices()
                  and (abs(edge.Center().x-(extent.xmin+extent.xmax)/2)>.41*extent.xlen
                       or abs(edge.Center().y-(extent.ymin+extent.ymax)/2)>.41*extent.ylen)]
    if not candidates:
        return body,False
    low = min(z for _,z in candidates)
    edges = [edge for edge,z in candidates if z<=low+selection_band]
    from OCP.Standard import Standard_Failure

    volume = body.Volume()
    radii = (radius,.85*radius,.7*radius)
    for trial in dict.fromkeys(radii):
        if trial<=0:
            continue
        try:
            candidate = body.fillet(trial,edges)
            if (candidate.isValid() and len(candidate.Solids()) == 1
                    and .95*volume <= candidate.Volume() < volume):
                return candidate,True
        except (ValueError,RuntimeError,Standard_Failure):
            continue
    return body,False
