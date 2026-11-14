"""Recognize scan islands with a common cylindrical axis over their height."""
import cadquery as cq
import numpy as np
from OCP.Standard import Standard_Failure
from scipy.optimize import least_squares


def _fit(rings, tolerance=.05, diameter_step=.1, *, sample_segments=False):
    levels = []
    for ring in rings:
        samples = np.asarray(ring.coords)[:-1]
        if len(samples)<8:
            return None
        # Sample along segments as well: polygon vertices alone can all
        # lie on a circle even when the actual contour is faceted.
        if sample_segments:
            middle = (samples+np.roll(samples,-1,axis=0))/2
            samples = np.vstack((samples,middle))
        levels.append(samples)
    points = np.vstack(levels)
    origin = points.mean(axis=0)
    xy = points-origin
    design = np.c_[2*xy,np.ones(len(xy))]
    x,y,c = np.linalg.lstsq(design,np.sum(xy*xy,axis=1),rcond=None)[0]
    radius2 = c+x*x+y*y
    if radius2<=0:
        return None
    weights = np.repeat([1/np.sqrt(len(level)) for level in levels],[len(level) for level in levels])
    fit = least_squares(lambda q:(np.linalg.norm(xy-q[:2],axis=1)-q[2])*weights,
                        [x,y,np.sqrt(radius2)])
    if not fit.success:
        return None
    center, radius = fit.x[:2]+origin,fit.x[2]
    if radius<.25:
        return None
    diameter = round(round(2*radius/diameter_step)*diameter_step,2)
    for level in levels:
        relative = level-center
        angles = np.arctan2(relative[:,1],relative[:,0])
        angles = np.sort(angles)
        if np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))>np.pi/3:
            return None
        residual = np.linalg.norm(relative,axis=1)-radius
        # Measure the envelope after excluding the outer 5% of sample
        # residuals at each height; coherent deviations still fail.
        if np.quantile(abs(residual),.95)>=tolerance or np.max(abs(residual))>=2*tolerance:
            return None
    return center, diameter


def cylindrical_island(sections, low, high):
    """Return a cylinder with optional bore, or None for an unsupported island."""
    if len(sections)<3 or len({z for z,_ in sections})!=len(sections) or high<=low:
        return None
    polys = [poly for _,poly in sections]
    holes = {len(poly.interiors) for poly in polys}
    if holes not in ({0},{1}):
        return None
    outer = _fit([poly.exterior for poly in polys])
    inner = _fit([poly.interiors[0] for poly in polys]) if holes=={1} else None
    if outer is None or (holes=={1} and inner is None):
        return None
    center,diameter = outer
    try:
        body = cq.Solid.makeCylinder(diameter/2,high-low,cq.Vector(*center,low))
        bore = None
        if inner is not None:
            bore_center,bore = inner
            if np.linalg.norm(center-bore_center)+bore/2>=diameter/2:
                return None
            body = body.cut(cq.Solid.makeCylinder(bore/2,high-low+.2,cq.Vector(*bore_center,low-.1)))
        if body.isValid() and len(body.Solids())==1:
            return body,dict(center_mm=center.tolist(),diameter_mm=diameter,bore_diameter_mm=bore)
    except (RuntimeError,ValueError,Standard_Failure):
        pass
    return None


def guarded_exact_islands(scan, body, info, log, assemble_stock, *, mode=True):
    try:
        trial,details = assemble_stock(scan,log=log,exact_islands=mode,
            fit_pocket_floor=bool(info.get("smooth_pocket_floors")),
            void_n=info.get("void_n"),periodic_wall=bool(info.get("periodic_wall")),
            fit_upper_wall=bool(info.get("upper_wall_fit")),
            fit_variable=bool(info.get("top_fillet_law") or info.get("bottom_fillet_law")))
        if details.get("cylindrical_islands"):
            from candidate_guard import compare_candidate

            accepted,audit = compare_candidate(scan,body,trial,allowance=0.)
            if accepted and any(info.get(key) and not details.get(key)
                                for key in ("top_fillet","bottom_fillet","smooth_pocket_floors")):
                accepted = False
                audit["reason"] = "existing_feature_lost"
            if accepted:
                body,info = trial,{**info,**details}
            info = {**info,"exact_islands_guard":{**audit,"accepted":accepted}}
            log(f"    exact islands guard: {info['exact_islands_guard']}")
    except (ValueError,RuntimeError,Standard_Failure) as error:
        log(f"    exact islands rollback: {error}")
    return body,info
