"""Task 50. Robust cubic rim fit balanced along the perimeter."""
import numpy as np
from shapely.geometry import Polygon
import shapely
import trimesh


def _basis(y):
    return np.column_stack((np.ones(len(y)),y,y*y,y*y*y))


def fit_rim(scan, ring, top, *, band=1.7):
    """Return a height evaluator after bounding the fit at sampled rim stations."""
    xy = np.asarray(ring)
    polygon = Polygon(xy)
    samples, ids = trimesh.sample.sample_surface(scan,80000,seed=50)
    candidates = samples[(scan.face_normals[ids,2]>.52) & (samples[:,2]>top-1.65)]
    if len(candidates)<100:
        return None
    distance = shapely.distance(polygon.exterior,shapely.points(candidates[:,:2]))
    pts = candidates[distance<band]
    if len(pts)<100:
        return None
    center = (xy[:,1].min()+xy[:,1].max())/2
    radius = np.ptp(xy[:,1])/2
    if radius<=0:
        return None
    edges = np.linspace(-1.001,1.001,33)
    stations = []
    for lo,hi in zip(edges,edges[1:]):
        group = pts[((pts[:,1]-center)/radius>=lo)&((pts[:,1]-center)/radius<hi)]
        if len(group)>=12:
            stations.append((np.median(group[:,1]),np.median(group[:,2])))
    stations = np.asarray(stations)
    if len(stations)<16:
        return None
    y = (stations[:,0]-center)/radius
    z = stations[:,1]
    if y.min()>-.9 or y.max()<.9:
        return None
    design = _basis(y)
    weights = np.ones(len(z))
    for _ in range(4):
        coef = np.linalg.lstsq(design*weights[:,None],z*weights,rcond=None)[0]
        residual = z-design@coef
        weights = np.sqrt(np.minimum(1.,.035/np.maximum(abs(residual),1e-9)))
    # Extend the height surface only as far as measured support, with
    # constant continuation beyond it when constructing the cutting tool.
    def evaluate(x,y_values):
        ys = np.asarray(y_values)
        t = np.clip((ys-center)/radius,y.min(),y.max())
        return np.polynomial.polynomial.polyval(t,coef)

    return evaluate


def cut_rim(body, height, bbox, *, below=False):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt

    grid = TColgp_Array2OfPnt(1,13,1,13)
    x0,y0,x1,y1 = bbox
    for i,x in enumerate(np.linspace(x0-3,x1+3,13),1):
        for j,y in enumerate(np.linspace(y0-3,y1+3,13),1):
            grid.SetValue(i,j,gp_Pnt(x,y,float(height(x,y))))
    surface = GeomAPI_PointsToBSplineSurface(grid,3,3,GeomAbs_C2,1e-5).Surface()
    face = BRepBuilderAPI_MakeFace(surface,1e-6).Face()
    x,y = (x0+x1)/2,(y0+y1)/2
    tool = BRepPrimAPI_MakeHalfSpace(face,gp_Pnt(x,y,float(height(x,y))+(-50 if below else 50))).Solid()
    return body.cut(cq.Solid(tool))
