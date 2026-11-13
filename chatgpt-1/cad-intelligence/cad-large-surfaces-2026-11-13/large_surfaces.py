"""Fit a shared cubic spline over a rectangular free zone and sew it to planar neighbors."""
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix, eye, kron, vstack
from scipy.sparse.linalg import lsqr
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.Geom import Geom_BSplineSurface
from OCP.ShapeFix import ShapeFix_Edge
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pln, gp_Pnt, gp_Vec

from patch_faces import _loops, _mesh_cover, topology


@dataclass(frozen=True)
class Chart:
    origin: np.ndarray
    axes: np.ndarray
    size: tuple[float, float]
    bottom: float

    def xyz(self, u, v, h):
        return self.origin + np.array([u*self.size[0], v*self.size[1], h]) @ self.axes


def rectangular_chart(mesh, zone, support_normals, support_origins, bottom_point, tol=.3):
    """Infer a chart from four known opposite support planes and the scan zone."""
    normals=np.asarray(support_normals,dtype=float)
    origins=np.asarray(support_origins,dtype=float)
    if normals.shape!=(4,3) or origins.shape!=(4,3):
        raise ValueError("four lateral supports required")
    norms=np.linalg.norm(normals,axis=1)
    if np.any(norms<1e-9):
        raise ValueError("invalid support normal")
    normals=normals/norms[:,None]
    x=normals[0]
    opposite=int(np.argmin(normals@x))
    other=[i for i in range(1,4) if i!=opposite]
    y=normals[other[0]]
    if (normals[opposite]@x > -.99999 or normals[other[1]]@y > -.99999
            or abs(x@y)>1e-5):
        raise ValueError("supports are not rectangular")
    y=y-(x@y)*x
    y/=np.linalg.norm(y)
    z=np.cross(x,y)
    areas=mesh.area_faces[zone]
    mean=np.average(mesh.face_normals[zone],axis=0,weights=areas)
    if mean@z<0:
        x,z=-x,-z
    if np.quantile(mesh.face_normals[zone]@z,.05)<.15:
        raise ValueError("zone contains an overhang or unaligned normals")
    planes=[sorted(origins[[0,opposite]]@x),sorted(origins[other]@y)]
    triangles=mesh.faces[zone]
    loops=_loops(topology(triangles))
    if loops is None or len(loops)!=1:
        raise ValueError("zone must have one closed boundary without holes")
    points=mesh.vertices[np.unique(triangles)]
    offset=float(np.median(points@z))
    origin=planes[0][0]*x+planes[1][0]*y+offset*z
    sizes=tuple(high-low for low,high in planes)
    bottom=float((bottom_point-origin)@z)
    chart=Chart(origin,np.array([x,y,z]),sizes,bottom)
    uv=(points-origin)@chart.axes[:2].T
    for axis in range(2):
        if uv[:,axis].min()>tol or uv[:,axis].max()<sizes[axis]-tol:
            raise ValueError("zone does not span its support planes")
    ring=(mesh.vertices[loops[0]]-origin) @ chart.axes[:2].T
    if np.any(np.min(np.abs(np.column_stack((ring,ring-np.asarray(sizes)))),axis=1)>tol):
        raise ValueError("zone boundary leaves the four support planes")
    projected=(mesh.vertices[triangles]-origin) @ chart.axes[:2].T
    a=projected[:,1]-projected[:,0]
    b=projected[:,2]-projected[:,0]
    signed=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if np.any(signed<=1e-12):
        raise ValueError("chart projection folds over itself")
    footprint=float(signed.sum()/2)
    if not .98<=footprint/np.prod(sizes)<=1.02:
        raise ValueError("zone does not cover one rectangular footprint")
    return points,chart


def _axis(length, span):
    breaks = np.linspace(0, 1, max(1, round(length/span))+1)
    knots = np.r_[np.zeros(3),breaks,np.ones(3)]
    greville = np.array([knots[i+1:i+4].mean() for i in range(len(knots)-4)])
    return breaks, knots, greville


def _basis(uv, ku, kv):
    u = BSpline.design_matrix(uv[:,0], ku, 3).tocoo()
    v = BSpline.design_matrix(uv[:,1], kv, 3).tocoo()
    return coo_matrix(((u.data.reshape(-1,4)[:,:,None]*v.data.reshape(-1,4)[:,None,:]).ravel(),
                       (np.repeat(np.arange(len(uv)),16),
                        (u.col.reshape(-1,4)[:,:,None]*v.shape[1]+v.col.reshape(-1,4)[:,None,:]).ravel())),
                      shape=(len(uv),u.shape[1]*v.shape[1])).tocsr()


def _array(values, cls):
    result = cls(1,len(values))
    for i,value in enumerate(values,1):
        result.SetValue(i,value)
    return result


def fit_surface(points, chart, span=12., smoothing=.001, tol=.3):
    """Area-balanced, robust least squares with a second-difference penalty."""
    if not 5 <= span <= 15 or not np.isfinite(smoothing) or smoothing <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("invalid fit parameters")
    lengths=np.asarray(chart.size)
    frame=np.asarray(chart.axes)
    if (lengths.shape!=(2,) or not np.isfinite(lengths).all() or not np.isfinite(chart.bottom)
            or frame.shape!=(3,3) or not np.allclose(frame @ frame.T,np.eye(3),atol=1e-6)
            or np.linalg.det(frame)<0 or np.min(lengths)<=0):
        raise ValueError("invalid chart")
    local=(np.asarray(points)-chart.origin) @ frame.T
    if len(local)<16 or not np.isfinite(local).all():
        raise ValueError("insufficient scan samples")
    uv=local[:,:2]/lengths
    if np.any(uv < -tol/lengths) or np.any(uv > 1+tol/lengths):
        raise ValueError("scan extends outside support planes")
    uv=np.clip(uv,0,1)
    h=local[:,2]
    if h.min() <= chart.bottom + tol:
        raise ValueError("free surface touches bottom")
    bu,ku,gu=_axis(lengths[0],span)
    bv,kv,gv=_axis(lengths[1],span)
    if len(gu)*len(gv)>4096:
        raise ValueError("chart exceeds the control-point budget")
    cells=np.minimum((uv*[len(bu)-1,len(bv)-1]).astype(int),[len(bu)-2,len(bv)-2])
    cell_ids=cells[:,0]*(len(bv)-1)+cells[:,1]
    counts=np.bincount(cell_ids,minlength=(len(bu)-1)*(len(bv)-1))
    if not counts.all():
        raise ValueError("unsupported quadrilateral cell")
    base=np.sqrt(np.median(counts)/counts[cell_ids])
    A=_basis(uv,ku,kv)
    # Normalize the finite difference at repeated end knots using pole coordinates.
    def regularizer(g):
        out=[]
        for i in range(len(g)-2):
            row=np.zeros(len(g))
            a,b=np.diff(g[i:i+3])
            row[i:i+3]=[1/a,-1/a-1/b,1/b]
            out.append(row*(a+b)/2)
        return np.asarray(out)
    du,dv=regularizer(gu),regularizer(gv)
    penalty=vstack([kron(du,eye(len(gv))),kron(eye(len(gu)),dv)]).tocsr()
    penalty*=np.sqrt(smoothing*len(points)/A.shape[1])
    weights=base
    for _ in range(3):
        system=vstack([A.multiply(weights[:,None]),penalty]).tocsr()
        solution=lsqr(system,np.r_[h*weights,np.zeros(penalty.shape[0])],atol=1e-9,btol=1e-9,iter_lim=1200)
        if solution[1] not in (0,1,2,4,5):
            raise ValueError("least squares did not converge")
        coeff=solution[0]
        residual=abs(A@coeff-h)
        weights=base*np.sqrt(np.minimum(1,tol/2/np.maximum(residual,1e-12)))
    if coeff.min() <= chart.bottom+1e-6:
        raise ValueError("fitted spline cannot be certified above the bottom")
    poles=TColgp_Array2OfPnt(1,len(gu),1,len(gv))
    for i,u in enumerate(gu):
        for j,v in enumerate(gv):
            poles.SetValue(i+1,j+1,gp_Pnt(*chart.xyz(u,v,coeff[i*len(gv)+j])))
    u_mult=[4]+[1]*(len(bu)-2)+[4]
    v_mult=[4]+[1]*(len(bv)-2)+[4]
    spline=Geom_BSplineSurface(poles,_array(bu,TColStd_Array1OfReal),_array(bv,TColStd_Array1OfReal),
                             _array(u_mult,TColStd_Array1OfInteger),_array(v_mult,TColStd_Array1OfInteger),3,3)
    return spline,bu,bv,{"fit_p95_mm":float(np.quantile(residual,.95)), "fit_max_mm":float(residual.max())}


def _point(p):
    return np.array(p.Coord())


def _line(a,b):
    return cq.Edge.makeLine(cq.Vector(*a),cq.Vector(*b)).wrapped


def _side(chart, edges, axis, side):
    if not edges:
        raise ValueError("missing shared edges on support plane")
    ordered=[]
    for edge in edges:
        curve=BRepAdaptor_Curve(edge)
        a,b=(_point(curve.Value(t)) for t in (curve.FirstParameter(),curve.LastParameter()))
        if (b-a)@chart.axes[1-axis] < 0:
            a,b=b,a
            edge=TopoDS.Edge_s(edge.Reversed())
        ordered.append((float((a-chart.origin)@chart.axes[1-axis]),edge,a,b))
    ordered.sort(key=lambda item:item[0])
    first,last=ordered[0][2],ordered[-1][3]
    n=chart.axes[axis]*(1 if side else -1)
    plane=gp_Pln(gp_Pnt(*first),cq.Vector(*n).toDir())
    carrier=BRepBuilderAPI_MakeFace(plane).Face()
    fix=ShapeFix_Edge()
    wire=BRepBuilderAPI_MakeWire()
    for _,e,_,_ in ordered:
        fix.FixAddPCurve(e,carrier,False,1e-8)
        fix.FixSameParameter(e)
        wire.Add(e)
    down=chart.axes[2]
    bottom_a=first+(chart.bottom-(first-chart.origin)@down)*down
    bottom_b=last+(chart.bottom-(last-chart.origin)@down)*down
    for a,b in ((last,bottom_b),(bottom_b,bottom_a),(bottom_a,first)):
        wire.Add(_line(a,b))
    if not wire.IsDone():
        raise ValueError("support contour did not close")
    face=BRepBuilderAPI_MakeFace(plane,wire.Wire(),True)
    if not face.IsDone():
        raise ValueError("support face failed")
    return face.Face()


def build(points, chart, span=12., smoothing=.001, tol=.3):
    """Build a solid for a graph-like zone bounded by four orthogonal planes."""
    surface,u,v,audit=fit_surface(points,chart,span,smoothing,tol)
    sew=BRepBuilderAPI_Sewing(1e-5)
    boundary={(i,j):[] for i in range(2) for j in range(2)}
    for i,(u0,u1) in enumerate(zip(u,u[1:])):
        for j,(v0,v1) in enumerate(zip(v,v[1:])):
            face=BRepBuilderAPI_MakeFace(surface,float(u0),float(u1),float(v0),float(v1),1e-8).Face()
            sew.Add(face)
            for edge in cq.Face(face).Edges():
                c=BRepAdaptor_Curve(edge.wrapped)
                midpoint=_point(c.Value((c.FirstParameter()+c.LastParameter())/2))
                uv=(midpoint-chart.origin) @ chart.axes[:2].T / chart.size
                for axis in range(2):
                    for side in (0,1):
                        if abs(uv[axis]-side)<1e-7:
                            boundary[axis,side].append(edge.wrapped)
    for (axis,side),edges in boundary.items():
        sew.Add(_side(chart,edges,axis,side))
    bottom=cq.Wire.makePolygon([cq.Vector(*chart.xyz(x,y,chart.bottom)) for x,y in ((0,0),(1,0),(1,1),(0,1))],close=True)
    sew.Add(cq.Face.makeFromWires(bottom).wrapped)
    sew.Perform()
    shell=sew.SewedShape()
    if shell.ShapeType()!=TopAbs_SHELL or sew.NbFreeEdges() or sew.NbMultipleEdges():
        raise ValueError(f"shell not closed: {sew.NbFreeEdges()} free edges")
    solid=BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell)).Solid()
    if not BRepLib.OrientClosedSolid_s(solid) or not BRepCheck_Analyzer(solid).IsValid():
        raise ValueError("invalid sewn solid")
    return cq.Solid(solid),{**audit,"quad_faces":(len(u)-1)*(len(v)-1)}


def step_audit(body, destination=None):
    """Reopen STEP and measure both meshing and physical tangent continuity."""
    if destination is None:
        with TemporaryDirectory() as temp:
            return step_audit(body,Path(temp)/"model.step")
    cq.exporters.export(body,str(destination))
    restored=cq.importers.importStep(str(destination)).val()
    incidences={}
    smooth=[]
    minimum=1.
    for face in restored.Faces():
        minimum=min(minimum,_mesh_cover(face.wrapped,face.Area()))
        if face.geomType()=="BSPLINE":
            smooth.append(face)
            for edge in face.Edges():
                incidences.setdefault(edge,[]).append(face.wrapped)
    maximum=0.
    shared=0
    for edge,faces in incidences.items():
        if len(faces)>2:
            raise ValueError("non-manifold spline seam after STEP round trip")
        if len(faces)!=2:
            continue
        shared+=1
        curve=BRepAdaptor_Curve(edge.wrapped)
        for p in np.linspace(.05,.95,11):
            t=(1-p)*curve.FirstParameter()+p*curve.LastParameter()
            normals=[]
            for face in faces:
                pc=BRep_Tool.CurveOnSurface_s(edge.wrapped,face,0.,0.)
                if pc is None or not BRep_Tool.SameParameter_s(edge.wrapped):
                    raise ValueError("unmeasurable spline seam after STEP round trip")
                uv=pc.Value(t)
                du,dv=gp_Vec(),gp_Vec()
                BRepAdaptor_Surface(face).D1(uv.X(),uv.Y(),gp_Pnt(),du,dv)
                normal=du.Crossed(dv)
                normal.Normalize()
                normals.append(normal)
            maximum=max(maximum,float(np.degrees(np.arccos(np.clip(normals[0].Dot(normals[1]),-1,1)))))
    return restored,{"valid":bool(restored.isValid() and len(restored.Solids())==1),"mesh_cover_min":minimum,
                     "spline_faces":len(smooth),"spline_area_mm2":sum(f.Area() for f in smooth),
                     "shared_seams":shared,"seam_angle_max_deg":maximum}
