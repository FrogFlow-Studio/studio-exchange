"""Estimate a nonconstant convex fillet law in the edge's curve parameter."""
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.Standard import Standard_Failure
from OCP.TColgp import TColgp_Array1OfPnt2d
from OCP.gp import gp_Pnt2d
from scipy.interpolate import UnivariateSpline

import cadquery as cq
from fillet_radius import _slice_radius


def _law(body, edge, values):
    builder = BRepFilletAPI_MakeFillet(body.wrapped)
    array = TColgp_Array1OfPnt2d(1,len(values))
    for i,(t,r) in enumerate(values,1):
        array.SetValue(i,gp_Pnt2d(float(t),float(r)))
    builder.Add(array,edge.wrapped)
    builder.Build()
    if builder.IsDone():
        result = cq.Shape.cast(builder.Shape())
        if result.isValid() and len(result.Solids())==1:
            return result
    return None


def radius_profile(scan, body, edge):
    faces = [f for f in body.Faces() if any(e.isSame(edge) for e in f.Edges())]
    if len(faces)!=2 or edge.Length()<8:
        return None
    curve = BRepAdaptor_Curve(edge.wrapped)
    stations = np.linspace(.04,.96,25)
    data = []
    for t in stations:
        u = (1-t)*curve.FirstParameter()+t*curve.LastParameter()
        point = np.array(curve.Value(u).Coord())
        location = cq.Vector(*point)
        tangent = np.array(edge.tangentAt(float(u),mode="parameter").toTuple())
        tangent /= np.linalg.norm(tangent)
        normals = np.array([face.normalAt(location).toTuple() for face in faces])
        normals -= (normals@tangent)[:,None]*tangent
        length = np.linalg.norm(normals,axis=1)
        if np.min(length)<1e-6:
            continue
        normals /= length[:,None]
        if not body.isInside(cq.Vector(*(point-.18*normals.sum(axis=0))),tolerance=1e-5):
            continue
        result = _slice_radius(scan,point,tangent,normals)
        if result is not None and result[1]<.85:
            data.append((t,result[0]))
    if len(data)<14:
        return None
    t,r = np.asarray(data).T
    if t[0]>.15 or t[-1]<.85 or np.max(np.diff(t))>.22:
        return None
    t_eval = np.linspace(0,1,9)
    fitted = UnivariateSpline(t,r,k=2,s=len(t)*.045**2)(np.clip(t_eval,t[0],t[-1]))
    if edge.IsClosed():
        fitted[[0,-1]] = np.mean(fitted[[0,-1]])
    fitted = np.clip(fitted,.17,1.1)
    if np.ptp(fitted)<.12:
        return None
    return np.column_stack((t_eval,fitted))


def variable_fillet(scan,body,edge):
    try:
        law = radius_profile(scan,body,edge)
        if law is None:
            return None
        result = _law(body,edge,law)
        if result is not None and .96*body.Volume()<result.Volume()<body.Volume():
            return result,law
    except (RuntimeError,ValueError,Standard_Failure):
        pass
    return None
