"""Splice a local analytic replacement into a previously assembled STEP body."""
import numpy as np
from scipy.spatial import cKDTree
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS
import cadquery as cq


def _support(face, tree, radius):
    vertices = np.asarray([v.Center().toTuple() for v in face.Vertices()])
    if not len(vertices):
        return False
    probes = np.vstack((vertices, [edge.positionAt(.5).toTuple() for edge in face.Edges()]))
    return bool((tree.query(probes)[0] <= radius).all())


def _distance(face, tree):
    vertices = np.asarray([v.Center().toTuple() for v in face.Vertices()])
    return float(tree.query(vertices)[0].min()) if len(vertices) else np.inf


def splice(stock, candidate, points, radius, sew_tol=.001, *, include_incident=False):
    """Reuse every outside face by identity. Reject if the two boundaries disagree."""
    if not len(points) or radius <= 0:
        return stock, {"accepted": False, "reason": "empty_scope"}
    tree = cKDTree(points)
    old = stock.Faces()
    def local(face):
        if _support(face,tree,radius):
            return True
        return include_incident and face.geomType()=="PLANE" and _distance(face,tree) <= radius/2
    outside = [f for f in old if not local(f)]
    inside = [f for f in candidate.Faces() if local(f)]
    new_cylinders = sum(f.geomType()=="CYLINDER" for f in inside)
    old_cylinders = sum(f.geomType()=="CYLINDER" for f in old)-sum(f.geomType()=="CYLINDER" for f in outside)
    audit = {"reused_faces":len(outside),"candidate_faces":len(inside),"local_cylinders":new_cylinders}
    if new_cylinders <= old_cylinders:
        return stock, {**audit,"accepted":False,"reason":"no_local_cylinder_gain"}
    sewing = BRepBuilderAPI_Sewing(sew_tol)
    for face in outside+inside:
        sewing.Add(face.wrapped)
    sewing.Perform()
    shell = sewing.SewedShape()
    if shell.ShapeType()!=TopAbs_SHELL or sewing.NbFreeEdges() or sewing.NbMultipleEdges():
        return stock, {**audit,"accepted":False,"reason":"boundary_mismatch","free_edges":sewing.NbFreeEdges()}
    solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell)).Solid()
    if not BRepLib.OrientClosedSolid_s(solid) or not BRepCheck_Analyzer(solid).IsValid():
        return stock, {**audit,"accepted":False,"reason":"invalid_solid"}
    result = cq.Solid(solid)
    # Sewing may replace an edge shared across the cut, but the faces far
    # from the cut must be literally present as the previous OCC faces.
    known = {face for face in result.Faces()}
    unchanged = sum(face in known for face in outside)
    if unchanged != len(outside):
        return stock, {**audit,"accepted":False,"reason":"outside_face_changed","outside_identical_faces":unchanged}
    return result, {**audit,"accepted":True,"outside_identical_faces":unchanged}
