"""Fit a trimmed B-spline to a connected scan zone and its analytic neighbors."""
import cadquery as cq
import numpy as np
import trimesh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.Geom import Geom_CylindricalSurface, Geom_Plane
from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.ShapeFix import ShapeFix_Edge, ShapeFix_Face
from OCP.Standard import Standard_Failure
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt

from large_surfaces import parametric_surface
from compare import compare
from nurbs import body_mesh
from patch_faces import _loops, _mesh_cover, topology


def _signed_area(uv, faces):
    a, b, c = uv[faces].transpose(1, 0, 2)
    x, y = (b-a).T
    u, v = (c-a).T
    return x*v-y*u


def _parameterize(points, faces, outer, holes, cross):
    # An orthographic chart preserves physical scale on shallow surfaces.
    # Fall back to a positive-weight harmonic map for a wrapped disk.
    origin = points.mean(axis=0)
    normal = cross.sum(axis=0)
    if np.linalg.norm(normal) > 1e-9:
        normal /= np.linalg.norm(normal)
        centered = points-origin
        flat = centered-np.outer(centered@normal, normal)
        axis = np.linalg.svd(flat, full_matrices=False)[2][0]
        uv = flat @ np.array([axis, np.cross(normal, axis)]).T
        if np.all(_signed_area(uv, faces) > 1e-12):
            return uv
    if holes:
        raise ValueError("wrapped chart with holes requires cuts")
    boundary = points[outer]
    distances = np.linalg.norm(boundary-np.roll(boundary, -1, axis=0), axis=1)
    perimeter = distances.sum()
    theta = 2*np.pi*np.r_[0., np.cumsum(distances[:-1])]/perimeter
    uv = np.empty((len(points),2))
    uv[outer] = (perimeter/(2*np.pi))*np.column_stack((np.cos(theta),np.sin(theta)))
    inner = np.setdiff1d(np.arange(len(points)),outer)
    edges = np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    edges = np.vstack((edges,edges[:,::-1]))
    weight = 1/np.maximum(1e-8,np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1))
    graph = coo_matrix((weight,edges.T),shape=(len(points),len(points))).tocsr()
    laplacian = diags(np.asarray(graph.sum(axis=1)).ravel())-graph
    if len(inner):
        uv[inner] = spsolve(laplacian[inner][:,inner], -(laplacian[inner][:,outer]@uv[outer]))
    if not np.isfinite(uv).all() or np.any(_signed_area(uv,faces) <= 1e-12):
        raise ValueError("folded or disconnected parameterization")
    return uv


def _carrier(neighbor):
    if neighbor["kind"] == "plane":
        return Geom_Plane(gp_Pnt(*neighbor["origin"]),gp_Dir(*neighbor["normal"]))
    if neighbor["kind"] == "cylinder":
        return Geom_CylindricalSurface(gp_Ax3(gp_Pnt(*neighbor["origin"]),gp_Dir(*neighbor["axis"])),
                                       float(neighbor["r"]))
    raise ValueError("unsupported neighbor")


def _intersection_wire(surface, carrier, ring, tol):
    intersection = GeomAPI_IntSS(surface,carrier,1e-7)
    if not intersection.IsDone():
        raise ValueError("neighbor intersection failed")
    choices = []
    tree = cKDTree(ring)
    step = np.median(np.linalg.norm(ring-np.roll(ring,1,axis=0),axis=1))
    for i in range(1,intersection.NbLines()+1):
        curve = intersection.Line(i)
        curve_points = np.asarray([curve.Value(t).Coord() for t in
                                   np.linspace(curve.FirstParameter(),curve.LastParameter(),361)])
        if np.quantile(tree.query(curve_points)[0],.95) > 2*tol + step:
            continue
        choices.append(curve)
    if not choices:
        raise ValueError("no intersection near boundary")
    distances = np.full(len(ring),np.inf)
    wire = BRepBuilderAPI_MakeWire()
    for curve in choices:
        for i,p in enumerate(ring):
            project = GeomAPI_ProjectPointOnCurve(gp_Pnt(*p),curve)
            if project.NbPoints():
                distances[i] = min(distances[i],project.LowerDistance())
        wire.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
        if not wire.IsDone():
            raise ValueError("disconnected intersection")
    if not wire.Wire().Closed() or np.quantile(distances,.9)>1.2*tol or distances.max()>3*tol:
        raise ValueError("open or distant intersection")
    return wire.Wire()


def _trimmed_face(surface, boundaries):
    maker = BRepBuilderAPI_MakeFace(surface,boundaries[0],True)
    if not maker.IsDone():
        return None
    for wire in boundaries[1:]:
        maker.Add(wire)
    fix = ShapeFix_Face(maker.Face())
    fix.SetPrecision(1e-6)
    fix.FixOrientation()
    fix.Perform()
    return TopoDS.Face_s(fix.Face())


def fit_zone_any(V, F, faces, neighbours, *, tol=.3, span=10., min_scan_coverage=.97):
    """Return a checked OCC face or None.

    Neighbours maps sorted boundary vertex pairs to a common plane/cylinder
    record per loop, or a shared TopoDS wire per loop. Shared wires are indexed
    by the smallest vertex index in the loop. A caller must sew the same
    intersection contours into the adjacent faces.
    """
    try:
        if not np.isfinite(tol) or tol <= 0 or not 5 <= span <= 15 or not .95 <= min_scan_coverage <= 1:
            return None
        V, F, faces = np.asarray(V,float),np.asarray(F,int),np.asarray(faces,int)
        if faces.ndim != 1 or len(faces)<12 or faces.min()<0 or faces.max()>=len(F):
            return None
        triangles = F[faces]
        loops = _loops(topology(triangles))
        if not loops:
            return None
        loops.sort(key=lambda loop: -np.linalg.norm(np.cross(V[loop],np.roll(V[loop],-1,axis=0)).sum(axis=0)))
        vertex_ids, inv = np.unique(triangles,return_inverse=True)
        local = inv.reshape(triangles.shape)
        points = V[vertex_ids]
        if not np.isfinite(points).all():
            return None
        xyz = points[local]
        cross = np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        if np.any(np.linalg.norm(cross,axis=1)<1e-10):
            return None
        outer = np.searchsorted(vertex_ids,loops[0])
        uv = _parameterize(points,local,outer,loops[1:],cross)
        lo, hi = uv.min(axis=0),uv.max(axis=0)
        if np.min(hi-lo)<tol:
            return None
        # A small margin permits intersections to lie beyond boundary samples.
        pad = .07*(hi-lo)
        uv = (uv-lo+pad)/(hi-lo+2*pad)
        surface = parametric_surface(uv,points,hi-lo,span=span,smoothing=.00003,tol=tol)
        wires = []
        for loop in loops:
            edge_keys = [tuple(sorted((int(a),int(b)))) for a,b in zip(loop,np.roll(loop,-1))]
            neighboring = [neighbours.get(key) for key in edge_keys]
            supplied = neighbours.get(int(min(loop)))
            if supplied is not None:
                wire = TopoDS.Wire_s(BRepBuilderAPI_Copy(supplied).Shape())
            elif neighboring[0] is not None and all(row is neighboring[0] for row in neighboring):
                wire = _intersection_wire(surface,_carrier(neighboring[0]),V[loop],tol)
            else:
                return None
            fixer = ShapeFix_Edge()
            explorer = BRepTools_WireExplorer(wire)
            while explorer.More():
                edge = explorer.Current()
                fixer.FixAddPCurve(edge,surface,TopLoc_Location(),False,1e-6)
                fixer.FixSameParameter(edge)
                explorer.Next()
            wires.append(wire)
        face = _trimmed_face(surface,wires)
        if face is None or not BRepCheck_Analyzer(face).IsValid():
            return None
        area = cq.Face(face).Area()
        if _mesh_cover(face,area) < .98:
            return None
        model = body_mesh(cq.Face(face),min(.04,tol/6))
        model.merge_vertices()
        source = trimesh.Trimesh(points,local,process=False)
        result = compare(source,model,tol,n=3000,seed=48)
        if min(result["within_tol_share"],1-result["added_surface_share"]) < min_scan_coverage:
            return None
        return [face]
    except (ValueError, RuntimeError, TypeError, KeyError, IndexError, np.linalg.LinAlgError, Standard_Failure):
        return None
