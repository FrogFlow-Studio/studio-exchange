"""Estimate a convex edge radius in transverse slices of an aligned scan."""
import numpy as np
import trimesh
from scipy.optimize import least_squares


def _slice_radius(scan, point, tangent, normals):
    # In this oblique coordinate system, a sharp edge is at (0,0), and the
    # two unrounded supporting faces have signed normal coordinates zero.
    dot = normals[0]@normals[1]
    if not -.45<dot<.45:
        return None
    segments = trimesh.intersections.mesh_plane(scan,tangent,point)
    if not len(segments):
        return None
    mid = segments.mean(axis=1)-point
    segments = segments[np.linalg.norm(mid,axis=1)<2.25]
    if len(segments)<5:
        return None
    count = np.maximum(1,np.ceil(np.linalg.norm(segments[:,1]-segments[:,0],axis=1)/.035).astype(int))
    points = np.vstack([a+(b-a)*((np.arange(n)+.5)/n)[:,None] for (a,b),n in zip(segments,count)])
    q = (points-point)@normals.T
    a = q[(q[:,1]<-.95)&(abs(q[:,0])<.25),0]
    b = q[(q[:,0]<-.95)&(abs(q[:,1])<.25),1]
    if min(len(a),len(b))<5:
        return None
    q -= [np.median(a),np.median(b)]
    q = q[(q.max(axis=1)<.22)&(q.min(axis=1)>-1.1)]
    if len(q)<12:
        return None
    inverse = np.linalg.inv(normals@normals.T)

    def distances(radius):
        rel = q+radius
        projection = rel@inverse
        arc = np.sqrt(np.maximum(0.,np.einsum("ij,ij->i",projection,rel)))-radius
        return np.where((projection>0).all(axis=1),arc,np.max(q,axis=1))

    def fit(initial):
        result = least_squares(lambda x:distances(x[0]),[initial],bounds=(.015,1.25),
                               loss="soft_l1",f_scale=.035)
        return result.x[0],result.cost

    radius,cost = min((fit(r) for r in (.2,.55,.9)),key=lambda row:row[1])
    sharp = np.max(q,axis=1)
    sharp_cost = .035**2*np.sum(np.sqrt(1+(sharp/.035)**2)-1)
    interior = ((q+radius)@inverse>radius*.12).all(axis=1)
    if radius<.12 or interior.sum()<6 or sharp_cost<=0:
        return None
    return float(radius),float(cost/sharp_cost)


def fillet_radius(scan, body, edge):
    """Return a robust radius or None when the edge lacks repeatable arc evidence.

    The supplied body carries the two pre-fillet faces that meet at the edge.
    Inputs are aligned in millimetres. Convex corners near a right angle are
    supported; non-manifold, tangent and concave edges are rejected.
    """
    import cadquery as cq

    faces = [face for face in body.Faces() if any(e.isSame(edge) for e in face.Edges())]
    if len(faces)!=2 or edge.Length()<2.5:
        return None
    radii = []
    for station in np.linspace(.08,.92,21):
        location = edge.positionAt(float(station))
        point = np.array(location.toTuple())
        tangent = np.array(edge.tangentAt(float(station)).toTuple())
        normals = np.array([face.normalAt(location).toTuple() for face in faces])
        normals -= (normals@tangent)[:,None]*tangent
        lengths = np.linalg.norm(normals,axis=1)
        if np.min(lengths)<1e-6:
            continue
        normals /= lengths[:,None]
        probe = cq.Vector(*(point-.18*normals.sum(axis=0)))
        if not body.isInside(probe,tolerance=1e-5):
            continue
        radius = _slice_radius(scan,point,tangent,normals)
        if radius is not None:
            radii.append(radius)
    if len(radii)<12:
        return None
    values, ratios = np.asarray(radii).T
    median = np.median(values)
    if median<.17 or np.median(ratios)>.86 or np.median(abs(values-median))>.10:
        return None
    return float(median)
