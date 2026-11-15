"""Build valid planar section polygons, with fallback for ambiguous linework."""
import os

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def fast_polygons(mesh, z):
    """Return polygons in XY, or None when legacy contour repair is needed."""
    segments = trimesh.intersections.mesh_plane(mesh, (0., 0., 1.), (0., 0., z))
    if not len(segments):
        return []
    # Close floating-point gaps from opposite interpolations of a shared mesh edge.
    xy = np.round(segments[:, :, :2], 8)
    if not np.isfinite(xy).all() or np.any(np.all(xy[:, 0] == xy[:, 1], axis=1)):
        return None
    faces, cuts, loose, invalid = shapely.polygonize_full(shapely.linestrings(xy))
    if not cuts.is_empty or not loose.is_empty or not invalid.is_empty or faces.is_empty:
        return None
    polygons = list(faces.geoms)
    if len(polygons) == 1:
        return polygons if polygons[0].is_valid and not polygons[0].interiors else None
    shells = [Polygon(poly.exterior) for poly in polygons]
    tree = STRtree(shells)
    result = []
    for i, (poly, shell) in enumerate(zip(polygons, shells)):
        parents = [j for j in tree.query(shell, predicate="within") if j != i]
        if len(parents) % 2 == 0:
            result.append(poly)
    return result if all(p.is_valid for p in result) else None


def section_polygons(mesh, z, *, fast=None):
    """Opt-in fast contours, with the original trimesh repair on difficult sections."""
    if fast is None:
        fast = os.environ.get("SEC_FAST") == "1"
    if fast:
        from shapely.errors import GEOSException

        try:
            polygons = fast_polygons(mesh, z)
        except GEOSException:
            polygons = None
        if polygons is not None:
            return polygons
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []
    planar, _ = section.to_2D(to_2D=np.eye(4))
    return list(planar.polygons_full)
