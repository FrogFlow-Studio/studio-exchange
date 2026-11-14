"""Fast structural diagnostics for a finished 0с shape."""
from collections import Counter
from math import cos, radians

from OCP.BRepAdaptor import BRepAdaptor_Surface


def horizontal_planes(shape, angle_deg=1.):
    """Return almost horizontal planar faces and their centroid heights."""
    minimum = cos(radians(angle_deg))
    rows = []
    for face in shape.Faces():
        if face.geomType() == "PLANE":
            plane = BRepAdaptor_Surface(face.wrapped).Plane()
            if abs(plane.Axis().Direction().Z()) >= minimum:
                rows.append((face, face.Center().z))
    return rows


def cad_cleanliness(shape, *, narrow_mm=.08, height_mm=.05):
    """Measure face counts and shares on an aligned CadQuery shape."""
    faces = shape.Faces()
    areas = Counter()
    narrow = 0
    for face in faces:
        area = face.Area()
        areas[face.geomType()] += area
        # Count all boundary wires, including holes.
        perimeter = sum(wire.Length() for wire in face.Wires())
        narrow += perimeter > 0 and 2*area/perimeter < narrow_mm
    total = sum(areas.values())
    heights = sorted(z for _,z in horizontal_planes(shape))
    near = set()
    for i,(a,b) in enumerate(zip(heights,heights[1:])):
        if b-a < height_mm:
            near.update((i,i+1))
    plane, cylinder = (areas[k]/total if total else 0. for k in ("PLANE","CYLINDER"))
    return dict(faces=len(faces), narrow_faces=narrow, plane_area_share=plane,
                cylinder_area_share=cylinder, plane_cylinder_area_share=plane+cylinder,
                horizontal_planes=len(heights), near_height_horizontal_faces=len(near))
