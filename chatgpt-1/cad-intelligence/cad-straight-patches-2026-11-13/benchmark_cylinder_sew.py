"""Sew an exact tangent cylinder in place of the matching reference face."""
from __future__ import annotations

import json

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS

from straight_plus import cylindrical_face


def main():
    radius, length = 5., 19.
    diagonal = radius / np.sqrt(2.)
    reference = (cq.Workplane("XY").moveTo(radius, -8).lineTo(radius, 0)
                 .threePointArc((diagonal, diagonal), (0, radius)).lineTo(-8, radius)
                 .lineTo(-8, -8).close().extrude(length).val())
    planes = [{"normal": [1., 0., 0.], "origin": [radius, 0., 0.]},
              {"normal": [0., 1., 0.], "origin": [0., radius, 0.]}]
    blend = {"planes": (0, 1), "radius": radius, "axis": [0., 0., 1.],
             "origin": [0., 0., 0.], "axis_bounds_mm": (0., length), "sense": "convex"}
    face = cylindrical_face(blend, planes)
    edges = sorted(e.geomType() for e in cq.Face(face).Edges())
    sewing = BRepBuilderAPI_Sewing(1e-3)
    sewing.Add(face)
    for support in reference.Faces():
        if support.geomType() != "CYLINDER":
            sewing.Add(support.wrapped)
    sewing.Perform()
    shape = sewing.SewedShape()
    ok = shape.ShapeType() == TopAbs_SHELL and sewing.NbFreeEdges() == 0 and sewing.NbMultipleEdges() == 0
    error = None
    if ok:
        solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shape)).Solid()
        ok = bool(BRepLib.OrientClosedSolid_s(solid) and BRepCheck_Analyzer(solid).IsValid())
        error = abs(cq.Solid(solid).Volume() - reference.Volume())
    print(json.dumps({"edges": edges, "free_edges": sewing.NbFreeEdges(), "valid_solid": ok,
                      "volume_error_mm3": error}, indent=2))
    return 0 if ok and edges == ["CIRCLE", "CIRCLE", "LINE", "LINE"] and error < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
