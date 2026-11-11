from collections import Counter
from pathlib import Path
import tempfile
import unittest

import cadquery as cq
import numpy as np

from geometry_acceptance import Tolerance, evaluate, extract, match, topology, transform_shape

TOL = Tolerance(samples=600)


class GeometryAcceptanceTests(unittest.TestCase):
    def test_holes_missing_extra_parameter_error(self):
        def plate(holes):
            shape = cq.Workplane("XY").box(70, 42, 8).val()
            for x, y, r in holes:
                shape = shape.cut(cq.Solid.makeCylinder(r, 10, cq.Vector(x, y, -5)))
            return shape

        truth = plate([(-20, 0, 3), (0, 0, 3), (20, 0, 3)])
        model = plate([(-20, 0, 3.55), (20, 0, 3), (28, 12, 3)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, shape in (("truth", truth), ("model", model)):
                cq.exporters.export(shape, str(path / f"{name}.step"))
            result = evaluate(path / "model.step", path / "truth.step", tol=TOL)
        holes = [row for row in result["features"]
                 if (row["reference"] or row["model"])["role"] == "bore"]
        self.assertEqual(Counter(row["status"] for row in holes),
                         {"ok": 1, "parameter_error": 1, "missing": 1, "extra": 1})
        self.assertTrue(result["model_topology"]["closed"])

    def test_cylinder_axis_offset_and_rotation(self):
        shape = cq.Solid.makeCylinder(5, 20)
        moved = shape.rotate((0, 0, 0), (1, 0, 0), 4).translate((0.8, 0, 0))
        rows = match(extract(shape, TOL), extract(moved, TOL), TOL)
        errors = next(r for r in rows if r["reference"] and r["reference"]["kind"] == "cylinder")
        self.assertEqual(errors["status"], "parameter_error")
        self.assertIn("axis_angle_deg", errors["failed"])
        self.assertIn("axis_offset_mm", errors["failed"])

    def test_rigid_transform_and_plane(self):
        shape = cq.Workplane("XY").box(40, 30, 10).val()
        matrix = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 3], [0, 0, 0, 1]])
        model = shape.rotate((0, 0, 0), (0, 0, 1), 90).translate((10, 20, 3))
        rows = match(extract(transform_shape(shape, matrix), TOL), extract(model, TOL), TOL)
        self.assertEqual(Counter(r["status"] for r in rows), {"ok": 6})
        larger = cq.Workplane("XY").box(40, 30, 10.8).val()
        rows = match(extract(shape, TOL), extract(larger, TOL), TOL)
        self.assertEqual(Counter(r["status"] for r in rows), {"ok": 4, "parameter_error": 2})

    def test_cone(self):
        truth = cq.Solid.makeCone(9, 6, 12)
        model = cq.Solid.makeCone(9, 6.6, 12)
        rows = match(extract(truth, TOL), extract(model, TOL), TOL)
        cone = next(r for r in rows if r["reference"] and r["reference"]["kind"] == "cone")
        self.assertEqual(cone["status"], "parameter_error")
        self.assertIn("semi_angle_deg", cone["failed"])

    def test_cylindrical_fillet(self):
        body = cq.Workplane("XY").box(30, 20, 10).edges("|Z").fillet(2).val()
        fillets = [f for f in extract(body, TOL) if f["role"] == "cylindrical_fillet"]
        self.assertEqual(len(fillets), 4)
        self.assertTrue(all(abs(f["radius"] - 2) < 1e-6 for f in fillets))

    def test_topology_distinguishes_open_and_disconnected(self):
        a = cq.Workplane("XY").box(10, 10, 10).val()
        self.assertTrue(topology(a)["connected_solid"])
        self.assertFalse(topology(a.Faces()[0])["closed"])
        separate = cq.Compound.makeCompound([a, a.translate((30, 0, 0))])
        self.assertTrue(topology(separate)["closed"])
        self.assertFalse(topology(separate)["connected_solid"])
        edge = cq.Edge.makeLine(cq.Vector(15, 0, 0), cq.Vector(20, 0, 0))
        extra = topology(cq.Compound.makeCompound([a, edge]))
        self.assertEqual(extra["edges_outside_solids"], 1)
        self.assertFalse(extra["connected_solid"])


if __name__ == "__main__":
    unittest.main()
