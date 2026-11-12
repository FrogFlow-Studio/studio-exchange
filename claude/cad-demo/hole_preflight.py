"""Conservative rejection tests for a cylinder cut on a valid, single CAD solid."""
from itertools import product

import numpy as np


class HolePreflight:
    def __init__(self):
        self.body = None
        self.bounds = None
        self.valid = None
        self.vertices = None

    def _load(self, body):
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        self.body = body
        self.bounds = self.valid = self.vertices = None
        if body.ShapeType() != "Solid":
            return
        box = Bnd_Box()
        # Conservative B-rep bounds with shape tolerances; no STL or tessellation.
        BRepBndLib.Add_s(body.wrapped, box, False)
        if not box.IsVoid() and not box.IsOpen():
            bounds = np.array(box.Get(), dtype=float).reshape(2, 3)
            if np.isfinite(bounds).all():
                self.bounds = bounds

    def _valid_body(self):
        if self.valid is None:
            from OCP.BRepClass3d import BRepClass3d_SolidClassifier
            from OCP.Precision import Precision
            from OCP.TopAbs import TopAbs_OUT

            self.valid = False
            if self.body.isValid():
                classifier = BRepClass3d_SolidClassifier(self.body.wrapped)
                classifier.PerformInfinitePoint(Precision.Confusion_s())
                self.valid = classifier.State() == TopAbs_OUT
        return self.valid

    def reject(self, body, spec):
        """Return a proved rejection reason, otherwise let OCC decide as before."""
        from OCP.Precision import Precision

        if body is not self.body:
            self._load(body)
        if self.bounds is None:
            return None
        c, d, r, h = spec
        origin, axis = np.asarray(c, float), np.asarray(d, float)
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm <= 0 or not (r > 0 and h > 0):
            return None
        axis = axis / norm
        low, high = self.bounds
        scale = max(1.0, np.abs(self.bounds).max(), np.abs(origin).max(), r, h)
        margin = max(10 * Precision.Confusion_s(), 128 * np.finfo(float).eps * scale)

        def distances(points):
            q = points - origin
            axial = q @ axis
            radial = np.linalg.norm(q - axial[:, None] * axis, axis=1)
            return axial, radial

        def all_inside(points):
            axial, radial = distances(points)
            return np.all((axial > margin) & (axial < h - margin) & (radial < r - margin))

        # The cylinder is convex. Containing all bounding-box corners means
        # containing the entire body, so the result cannot have one solid.
        if all_inside(np.array(list(product(*self.bounds.T)))):
            return "цилиндр охватывает всё тело" if self._valid_body() else None

        # An entire plane section inside the tool cannot connect surviving
        # material on its opposite sides. Exact B-rep vertices witness that
        # both sides survive; the enclosing box itself is not a witness.
        middle = origin + axis * (h / 2)
        for i, at in enumerate(middle):
            if not low[i] + margin < at < high[i] - margin:
                continue
            axes = list(self.bounds.T)
            axes[i] = (at,)
            if not all_inside(np.array(list(product(*axes)))):
                continue
            if not self._valid_body():
                return None
            if self.vertices is None:
                from OCP.BRep import BRep_Tool

                vertices = body.Vertices()
                self.vertices = (
                    np.array([v.toTuple() for v in vertices], dtype=float).reshape(-1, 3),
                    np.array([BRep_Tool.Tolerance_s(v.wrapped) for v in vertices], dtype=float),
                )
            xyz, vertex_tol = self.vertices
            clearance = margin + vertex_tol
            axial, radial = distances(xyz)
            survives = (axial < -clearance) | (axial > h + clearance) | (radial > r + clearance)
            side = xyz[:, i] - at
            if np.any(survives & (side < -clearance)) and np.any(survives & (side > clearance)):
                return "цилиндр перекрывает сечение, материал остаётся с обеих сторон"
        return None
