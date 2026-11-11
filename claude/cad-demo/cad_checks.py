"""Shared checks and measurements for local Boolean operations."""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measured(timings, key):
    if timings is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + perf_counter() - start


class VolumeMemo:
    """Remember the last accepted shape's volume using the original integrator."""

    def __init__(self):
        self.shape = None
        self.volume = None

    def get(self, shape):
        if shape is not self.shape:
            self.remember(shape, shape.Volume())
        return self.volume

    def remember(self, shape, volume):
        self.shape, self.volume = shape, volume

    def remember_cut(self, result, solid, volume):
        if result.wrapped.IsEqual(solid.wrapped):
            self.remember(solid, volume)
        elif result.ShapeType() == "Compound":
            # A one-child compound integrates the same oriented faces as its child.
            children = list(result)
            if len(children) == 1 and children[0].wrapped.IsEqual(solid.wrapped):
                self.remember(solid, volume)


class InsideMemo:
    """Reuse OCC's classifier while probing an unchanged accepted body."""

    def __init__(self):
        self.shape = None
        self.classifier = None

    def contains(self, shape, point, tol=1e-6):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN

        if shape is not self.shape:
            self.classifier = BRepClass3d_SolidClassifier(shape.wrapped)
            self.shape = shape
        self.classifier.Perform(gp_Pnt(*map(float, point)), tol)
        return self.classifier.State() == TopAbs_IN or self.classifier.IsOnAFace()
