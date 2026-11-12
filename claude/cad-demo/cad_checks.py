"""Shared checks and measurements for local Boolean operations."""
import os
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


def add_count(timings, key, count=1):
    if timings is not None:
        timings[key] = timings.get(key, 0) + count


def disjoint_tool_boxes(groups):
    """Skip trial batching when different jobs' tools may touch one another."""
    boxes = []
    try:
        for group in groups:
            bounds = [tool.BoundingBox() for tool in group]
            boxes.append((min(b.xmin for b in bounds), min(b.ymin for b in bounds), min(b.zmin for b in bounds),
                          max(b.xmax for b in bounds), max(b.ymax for b in bounds), max(b.zmax for b in bounds)))
    except Exception:
        return False
    for i, a in enumerate(boxes):
        for b in boxes[:i]:
            if all(a[axis + 3] >= b[axis] and b[axis + 3] >= a[axis] for axis in range(3)):
                return False
    return True


def boolean_op(base, tools, *, fuse=False, preserve_inputs=False):
    """Use the original flags, with optional OBB and non-destructive trial mode."""
    legacy = base.fuse if fuse else base.cut
    obb = os.environ.get("CAD_BOOLEAN_OBB", "0") == "1"
    if not obb and not preserve_inputs:
        return legacy(*tools)

    import cadquery as cq
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    try:
        args, operands = TopTools_ListOfShape(), TopTools_ListOfShape()
        args.Append(base.wrapped)
        for tool in tools:
            operands.Append(tool.wrapped)
        op = BRepAlgoAPI_Fuse() if fuse else BRepAlgoAPI_Cut()
        op.SetArguments(args)
        op.SetTools(operands)
        op.SetRunParallel(True)
        if obb:
            op.SetUseOBB(True)
        if preserve_inputs:
            op.SetNonDestructive(True)
        op.Build()
        if op.IsDone():
            result = op.Shape()
            if not result.IsNull():
                return cq.Shape.cast(result)
    except Exception:  # optional OCC path: retry with the original flags
        if preserve_inputs:
            raise
    if preserve_inputs:
        raise RuntimeError("пробная булева операция OCC не завершилась")
    return legacy(*tools)


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




_CHILD = ("import sys, cadquery as cq\n"
          "b = cq.Shape.importBrep(sys.argv[1])\n"
          "if sys.argv[2] != '-': b = b.cut(cq.Shape.importBrep(sys.argv[2]))\n"
          "if sys.argv[3] != '-': b = b.fuse(cq.Shape.importBrep(sys.argv[3]))\n"
          "b.exportBrep(sys.argv[4])\n")


def boolean_limited(base, cut_tools=(), fuse_tools=(), timeout_s: float = 60.0):
    """Вырез, затем вставка, с пределом времени. OCC нельзя прервать изнутри, поэтому операция
    идёт в отдельном процессе Python (так же на Windows), тела передаются файлами BRep.
    Не уложилась — TimeoutError, исходное тело не тронуто."""
    import subprocess
    import sys
    import tempfile

    import cadquery as cq

    def one(tools):
        tools = list(tools)
        return None if not tools else tools[0] if len(tools) == 1 else cq.Compound.makeCompound(tools)

    with tempfile.TemporaryDirectory(prefix="cadbool-") as d:
        paths = [os.path.join(d, n) for n in ("base.brep", "cut.brep", "fuse.brep", "out.brep")]
        base.exportBrep(paths[0])
        for shape, path in ((one(cut_tools), paths[1]), (one(fuse_tools), paths[2])):
            if shape is None:
                paths[paths.index(path)] = "-"
            else:
                shape.exportBrep(path)
        try:
            r = subprocess.run([sys.executable, "-c", _CHILD, *paths], capture_output=True, timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"не уложилась в {timeout_s:.0f} с") from None
        if r.returncode != 0 or not os.path.exists(paths[3]):
            tail = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"дочерний процесс: код {r.returncode} {tail[0][:120]}")
        return cq.Shape.importBrep(paths[3])


def boolean_cut_limited(base, tool, timeout_s: float):
    return boolean_limited(base, (tool,), (), timeout_s)
