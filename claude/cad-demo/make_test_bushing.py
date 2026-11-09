"""Третья тестовая деталь (НЕ деталь студии): втулка с буртом и шпоночным пазом,
у бурта отломан кусок."""
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

OUT = Path(__file__).parent / "samples"


def truth() -> cq.Workplane:
    flange = cq.Workplane("XY").circle(25).extrude(6)
    sleeve = cq.Workplane("XY").workplane(offset=6).circle(15).extrude(34)
    body = flange.union(sleeve)
    body = body.cut(cq.Workplane("XY").circle(8).extrude(40))
    key = cq.Workplane("XY").center(0, 8 + 1.15).rect(5, 2.3 + 0.6).extrude(40)  # паз 5 × 2.3 от края отверстия
    return body.cut(key.translate((0, 0.3, 0)))


def main() -> None:
    part = truth()
    cq.exporters.export(part, str(OUT / "bushing_truth.step"))
    breaker = cq.Workplane("XY").polyline([(-30, -2), (-17, -9), (-9, -30), (-30, -30)]).close().extrude(10).translate((0, 0, -2))
    broken = part.cut(breaker)
    tmp = OUT / "_tmp.stl"
    cq.exporters.export(broken, str(tmp), tolerance=0.02, angularTolerance=0.1)
    mesh = trimesh.load(tmp).subdivide_to_size(3.0)
    tmp.unlink()
    mesh.vertices += np.random.default_rng(5).normal(0, 0.04, mesh.vertices.shape)
    mesh.export(OUT / "bushing_scan.stl")
    print("ok", len(mesh.faces))


if __name__ == "__main__":
    main()
