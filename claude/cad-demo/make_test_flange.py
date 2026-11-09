"""Вторая тестовая деталь (НЕ деталь студии): фланец с шестью отверстиями по кругу,
ступицей и отломанным куском края вместе с одним отверстием."""
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

OUT = Path(__file__).parent / "samples"


def truth() -> cq.Workplane:
    disk = cq.Workplane("XY").circle(45).extrude(10)
    hub = cq.Workplane("XY").workplane(offset=10).circle(22).extrude(15)
    body = disk.union(hub).faces(">Z").workplane().hole(25)
    body = body.faces("<Z").workplane().polarArray(34, 0, 360, 6).hole(8)
    return body


def main() -> None:
    part = truth()
    cq.exporters.export(part, str(OUT / "flange_truth.step"))
    breaker = cq.Workplane("XY").polyline([(28, -12), (60, -30), (60, 20), (31, 8)]).close().extrude(40).translate((0, 0, -5))
    broken = part.cut(breaker)
    tmp = OUT / "_tmp.stl"
    cq.exporters.export(broken, str(tmp), tolerance=0.02, angularTolerance=0.1)
    mesh = trimesh.load(tmp).subdivide_to_size(3.0)
    tmp.unlink()
    mesh.vertices += np.random.default_rng(11).normal(0, 0.05, mesh.vertices.shape)
    mesh.export(OUT / "flange_scan.stl")
    print("ok", len(mesh.faces))


if __name__ == "__main__":
    main()
