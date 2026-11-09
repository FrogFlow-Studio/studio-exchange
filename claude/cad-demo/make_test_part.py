"""Тестовая «сломанная деталь» для демо (НЕ деталь студии).

Строит кронштейн в CadQuery (это эталон, ИИ его не видит), превращает в сетку,
откалывает угол и добавляет шум, как у сканера. Результат: samples/bracket_scan.stl.
Эталон сохраняется отдельно (samples/bracket_truth.step) только для честной
проверки в конце: насколько восстановленная модель совпала с исходной.
"""
import sys
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

OUT = Path(__file__).parent / "samples"


def truth() -> cq.Workplane:
    base = (
        cq.Workplane("XY")
        .box(80, 50, 8, centered=(True, True, False))
        .edges("|Z").fillet(5)
        .faces(">Z").workplane()
        .rect(64, 34, forConstruction=True).vertices().hole(6.5)
    )
    boss = cq.Workplane("XY").workplane(offset=8).circle(13).extrude(18)
    rib = (
        cq.Workplane("YZ").workplane(offset=-3)
        .polyline([(-25, 8), (25, 8), (12, 20), (-12, 20)]).close().extrude(6)
    )
    body = base.union(boss).union(rib)
    body = body.faces(">Z").workplane().hole(12)
    return body


def main(seed: int = 7) -> None:
    OUT.mkdir(exist_ok=True)
    part = truth()
    cq.exporters.export(part, str(OUT / "bracket_truth.step"))
    cq.exporters.export(part, str(OUT / "_truth_tmp.stl"), tolerance=0.02, angularTolerance=0.1)
    # «Скол»: отрезаем угол плиты неровной плоскостью (часть детали отсутствует).
    breaker = (
        cq.Workplane("XY").polyline([(22, 30), (45, 30), (45, -2), (40, 6), (31, 14)]).close()
        .extrude(30).translate((0, 0, -5))
    )
    broken = part.cut(breaker)
    cq.exporters.export(broken, str(OUT / "_broken_tmp.stl"), tolerance=0.02, angularTolerance=0.1)
    mesh = trimesh.load(OUT / "_broken_tmp.stl")
    mesh = mesh.subdivide_to_size(3.0)
    rng = np.random.default_rng(seed)
    mesh.vertices += rng.normal(0, 0.04, mesh.vertices.shape)  # шум сканера ~0.04 мм
    mesh.export(OUT / "bracket_scan.stl")
    for f in ("_truth_tmp.stl", "_broken_tmp.stl"):
        (OUT / f).unlink()
    print(f"scan: {len(mesh.faces)} граней -> {OUT/'bracket_scan.stl'}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
