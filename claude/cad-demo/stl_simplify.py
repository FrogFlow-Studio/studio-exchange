"""Упрощение огромной сетки модели (model.stl) до 1 млн треугольников.

Запускается отдельным процессом из cad_runner.py: в самом раннере уже занята память ядром
CAD (тесселяция тысяч сплайновых граней), и под его ограничением в 3 ГБ разбор сетки на
4,7 млн треугольников падал с MemoryError — модель 0к терялась целиком (соковыжималка,
крышка студии 12.11). Здесь адресное пространство свежее.

Файл заменяется, только если объём изменился не больше чем на 0,1 % и замкнутость не
потеряна. Печатает число исходных треугольников, если замена сделана.
"""
import sys

import fast_simplification
import numpy as np
import trimesh

TARGET = 1_000_000


def main(path: str) -> None:
    full = trimesh.load(path, force="mesh")
    n = len(full.faces)
    if n <= TARGET:
        return
    v, f = fast_simplification.simplify(full.vertices, full.faces, 1 - TARGET / n)
    small = trimesh.Trimesh(v, f, process=True)
    # замкнутость сетки STL из OCC и так не гарантирована (грани стыкуются с зазором в микроны);
    # проверяется, что упрощение не изменило объём (крышка: 24 193 → 24 185 мм³, p99 0,05 мм)
    vol_full = float(full.volume)
    tight_full = bool(full.is_watertight)
    del full
    if (small.is_watertight or not tight_full) and abs(small.volume - vol_full) <= 1e-3 * abs(vol_full):
        small.export(path)
        print(n)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(sys.argv[1])
