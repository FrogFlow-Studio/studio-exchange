"""Запись STL из тела CAD с проверкой объёма.

Сетка OCC иногда выходит битой: часть граней вывернута, и объём сетки становится
отрицательным или сильно меньше объёма тела (кнопка 13.11: тело 8093 мм³, STL −1794).
Сравнение такой сетки со сканом даёт ложные 66 %. Поэтому после записи объём сетки
сверяется с объёмом тела; при расхождении грани чистятся и сетка строится заново
с другим допуском.
"""
from pathlib import Path

import numpy as np


def stl_volume(path) -> float:
    """Объём бинарного STL без загрузки в trimesh (быстро и без лишней памяти)."""
    data = Path(path).read_bytes()
    n = int.from_bytes(data[80:84], "little")
    tri = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)["v"].astype(np.float64)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def write_stl_checked(shape, path, lin=0.02, rel=True, ang=0.1, tol=0.05):
    """Пишет STL; возвращает (объём сетки / объём тела, попытка). shape — TopoDS_Shape."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.StlAPI import StlAPI_Writer

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    vol = abs(props.Mass()) or 1.0
    tries = [(lin, rel, ang), (0.02, False, 0.3), (0.05, False, 0.5), (0.01, False, 0.2)]
    best = None
    for k, (l, r, a) in enumerate(tries):
        if k:
            BRepTools.Clean_s(shape)
        BRepMesh_IncrementalMesh(shape, l, r, a, True)
        w = StlAPI_Writer()
        w.ASCIIMode = False
        if not w.Write(shape, str(path)):
            raise RuntimeError("не удалось записать STL")
        ratio = stl_volume(path) / vol
        if abs(ratio - 1) <= tol:
            return ratio, k
        if best is None or abs(ratio - 1) < abs(best[0] - 1):
            best = (ratio, k)
    # тело прямо из построения иногда мешится с дырами на всех допусках (кнопка: 0,25),
    # а то же тело, записанное в STEP и прочитанное обратно, — нормально (0,96): запись
    # STEP заново строит кривые на гранях. Последняя попытка — через такой файл.
    import tempfile

    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer

    with tempfile.TemporaryDirectory() as tmp:
        f = str(Path(tmp) / "body.step")
        sw = STEPControl_Writer()
        sw.Transfer(shape, STEPControl_AsIs)
        sw.Write(f)
        rd = STEPControl_Reader()
        if rd.ReadFile(f) != IFSelect_RetDone:
            raise RuntimeError("не удалось перечитать STEP для сетки")
        rd.TransferRoots()
        fresh = rd.OneShape()
    BRepMesh_IncrementalMesh(fresh, lin, rel, ang, True)
    w = StlAPI_Writer()
    w.ASCIIMode = False
    w.Write(fresh, str(path))
    ratio = stl_volume(path) / vol
    if abs(ratio - 1) <= tol or abs(ratio - 1) < abs(best[0] - 1):
        return ratio, len(tries)
    if best[1] != len(tries) - 1:  # вернуть лучшую сетку
        l, r, a = tries[best[1]]
        BRepTools.Clean_s(shape)
        BRepMesh_IncrementalMesh(shape, l, r, a, True)
        w = StlAPI_Writer()
        w.ASCIIMode = False
        w.Write(shape, str(path))
    return best
