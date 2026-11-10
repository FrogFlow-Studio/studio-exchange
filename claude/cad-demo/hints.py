"""Подсказки CADENA проверяет наш алгоритм: каждое отверстие из модели CADENA ищется в нашей
модели и в скане. Есть в скане, а у нас нет — достраиваем сами (по точкам скана, не по CADENA);
в скане не подтверждено — только строка «проверить» в отчёте.

python hints.py наша_model.py наша_model.step cadena.step скан.stl новая_model.py отчёт.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def cylinders(step: Path, holes_only: bool = True) -> list[dict]:
    """Цилиндрические грани: ось (точка, направление), радиус; holes_only — только отверстия."""
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    s = cq.importers.importStep(str(step)).val()
    out = []
    for f in s.Faces():
        if f.geomType() != "CYLINDER":
            continue
        c = BRepAdaptor_Surface(f.wrapped).Cylinder()
        ax = c.Axis()
        o = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        p = f.positionAt(0.5, 0.5)
        p = np.array([p.x, p.y, p.z])
        n = f.normalAt(f.positionAt(0.5, 0.5))
        foot = o + d * np.dot(p - o, d)
        hole = float(np.dot([n.x, n.y, n.z], foot - p)) > 0
        if hole or not holes_only:
            out.append({"o": o, "d": d / np.linalg.norm(d), "r": float(c.Radius()), "hole": hole})
    return out


def same(a: dict, b: dict, tol: float = 1.0) -> bool:
    if abs(abs(np.dot(a["d"], b["d"])) - 1) > 0.01:
        return False
    off = b["o"] - a["o"]
    dist = np.linalg.norm(off - a["d"] * np.dot(off, a["d"]))
    return dist < tol and abs(a["r"] - b["r"]) < 0.5


def confirm(h: dict, pts: np.ndarray, nrm: np.ndarray, band: float = 0.3):
    """Есть ли в скане стенка отверстия: точки на радиусе, по окружности не меньше чем на 200°.
    Возвращает уточнённые по скану центр, радиус и пределы по оси или None."""
    d, o = h["d"], h["o"]
    rel = pts - o
    t = rel @ d
    radial = rel - np.outer(t, d)
    dist = np.linalg.norm(radial, axis=1)
    # стенка отверстия: точка на радиусе и нормаль скана смотрит к оси (иначе это торец или плоскость,
    # пересекающая ту же окружность)
    to_axis = -radial / np.maximum(dist, 1e-9)[:, None]
    sel = (np.abs(dist - h["r"]) < band) & (np.abs(nrm @ d) < 0.3) & (np.einsum("ij,ij->i", nrm, to_axis) > 0.7)
    if sel.sum() < 40:
        return None
    u = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    q = np.c_[radial[sel] @ u, radial[sel] @ v]
    ang = np.sort(np.degrees(np.arctan2(q[:, 1], q[:, 0])))
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    if 360 - gaps.max() < 200:
        return None
    # окружность по точкам скана (наименьшие квадраты)
    A = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    if abs(r - h["r"]) > 0.5:
        return None
    centre = o + cx * u + cy * v
    return {"o": centre, "d": d, "r": r, "t0": float(t[sel].min()), "t1": float(t[sel].max())}


def main() -> None:
    code_p, ours_step, cad_step, scan_p, dst, rep = (Path(x) for x in sys.argv[1:7])
    import trimesh

    ours = cylinders(ours_step)
    hints = cylinders(cad_step)
    m = trimesh.load(scan_p, force="mesh")
    pts, fi = trimesh.sample.sample_surface(m, 300000, seed=3)
    nrm = m.face_normals[fi]
    code = code_p.read_text(encoding="utf-8")
    added, checked, unconfirmed = [], 0, []
    seen: list[dict] = []
    for h in hints:
        if any(same(h, s) for s in seen):
            continue  # у CADENA одно отверстие бывает несколькими гранями
        seen.append(h)
        if any(same(h, o) for o in ours):
            checked += 1
            continue
        c = confirm(h, pts, nrm)
        where = ", ".join(f"{x:.1f}" for x in h["o"])
        if c is None:
            unconfirmed.append(f"CADENA предлагает отверстие Ø{2 * h['r']:.2f} (ось через [{where}]): в скане не подтверждено")
            continue
        # глубина — там, где в скане есть стенка отверстия, с запасом 0,5 мм на выход
        span = c["t1"] - c["t0"] + 1.0
        start = c["o"] + c["d"] * (c["t0"] - 0.5)
        code += ("\n# Отверстие подсказала CADENA, проверено и построено по точкам скана\n"
                 f"result = result.cut(cq.Workplane().add(cq.Solid.makeCylinder({c['r']:.3f}, {span:.3f}, "
                 f"cq.Vector({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}), "
                 f"cq.Vector({c['d'][0]:.4f}, {c['d'][1]:.4f}, {c['d'][2]:.4f}))))\n")
        added.append(f"отверстие Ø{2 * c['r']:.2f} мм (подсказка CADENA, подтверждено сканом)")
    dst.write_text(code, encoding="utf-8")
    rep.write_text(json.dumps({"added": added, "confirmed": checked, "unconfirmed": unconfirmed,
                               "hints": len(seen)}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
