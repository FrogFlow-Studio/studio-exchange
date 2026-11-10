"""Объединение: размеры, уточнённые Point2STEP, записываются обратно в нашу программу построения.

Наша программа (CadQuery) редактируемая и с достроенным сколом, Point2STEP точнее подгоняет видимые
грани. Здесь каждое отверстие программы (`.center(x, y).circle(r)`) и каждый размер профиля
вращения (радиусы и высоты в `polyline` перед `revolve`) сверяется с гранями Point2STEP и, если
найдена та же грань, получает её размер. На выходе — одна программа с точными размерами.

python merge.py model.py p2s.step скан.stl новая_model.py отчёт.json
python merge.py model.py model.step скан.stl новая_model.py отчёт.json --fit   (без Point2STEP: fitfaces.py)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUM = r"-?\d+(?:\.\d+)?"
HOLE = re.compile(rf"\.center\(({NUM}), ({NUM})\)\.circle\(({NUM})\)")


def faces(step: Path):
    """Цилиндры с осью вдоль Z (ось x, y, радиус) и плоскости, перпендикулярные Z (высота)."""
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    s = cq.importers.importStep(str(step)).val()
    cyl, pl = [], []
    for f in s.Faces():
        g = BRepAdaptor_Surface(f.wrapped)
        if f.geomType() == "CYLINDER":
            c = g.Cylinder()
            ax = c.Axis()
            d, o = ax.Direction(), ax.Location()
            if abs(abs(d.Z()) - 1) < 1e-3:
                cyl.append((o.X(), o.Y(), c.Radius(), f.Area()))
        elif f.geomType() == "PLANE":
            p = g.Plane()
            n = p.Axis().Direction()
            if abs(abs(n.Z()) - 1) < 1e-3:
                pl.append((p.Location().Z(), f.Area()))
    return cyl, pl


def refine(code: str, cyl: list, pl: list, pts=None, pos_tol: float = 0.3, r_tol: float = 0.15):
    """pts — точки скана: правка принимается, только если новая поверхность ближе к ним, чем старая."""
    import numpy as np

    changes = []
    refined: list[tuple[float, float]] = []  # (радиус в программе, уточнённый) у отверстий, найденных в скане

    def cyl_err(x, y, r, band=0.5):
        if pts is None:
            return 0.0
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        sel = np.abs(d - r) < band
        return float(np.median(np.abs(d[sel] - r))) if sel.sum() > 20 else None

    def cyl_better(old, new):
        if pts is None:
            return True
        a = cyl_err(*old, band=0.5)
        b = cyl_err(*new, band=0.5)
        return a is not None and b is not None and b < a

    def plane_better(z0, z1):
        if pts is None:
            return True
        sel = np.abs(pts[:, 2] - z0) < 0.3
        if sel.sum() < 20:
            return False
        return float(np.median(np.abs(pts[sel, 2] - z1))) < float(np.median(np.abs(pts[sel, 2] - z0)))

    def hole(m: re.Match) -> str:
        x, y, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        near = [c for c in cyl if ((c[0] - x) ** 2 + (c[1] - y) ** 2) ** 0.5 < pos_tol and abs(c[2] - r) < r_tol]
        if not near:
            return m.group(0)
        cx, cy, cr, _ = max(near, key=lambda c: c[3])
        if (abs(cr - r) < 0.001 and abs(cx - x) < 0.001 and abs(cy - y) < 0.001) or not cyl_better((x, y, r), (cx, cy, cr)):
            refined.append((r, r))
            return m.group(0)
        refined.append((r, cr))
        changes.append(f"отверстие в [{x:.2f}, {y:.2f}]: Ø{2 * r:.3f} → Ø{2 * cr:.3f} мм, центр сдвинут на "
                       f"{((cx - x) ** 2 + (cy - y) ** 2) ** 0.5:.3f} мм")
        return f".center({cx:.3f}, {cy:.3f}).circle({cr:.3f})"

    code = HOLE.sub(hole, code)

    # достроенное отверстие (в скане его нет, Point2STEP его не видел) — того же размера, что его
    # соседи из того же массива: берём медиану их уточнённых радиусов
    def restored(line: str) -> str:
        m = HOLE.search(line)
        if not m or "# восстановлено" not in line:
            return line
        r = float(m.group(3))
        sib = sorted(n for o, n in refined if abs(o - r) < 0.05)
        if len(sib) < 2:
            return line
        nr = sib[len(sib) // 2] if len(sib) % 2 else (sib[len(sib) // 2 - 1] + sib[len(sib) // 2]) / 2
        if abs(nr - r) < 0.001:
            return line
        changes.append(f"достроенное отверстие в [{float(m.group(1)):.2f}, {float(m.group(2)):.2f}]: Ø{2 * r:.3f} → "
                       f"Ø{2 * nr:.3f} мм, как у уточнённых соседей ({len(sib)} шт.)")
        return line[:m.start(3)] + f"{nr:.3f}" + line[m.end(3):]

    code = "\n".join(restored(x) for x in code.split("\n"))

    # профиль вращения: точки [радиус, высота]; ось вращения — по translate после revolve
    def profile(m: re.Match) -> str:
        pts = json.loads(m.group(1))
        tx, ty = float(m.group(3)), float(m.group(4))
        out = []
        for r, z in pts:
            nr, nz = r, z
            if r > 0.5:
                near = [c for c in cyl if ((c[0] - tx) ** 2 + (c[1] - ty) ** 2) ** 0.5 < pos_tol and abs(c[2] - r) < r_tol]
                if near:
                    cand = max(near, key=lambda c: c[3])[2]
                    if cyl_better((tx, ty, r), (tx, ty, cand)):
                        nr = cand
            if z > 0.05:
                nearz = [p for p in pl if abs(p[0] - z) < r_tol]
                if nearz:
                    cand = max(nearz, key=lambda p: p[1])[0]
                    if plane_better(z, cand):
                        nz = cand
            if abs(nr - r) >= 0.001:
                changes.append(f"радиус профиля {r:.3f} → {nr:.3f} мм (Ø{2 * nr:.3f})")
            if abs(nz - z) >= 0.001:
                changes.append(f"высота {z:.3f} → {nz:.3f} мм")
            out.append([round(nr, 3), round(nz, 3)])
        return (f"polyline({json.dumps(out)}){m.group(2)}.translate(({tx}, {ty}, 0))")

    code = re.sub(rf"polyline\((\[\[.*?\]\])\)(\.close\(\)\.revolve\(360, \(0, 0, 0\), \(0, 1, 0\)\))"
                  rf"\.translate\(\(({NUM}), ({NUM}), 0\)\)", profile, code)
    # одна и та же правка профиля встречается дважды (две точки на одном радиусе) — без повторов
    return code, list(dict.fromkeys(changes))


def main() -> None:
    src, step, scan, dst, rep = (Path(x) for x in sys.argv[1:6])
    own = "--fit" in sys.argv[6:]  # step — наша модель, поверхности подгоняются по скану здесь же
    import trimesh

    m = trimesh.load(scan, force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, 200000, seed=2)
    if own:
        from fitfaces import fit

        cyl, pl = fit(step, pts)
    else:
        cyl, pl = faces(step)
    code, changes = refine(src.read_text(encoding="utf-8"), cyl, pl, pts)
    if changes:
        code = code.replace("import cadquery as cq\n",
                            "import cadquery as cq\n# Размеры видимых граней уточнены по скану"
                            + ("" if own else " (Point2STEP)") + ".\n", 1)
    dst.write_text(code, encoding="utf-8")
    rep.write_text(json.dumps({"changes": changes}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
