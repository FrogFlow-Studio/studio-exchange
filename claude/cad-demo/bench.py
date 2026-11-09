"""Проверка на случайных деталях: python bench.py [кол-во] [seed]

Генерирует случайные детали четырёх семейств (плита с отверстиями и бобышкой,
фланец, втулка с буртом и пазом, ступенчатый вал-ось), случайно откалывает кусок
края, добавляет шум сканера и случайно поворачивает. Затем строит модель
алгоритмом без ИИ и сравнивает с целой деталью до поломки. Так алгоритм нельзя
подогнать под три известные детали.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

import compare as cmpmod
import demo
import geometric
import scan_analysis as sa


def r_(rng, a, b, step=0.5):
    return round(float(rng.uniform(a, b)) / step) * step


def plate(rng):
    L, W, T = r_(rng, 50, 120), r_(rng, 30, 70), r_(rng, 5, 12)
    hd = r_(rng, 4, 9)
    m = r_(rng, hd + 3, min(L, W) / 4)
    body = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    if rng.random() < 0.7:
        body = body.edges("|Z").fillet(r_(rng, 2, min(6, m - hd / 2 - 0.5), 0.5))
    body = body.faces(">Z").workplane().rect(L - 2 * m, W - 2 * m, forConstruction=True).vertices().hole(hd)
    if rng.random() < 0.6:
        bd = r_(rng, 14, min(W - 2 * m - hd, 40)) if W - 2 * m - hd > 15 else 14
        bh = r_(rng, 6, 25)
        body = body.union(cq.Workplane("XY").workplane(offset=T).circle(bd / 2).extrude(bh))
        body = body.faces(">Z").workplane().hole(r_(rng, 5, bd - 6))
    return body, f"плита {L}×{W}×{T}, 4×Ø{hd}"


def flange(rng):
    D, T = r_(rng, 60, 140), r_(rng, 6, 14)
    hub_d = r_(rng, D * 0.3, D * 0.5)
    bore = r_(rng, hub_d * 0.35, hub_d * 0.7)
    n = int(rng.choice([3, 4, 5, 6, 8]))
    hd = r_(rng, 5, 11)
    pcd = r_(rng, (hub_d + D) / 2 - 2, (hub_d + D) / 2 + 2)
    body = cq.Workplane("XY").circle(D / 2).extrude(T)
    body = body.union(cq.Workplane("XY").workplane(offset=T).circle(hub_d / 2).extrude(r_(rng, 6, 25)))
    body = body.faces(">Z").workplane().hole(bore)
    a0 = float(rng.uniform(0, 360 / n))
    pts = [(pcd / 2 * math.cos(math.radians(a0 + 360 * k / n)), pcd / 2 * math.sin(math.radians(a0 + 360 * k / n))) for k in range(n)]
    body = body.cut(cq.Workplane("XY").pushPoints(pts).circle(hd / 2).extrude(T + 1).translate((0, 0, -0.5)))
    return body, f"фланец Ø{D}, {n}×Ø{hd} на Ø{pcd}"


def bushing(rng):
    Dc, Tc = r_(rng, 35, 70), r_(rng, 4, 10)
    Ds = r_(rng, Dc * 0.45, Dc * 0.75)
    H = r_(rng, 15, 45)
    bore = r_(rng, Ds * 0.35, Ds * 0.6)
    body = cq.Workplane("XY").circle(Dc / 2).extrude(Tc).union(
        cq.Workplane("XY").workplane(offset=Tc).circle(Ds / 2).extrude(H))
    body = body.cut(cq.Workplane("XY").circle(bore / 2).extrude(Tc + H))
    if rng.random() < 0.6:
        kw = r_(rng, 3, max(3.5, bore * 0.35))
        kd = r_(rng, 1.5, min(4, (Ds - bore) / 4))
        body = body.cut(cq.Workplane("XY").center(0, bore / 2 + kd / 2 - 0.3).rect(kw, kd + 0.6).extrude(Tc + H))
    return body, f"втулка Ø{Dc}/Ø{Ds}×{Tc + H}"


def shaft(rng):
    """Ступенчатая ось: лежит на торце, 2–4 ступени."""
    n = int(rng.integers(2, 5))
    d = sorted([r_(rng, 12, 50) for _ in range(n)], reverse=True)
    z, body = 0.0, None
    for di in d:
        h = r_(rng, 6, 25)
        s = cq.Workplane("XY").workplane(offset=z).circle(di / 2).extrude(h)
        body = s if body is None else body.union(s)
        z += h
    if rng.random() < 0.5:
        body = body.faces(">Z").workplane().hole(r_(rng, 4, d[-1] * 0.5))
    return body, f"ось {n} ступ., Ø{d[0]}…Ø{d[-1]}"


FAMILIES = [plate, flange, bushing, shaft]


def break_part(part, rng):
    """Откол: клин с краю детали в нижней части (там, где деталь держится)."""
    bb = part.val().BoundingBox()
    cx, cy = (bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2
    R = max(bb.xlen, bb.ylen) / 2
    ang = float(rng.uniform(0, 2 * math.pi))
    depth = float(rng.uniform(0.12, 0.25)) * R
    w = float(rng.uniform(0.35, 0.7)) * R
    ux, uy = math.cos(ang), math.sin(ang)
    vx, vy = -uy, ux
    p0 = (cx + ux * (R - depth), cy + uy * (R - depth))
    poly = [(p0[0] + vx * w * s + ux * t, p0[1] + vy * w * s + uy * t)
            for s, t in ((-1, R), (-0.4, depth * 0.2), (0.3, -depth * 0.1), (1, R))]
    poly.append((p0[0] + ux * 3 * R, p0[1] + uy * 3 * R))
    hcut = float(rng.uniform(0.5, 1.0)) * bb.zlen
    br = cq.Workplane("XY").polyline(poly).close().extrude(hcut + 1).translate((0, 0, bb.zmin - 0.5))
    return part.cut(br)


def make(seed: int, d: Path):
    rng = np.random.default_rng(seed)
    fam = FAMILIES[seed % len(FAMILIES)]
    part, desc = fam(rng)
    broken = break_part(part, rng)
    cq.exporters.export(part, str(d / "truth.stl"), tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(broken, str(d / "broken.stl"), tolerance=0.02, angularTolerance=0.1)
    m = trimesh.load(d / "broken.stl").subdivide_to_size(float(rng.uniform(1.5, 3.0)))
    m.vertices += rng.normal(0, float(rng.uniform(0.01, 0.06)), m.vertices.shape)
    T = trimesh.transformations.random_rotation_matrix(rng.random(3))
    T[:3, 3] = rng.uniform(-200, 200, 3)
    m.apply_transform(T)
    m.export(d / "scan.stl")
    truth = trimesh.load(d / "truth.stl")
    truth.apply_transform(T)  # эталон в тех же координатах, что и «скан»
    return desc, truth


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    tmp = Path(tempfile.mkdtemp())
    rows = []
    for k in range(n):
        seed = seed0 + k
        d = tmp / f"p{seed}"
        d.mkdir()
        t0 = time.time()
        try:
            desc, truth = make(seed, d)
        except Exception as e:  # noqa: BLE001 — неудачная генерация не считается
            print(f"{seed}: генерация не удалась ({type(e).__name__}), пропуск")
            continue
        row = {"seed": seed, "part": desc}
        try:
            sheet, scan = sa.measurement_sheet(str(d / "scan.stl"))
            code, meta = geometric.build(scan, sheet)
            run = demo.run_code(code, d)
            row["valid"] = bool(run.get("ok") and run.get("valid"))
            if run.get("ok"):
                model = trimesh.load(d / "model.stl", force="mesh")
                truth.apply_transform(np.array(sheet["transform_from_scan_4x4"]))
                tol = max(0.3, 3 * sheet["diagnostics"].get("noise_mm", 0))
                c_scan = cmpmod.compare(scan, model, tol=tol)
                c_truth = cmpmod.compare(truth, model, tol=tol)
                row.update(scan_fit=c_scan["within_tol_share"],
                           # симметрично: и эталон близко к модели, и модель не выходит за эталон
                           truth_fit=min(c_truth["within_tol_share"], 1 - c_truth["added_surface_share"]),
                           truth_p95=c_truth["p95_abs_dev_mm"], restored=len([r for r in meta["restored"] if "ничего" not in r]))
            else:
                row["error"] = run.get("error", "")[:120]
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:160]
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    ok = [r for r in rows if r.get("truth_fit") is not None]
    print("\nИТОГ:", f"деталей {len(rows)}, построено {len(ok)}")
    if ok:
        tf = np.array([r["truth_fit"] for r in ok])
        print(f"  совпадение с целой деталью: медиана {np.median(tf)*100:.1f}%, худшая {tf.min()*100:.1f}%, "
              f"≥95%: {(tf >= 0.95).sum()} из {len(ok)}")
    (Path(__file__).parent / "out").mkdir(exist_ok=True)
    (Path(__file__).parent / "out" / f"bench-{seed0}-{n}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
