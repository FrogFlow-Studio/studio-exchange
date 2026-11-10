"""Проверка черновика из стенок на случайных тонкостенных деталях (не только на кронштейне).

python thinbench.py [число_деталей] [первый_seed]

Деталь: основание, 2–4 боковые стенки (часть наклонные), рёбра, толстый прилив, отверстия,
иногда скругления. Скан = поверхность, измельчённая и зашумлённая (0,15 мм). Черновик строится
по зашумлённому скану, сравнивается с целой деталью (эталоном) в допуске 0,5 мм.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import compare  # noqa: E402
import demo  # noqa: E402
import walls  # noqa: E402


def make_part(seed: int):
    r = np.random.default_rng(seed)
    L, W, t = r.uniform(60, 120), r.uniform(40, 80), r.uniform(1.6, 3.0)
    H = r.uniform(20, 45)
    part = cq.Workplane("XY").box(L, W, t, centered=(True, True, False))
    sides = r.choice(4, size=int(r.integers(2, 5)), replace=False)
    for s in sides:
        tilt = float(r.choice([0, 0, r.uniform(10, 35)]))
        if s in (0, 1):  # вдоль X
            y = (W / 2 - t / 2) * (1 if s == 0 else -1)
            wall = cq.Workplane("XY").box(L, t, H, centered=(True, True, False))
            wall = wall.rotate((0, 0, 0), (1, 0, 0), tilt * (1 if s == 0 else -1)).translate((0, y, 0))
        else:
            x = (L / 2 - t / 2) * (1 if s == 2 else -1)
            wall = cq.Workplane("XY").box(t, W, H, centered=(True, True, False))
            wall = wall.rotate((0, 0, 0), (0, 1, 0), -tilt * (1 if s == 2 else -1)).translate((x, 0, 0))
        part = part.union(wall)
    xs: list[float] = []
    for _ in range(int(r.integers(1, 4))):  # рёбра, не ближе 6 мм друг к другу (иначе щель 0,3 мм)
        x = r.uniform(-L / 3, L / 3)
        if any(abs(x - q) < 6 for q in xs):
            continue
        xs.append(x)
        part = part.union(cq.Workplane("XY").box(t, W * 0.8, r.uniform(8, H * 0.8),
                                                 centered=(True, True, False)).translate((x, 0, 0)))
    if r.random() < 0.7:  # толстый прилив
        bx, by = r.uniform(-L / 4, L / 4), r.uniform(-W / 4, W / 4)
        part = part.union(cq.Workplane("XY").box(r.uniform(10, 20), r.uniform(10, 20), r.uniform(6, 15),
                                                 centered=(True, True, False)).translate((bx, by, 0)))
    try:  # отверстия
        part = part.faces("<Z").workplane().pushPoints(
            [(r.uniform(-L / 3, L / 3), r.uniform(-W / 3, W / 3)) for _ in range(int(r.integers(1, 4)))]
        ).hole(r.uniform(3, 8))
    except Exception:  # noqa: BLE001
        pass
    if r.random() < 0.5:
        try:
            part = part.edges("|Z").fillet(min(1.0, t * 0.4))
        except Exception:  # noqa: BLE001
            pass
    return part


def noisy_scan(part, path: Path, seed: int, noise: float = 0.15) -> trimesh.Trimesh:
    cq.exporters.export(part, str(path), tolerance=0.05, angularTolerance=0.2)
    m = trimesh.load(path, force="mesh")
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=2.0)
    m = trimesh.Trimesh(v, f, process=True)
    m.merge_vertices()
    rng = np.random.default_rng(seed)
    m.vertices = m.vertices + m.vertex_normals * rng.normal(0, noise, (len(m.vertices), 1))
    return m


def run(n: int, first: int) -> list[dict]:
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="thinbench_"))
    for seed in range(first, first + n):
        t0 = time.time()
        part = make_part(seed)
        truth_p = tmp / f"truth{seed}.stl"
        scan = noisy_scan(part, truth_p, seed)
        truth = trimesh.load(truth_p, force="mesh")
        ws = walls.walls(scan)
        it = tmp / f"it{seed}"
        it.mkdir()
        res = demo.run_code(walls.draft_code(ws), it, timeout=420)
        row = {"seed": seed, "walls": len(ws), "ok": bool(res.get("ok")), "solids": res.get("solids"),
               "voids": res.get("voids")}
        if res.get("ok"):
            m = trimesh.load(it / "model.stl", force="mesh")
            row["truth_fit"] = compare.compare(truth, m, tol=0.5)["within_tol_share"]
        row["sec"] = round(time.time() - t0)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    fits = [r.get("truth_fit", 0) for r in rows]
    print(f"ИТОГ: деталей {len(rows)}, одно тело {sum(r.get('solids') == 1 for r in rows)}, "
          f"совпадение с эталоном: среднее {np.mean(fits)*100:.1f}%, худшее {min(fits)*100:.1f}%, "
          f"≥95%: {sum(f >= 0.95 for f in fits)}")
    return rows


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 6, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
