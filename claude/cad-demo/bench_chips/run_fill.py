"""Достройка без симметрии (chipfill) на синтетике со сколами: совпадение с эталоном до и после."""
import glob, sys, time
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chipfill, nurbs, surface
from chipfill import restore_all

for f in sorted(sys.argv[2:] or glob.glob(sys.argv[1] + "/*_[0-9].stl")):
    name = f.split("/")[-1][:-6]
    scan = trimesh.load(f, force="mesh")
    truth = nurbs.body_mesh(cq.importers.importStep(f"{sys.argv[1]}/{name}_truth.step").val(), 0.03)
    clean = surface.surface_mesh(scan)
    t = time.time(); r, log, info = restore_all(scan, clean); t = time.time() - t
    P = trimesh.sample.sample_surface(truth, 15000, seed=2)[0]
    def cov(m):
        a = trimesh.proximity.closest_point(m, P)[1]
        b = trimesh.proximity.closest_point(truth, trimesh.sample.sample_surface(m, 15000, seed=3)[0])[1]
        return f"{np.mean(a<0.3)*100:.1f}/{np.mean(b<0.3)*100:.1f}%"
    print(f.split("/")[-1], f"ядро {info['core']}, достроек {len(log)}, чистая {cov(clean)} → {cov(r)}, {t:.0f} с", flush=True)
    for e in log:
        near = np.linalg.norm(P - e["center"], axis=1) < e["radius"]
        loc = lambda m: np.mean(trimesh.proximity.closest_point(m, P[near])[1] < 0.3) * 100
        print("    в месте скола эталон покрыт: %.1f%% → %.1f%%" % (loc(clean), loc(r)))
        print("    центр", np.round(e["center"], 1), "площадь %.0f глубина %.1f объём +%.0f, %s" % (e["area"], e["depth"], e["volume"], e["how"]), e.get("detail"))
