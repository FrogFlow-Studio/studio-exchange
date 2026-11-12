import sys, time, glob, numpy as np, trimesh, cadquery as cq
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import symmetry, nurbs
d = sys.argv[1]
for f in sorted(glob.glob(d + "/*_[0-9].stl")):
    name = f.split("/")[-1][:-6]
    scan = trimesh.load(f, force="mesh")
    truth = nurbs.body_mesh(cq.importers.importStep(f"{d}/{name}_truth.step").val(), 0.03)
    t = time.time(); r, log, pl = symmetry.restore_all(scan); t = time.time() - t
    P = trimesh.sample.sample_surface(truth, 15000, seed=2)[0]
    def cov(m):
        a = trimesh.proximity.closest_point(m, P)[1]
        b = trimesh.proximity.closest_point(truth, trimesh.sample.sample_surface(m, 15000, seed=3)[0])[1]
        return f"{np.mean(a<0.3)*100:.1f}/{np.mean(b<0.3)*100:.1f}%"
    print(f"{f.split('/')[-1]}: плоскостей {len(pl)} (лучшая {pl[0]['share']*100:.0f}%)" if pl else f"{f.split('/')[-1]}: плоскостей 0", f"достроек {len(log)}, скан {cov(scan)} → {cov(r)}, {t:.0f} с", flush=True)
    for e in log: print("    плоскость", np.round(e["normal"], 2), "площадь %.0f, глубина %.1f, объём +%.0f" % (e["area"], e["depth"], e["volume"]))
