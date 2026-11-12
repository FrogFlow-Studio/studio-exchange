"""Синтетика для достройки без симметрии: несимметричные кривые детали, случайный скол (шероховатый излом)."""
import sys, numpy as np, trimesh, cadquery as cq, manifold3d as m3
out = sys.argv[1]
def man(t): return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
def tri(m):
    o = m.to_mesh(); return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
def parts():
    # 1) корпус-лофт: эллипсы со сдвигом и поворотом (симметрии нет)
    w = (cq.Workplane("XY").ellipse(32, 20).workplane(offset=15).center(7, 3).transformed(rotate=(0, 0, 25))
         .ellipse(22, 13).workplane(offset=14).center(4, -2).ellipse(10, 7).loft())
    yield "loft", w
    # 2) кронштейн с наклонной кривой полкой: коробка + повёрнутый цилиндр + скругления
    b = (cq.Workplane("XY").box(70, 40, 12).edges("|Z").fillet(6)
         .union(cq.Workplane("XZ").transformed(rotate=(0, 18, 0)).center(18, 14).circle(11).extrude(15, both=True))
         .union(cq.Workplane("XY").center(-22, 8).circle(7).extrude(26)))
    yield "bracket", b
    # 3) ручка: сплайновый профиль, вытянутый с вращением
    pts = [(0, 0), (30, 6), (55, 2), (75, 12), (75, 22), (50, 16), (28, 20), (0, 14)]
    h = cq.Workplane("XY").spline(pts, includeCurrent=False).close().twistExtrude(18, 12)
    yield "handle", h
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
for name, wp in parts():
    sh = wp.val(); cq.exporters.export(sh, f"{out}/{name}_truth.step")
    v, f = sh.tessellate(0.002, 0.1); T = trimesh.Trimesh([(q.x, q.y, q.z) for q in v], f)
    print(name, "эталон замкнут", T.is_watertight, "объём %.0f" % T.volume)
    for k in range(2):
        p = trimesh.sample.sample_surface(T, 1, seed=int(rng.integers(1e6)))[0][0]
        r = rng.uniform(5, 9)
        cutter = trimesh.creation.icosphere(3, r); cutter.vertices += p + rng.normal(0, r * 0.3, 3)
        cutter.vertices += rng.normal(0, 0.25, cutter.vertices.shape)
        S_ = tri(man(T) - man(cutter))
        S_ = S_.subdivide_to_size(1.0); S_.merge_vertices()
        S_.vertices += rng.normal(0, 0.04, S_.vertices.shape)
        S_.export(f"{out}/{name}_{k}.stl")
        print(name, k, len(S_.faces), "замкнут", S_.is_watertight, "убрано объёма %.0f" % (T.volume - S_.volume))
