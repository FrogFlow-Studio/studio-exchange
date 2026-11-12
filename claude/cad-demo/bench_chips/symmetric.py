"""Синтетика для этапа Д: симметричные кривые детали, случайный скол, односторонняя бобышка."""
import sys, numpy as np, trimesh, cadquery as cq, manifold3d as m3
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import nurbs
out = sys.argv[1]
def man(t): return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
def tri(m):
    o = m.to_mesh(); return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
def parts():
    # 1) кривая крышка: тело вращения по сплайну с двумя ушами (симметрия XZ и YZ)
    prof = cq.Workplane("XY").sphere(34).translate((0, 0, -12)).intersect(cq.Workplane("XY").box(100, 100, 40).translate((0, 0, 20)))
    ears = cq.Workplane("XY").rect(84, 14).extrude(4).faces(">Z").workplane().pushPoints([(-37, 0), (37, 0)]).hole(5)
    yield "cover", prof.union(ears), None
    # 2) обтекаемый корпус: лофт эллипсов (симметрия только XZ) с бобышкой на одной стороне от плоскости
    body = cq.Workplane("XY").ellipse(30, 18).workplane(offset=14).center(6, 0).ellipse(20, 12).workplane(offset=12).ellipse(8, 6).loft()
    yield "hull", body, None
    boss = cq.Workplane("XY").workplane(offset=0).center(-8, 11).circle(4).extrude(22)
    yield "hull_boss", body.union(boss), "one-sided"
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
for name, wp, note in parts():
    sh = wp.val(); cq.exporters.export(sh, f"{out}/{name}_truth.step")
    v, f = sh.tessellate(0.002, 0.1); T = trimesh.Trimesh([(q.x, q.y, q.z) for q in v], f); print(name, 'эталон замкнут', T.is_watertight)
    for k in range(3 if note is None else 1):
        if note is None:
            p = trimesh.sample.sample_surface(T, 1, seed=int(rng.integers(1e6)))[0][0]
            r = rng.uniform(5, 11)
            cutter = trimesh.creation.icosphere(3, r); cutter.vertices += p + rng.normal(0, r * 0.3, 3)
            # излом шероховатый: шум по вершинам резака
            cutter.vertices += rng.normal(0, 0.25, cutter.vertices.shape)
            S_ = tri(man(T) - man(cutter))
        else:
            S_ = T.copy()
        S_ = S_.subdivide_to_size(1.0); S_.merge_vertices()
        S_.vertices += rng.normal(0, 0.04, S_.vertices.shape)
        S_.export(f"{out}/{name}_{k}.stl"); print(name, k, len(S_.faces), "watertight", S_.is_watertight, "объём скана %.0f, эталона %.0f" % (S_.volume, T.volume))
