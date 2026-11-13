import os
"""Прототип «как в Компасе»: деталь = стопка операций «по сечениям» вдоль оси z.
Сечения скана каждые dz; слой = промежуток, где набор контуров не меняется; каждый контур слоя —
гладкий сплайн; тело слоя — лофт через несколько сечений; дыры контура вычитаются тем же способом."""
import sys, time, numpy as np, trimesh, shapely
from shapely.geometry import Polygon
from scipy.interpolate import splprep, splev
import cadquery as cq
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.gp import gp_Pnt
from OCP.GeomAbs import GeomAbs_C2

MIN_AREA = 1.0
TOPINFO = {}
STEP_JUMP = float(os.environ.get('STEP_JUMP', '1.0'))

def sections(mesh, dz):
    z0, z1 = mesh.bounds[:, 2]
    out = []
    for z in np.arange(z0 + dz / 2, z1, dz):
        s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        polys = []
        if s is not None:
            P, _ = s.to_2D(to_2D=np.eye(4))
            polys = [p.buffer(0) for p in P.polygons_full if p.area >= MIN_AREA]
            polys = [Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area >= MIN_AREA]) for p in polys if p.geom_type == "Polygon"]
        out.append((z, polys))
    return out

def signature(polys):
    return tuple(sorted(len(p.interiors) for p in polys)), len(polys)

def similar(a, b, tol):
    """соседние сечения одного слоя: тот же набор контуров и нет ступеньки (резкого скачка границы)"""
    if signature(a) != signature(b):
        return False
    A = shapely.union_all(a); B = shapely.union_all(b)
    return A.hausdorff_distance(B) < STEP_JUMP

def layers(secs, tol):
    out, cur = [], [secs[0]]
    for s in secs[1:]:
        if s[1] and cur[-1][1] and similar(cur[-1][1], s[1], tol):
            cur.append(s)
        else:
            out.append(cur); cur = [s]
    out.append(cur)
    return [l for l in out if any(p for _, p in l)]

def ring_pts(coords, n, smooth, ref_angle_pt=None):
    p = np.asarray(coords)[:-1]
    # ориентация против часовой
    if np.sum((p[:, 0] - np.roll(p[:, 0], 1)) * (p[:, 1] + np.roll(p[:, 1], 1))) > 0:
        p = p[::-1]
    c = p.mean(0)
    start = np.argmin(np.linalg.norm(p - (c + [1e3, 0]), axis=1))
    p = np.roll(p, -start, axis=0)
    p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1); keep = np.r_[True, seg > 1e-6]
    p = p[keep]
    tck, _ = splprep(p.T, s=len(p) * smooth ** 2, per=1)
    u = np.linspace(0, 1, n, endpoint=False)
    return np.array(splev(u, tck)).T

def wire3d(pts2, z):
    arr = TColgp_Array1OfPnt(1, len(pts2) + 1)
    for i, (x, y) in enumerate(np.vstack([pts2, pts2[:1]])):
        arr.SetValue(i + 1, gp_Pnt(float(x), float(y), float(z)))
    curve = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-3).Curve()
    return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(curve).Edge()).Wire()

def prism(pts2, za, zb):
    w = cq.Workplane("XY").workplane(offset=float(za)).spline([tuple(map(float, p)) for p in pts2], periodic=True).close()
    return w.extrude(float(zb - za)).val()

def _wire(pts2, z):
    return cq.Wire.assembleEdges([cq.Edge.makeSpline([cq.Vector(float(x), float(y), float(z)) for x, y in pts2], periodic=True)])

def _loft_child(rings_z, ruled, q):
    try:
        if os.environ.get("LOFT_SMOOTH", "0") == "1" and not ruled and len(rings_z) > 2:
            ts = BRepOffsetAPI_ThruSections(True, False, 1e-6)
            ts.SetSmoothing(True)
            ts.SetMaxDegree(5)
            for p, z in rings_z:
                ts.AddWire(_wire(p, z).wrapped)
            ts.CheckCompatibility(False)
            ts.Build()
            s = cq.Solid(ts.Shape())
            if not s.isValid():
                s = cq.Solid.makeLoft([_wire(p, z) for p, z in rings_z], ruled)
        else:
            s = cq.Solid.makeLoft([_wire(p, z) for p, z in rings_z], ruled)
        import io
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape
        fn = os.path.join(__import__("tempfile").gettempdir(), "_kloft%d.brep" % os.getpid())
        BRepTools.Write_s(s.wrapped, fn)
        q.put(fn if s.isValid() else None)
    except Exception as e:
        q.put(None)

def align(rings_z):
    """каждое следующее кольцо поворачиваем по кругу так, чтобы точки шли напротив точек предыдущего (без закрутки)"""
    out = [rings_z[0]]
    for p, z in rings_z[1:]:
        q = out[-1][0]
        cost = [np.sum((np.roll(p, -k, axis=0) - q) ** 2) for k in range(len(p))]
        out.append((np.roll(p, -int(np.argmin(cost)), axis=0), z))
    return out

def loft(rings_z, ruled=False):
    rings_z = align(rings_z)
    import multiprocessing as mp
    ctx = mp.get_context("fork"); q = ctx.Queue()
    p = ctx.Process(target=_loft_child, args=(rings_z, ruled, q)); p.start(); p.join(60)
    if p.is_alive(): p.kill(); raise RuntimeError("loft timeout")
    if p.exitcode != 0: raise RuntimeError("loft crash %s" % p.exitcode)
    fn = q.get(timeout=5)
    if not fn: raise RuntimeError("loft invalid")
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    from OCP.BRep import BRep_Builder
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, fn, BRep_Builder()); os.remove(fn)
    return cq.Solid(sh)

def track(layer):
    """контуры слоя по уровням: список цепочек [(z, polygon), ...] по ближайшему центру"""
    z0, base = layer[0]
    chains = [[(z0, p)] for p in base]
    for z, polys in layer[1:]:
        used = set()
        for ch in chains:
            c = np.array(ch[-1][1].centroid.coords[0])
            d = [np.linalg.norm(np.array(p.centroid.coords[0]) - c) if i not in used else 1e9 for i, p in enumerate(polys)]
            j = int(np.argmin(d)); used.add(j); ch.append((z, polys[j]))
    return chains

def build(mesh, dz=0.2, tol=0.3, n=96, smooth=0.05, step=1.0, secs=None, fuse=True):
    secs = secs if secs is not None else sections(mesh, dz)
    L = layers(secs, tol)
    pieces = []; info = []
    for layer in L:
        za, zb = layer[0][0] - dz / 2, layer[-1][0] + dz / 2
        k = max(2, int(round((zb - za) / step)) + 1)
        same = len({signature(p) for _, p in layer}) == 1
        if not same:
            # контуры от уровня к уровню меняются (шум, прорези): прямое выдавливание среднего уровня
            mid = layer[len(layer) // 2][1]
            for p in mid:
                try:
                    solid = cq.Workplane().add(prism(ring_pts(p.exterior.coords, n, smooth), za, zb))
                    for h in p.interiors:
                        if Polygon(h).area < 2: continue
                        solid = solid.cut(cq.Workplane().add(prism(ring_pts(h.coords, n, smooth), za - 1e-3, zb + 1e-3)))
                    piece = solid.val()
                except Exception as e:
                    info.append(("fail-ext", round(za, 2), round(zb, 2), str(e)[:60])); continue
                pieces.append(piece)
                info.append(("ext", round(za, 2), round(zb, 2))); print(info[-1], flush=True)
            continue
        for ch in track(layer):
            idx = np.unique(np.linspace(0, len(ch) - 1, k).round().astype(int))
            zs = np.linspace(za, zb, len(idx)) if len(idx) > 1 else [za, zb]
            try:
                outer = loft([(ring_pts(ch[i][1].exterior.coords, n, smooth), z) for i, z in zip(idx, zs)])
                solid = cq.Workplane().add(outer)
                for h in range(len(ch[idx[0]][1].interiors)):
                    hole = loft([(ring_pts(ch[i][1].interiors[h].coords, n, smooth), z) for i, z in zip(idx, zs)])
                    solid = solid.cut(cq.Workplane().add(hole))
                piece = solid.val()
            except Exception as e:
                info.append(("fail", round(za, 2), round(zb, 2), str(e)[:60])); print(info[-1], flush=True)
                try:
                    p = ch[len(ch) // 2][1]
                    solid = cq.Workplane().add(prism(ring_pts(p.exterior.coords, n, smooth), za, zb))
                    for h in p.interiors:
                        if Polygon(h).area < 2: continue
                        solid = solid.cut(cq.Workplane().add(prism(ring_pts(h.coords, n, smooth), za - 1e-3, zb + 1e-3)))
                    piece = solid.val()
                except Exception:
                    continue
            pieces.append(piece)
            info.append(("ok", round(za, 2), round(zb, 2), len(idx))); print(info[-1], flush=True)
    if not fuse:
        return pieces, info
    vol = sum(p.Volume() for p in pieces)
    cq.exporters.export(cq.Compound.makeCompound(pieces), sys.argv[2] + "_pieces.stl", tolerance=0.01, angularTolerance=0.1)
    print("pieces written", round(time.time() - T0, 1), flush=True)
    body = pieces[0].fuse(*pieces[1:], glue=True).clean() if len(pieces) > 1 else pieces[0]
    print("pieces", len(pieces), "sum vol", round(vol, 1), "fused vol", round(body.Volume(), 1), flush=True)
    return body, info



def pick_levels(zs, rings, jump=0.8, dzmax=2.0, dzmin=0.6):
    """уровни для лофта: новый уровень, когда контур ушёл на jump мм или прошло dzmax мм"""
    keep = [0]
    for i in range(1, len(zs)):
        A = Polygon(rings[keep[-1]]); B = Polygon(rings[i])
        if zs[i] - zs[keep[-1]] >= dzmin and (A.hausdorff_distance(B) > jump or zs[i] - zs[keep[-1]] > dzmax):
            keep.append(i)
    if keep[-1] != len(zs) - 1:
        if zs[-1] - zs[keep[-1]] < dzmin and len(keep) > 1: keep[-1] = len(zs) - 1
        else: keep.append(len(zs) - 1)
    return keep

def build2(mesh, dz=0.2, n=int(os.environ.get('RING_N', '32')), smooth=0.05):
    """наружная стенка — один лофт по всей высоте; внутренние объёмы (карманы, пазы) вычитаются по слоям"""
    secs = sections(mesh, dz)
    secs = [(z, p) for z, p in secs if p]
    # наружный контур: самый большой многоугольник, пока он не распался
    outer = []
    for z, polys in secs:
        big = max(polys, key=lambda p: Polygon(p.exterior).area)
        a = Polygon(big.exterior).area
        if outer and a < 0.7 * outer[-1][2]: break
        outer.append((z, big, a))
    # нижние срезы-обрезки (касание дна) дают скачок формы — дно начинаем с первого полного среза
    pads = []
    while len(outer) > 2 and outer[0][2] < 0.6 * outer[1][2]:
        pads.append(outer.pop(0))
    TOPINFO["pads"] = pads
    zs = [o[0] for o in outer]
    rings = [ring_pts(o[1].exterior.coords, n, smooth) for o in outer]
    # сглаживание вдоль высоты: каждая точка кольца — гладкая функция z (шум скана не повторяем)
    from scipy.interpolate import UnivariateSpline
    al = align([(r, z) for r, z in zip(rings, zs)])
    R = np.array([r for r, _ in al]); Z = np.array(zs)
    sm = float(os.environ.get("Z_SMOOTH", "0.08"))
    if len(Z) >= 6:
        R2 = np.empty_like(R)
        W = 1.0 + 6.0 * np.exp(-(Z - Z[0]) / 1.0)  # у дна форма меняется быстро — держимся ближе к срезам
        for i in range(R.shape[1]):
            for d in range(2):
                R2[:, i, d] = UnivariateSpline(Z, R[:, i, d], w=W, k=3, s=len(Z) * sm ** 2)(Z)
        rings = list(R2)
    keep = pick_levels(zs, rings)
    za, zb = zs[0] - dz / 2, zs[-1] + dz / 2
    lz = np.array([zs[i] for i in keep]); lz[0] = za; lz[-1] = zb
    print("outer levels", len(keep), round(za, 2), round(zb, 2), flush=True)
    coef = top_surface(mesh, rings[-1], zb)
    ztop = zb
    levels = [(rings[i], z) for i, z in zip(keep, lz)]
    if coef is not None:
        xs, ys = rings[-1][:, 0], rings[-1][:, 1]
        zz = coef[0] + coef[1] * xs + coef[2] * ys + coef[3] * xs * xs + coef[4] * xs * ys + coef[5] * ys * ys
        print("top z", round(float(zz.min()), 2), round(float(zz.max()), 2), flush=True)
        if zz.max() + 0.5 > zb + 0.3:
            ztop = float(zz.max()) + 0.5
            levels[-1] = (levels[-1][0], ztop)  # верхнее сечение поднимаем, без двойного кольца (иначе лофт раздувает)
    bcoef = top_surface(mesh, rings[keep[0]], zs[0], down=True) if os.environ.get("BOTTOM_FIT", "1") == "1" else None
    if bcoef is not None:
        xs, ys = rings[keep[0]][:, 0], rings[keep[0]][:, 1]
        zz = bcoef[0] + bcoef[1] * xs + bcoef[2] * ys + bcoef[3] * xs * xs + bcoef[4] * xs * ys + bcoef[5] * ys * ys
        print("bottom z", round(float(zz.min()), 2), round(float(zz.max()), 2), flush=True)
        levels[0] = (levels[0][0], min(levels[0][1], float(zz.min()) - 0.3))
    TOPINFO["bcoef"] = bcoef
    body = cq.Workplane().add(loft(levels))
    TOPINFO.update(coef=coef, ztop=ztop, ring=rings[-1])
    # пустоты внутри наружного контура по слоям
    voids = []
    for z, polys in secs:
        if z > zb: break
        big = max(polys, key=lambda p: Polygon(p.exterior).area)
        v = Polygon(big.exterior).difference(shapely.union_all(polys)).buffer(-0.15).buffer(0.15)
        geoms = [g for g in getattr(v, "geoms", [v]) if not g.is_empty and g.area >= 1.0]
        voids.append((z, geoms))
    L = []
    for z, g in voids:
        sig = len(g)
        if L and L[-1][-1][2] == sig and shapely.union_all(L[-1][-1][1] or [Polygon()]).symmetric_difference(shapely.union_all(g or [Polygon()])).area < 0.5 + 0.03 * sum(x.area for x in g):
            L[-1].append((z, g, sig))
        else:
            L.append([(z, g, sig)])
    cut = []
    for lay in L:
        z0, z1 = lay[0][0] - dz / 2, lay[-1][0] + dz / 2
        if z1 >= zb - 1e-6: z1 = TOPINFO['ztop'] + 0.5
        mid = lay[len(lay) // 2][1]
        if len(lay) >= 3 and len({len(x[1]) for x in lay}) == 1:
            # стенки кармана — лофт по отслеженным контурам, а не лесенка
            chains = track([(z, g) for z, g, _ in lay])
            done = True
            pieces_l = []
            for ch in chains:
                try:
                    zs_c = [z for z, _ in ch]
                    rings_c = [ring_pts(g.exterior.coords, n, 0.03) for _, g in ch]
                    kk = pick_levels(zs_c, rings_c, jump=0.5, dzmax=1.5, dzmin=0.4)
                    if len(kk) < 2: raise RuntimeError("short")
                    lz = [zs_c[i] for i in kk]; lz[0] = z0 - 0.02; lz[-1] = z1 + 0.02
                    sol = cq.Workplane().add(loft([(rings_c[i], z) for i, z in zip(kk, lz)]))
                    pieces_l.append(sol.val())
                except Exception as e:
                    print("void loft fail", round(z0, 2), str(e)[:50], flush=True); done = False; break
            if done:
                cut.extend(pieces_l); print("void loft", round(z0, 2), round(z1, 2), len(pieces_l), flush=True)
                continue
        for g in mid:
            try:
                s = cq.Workplane().add(prism(ring_pts(g.exterior.coords, n, 0.03), z0 - 0.02, z1 + 0.02))
                cut.append(s.val())
            except Exception as e:
                print("void fail", round(z0, 2), str(e)[:50], flush=True)
    print("void layers", len(L), "void prisms", len(cut), flush=True)
    pieces = [body.val()]
    isl = islands(secs, zb, dz, n)
    # выступ под дном (площадка): выдавливание самого большого из отрезанных нижних срезов до дна
    if TOPINFO.get("pads"):
        pz = [p[0] for p in TOPINFO["pads"]]; big = max(TOPINFO["pads"], key=lambda p: p[2])[1]
        try:
            isl.append(prism(ring_pts(big.exterior.coords, n, 0.03), min(pz) - dz / 2, outer[0][0] + 0.3))
            print("pad", round(min(pz) - dz / 2, 2), round(Polygon(big.exterior).area, 1), flush=True)
        except Exception as e:
            print("pad fail", str(e)[:50], flush=True)
    return pieces, cut, isl


def islands(secs, zb, dz=0.2, n=32, min_area=3.0):
    """бобышки и рёбра внутри кармана и над краем: отдельные многоугольники сечений,
    сгруппированные по перекрытию в плане; каждое — выдавливание своего контура с отверстиями"""
    items = []
    for z, polys in secs:
        big = max(polys, key=lambda p: Polygon(p.exterior).area)
        for p in polys:
            if (p is big and z <= zb) or p.area < min_area:
                continue
            items.append((z, p))
    groups = []
    for z, p in items:
        f = Polygon(p.exterior).buffer(0.3)
        hit = [g for g in groups if g["foot"].intersects(f)]
        if hit:
            g = hit[0]
            for o in hit[1:]:
                g["polys"] += o["polys"]; g["foot"] = g["foot"].union(o["foot"]); groups.remove(o)
            g["polys"].append((z, p)); g["foot"] = g["foot"].union(f)
        else:
            groups.append({"polys": [(z, p)], "foot": f})
    for g in groups:
        g["polys"].sort(key=lambda t: t[0])
    solids = []
    wall = Polygon(TOPINFO["ring"]).exterior if TOPINFO.get("ring") is not None else None
    for g in groups:
        ps = g["polys"]
        if wall is not None and g["foot"].distance(wall) < 0.8:
            print("island at wall skipped", round(g["foot"].area, 1), flush=True); continue
        zset = sorted({z for z, _ in ps})
        if len(zset) < 2: continue
        za, zb2 = zset[0] - dz / 2, zset[-1] + dz / 2
        with_holes = [q for _, q in ps if len(q.interiors)]
        ref = max(with_holes or [q for _, q in ps], key=lambda q: Polygon(q.exterior).area)
        single = len(zset) == len(ps)
        try:
            zs_c = [z for z, _ in ps]
            rings_c = [ring_pts(q.exterior.coords, n, 0.03) for _, q in ps]
            kk = pick_levels(zs_c, rings_c, jump=0.5, dzmax=1.5, dzmin=0.4) if single and not with_holes else []
            if len(kk) >= 2:
                lz = [zs_c[i] for i in kk]; lz[0] = za; lz[-1] = zb2
                sol = cq.Workplane().add(loft([(rings_c[i], z) for i, z in zip(kk, lz)]))
            else:
                sol = cq.Workplane().add(prism(ring_pts(ref.exterior.coords, n, 0.03), za, zb2))
            for h in ref.interiors:
                if Polygon(h).area < 1: continue
                sol = sol.cut(cq.Workplane().add(prism(ring_pts(h.coords, n, 0.03), za - 0.05, zb2 + 0.05)))
            solids.append(sol.val())
            print("island", round(za, 2), round(zb2, 2), round(ref.area, 1), len(ref.interiors), flush=True)
        except Exception as e:
            print("island fail", round(za, 2), str(e)[:50], flush=True)
    return solids


def top_surface(mesh, ring, zb, band=3.0, down=False):
    """верх стенки: гладкая поверхность z = f(x, y) (квадратичная) по точкам скана, смотрящим вверх у края"""
    from shapely.geometry import Point
    c = mesh.triangles_center; nz = mesh.face_normals[:, 2]
    sel = ((nz < -0.8) & (c[:, 2] < zb + 1.5)) if down else ((nz > 0.8) & (c[:, 2] > zb - 1.5))
    P = c[sel]
    edge = Polygon(ring).exterior
    d = np.array([edge.distance(Point(x, y)) for x, y in P[:, :2]])
    inside = np.array([Polygon(ring).contains(Point(x, y)) for x, y in P[:, :2]])
    P = P[((d < band) | down) & inside]
    if len(P) < 30:
        return None
    x, y, z = P.T
    A = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
    w = np.ones_like(z)
    for _ in range(5):  # робастно: отбрасываем далёкие точки
        coef = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
        r = z - A @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-3
        w = (np.abs(r) < 3 * s).astype(float)
    print("top fit", len(P), "rms", round(float(np.sqrt(np.mean((r[w > 0]) ** 2))), 3), flush=True)
    return coef

def cut_above(solid, coef, bbox, margin=5.0, below=False):
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
    x0, y0, x1, y1 = bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin
    k = 9
    arr = TColgp_Array2OfPnt(1, k, 1, k)
    for i, x in enumerate(np.linspace(x0, x1, k)):
        for j, y in enumerate(np.linspace(y0, y1, k)):
            z = coef @ [1, x, y, x * x, x * y, y * y]
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
    surf = GeomAPI_PointsToBSplineSurface(arr, 3, 3, GeomAbs_C2, 1e-4).Surface()
    face = BRepBuilderAPI_MakeFace(surf, 1e-6).Face()
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    zc = coef @ [1, xc, yc, xc * xc, xc * yc, yc * yc] + (-50 if below else 50)
    hs = BRepPrimAPI_MakeHalfSpace(face, gp_Pnt(float(xc), float(yc), float(zc))).Solid()
    return solid.cut(cq.Solid(hs))

T0 = time.time()
if __name__ == "__main__":
    t = time.time()
    m = trimesh.load(sys.argv[1], force="mesh")
    body, info = build(m)
    for i in info: print(i)
    print("valid", body.isValid(), "faces", len(body.Faces()), "solids", len(body.Solids()), round(time.time() - t, 1), "s")
    cq.exporters.export(body, sys.argv[2] + ".step")
    cq.exporters.export(body, sys.argv[2] + ".stl", tolerance=0.01, angularTolerance=0.1)


def correction_prisms(scan, model_mesh, dz=0.2, n=32, extra=False, min_area=0.6, shrink=0.12):
    """поправка по сечениям: где в модели есть материал, а в скане нет (пазы, канавки), — вырез;
    при extra=True наоборот — где в скане есть, а в модели нет — добавка. Слои по перекрытию, выдавливание."""
    z0 = max(scan.bounds[0, 2], model_mesh.bounds[0, 2]); z1 = min(scan.bounds[1, 2], model_mesh.bounds[1, 2])
    def sec(mesh, z):
        s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is None: return Polygon()
        P, _ = s.to_2D(to_2D=np.eye(4))
        return shapely.union_all([p.buffer(0) for p in P.polygons_full])
    layers_ = []
    for z in np.arange(z0 + dz, z1 - dz / 2, dz):
        a, b = sec(model_mesh, z), sec(scan, z)
        D = (b.difference(a) if extra else a.difference(b)).buffer(-shrink).buffer(shrink)
        geoms = [g for g in getattr(D, "geoms", [D]) if not g.is_empty and g.geom_type == "Polygon" and g.area >= min_area]
        for g in geoms:
            for L in layers_:
                if abs(L["z1"] - z) < dz * 1.5 and L["g"].intersection(g).area > 0.5 * min(L["g"].area, g.area):
                    L["z1"] = z; L["all"].append(g); break
            else:
                layers_.append({"z0": z, "z1": z, "g": g, "all": [g]})
    out = []
    for L in layers_:
        if len(L["all"]) < 2: continue
        g = shapely.intersection_all(L["all"]) if not extra else L["all"][len(L["all"]) // 2]
        if g.is_empty or g.area < min_area: g = L["all"][len(L["all"]) // 2]
        if g.geom_type != "Polygon": g = max(g.geoms, key=lambda q: q.area)
        try:
            out.append(prism(ring_pts(g.exterior.coords, n, 0.02), L["z0"] - dz / 2 - 0.02, L["z1"] + dz / 2 + 0.02))
        except Exception:
            pass
    return out
