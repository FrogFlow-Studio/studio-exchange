import os
"""Прототип «как в Компасе»: деталь = стопка операций «по сечениям» вдоль оси z.
Сечения скана каждые dz; слой = промежуток, где набор контуров не меняется; каждый контур слоя —
гладкий сплайн; тело слоя — лофт через несколько сечений; дыры контура вычитаются тем же способом."""
import sys, time, numpy as np, trimesh, shapely
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # stl_out.py рядом с demo.py
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

def prism_poly(poly, za, zb, tol=0.08):
    """выдавливание многоугольника с прямыми сторонами (паз, шпонка): углы остаются острыми, без сплайна"""
    q = poly.simplify(tol, preserve_topology=True)
    pts = [tuple(map(float, p)) for p in list(q.exterior.coords)[:-1]]
    w = cq.Workplane("XY").workplane(offset=float(za)).polyline(pts).close()
    return w.extrude(float(zb - za)).val()

def prism(pts2, za, zb):
    w = cq.Workplane("XY").workplane(offset=float(za)).spline([tuple(map(float, p)) for p in pts2], periodic=True).close()
    return w.extrude(float(zb - za)).val()

def _wire(pts2, z):
    zz = np.broadcast_to(np.asarray(z, float), (len(pts2),))  # кольцо может лежать не в плоскости (скругление по дну)
    return cq.Wire.assembleEdges([cq.Edge.makeSpline([cq.Vector(float(x), float(y), float(h)) for (x, y), h in zip(pts2, zz)], periodic=True)])


def _inset(ring, d):
    """кольцо, сдвинутое внутрь на d по нормали (кольцо гладкое, d мало по сравнению с размером)"""
    t = np.roll(ring, -1, 0) - np.roll(ring, 1, 0)
    nrm = np.c_[t[:, 1], -t[:, 0]] / np.linalg.norm(t, axis=1)[:, None]  # наружная для кольца против часовой
    return ring - nrm * d


def round_rings(ring, zfun, r, k=int(os.environ.get("ROUND_K", "4"))):
    """кольца скругления кромки дна радиусом r: четверть окружности от дна (z = zfun(x, y)) вверх на r"""
    # нижнее кольцо — плоское (торец лофта закрывается плоской гранью), ниже дна; дно потом срезает
    out = [(_inset(ring, r), float(np.min(zfun(ring[:, 0], ring[:, 1]))) - 0.3)]
    for h in r * (1 - np.cos(np.linspace(0, np.pi / 2, k)))[1:]:
        d = r - np.sqrt(max(0.0, r * r - (r - h) ** 2))
        q = _inset(ring, d)
        out.append((q, zfun(ring[:, 0], ring[:, 1]) + h))
    return out

def _loft_child(rings_z, ruled, q, deg=None):
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
        elif deg and not ruled:
            # степень лофта вдоль высоты ограничена: квадратичный лофт меньше раздувается между сечениями,
            # и на его кромках строятся скругления (при степени 6 ядро в скруглении отказывает)
            ts = BRepOffsetAPI_ThruSections(True, False, 1e-6)
            ts.SetMaxDegree(int(deg))
            ts.CheckCompatibility(False)
            for p, z in rings_z:
                ts.AddWire(_wire(p, z).wrapped)
            ts.Build()
            s = cq.Solid(ts.Shape())
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
        k = int(np.argmin(cost))
        out.append((np.roll(p, -k, axis=0), np.roll(z, -k) if np.ndim(z) else z))
    return out

def loft(rings_z, ruled=False, deg=None):
    rings_z = align(rings_z)
    import multiprocessing as mp
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork"); q = ctx.Queue()
    p = ctx.Process(target=_loft_child, args=(rings_z, ruled, q, deg)); p.start(); p.join(60)
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



def pick_levels(zs, rings, jump=float(os.environ.get("LEV_JUMP", "0.8")), dzmax=float(os.environ.get("LEV_DZMAX", "2.0")), dzmin=0.6):
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

def _region_below(coef, poly, z, h=0.2):
    """часть многоугольника, где поверхность верха выше z (по сетке h, полосами строк)"""
    from shapely.geometry import box
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0, x1 + h, h); boxes = []
    for y in np.arange(y0, y1 + h, h):
        ok = np.asarray(surf_eval(coef, xs + h / 2, np.full_like(xs, y + h / 2))) > z
        d = np.diff(np.r_[0, ok.astype(int), 0]); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
        boxes += [box(xs[a], y, xs[b - 1] + h, y + h) for a, b in zip(st, en)]
    return shapely.union_all(boxes).intersection(poly) if boxes else Polygon()


def _top_open_levels(top_v, min_area=20.0):
    """главное углубление выше верха стенки по уровням, пока оно не распалось и не сжалось вдвое"""
    lev = []
    for z, gs in top_v:
        if not gs: break
        g = max(gs, key=lambda q: q.area)
        if g.area < min_area or (lev and g.area < 0.5 * lev[-1][1].area): break
        if lev and lev[-1][1].intersection(g).area < 0.5 * min(lev[-1][1].area, g.area): break
        lev.append((z, g))
    return lev


def build2(mesh, dz=0.2, n=int(os.environ.get('RING_N', '32')), smooth=float(os.environ.get('RING_SMOOTH', '0.05')), *,
           periodic_wall=False):
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
    if periodic_wall:
        from wall_profiles import fit_profiles

        contours = [o[1].exterior.coords for o in outer]
        fitted = fit_profiles(contours, Z)
        if fitted is None and len(Z)>=9:
            # The first and last cuts may touch the caps. Interior contours
            # still describe the wall and can extend to the trim surfaces.
            middle = fit_profiles(contours[2:-2], Z[2:-2])
            if middle is not None:
                fitted = np.pad(middle,((2,2),(0,0),(0,0)),mode="edge")
        if fitted is not None:
            R = fitted
            TOPINFO["periodic_wall"] = True
            rings = list(R)
    sm = float(os.environ.get("Z_SMOOTH", "0.08"))
    if len(Z) >= 6 and not TOPINFO.get("periodic_wall"):
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
    if os.environ.get("TOP_RIM", "0") == "1":  # опыт: подгонка борта Севера (задача 50), высота по длине
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                        "chatgpt-1", "cad-intelligence", "cad-kompas-button-2026-11-13"))
        from rim_fit import fit_rim
        ev = fit_rim(mesh, rings[-1], zb)
        if ev is not None:
            coef = lambda x, y, ev=ev: np.broadcast_to(np.asarray(ev(x, y), float), np.broadcast(np.asarray(x), np.asarray(y)).shape).copy() if np.ndim(x) or np.ndim(y) else float(ev(x, y))
    ztop = zb
    levels = [(rings[i], z) for i, z in zip(keep, lz)]
    if coef is not None:
        xs, ys = rings[-1][:, 0], rings[-1][:, 1]
        zz = surf_eval(coef, xs, ys)
        print("top z", round(float(zz.min()), 2), round(float(zz.max()), 2), flush=True)
        if zz.max() + 0.5 > zb + 0.3:
            ztop = float(zz.max()) + 0.5
            levels[-1] = (levels[-1][0], ztop)  # верхнее сечение поднимаем, без двойного кольца (иначе лофт раздувает)
    bcoef = top_surface(mesh, rings[keep[0]], zs[0], down=True) if os.environ.get("BOTTOM_FIT", "1") == "1" else None
    if bcoef is not None:
        xs, ys = rings[keep[0]][:, 0], rings[keep[0]][:, 1]
        zz = surf_eval(bcoef, xs, ys)
        print("bottom z", round(float(zz.min()), 2), round(float(zz.max()), 2), flush=True)
        levels[0] = (levels[0][0], min(levels[0][1], float(zz.min()) - 0.3))
    TOPINFO["bcoef"] = bcoef
    TOPINFO["dome"] = False
    rr = os.environ.get("BOTTOM_ROUND", "0")
    if rr != "0" and bcoef is not None:
        r = None
        if rr == "auto":  # радиус по тому, как сужаются нижние сечения (оценка Севера, задача 49)
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                            "chatgpt-1", "cad-intelligence", "cad-kompas-button-2026-11-13"))
            from bottom_round import estimate_radius
            allo = sorted(pads, key=lambda p: p[0]) + outer
            zb0 = float(np.median(surf_eval(bcoef, rings[keep[0]][:, 0], rings[keep[0]][:, 1])))
            r = estimate_radius([np.asarray(p[1].exterior.coords) for p in allo], [p[0] for p in allo], zb0)
        else:
            r = float(rr)
        if r:
            zf = lambda x, y: surf_eval(bcoef, x, y)
            base = rings[keep[0]]
            ztop_r = float(np.max(zf(base[:, 0], base[:, 1]))) + r
            rest = [(p, z) for p, z in levels[1:] if z > ztop_r + 0.15]
            levels = round_rings(base, zf, r) + rest
            TOPINFO["bottom_round"] = r
            print("bottom round", round(r, 3), "levels", len(levels), flush=True)
    if os.environ.get("BOTTOM_DOME", "0") == "1" and pads:
        # дно-купол: отдельный лофт по срезам низа (pad ниже) срастается со стенкой; стенку вниз не тянем
        # и плоскостью дна не режем — иначе на дне остаётся уступ по контуру купола
        TOPINFO["bcoef"] = None; TOPINFO["dome"] = True
        levels[0] = (levels[0][0], zs[keep[0]] - dz / 2)
    degree = int(os.environ.get("WALL_DEG", "2")) or None
    try:
        wall = loft(levels, deg=degree)
    except RuntimeError:
        if (os.environ.get("CAD_KOMPAS_END_RECOVERY", "0") != "1"
                or len(levels)<4 or TOPINFO.get("bottom_round")):
            raise
        # End sections may cut the curved bottom or the rising rim instead
        # of the wall. Extend the neighboring full profiles to the trim tools.
        recovery = [(levels[1][0],levels[0][1]),*levels[1:-1],
                    (levels[-2][0],levels[-1][1])]
        wall = loft(recovery, deg=degree)
        TOPINFO["wall_end_recovery"] = True
    body = cq.Workplane().add(wall)
    TOPINFO.update(coef=coef, ztop=ztop, ring=rings[-1])
    # пустоты внутри наружного контура по слоям
    voids = []; top_v = []
    # выше верха стенки (чаша, углубление у края) — пустоты считаем внутри последнего полного контура:
    # иначе их форму несёт только гладкая поверхность верха и на месте чаши остаётся ровная площадка
    above = float(os.environ.get("VOID_ABOVE", "0.6"))
    ref_top = Polygon(outer[-1][1].exterior).buffer(-above) if above > 0 else None
    for z, polys in secs:
        if z > zb:
            if ref_top is None or ref_top.is_empty: break
            # только там, где по поверхности верха тело ещё есть: выше неё срез верха и так всё снимет,
            # а вырез у края оставил бы тонкую кожуру отдельным телом
            live = _region_below(coef, ref_top, z + 0.25) if coef is not None else ref_top
            v = live.difference(shapely.union_all(polys)).buffer(-0.15).buffer(0.15)
            geoms = [g for g in getattr(v, "geoms", [v]) if not g.is_empty and g.area >= 1.0]
            top_v.append((z, geoms))
            continue
        big = max(polys, key=lambda p: Polygon(p.exterior).area)
        v = Polygon(big.exterior).difference(shapely.union_all(polys)).buffer(-0.15).buffer(0.15)
        geoms = [g for g in getattr(v, "geoms", [v]) if not g.is_empty and g.area >= 1.0]
        voids.append((z, geoms))
    if os.environ.get("VOIDS", "chains") == "chains":
        # углубление выше верха стенки продолжает цепочку кармана, с которым сливается: один лофт,
        # стенка чаши — наклонная грань (отдельный вырез рядом с карманом ломал вычитание)
        cut = void_chains(voids + [(z, [g]) for z, g in _top_open_levels(top_v)], dz, n, zb)
        print("void chains", len(cut), flush=True)
    else:
        cut = void_layers(voids, dz, n, zb)
    pieces = [body.val()]
    isl = islands(secs, zb, dz, n)
    # выступ под дном (площадка): выдавливание самого большого из отрезанных нижних срезов до дна
    if TOPINFO.get("pads") and os.environ.get("PADS", "0") == "1":
        pz = [p[0] for p in TOPINFO["pads"]]; big = max(TOPINFO["pads"], key=lambda p: p[2])[1]
        try:
            # дно-купол: лофт по срезам низа (от самого нижнего, маленького, к полному), а не плоская площадка со ступенькой
            pd = sorted(TOPINFO["pads"], key=lambda p: p[0])
            done = False
            if os.environ.get("PAD_LOFT", "1") == "1" and len(pd) >= 1:
                try:
                    lv = [(ring_pts(p[1].exterior.coords, n, 0.03), p[0]) for p in pd]
                    lv[0] = (lv[0][0], pd[0][0] - dz / 2)
                    lv.append((ring_pts(outer[0][1].exterior.coords, n, 0.03), outer[0][0] + 0.3))
                    sol = loft(lv)
                    bb = sol.BoundingBox()
                    if sol.isValid() and sol.Volume() > 0 and bb.zmin > pd[0][0] - dz / 2 - 0.3:
                        isl.append(sol); done = True
                except Exception as e:
                    print("pad loft fail", str(e)[:50], flush=True)
            if not done:
                isl.append(prism(ring_pts(big.exterior.coords, n, 0.03), min(pz) - dz / 2, outer[0][0] + 0.3))
            print("pad", round(min(pz) - dz / 2, 2), round(Polygon(big.exterior).area, 1), flush=True)
        except Exception as e:
            print("pad fail", str(e)[:50], flush=True)
    return pieces, cut, isl



def void_layers(voids, dz, n, zb):
    """прежний способ: слои с одинаковой формой пустот, лофт или призма на слой"""
    if True:
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
            print("void prism", round(z0, 2), round(z1, 2), len(mid), flush=True)
            for g in mid:
                try:
                    s = cq.Workplane().add(prism(ring_pts(g.exterior.coords, n, 0.03), z0 - 0.02, z1 + 0.02))
                    cut.append(s.val())
                except Exception as e:
                    print("void fail", round(z0, 2), str(e)[:50], flush=True)
        print("void layers", len(L), "void prisms", len(cut), flush=True)
        return cut


def void_chains(voids, dz, n, zb, jump=0.5):
    """каждое отделение кармана — своя цепочка контуров по всей высоте и один лофт.
    Отделение, которое выше сливается с большим (начались рёбра), продолжается
    вверх до конца того, с чем слилось: его стенки не обрываются ступенькой."""
    chains = []  # {"items": [(z, poly)], "open": bool, "into": idx}
    for z, geoms in voids:
        open_ = [k for k, c in enumerate(chains) if c["open"]]
        claim = {}
        for gi, g in enumerate(geoms):
            for k in open_:
                last = chains[k]["items"][-1][1]
                if last.intersection(g).area > 0.3 * min(last.area, g.area):
                    claim.setdefault(gi, []).append(k)
        owners = {}
        for gi, ks in claim.items():
            for k in ks: owners.setdefault(k, []).append(gi)
        cont = set()
        for gi, g in enumerate(geoms):
            ks = claim.get(gi, [])
            if len(ks) == 1 and len(owners[ks[0]]) == 1:
                chains[ks[0]]["items"].append((z, g)); cont.add(ks[0])
            else:
                chains.append({"items": [(z, g)], "open": True, "into": None, "from": ks})
                for k in ks:
                    if len(owners[k]) == 1 and Polygon(chains[k]["items"][-1][1].exterior).difference(g.buffer(0.3)).area < 0.1 * chains[k]["items"][-1][1].area:
                        chains[k]["into"] = len(chains) - 1
        for k in open_:
            if k not in cont: chains[k]["open"] = False
    def top_of(k, seen=()):
        c = chains[k]
        if c["into"] is not None and c["into"] not in seen:
            return top_of(c["into"], seen + (k,))
        return c["items"][-1][0]
    segs = []
    for k, c in enumerate(chains):
        # скачок формы внутри цепочки (верх ребра, уступ) — настоящая ступень: делим цепочку
        cur = [c["items"][0]]
        for z, g in c["items"][1:]:
            if cur[-1][1].hausdorff_distance(g) > float(os.environ.get("VOID_JUMP", "2.0")):
                segs.append((cur, cur[-1][0] + dz / 2)); cur = []
            cur.append((z, g))
        segs.append((cur, top_of(k) + dz / 2))
    cut = []
    for it, z1 in segs:
        if (os.environ.get("CAD_KOMPAS_VOID_PERSISTENCE", "0") == "1"
                and len(it)==1 and it[0][1].area<12):
            # A small, isolated slice can be caused by floor noise. It
            # must not inherit the height of a later, persistent pocket.
            continue
        z0 = it[0][0] - dz / 2
        if z1 >= zb - 1e-6: z1 = TOPINFO['ztop'] + 0.5
        try:
            zs_c = [z for z, _ in it]
            rings_c = [ring_pts(g.exterior.coords, n, 0.03) for _, g in it]
            if len(it) >= 3:
                kk = pick_levels(zs_c, rings_c, jump=jump, dzmax=1.5, dzmin=0.4)
                lv = [(rings_c[i], zs_c[i]) for i in kk]
                lv[0] = (lv[0][0], z0 - 0.02)
                if len(lv) < 2: lv.append((lv[0][0], z1 + 0.02))
                else: lv[-1] = (lv[-1][0], z1 + 0.02) if z1 - zs_c[-1] < 0.5 else lv[-1]
                if lv[-1][1] < z1: lv.append((rings_c[-1], z1 + 0.02))
                lv[0] = (lv[0][0], z0 - 0.1)
                sol = loft(lv)
                bb = sol.BoundingBox()
                if sol.Volume() <= 0 or bb.zmin < z0 - 0.3 or bb.zmax > lv[-1][1] + 0.3:
                    sol = loft(lv, ruled=True)  # гладкий лофт выгнуло за свои уровни — прямые переходы
                    if sol.Volume() <= 0: raise RuntimeError("negative")
            else:
                sol = prism(rings_c[len(it) // 2], z0 - 0.02, z1 + 0.02)
            sol = sol if isinstance(sol, cq.Shape) else cq.Workplane().add(sol).val()
            vdeg = int(os.environ.get("VOID_DEG", "0"))
            if vdeg and len(it) >= 3:
                # квадратичный лофт кармана: на его кромках строятся скругления; если вырез им не
                # удастся, при вычитании берётся прежний лофт (_fallback)
                try:
                    s2 = loft(lv, deg=vdeg); b2 = s2.BoundingBox()
                    if s2.isValid() and s2.Volume() > 0 and b2.zmin >= z0 - 0.3 and b2.zmax <= lv[-1][1] + 0.3:
                        s2._fallback = sol; sol = s2
                except Exception:  # noqa: BLE001
                    pass
            cut.append(sol)
            print("void chain", round(z0, 2), round(z1, 2), len(it), flush=True)
        except Exception as e:
            print("void chain fail", round(z0, 2), str(e)[:50], flush=True)
            try:
                cut.append(prism(rings_c[len(it) // 2], z0 - 0.02, z1 + 0.02))
            except Exception:
                pass
    return cut

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
        # низ бобышки утапливаем в пол кармана, чтобы она срослась с телом, а не висела отдельно
        za, zb2 = zset[0] - dz / 2 - float(os.environ.get("CAD_SECTIONS_EMBED", "0.6")), zset[-1] + dz / 2
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
            if os.environ.get("BOSS_SLOTS", "1") == "1":
                sol, ns = _boss_slots(sol, ref, ps, dz, n)
            else:
                ns = 0
            solids.append(sol.val())
            print("island", round(za, 2), round(zb2, 2), round(ref.area, 1), len(ref.interiors), "slots", ns, flush=True)
        except Exception as e:
            print("island fail", round(za, 2), str(e)[:50], flush=True)
    return solids



def _boss_slots(sol, ref, ps, dz, n, min_area=0.4, shrink=0.12):
    """пазы в стенке бобышки: где на уровне z в сечении нет материала, а в контуре бобышки он есть.
    Соседние уровни с перекрывающимся пазом — один вырез на всю их высоту (как паз фрезой).
    Кольцевые недостачи (фаска, скругление по краю) не режем."""
    F = ref.buffer(0)
    byz = {}
    for z, q in ps:
        byz.setdefault(z, []).append(q)
    runs = []
    for z in sorted(byz):
        M = F.difference(shapely.union_all(byz[z]).buffer(0)).buffer(-shrink).buffer(shrink)
        for g in getattr(M, "geoms", [M]):
            if g.is_empty or g.geom_type != "Polygon" or g.area < min_area or len(g.interiors) or g.area > 0.3 * F.area:
                continue
            if g.buffer(-0.3).is_empty:  # щель уже 0.6 мм — шум сечения, не паз
                continue
            for r in runs:
                a0 = r["g"][-1].area
                if abs(r["z1"] - z) < dz * 1.5 and 0.75 < g.area / a0 < 1.33 and r["g"][-1].intersection(g).area > 0.4 * min(a0, g.area):
                    r["z1"] = z; r["g"].append(g); break
            else:
                runs.append({"z0": z, "z1": z, "g": [g]})
    # выступы внутрь отверстия (шпонка, упор): материал в сечении внутри отверстия контура
    tabs = []
    for h in ref.interiors:
        H = Polygon(h)
        if H.area < 1: continue
        for z in sorted(byz):
            T = shapely.union_all(byz[z]).buffer(0).intersection(H).buffer(-shrink).buffer(shrink)
            for g in getattr(T, "geoms", [T]):
                if g.is_empty or g.geom_type != "Polygon" or g.area < 0.25 or g.buffer(-0.15).is_empty or g.area > 0.5 * H.area:
                    continue
                for r in tabs:
                    a0 = r["g"][-1].area
                    if abs(r["z1"] - z) < dz * 1.5 and r["g"][-1].intersection(g).area > 0.3 * min(a0, g.area):
                        r["z1"] = z; r["g"].append(g); r["z"].append(z); break
                else:
                    tabs.append({"z0": z, "z1": z, "g": [g], "z": [z], "H": H})
    k = 0
    for r in tabs:
        if len(r["g"]) < 2: continue
        # выступ врастает в стенку отверстия; меняется по высоте (скос защёлки) — лофт по уровням, а не ступеньки
        gs = []
        for g in r["g"]:
            g = g.buffer(0.2, join_style=2).intersection(r["H"].buffer(0.2)).buffer(0)
            if g.geom_type != "Polygon": g = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
            gs.append(g)
        try:
            if os.environ.get("TAB_RAMP", "1") == "1":
                # скос защёлки: на уровень ниже и выше выступ сходит на нет к стенке отверстия
                from shapely.affinity import scale
                from shapely.ops import nearest_points
                lv = [(ring_pts(q.exterior.coords, n, 0.01), z) for q, z in zip(gs, r["z"])]
                def tip(q, strip):
                    # strip: тонкая полоска язычка у стенки отверстия — ширина та же, толщина сходит на нет (ровный скос);
                    # иначе язычок сжимается к точке у стенки (запасной вариант, лофт надёжнее)
                    t = None
                    if isinstance(strip, float):
                        t = q.intersection(r["H"].exterior.buffer(strip)).buffer(0)
                    elif strip == 2:
                        # сдвигаем язычок к стенке на его вылет и берём общую часть — прямоугольная полоска у стенки
                        from shapely.affinity import translate
                        o = nearest_points(r["H"].exterior, q.centroid)[0]; c = r["H"].centroid
                        d = np.array([o.x - c.x, o.y - c.y]); d /= max(np.linalg.norm(d), 1e-9)
                        depth = q.intersection(r["H"]).area / max(q.length / 2, 1e-6) + 0.1
                        t = q.intersection(translate(q, d[0] * depth, d[1] * depth)).buffer(0)
                    if t is None or t.geom_type != "Polygon" or t.area < 0.02:
                        o = nearest_points(r["H"].exterior, q.centroid)[0]
                        t = scale(q, 0.2, 0.2, origin=o)
                    return ring_pts(t.exterior.coords, n, 0.01)
                done = False
                for strip in (0.12, 0.18, 0.08, 0.25, 2, 0):
                    try:
                        lv2 = [(tip(gs[0], strip), r["z0"] - dz)] + lv + [(tip(gs[-1], strip), r["z1"] + dz)]
                        nb = sol.union(cq.Workplane().add(loft(lv2, ruled=True)))
                        ok = nb.val().isValid() and len(nb.val().Solids()) == 1
                        if os.environ.get("TAB_DEBUG"): print("tab try", strip, ok, len(nb.val().Solids()), flush=True)
                        if ok:
                            sol = nb; k += 1; done = True; break
                    except Exception as e:
                        if os.environ.get("TAB_DEBUG"): print("tab try", strip, "fail", str(e)[:60], flush=True)
                continue
            else:
                g = shapely.intersection_all(gs)
                if g.is_empty or g.geom_type != "Polygon": g = gs[len(gs) // 2]
                add = cq.Workplane().add(prism_poly(g, r["z0"] - dz / 2, r["z1"] + dz / 2))
            nb = sol.union(add)
            if nb.val().isValid() and len(nb.val().Solids()) == 1:
                sol = nb; k += 1
        except Exception:
            pass
    for r in runs:
        if len(r["g"]) < 2: continue
        g = shapely.intersection_all(r["g"])
        if g.is_empty or g.area < min_area: g = r["g"][len(r["g"]) // 2]
        if g.geom_type != "Polygon": g = max(g.geoms, key=lambda q: q.area)
        # паз, выходящий на край бобышки, продлеваем наружу, чтобы не осталось плёнки
        g = g.buffer(0.15, join_style=2)
        try:
            cutter = cq.Workplane().add(prism_poly(g, r["z0"] - dz / 2 - 0.1, r["z1"] + dz / 2 + 0.1))
            nb = sol.cut(cutter)
            if nb.val().isValid() and nb.val().Volume() > 0.7 * sol.val().Volume():
                sol = nb; k += 1
        except Exception:
            pass
    return sol, k

def _basis(x, y, k):
    """базис поверхности z = f(x, y): 6 членов — квадратичная, 10 — кубическая"""
    x = np.asarray(x, float); y = np.asarray(y, float); one = np.ones_like(x)
    b = [one, x, y, x * x, x * y, y * y]
    if k == 10:
        b += [x ** 3, x * x * y, x * y * y, y ** 3]
    return np.stack(b, -1)

def surf_eval(coef, x, y):
    if callable(coef):
        return coef(x, y)
    return _basis(x, y, len(coef)) @ coef

def top_surface(mesh, ring, zb, band=3.0, down=False):
    """верх стенки: гладкая поверхность z = f(x, y) (квадратичная) по точкам скана, смотрящим вверх у края"""
    from shapely.geometry import Point
    # точки по площади, а не центры треугольников: крупные плоские треугольники скана иначе почти не голосуют
    c, fi = trimesh.sample.sample_surface(mesh, int(os.environ.get("SURF_SAMPLES", "60000")), seed=1) if os.environ.get("SURF_SAMPLE", "1") == "1" else (mesh.triangles_center, np.arange(len(mesh.faces)))
    nz = mesh.face_normals[fi, 2]
    sel = ((nz < -0.8) & (c[:, 2] < zb + 1.5)) if down else ((nz > 0.8) & (c[:, 2] > zb - 1.5))
    P = c[sel]
    edge = Polygon(ring).exterior
    pts = shapely.points(P[:, 0], P[:, 1])
    d = shapely.distance(edge, pts)
    inside = shapely.contains_xy(Polygon(ring), P[:, 0], P[:, 1])
    P = P[((d < band) | down) & inside]
    if len(P) < 30:
        return None
    x, y, z = P.T
    # дно — кубика: она ловит купол и наклон дна одной гладкой гранью, без площадки-ступеньки снизу
    deg = int(os.environ.get("BOTTOM_DEG", "3") if down else os.environ.get("TOP_DEG", "3"))
    A = _basis(x, y, 10 if deg == 3 else 6)
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
            z = surf_eval(coef, x, y)
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
    surf = GeomAPI_PointsToBSplineSurface(arr, 3, 3, GeomAbs_C2, 1e-4).Surface()
    face = BRepBuilderAPI_MakeFace(surf, 1e-6).Face()
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    zc = surf_eval(coef, xc, yc) + (-50 if below else 50)
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


def correction_prisms(scan, model_mesh, dz=0.2, n=32, extra=False, min_area=0.6, shrink=0.12,
                      skin=float(os.environ.get("CORR_SKIN", "0.6"))):
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
        if skin and not extra and geoms and not a.is_empty:
            # тонкая корка снаружи (лишнее модели по наружной кромке, шире скана на доли мм) — не паз,
            # а погрешность стенки; вырез такой корки оставляет на детали щели-лепестки
            ext = max(getattr(a, "geoms", [a]), key=lambda q: q.area).exterior
            geoms = [g for g in geoms if not (g.distance(ext) < 0.05 and 2 * g.area / g.length < skin)]
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


def _nz(f):
    try:
        return f.normalAt().z
    except Exception:  # noqa: BLE001
        return 0.0


def _edge_fillets(body, info, scan=None):
    """скругления кромок, как операция «скругление» в CAD: дно (BOTTOM_FILLET) и наружная кромка верха борта
    (TOP_FILLET), радиус в мм; при отказе ядра радиус уменьшается, при неудаче тело остаётся как было"""
    # радиусы по умолчанию — малые технологические скругления; оценка радиуса по скану — задача 61 Севера
    for key, sign, dflt in (("BOTTOM_FILLET", -1, "0.4"), ("TOP_FILLET", 1, "0.3")):
        setting = os.environ.get(key, dflt)
        if setting != "auto" and float(setting) <= 0:
            continue
        faces = body.Faces()
        cap = [f for f in faces if sign * _nz(f) > 0.7]
        walls = [f for f in faces if abs(_nz(f)) < 0.5]
        if not cap or not walls:
            continue
        cap = max(cap, key=lambda f: f.Area()); wall = max(walls, key=lambda f: f.Area())
        we = {e.hashCode() for e in wall.Edges()}
        edges = [e for e in cap.Edges() if e.hashCode() in we]
        if not edges:
            continue
        if setting == "auto":
            if scan is None:
                continue
            from fillet_radius import fillet_radius

            rf = fillet_radius(scan,body,max(edges,key=lambda edge:edge.Length()))
            info[key.lower()+"_estimated"] = rf
            if rf is None:
                continue
        else:
            rf = float(setting)
        for r in (rf, rf * 0.7, rf * 0.5):
            try:
                nb = body.fillet(r, edges)
                if nb.isValid() and nb.Volume() > 0.95 * body.Volume():
                    body = nb; info[key.lower()] = round(r, 3)
                    break
            except Exception:  # noqa: BLE001
                pass
    rin = float(os.environ.get("TOP_IN_FILLET", "0"))
    if rin > 0 and TOPINFO.get("ring") is not None:
        # внутренняя кромка борта (край кармана сверху): рёбра верхней грани, далёкие от наружного контура
        tops = [f for f in body.Faces() if _nz(f) > 0.7]
        if tops:
            top = max(tops, key=lambda f: f.Area())
            outer = Polygon(TOPINFO["ring"]).exterior
            edges = [e for e in top.Edges()
                     if outer.distance(shapely.Point(e.Center().x, e.Center().y)) > 0.8]
            for r in (rin, rin * 0.6, rin * 0.4):
                try:
                    nb = body.fillet(r, edges)
                    if nb.isValid() and nb.Volume() > 0.95 * body.Volume():
                        body = nb; info["top_in_fillet"] = round(r, 3)
                        break
                except Exception:  # noqa: BLE001
                    pass
    return body


def assemble(scan, log=print):
    """Build 0с and validate an opt-in candidate against the stock result."""
    candidate = None
    experiment = None
    if os.environ.get("CAD_KOMPAS_STABLE_POCKETS", "0") == "1":
        from stable_pockets import assemble_stable

        candidate = assemble_stable(scan, log=log)
        experiment = "stable_pockets"
    baseline = _assemble_stock(scan, log=log)
    if candidate is None and os.environ.get("CAD_KOMPAS_PERIODIC_WALL", "0") == "1":
        experiment = "periodic_wall"
        try:
            trial = _assemble_stock(scan, log=log, periodic_wall=True)
            if trial[1].get("periodic_wall"):
                candidate = trial
        except (RuntimeError, ValueError) as error:
            log(f"    periodic wall rollback: {error}")
    if candidate is None:
        return baseline
    from candidate_guard import compare_candidate

    accepted, audit = compare_candidate(scan, baseline[0], candidate[0])
    audit["accepted"] = accepted
    log(f"    {experiment} guard: {audit}")
    body, info = candidate if accepted else baseline
    return body, {**info, experiment+"_guard": audit}


def _assemble_stock(scan, log=print, *, periodic_wall=False):
    """Сборка «как в Компасе» вдоль оси z выровненного скана. Возвращает (тело, сведения).
    Порядок операций: наружная стенка (лофт по сечениям, сглаженным по высоте) → вырез кармана слоями →
    срез верха и низа гладкими поверхностями по скану → бобышки и площадка выдавливанием →
    поправка по сечениям (пазы и канавки вырезом). Если деталь не «стопка» вдоль z, тело выйдет негодным
    или далёким от скана — тогда попытку отбрасывает сравнение со сканом."""
    import tempfile
    TOPINFO.clear()
    info = {}
    t = time.time()
    pieces, cut, isl = build2(scan, periodic_wall=periodic_wall)
    for key in ("periodic_wall", "wall_end_recovery"):
        if TOPINFO.get(key):
            info[key] = True
    body = pieces[0]
    if not body.isValid() or body.Volume() <= 0:
        raise RuntimeError("наружная стенка не построена")
    info["voids"] = len(cut)
    skipped = 0
    trace = os.environ.get("CAD_KOMPAS_TRACE", "0") == "1"
    void_checks = []
    def _cut_try(body, v):
        try:
            nb = body.cut(v)
            if nb.isValid() and nb.Volume() > 0.5 * body.Volume():
                return nb
        except Exception:  # noqa: BLE001
            pass
        fb = getattr(v, "_fallback", None)
        return body.cut(fb) if fb is not None else body.cut(v)

    for v in sorted(cut, key=lambda c: -c.Volume()):
        try:
            before = body.Volume() if trace else None
            nb = _cut_try(body, v)
            if nb.isValid() and nb.Volume() > 0.5 * body.Volume():
                body = nb
                reason = "accepted"
            else:
                skipped += 1
                reason = "empty" if not nb.Solids() else "invalid" if not nb.isValid() else "half_volume_guard"
            if trace:
                bounds = v.BoundingBox()
                void_checks.append(dict(z=[bounds.zmin,bounds.zmax],reason=reason,
                                        tool_volume=v.Volume(),before_volume=before,after_volume=nb.Volume()))
        except Exception as error:  # noqa: BLE001
            skipped += 1
            if trace:
                void_checks.append(dict(reason="exception",error=str(error)))
    info["voids_skipped"] = skipped
    if trace:
        info["void_checks"] = void_checks
    bb = pieces[0].BoundingBox()
    for key, below in (("coef", False), ("bcoef", True)):
        if TOPINFO.get(key) is not None:
            nb = cut_above(body, TOPINFO[key], (bb.xmin, bb.ymin, bb.xmax, bb.ymax), below=below)
            if nb.isValid() and nb.Volume() > 0:
                body = nb
                info["bottom_surface" if below else "top_surface"] = True
            if trace:
                info[key+"_check"] = dict(valid=nb.isValid(), volume=nb.Volume(), solids=len(nb.Solids()))
    body = body.clean()
    body = _edge_fillets(body, info, scan)
    added = 0
    for p in isl:
        try:
            nb = body.fuse(p)
            if nb.isValid() and len(nb.Solids()) <= len(body.Solids()):
                body = nb; added += 1
        except Exception:  # noqa: BLE001
            pass
    info["bosses"] = added
    body = body.clean()
    if os.environ.get("CAD_SECTIONS_CORRECT", "1") == "1":
        tf = os.path.join(tempfile.gettempdir(), "_kcorr%d.stl" % os.getpid())
        # сетка для поправки — с проверкой объёма: битая сетка давала ложные вырезы
        from stl_out import write_stl_checked
        write_stl_checked(body.wrapped, tf, 0.03, True, 0.15, tol=0.1)
        mm = trimesh.load(tf, force="mesh")
        nc = 0
        for v in sorted(correction_prisms(scan, mm), key=lambda c: -c.Volume()):
            try:
                nb = body.cut(v)
                if nb.isValid() and nb.Volume() > 0.9 * body.Volume():
                    body = nb; nc += 1
            except Exception:  # noqa: BLE001
                pass
        info["correction_cuts"] = nc
        body = body.clean()
        try:
            os.remove(tf)
        except OSError:
            pass
    sols = body.Solids()
    if len(sols) == 1 and len(body.Shells()) > 1:
        # одно тело с несколькими наружными оболочками (тонкий обрывок борта после среза верха):
        # в STEP оно читается как два тела — разбираем по оболочкам, как отдельные тела
        sols = [cq.Solid.makeSolid(sh) for sh in body.Shells()]
        sols = [x for x in sols if x.isValid()] or body.Solids()
    if len(sols) > 1:
        # мелкие отрезанные обрывки (тонкие перемычки между вырезами) — убираем, если их меньше 1% объёма
        sols = sorted(sols, key=lambda x: -x.Volume())
        rest = sum(x.Volume() for x in sols[1:])
        if rest < 0.01 * sols[0].Volume():
            body = sols[0]
            info["fragments_dropped"] = len(sols) - 1
        else:
            raise RuntimeError(f"сборка распалась на {len(sols)} тела")
    info["faces"] = len(body.Faces())
    info["seconds"] = round(time.time() - t, 1)
    log(f"    сборка по сечениям: стенка лофтом, вырезов кармана {len(cut) - skipped}, бобышек {added}, "
        f"пазов поправкой {info.get('correction_cuts', 0)}, граней {info['faces']}, {info['seconds']} с")
    return body, info
