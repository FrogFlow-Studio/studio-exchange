import os
"""Прототип «как в Компасе»: деталь = стопка операций «по сечениям» вдоль оси z.
Сечения скана каждые dz; слой = промежуток, где набор контуров не меняется; каждый контур слоя —
гладкий сплайн; тело слоя — лофт через несколько сечений; дыры контура вычитаются тем же способом."""
import sys, time, numpy as np, trimesh, shapely
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # stl_out.py рядом с demo.py
from shapely.geometry import Polygon
from shapely.ops import unary_union
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
VOID_OPEN = float(os.environ.get("VOID_OPEN", "0.15"))  # срез тонких перешейков пустоты
VOID_MIN = float(os.environ.get("VOID_MIN", "1.0"))  # мельче — шум сечения

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

def _adapt_u(tck, n, adapt, m=2048, win=0.5):
    """параметры n точек кольца, сгущённые на скруглениях: доля точек делится по углу поворота контура
    (вес adapt) и по длине (вес 1); у любого сечения на каждый угол приходится одно и то же число точек,
    поэтому точки соседних уровней стоят друг против друга и на углах стенка не «гранится»"""
    uu = np.linspace(0, 1, m, endpoint=False)
    d1 = np.array(splev(uu, tck, der=1)); d2 = np.array(splev(uu, tck, der=2))
    sp = np.hypot(d1[0], d1[1]) + 1e-12
    ds = sp / m; L = float(ds.sum())
    k = np.abs(d1[0] * d2[1] - d1[1] * d2[0]) / sp ** 3
    # кривизну сглаживаем окном win мм по длине: шум скана на прямых участках не должен стягивать точки
    sg = max(1.0, win / L * m); j = np.arange(m); j = np.minimum(j, m - j)
    g = np.exp(-0.5 * (j / sg) ** 2); g /= g.sum()
    k = np.real(np.fft.ifft(np.fft.fft(k) * np.fft.fft(g)))
    w = ds * (1.0 + adapt * k * L / (2 * np.pi))
    cum = np.r_[0.0, np.cumsum(w)]; cum /= cum[-1]
    return np.interp(np.arange(n) / n, cum, np.r_[uu, 1.0])


def ring_pts(coords, n, smooth, ref_angle_pt=None, adapt=0.0):
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
    if adapt > 0:
        u = _adapt_u(tck, n, adapt)
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

def _wire(pts2, z, par=None):
    zz = np.broadcast_to(np.asarray(z, float), (len(pts2),))  # кольцо может лежать не в плоскости (скругление по дну)
    # par — общие параметры точек для всех колец лофта: узлы сплайнов совпадают, и поверхность лофта
    # получает столько же полюсов по кругу, сколько точек в кольце (иначе ядро сливает узлы всех колец)
    return cq.Wire.assembleEdges([cq.Edge.makeSpline([cq.Vector(float(x), float(y), float(h)) for (x, y), h in zip(pts2, zz)],
                                                     parameters=None if par is None else [float(t) for t in par], periodic=True)])


def _common_par(rings_z):
    """общие параметры точек колец: накопленная длина хорд, средняя по всем уровням (от 0 до 1, замкнуто)"""
    seg = np.mean([np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1) for p, _ in rings_z], axis=0)
    t = np.r_[0.0, np.cumsum(seg)]
    return t / t[-1]


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

def _loft_child(rings_z, ruled, q, deg=None, par=None):
    try:
        if os.environ.get("LOFT_SMOOTH", "0") == "1" and not ruled and len(rings_z) > 2:
            ts = BRepOffsetAPI_ThruSections(True, False, 1e-6)
            ts.SetSmoothing(True)
            ts.SetMaxDegree(5)
            for p, z in rings_z:
                ts.AddWire(_wire(p, z, par).wrapped)
            ts.CheckCompatibility(False)
            ts.Build()
            s = cq.Solid(ts.Shape())
            if not s.isValid():
                s = cq.Solid.makeLoft([_wire(p, z, par) for p, z in rings_z], ruled)
        elif deg and not ruled:
            # степень лофта вдоль высоты ограничена: квадратичный лофт меньше раздувается между сечениями,
            # и на его кромках строятся скругления (при степени 6 ядро в скруглении отказывает)
            ts = BRepOffsetAPI_ThruSections(True, False, 1e-6)
            ts.SetMaxDegree(int(deg))
            ts.CheckCompatibility(False)
            for p, z in rings_z:
                ts.AddWire(_wire(p, z, par).wrapped)
            ts.Build()
            s = cq.Solid(ts.Shape())
        else:
            s = cq.Solid.makeLoft([_wire(p, z, par) for p, z in rings_z], ruled)
        import io
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape
        fn = os.path.join(__import__("tempfile").gettempdir(), "_kloft%d.brep" % os.getpid())
        BRepTools.Write_s(s.wrapped, fn)
        q.put(fn if s.isValid() else None)
    except Exception as e:
        q.put(None)

def _cut_child(body, tools, q, vol0):
    try:
        from OCP.BRepTools import BRepTools
        nb = body.cut(*tools)
        v = nb.Volume()
        # вырез не должен разрезать тело на куски (шумный вырез насквозь через стенку)
        ok = bool(nb.isValid() and 0.5 * vol0 < v <= vol0 + 1e-6
                  and len(nb.Solids()) <= max(1, len(body.Solids())))
        fn = os.path.join(__import__("tempfile").gettempdir(), "_kcut%d.brep" % os.getpid())
        BRepTools.Write_s(nb.wrapped, fn)
        q.put((fn, ok, v))
    except Exception:  # noqa: BLE001
        q.put(None)


def cut_limited(body, tools, seconds, vol0):
    """вычитание с проверкой тела в отдельном процессе с пределом времени: на шумном скане булева
    операция ядра и даже проверка тела (isValid, Volume) на сложной детали идут минутами или не
    кончаются. Возвращает (тело, годно, объём); по пределу — RuntimeError"""
    import multiprocessing as mp
    if os.name == "nt":
        nb = body.cut(*tools); v = nb.Volume()
        return nb, bool(nb.isValid() and 0.5 * vol0 < v <= vol0 + 1e-6), v
    ctx = mp.get_context("fork"); q = ctx.Queue()
    p = ctx.Process(target=_cut_child, args=(body, tools, q, vol0)); p.start(); p.join(seconds)
    if p.is_alive():
        p.kill(); p.join(); raise RuntimeError("cut timeout")
    r = q.get(timeout=5)
    if not r: raise RuntimeError("cut failed")
    fn, ok, v = r
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    from OCP.BRep import BRep_Builder
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, fn, BRep_Builder()); os.remove(fn)
    return cq.Shape.cast(sh), ok, v


def _shape_child(fn_call, q):
    try:
        from OCP.BRepTools import BRepTools
        nb = fn_call()
        ok = bool(nb.isValid())
        fn = os.path.join(__import__("tempfile").gettempdir(), "_kop%d.brep" % os.getpid())
        BRepTools.Write_s(nb.wrapped, fn)
        q.put((fn, ok))
    except Exception:  # noqa: BLE001
        q.put(None)


def _mesh_child(shape_brep, tf, q):
    try:
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape
        from OCP.BRep import BRep_Builder
        from stl_out import write_stl_checked
        sh = TopoDS_Shape(); BRepTools.Read_s(sh, shape_brep, BRep_Builder())
        write_stl_checked(sh, tf, 0.03, True, 0.15, tol=0.1, parallel=False)
        q.put(True)
    except Exception:  # noqa: BLE001
        q.put(False)


def _corr_mesh(body, tf, info):
    """сетка тела для поправок; на большой шумной детали сетка ядра с мелким допуском строилась
    больше 30 минут (крышка) — предел времени MESH_SEC, не успела — поправки пропускаются"""
    from stl_out import write_stl_checked
    sec = float(os.environ.get("MESH_SEC", "120"))
    big = body.BoundingBox().DiagonalLength > float(os.environ.get("MESH_FORK_DIAG", "120"))
    if os.name == "nt" or sec <= 0 or not big:  # небольшая деталь: сетка за секунды, как раньше
        write_stl_checked(body.wrapped, tf, 0.03, True, 0.15, tol=0.1)
        return trimesh.load(tf, force="mesh")
    import multiprocessing as mp
    from OCP.BRepTools import BRepTools
    fb = tf + ".brep"
    BRepTools.Write_s(body.wrapped, fb)
    ctx = mp.get_context("fork"); q = ctx.Queue()
    p = ctx.Process(target=_mesh_child, args=(fb, tf, q)); p.start(); p.join(sec)
    ok = False
    if p.is_alive():
        p.kill(); p.join()
        info["correction_mesh_timeout"] = info.get("correction_mesh_timeout", 0) + 1
        print("correction mesh timeout", flush=True)
    else:
        try:
            ok = q.get(timeout=5)
        except Exception:  # noqa: BLE001
            ok = False
    try:
        os.remove(fb)
    except OSError:
        pass
    return trimesh.load(tf, force="mesh") if ok else None


def _fillet_sane(nb, body):
    """скругление не должно ломать тело: объём почти тот же, тел столько же, граней не меньше половины
    (на шумной крышке ядро отдало «скруглённое» тело из 39 граней вместо 848)"""
    try:
        v0, v = body.Volume(), nb.Volume()
        return (nb.isValid() and 0.95 * v0 < v < 1.02 * v0 and len(nb.Solids()) == len(body.Solids())
                and len(nb.Faces()) >= 0.5 * len(body.Faces()))
    except Exception:  # noqa: BLE001
        return False


def op_limited(fn_call, seconds):
    """операция ядра (скругление и т.п.) с пределом времени в отдельном процессе; (тело, годно) или RuntimeError"""
    import multiprocessing as mp
    if os.name == "nt":
        nb = fn_call(); return nb, nb.isValid()
    ctx = mp.get_context("fork"); q = ctx.Queue()
    p = ctx.Process(target=_shape_child, args=(fn_call, q)); p.start(); p.join(seconds)
    if p.is_alive():
        p.kill(); p.join(); raise RuntimeError("op timeout")
    r = q.get(timeout=5)
    if not r: raise RuntimeError("op failed")
    fn, ok = r
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    from OCP.BRep import BRep_Builder
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, fn, BRep_Builder()); os.remove(fn)
    return cq.Shape.cast(sh), ok


def align(rings_z):
    """каждое следующее кольцо поворачиваем по кругу так, чтобы точки шли напротив точек предыдущего (без закрутки)"""
    out = [rings_z[0]]
    for p, z in rings_z[1:]:
        q = out[-1][0]
        cost = [np.sum((np.roll(p, -k, axis=0) - q) ** 2) for k in range(len(p))]
        k = int(np.argmin(cost))
        out.append((np.roll(p, -k, axis=0), np.roll(z, -k) if np.ndim(z) else z))
    return out

def loft(rings_z, ruled=False, deg=None, par=False):
    rings_z = align(rings_z)
    par = _common_par(rings_z) if par else None  # WALL_PAR: одна параметризация на все кольца
    import multiprocessing as mp
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork"); q = ctx.Queue()
    p = ctx.Process(target=_loft_child, args=(rings_z, ruled, q, deg, par)); p.start(); p.join(60)
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



HD_SIMP = float(os.environ.get("HD_SIMP", "0.02"))


def _hd(a, b):
    """расстояние Хаусдорфа по контурам, упрощённым до HD_SIMP мм (пороги здесь 0.3–2 мм, ошибка ≤ 0.04)"""
    if HD_SIMP <= 0:
        return a.hausdorff_distance(b)
    return a.simplify(HD_SIMP).hausdorff_distance(b.simplify(HD_SIMP))


def pick_levels(zs, rings, jump=float(os.environ.get("LEV_JUMP", "0.8")), dzmax=float(os.environ.get("LEV_DZMAX", "2.0")), dzmin=0.6):
    """уровни для лофта: новый уровень, когда контур ушёл на jump мм или прошло dzmax мм"""
    keep = [0]
    for i in range(1, len(zs)):
        A = Polygon(rings[keep[-1]]); B = Polygon(rings[i])
        if zs[i] - zs[keep[-1]] >= dzmin and (_hd(A, B) > jump or zs[i] - zs[keep[-1]] > dzmax):
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


def _outer_segments(allb, i0, i1, dz, jump=float(os.environ.get("SEG_JUMP", "1.0")),
                    min_h=float(os.environ.get("SEG_MIN_H", "1.0"))):
    """наружный контур по всей высоте — стопка участков. allb: [(z, big, area)] всех уровней;
    [i0, i1) — прежний наружный контур (от дна, пока не распался). Участок кончается на скачке формы
    (площадь меняется больше чем в 1/0,7 раза или контур уходит дальше jump мм); новый участок
    начинается, только если выше скачка контур держится ровным не меньше min_h мм и стоит на
    предыдущем (перекрытие в плане) — иначе выше, как раньше, работают острова и срез верха.
    Внутри прежнего контура делим только скачок, у которого с обеих сторон ровные участки:
    касание дна и кромки верха остаются как были. Возвращает список диапазонов [(a, b)]."""
    m = max(2, int(round(min_h / dz)))
    P = [Polygon(b[1].exterior) for b in allb]
    Ps = [q.simplify(HD_SIMP) for q in P]  # точный Хаусдорф по ~200 точкам на сотнях уровней — минуты
    def step(i):
        a0, a1 = allb[i - 1][2], allb[i][2]
        if allb[i][0] - allb[i - 1][0] > 1.5 * dz or a1 < 0.7 * a0 or a1 > a0 / 0.7:
            return True
        return Ps[i - 1].hausdorff_distance(Ps[i]) > jump
    st = [True] + [step(i) for i in range(1, len(allb))]
    def run(i):  # ровный участок от уровня i не короче m уровней
        return i + m <= len(allb) and not any(st[i + 1:i + m])
    def on(i):   # стоит на предыдущем
        a = P[i - 1].intersection(P[i]).area
        return a > 0.3 * min(P[i - 1].area, P[i].area)
    cuts = [j for j in range(i0 + m, i1 - m + 1) if st[j] and run(j) and not any(st[j - m + 1:j]) and on(j)]
    bounds = [i0] + cuts + [i1]
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    j = i1
    while j < len(allb) and j > 0 and run(j) and on(j):
        k = j + 1
        while k < len(allb) and not st[k]:
            k += 1
        segs.append((j, k)); j = k
    if len(segs) > 1 and os.environ.get("SEG_CONST_SPLIT", "1") == "1":
        # излом без скачка: участок, где контур сначала меняется (рёбра сходят на нет), а потом стоит, —
        # два участка: лофт и прямое выдавливание (гладкий лофт скруглял бы излом)
        cst = float(os.environ.get("SEG_CONST_HD", "0.3"))
        out = []
        for a, b in segs:
            j = b - 1
            while j - 1 >= a and Ps[j - 1].hausdorff_distance(Ps[b - 1]) < cst:
                j -= 1
            i = a
            while i + 1 < b and Ps[i + 1].hausdorff_distance(Ps[a]) < cst:
                i += 1
            cut_ = [c for c in (i + 1, j) if a + m <= c <= b - m]
            cut_ = sorted(set(cut_))
            if len(cut_) == 2 and cut_[1] - cut_[0] < m:
                cut_ = cut_[:1]
            bb_ = [a] + cut_ + [b]
            out += list(zip(bb_[:-1], bb_[1:]))
        segs = out
    return segs


def _const_ring(rings, polys, tol=float(os.environ.get("SEG_CONST", "0.15"))):
    """контур участка, если он по высоте не меняется (с точностью до шума скана), иначе None.
    Средний контур — кольцо среднего уровня, сдвинутое по нормали на среднее расстояние до контуров
    всех уровней; участок постоянный, если почти все точки среднего контура (90 %) лежат ближе tol
    к контуру каждого уровня"""
    ref = rings[len(rings) // 2]
    t = np.roll(ref, -1, 0) - np.roll(ref, 1, 0)
    nrm = np.c_[t[:, 1], -t[:, 0]] / np.linalg.norm(t, axis=1)[:, None]
    def sd(pts, P):  # знаковое расстояние до контура: снаружи +
        d = shapely.distance(P.exterior, shapely.points(pts[:, 0], pts[:, 1]))
        return np.where(shapely.contains_xy(P, pts[:, 0], pts[:, 1]), -d, d)
    D = np.array([sd(ref, P) for P in polys])
    avg = ref - nrm * np.median(D, axis=0)[:, None]
    if not Polygon(avg).is_valid:
        return None
    dev = max(float(np.percentile(np.abs(sd(avg, P)), 90)) for P in polys)
    return avg if dev < tol else None


def _split_z(mesh, lo, up, z_lo, z_hi):
    """высота уступа между участками: средняя высота горизонтальных граней скана в полосе, где
    контуры участков расходятся (полка нижнего участка смотрит вверх, свес верхнего — вниз)"""
    zm = 0.5 * (z_lo + z_hi)
    try:
        A, B = Polygon(lo.exterior), Polygon(up.exterior)
        c = mesh.triangles_center; nz = mesh.face_normals[:, 2]; w = mesh.area_faces
        zsel = (c[:, 2] > z_lo - 0.05) & (c[:, 2] < z_hi + 0.05)
        best = None
        for reg, sgn in ((A.difference(B).buffer(-0.3), 1), (B.difference(A).buffer(-0.3), -1)):
            if reg.is_empty: continue
            sel = zsel & (sgn * nz > 0.9)
            sel[sel] = shapely.contains_xy(reg, c[sel, 0], c[sel, 1])
            if w[sel].sum() > (best[0] if best else 0):
                best = (w[sel].sum(), float(np.average(c[sel, 2], weights=w[sel])))
        if best and best[0] > 1.0:
            return float(np.clip(best[1], z_lo, z_hi))
    except Exception:  # noqa: BLE001
        pass
    return zm


def build2(mesh, dz=0.2, n=int(os.environ.get('RING_N', '32')), smooth=float(os.environ.get('RING_SMOOTH', '0.05')), *,
           periodic_wall=False, void_n=None, fit_upper_wall=False, exact_islands=False):
    """наружная стенка — один лофт по всей высоте; внутренние объёмы (карманы, пазы) вычитаются по слоям"""
    secs = sections(mesh, dz)
    secs = [(z, p) for z, p in secs if p]
    # наружный контур: самый большой многоугольник каждого уровня
    allb = []
    for z, polys in secs:
        big = max(polys, key=lambda p: Polygon(p.exterior).area)
        allb.append((z, big, Polygon(big.exterior).area))
    # прежний наружный контур: от дна, пока он не распался
    i1 = 0
    while i1 < len(allb) and not (i1 and allb[i1][2] < 0.7 * allb[i1 - 1][2]):
        i1 += 1
    # нижние срезы-обрезки (касание дна) дают скачок формы — дно начинаем с первого полного среза
    i0 = 0
    while i1 - i0 > 2 and allb[i0][2] < 0.6 * allb[i0 + 1][2]:
        i0 += 1
    pads = allb[:i0]
    TOPINFO["pads"] = pads
    segr = [(i0, i1)]
    # WALL_SEGMENTS=0 — выключатель: стенка только до первого скачка формы, как раньше
    if (os.environ.get("WALL_SEGMENTS", "1") == "1" and not (periodic_wall or fit_upper_wall)
            and os.environ.get("BOTTOM_ROUND", "0") == "0" and os.environ.get("BOTTOM_DOME", "0") != "1"
            and os.environ.get("TOP_RIM", "0") != "1"):
        segr = _outer_segments(allb, i0, i1, dz)
    while True:
        segs = [allb[a:b] for a, b in segr]
        multi = len(segs) > 1
        splits = [None] + [_split_z(mesh, allb[b - 1][1], allb[b][1], allb[b - 1][0], allb[b][0]) for _, b in segr[:-1]]
        walls = []; seg_info = []
        try:
            for si, outer in enumerate(segs):
                last = si == len(segs) - 1
                if multi and len(outer) > 4:
                    # срез у самого уступа касается полки и шумит — контур участка берём без него
                    outer = outer[(1 if si else 0):(len(outer) - (0 if last else 1))]
                zs = [o[0] for o in outer]
                # RING_ADAPT > 0: точки кольца сгущаются на скруглениях углов (0 — равномерно по длине, как раньше)
                adapt = float(os.environ.get("RING_ADAPT", "0"))
                # большой контур: точек кольца по длине, не меньше одной на WALL_PT_MM мм (32 точки на фланец
                # Ø90 — 9 мм между точками, отломанный край и выступы сглаживались)
                per = max(Polygon(o[1].exterior).length for o in outer)
                n_s = max(n, int(per / float(os.environ.get("WALL_PT_MM", "5"))))
                if not multi:
                    TOPINFO["wall_points"] = n_s
                else:
                    # участки стопки: точек кольца по длине контура (не меньше n), гуще на скруглениях
                    n_s = max(n, int(Polygon(outer[len(outer) // 2][1].exterior).length / float(os.environ.get("SEG_RING_STEP", "1.5"))))
                    adapt = max(adapt, float(os.environ.get("SEG_RING_ADAPT", "1.0")))
                rings = [ring_pts(o[1].exterior.coords, n_s, smooth, adapt=adapt) for o in outer]
                if multi:
                    ok_ = [i for i, r in enumerate(rings) if Polygon(r).is_valid]
                    if len(ok_) < 2:
                        raise RuntimeError("кольца участка негодны")
                    outer = [outer[i] for i in ok_]; rings = [rings[i] for i in ok_]; zs = [zs[i] for i in ok_]
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
                if multi:
                    za = splits[si] if si else za
                    zb = splits[si + 1] if not last else zb
                lz = np.array([zs[i] for i in keep]); lz[0] = za; lz[-1] = zb
                print("outer levels", len(keep), round(za, 2), round(zb, 2), flush=True)
                coef = top_surface(mesh, rings[-1], zb) if last else None
                if os.environ.get("TOP_RIM", "0") == "1" and last:  # опыт: подгонка борта Севера (задача 50), высота по длине
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
                bcoef = top_surface(mesh, rings[keep[0]], zs[0], down=True) if os.environ.get("BOTTOM_FIT", "1") == "1" and si == 0 else None
                if bcoef is not None:
                    xs, ys = rings[keep[0]][:, 0], rings[keep[0]][:, 1]
                    zz = surf_eval(bcoef, xs, ys)
                    print("bottom z", round(float(zz.min()), 2), round(float(zz.max()), 2), flush=True)
                    levels[0] = (levels[0][0], min(levels[0][1], float(zz.min()) - 0.3))
                if si == 0:
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
                if os.environ.get("BOTTOM_DOME", "0") == "1" and pads and si == 0:
                    # дно-купол: отдельный лофт по срезам низа (pad ниже) срастается со стенкой; стенку вниз не тянем
                    # и плоскостью дна не режем — иначе на дне остаётся уступ по контуру купола
                    TOPINFO["bcoef"] = None; TOPINFO["dome"] = True
                    levels[0] = (levels[0][0], zs[keep[0]] - dz / 2)
                degree = int(os.environ.get("WALL_DEG", "2")) or None
                wpar = os.environ.get("WALL_PAR", "1") == "1"  # общая параметризация колец стенки: меньше полюсов, ровнее грань
                if fit_upper_wall and coef is not None and last:
                    from local_surfaces import upper_wall

                    extended = upper_wall(mesh,levels,rings[-1],zs[-1],ztop)
                    if extended is not None:
                        try:
                            wall = loft(extended,deg=degree,par=wpar)
                            levels = None
                            TOPINFO["upper_wall_fit"] = True
                        except RuntimeError:
                            pass
                const = None
                if multi:
                    # участок, у которого контур по высоте не меняется, — прямое выдавливание (как в CAD)
                    const = _const_ring(rings, [Polygon(o[1].exterior) for o in outer])
                try:
                    if const is not None:
                        z0e, z1e = levels[0][1], levels[-1][1]
                        wall = prism(const, z0e, z1e)
                        wall = wall if isinstance(wall, cq.Shape) else cq.Workplane().add(wall).val()
                    elif levels is not None:
                        wall = loft(levels, deg=degree, par=wpar)
                except RuntimeError:
                    if (os.environ.get("CAD_KOMPAS_END_RECOVERY", "0") != "1"
                            or len(levels)<4 or TOPINFO.get("bottom_round")):
                        raise
                    # End sections may cut the curved bottom or the rising rim instead
                    # of the wall. Extend the neighboring full profiles to the trim tools.
                    recovery = [(levels[1][0],levels[0][1]),*levels[1:-1],
                                (levels[-2][0],levels[-1][1])]
                    wall = loft(recovery, deg=degree, par=wpar)
                    TOPINFO["wall_end_recovery"] = True
                if multi:
                    # меняющийся участок — свой лофт; соседние участки стыкуются по плоскости уступа
                    kind = "ext" if const is not None else "loft"
                    print("wall segment", si, kind, round(za, 2), round(zb, 2), "ring", n_s, flush=True)
                    seg_info.append((za, zb, rings[-1], rings[0], kind))
                walls.append(wall)
            if multi:
                wall = walls[0].fuse(*walls[1:], glue=True).clean()
                if not wall.isValid() or len(wall.Solids()) != 1:
                    raise RuntimeError("стопка участков стенки не срослась")
                TOPINFO["wall_segments"] = [(round(a, 2), round(b, 2), k) for a, b, _, _, k in seg_info]
                TOPINFO["seg_rings"] = [(a, b, r1, r0) for a, b, r1, r0, _ in seg_info]
            break
        except Exception as e:  # noqa: BLE001
            if not multi:
                raise
            # стопка не построилась — как раньше, одна стенка до первого скачка
            print("wall segments fail", str(e)[:60], flush=True)
            for key in ("wall_segments", "seg_rings", "bottom_round", "periodic_wall", "upper_wall_fit", "wall_end_recovery"):
                TOPINFO.pop(key, None)
            segr = [(i0, i1)]
    body = cq.Workplane().add(wall)
    TOPINFO.update(coef=coef, ztop=ztop, ring=rings[-1])
    # пустоты внутри наружного контура по слоям
    voids = []; top_v = []; RAW_VOIDS.clear()
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
        raw = Polygon(big.exterior).difference(shapely.union_all(polys))
        RAW_VOIDS[round(z, 3)] = raw
        if os.environ.get("RAW_SMALL_HOLES", "1") == "1":
            # для продолжения канавок вниз нужны и мелкие дырки сечения (меньше MIN_AREA = 1 мм²):
            # острие V-канавки у бобышки в сечении — полоска 0,05–0,6 мм², sections() её отбрасывает
            s_ = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if s_ is not None:
                P_, _ = s_.to_2D(to_2D=np.eye(4))
                full = shapely.union_all([q.buffer(0) for q in P_.polygons_full if q.area >= MIN_AREA])
                RAW_VOIDS[round(z, 3)] = Polygon(big.exterior).difference(full)
        v = raw.buffer(-VOID_OPEN).buffer(VOID_OPEN)
        geoms = [g for g in getattr(v, "geoms", [v]) if not g.is_empty and g.area >= VOID_MIN]
        voids.append((z, geoms))
    if multi:
        # срезы у самого уступа между участками касаются горизонтальной полки: их контур шумит, и
        # разность «контур минус сечение» даёт крошки-«пустоты» вдоль полки. Здесь оставляем только
        # пустоты, которые продолжают пустоты соседних (не пограничных) уровней — отверстия, карманы
        bz = {round(allb[b][0], 3) for _, b in segr[:-1]} | {round(allb[b - 1][0], 3) for _, b in segr[:-1]}
        for idx, (z, geoms) in enumerate(voids):
            if round(z, 3) not in bz:
                continue
            nb_ = [g for j in (idx - 1, idx + 1) if 0 <= j < len(voids) and round(voids[j][0], 3) not in bz
                   for g in voids[j][1]]
            ref_ = unary_union(nb_) if nb_ else Polygon()
            keep_ = []
            for g in geoms:
                if g.intersection(ref_).area <= 0.5 * g.area:
                    continue
                # и сама пустота — не шире соседних: крошки у полки, приросшие к отверстию, срезаем
                q = g.intersection(ref_.buffer(0.1))
                q = max(getattr(q, "geoms", [q]), key=lambda t: t.area) if not q.is_empty else g
                keep_.append(q if q.geom_type == "Polygon" else g)
            voids[idx] = (z, keep_)
    if os.environ.get("VOIDS", "chains") == "chains":
        # углубление выше верха стенки продолжает цепочку кармана, с которым сливается: один лофт,
        # стенка чаши — наклонная грань (отдельный вырез рядом с карманом ломал вычитание)
        cut = void_chains(voids + [(z, [g]) for z, g in _top_open_levels(top_v)], dz, (void_n or n), zb)
        print("void chains", len(cut), flush=True)
    else:
        cut = void_layers(voids, dz, n, zb)
    pieces = [body.val()]
    isl = islands(secs, zb, dz, n, exact=exact_islands)
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
                    lv.append((ring_pts(segs[0][0][1].exterior.coords, n, 0.03), segs[0][0][0] + 0.3))
                    sol = loft(lv)
                    bb = sol.BoundingBox()
                    if sol.isValid() and sol.Volume() > 0 and bb.zmin > pd[0][0] - dz / 2 - 0.3:
                        isl.append(sol); done = True
                except Exception as e:
                    print("pad loft fail", str(e)[:50], flush=True)
            if not done:
                isl.append(prism(ring_pts(big.exterior.coords, n, 0.03), min(pz) - dz / 2, segs[0][0][0] + 0.3))
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


RAW_VOIDS = {}


def _extend_down(it, dz, max_levels=6, min_area=0.03):
    """отделение, сужающееся книзу (канавка клином у бобышки, конец сверла): ниже первого сечения
    пустота есть, но тоньше порога и отброшена. Продолжаем цепочку вниз по сырым сечениям, пока
    пустота под ней сужается, — дно выреза выходит клином, а не плоской ступенью выше настоящего"""
    zs = sorted(RAW_VOIDS)
    out = list(it)
    g0 = it[0][1]
    if g0.area > float(os.environ.get("VOID_TIP_AREA", "40")) or 2 * g0.area / max(g0.length, 1e-9) > 0.8:
        return out  # только узкие канавки: у больших отделений пологий край дна — не клин
    for _ in range(max_levels):
        z, g = out[0]
        zl = round(z - dz, 3)
        k = min(zs, key=lambda q: abs(q - zl)) if zs else None
        if k is None or abs(k - zl) > dz / 4:
            break
        raw = RAW_VOIDS[k].intersection(g.buffer(0.1))
        parts = [q for q in getattr(raw, "geoms", [raw]) if q.geom_type == "Polygon" and q.area >= min_area]
        if not parts:
            break
        q = max(parts, key=lambda q: q.area)
        if q.area > 0.8 * g.area:
            break
        out.insert(0, (k, q))
    return out


FLICKER_ZONES = []  # места в плане, где сечения скана доказанно мерцают (см. _deflicker_voids)


def _flicker_zone(g):
    return any(z.intersects(g) for z in FLICKER_ZONES)


def _deflicker_voids(voids, jump, w, min_events=int(os.environ.get("VOID_FLICKER_MIN", "3"))):
    """мерцание формы пустоты от уровня к уровню (шумный скан: сечение то замыкает паз хордой поперёк устья,
    то делит отверстие на куски, то оставляет внутри крошку-«остров») — не ступень. Пустота p уровня
    сравнивается со следующими: если на уровне i то, что продолжает p, отличается от неё больше jump
    (по Хаусдорфу), а не дальше w уровней выше снова идёт одна пустота, близкая к p, — уровни между
    ними шум: их продолжения p заменяются на p. Настоящая ступень (форма не возвращается) не тронута.
    Шум повторяется: правим только места, где таких возвратов не меньше min_events (одиночный короткий
    уступ — может быть настоящим). Эти места запоминаются в FLICKER_ZONES: там же не верим «островам»,
    добавкам-поправкам и тонким рёбрам, висящим в пустоте скана"""
    def over(a, b):
        return a.intersection(b).area > 0.3 * min(a.area, b.area)
    simp = {}
    def hd(a, b):
        # по упрощённым (0,02 мм) контурам: у сечения скана тысячи вершин, точное — 0,2 с на пару
        for q in (a, b):
            if id(q) not in simp: simp[id(q)] = (q, q.simplify(0.02))
        return simp[id(a)][1].hausdorff_distance(simp[id(b)][1])
    def run(allowed):
        out = [(z, list(gs)) for z, gs in voids]
        events = []; nfix = 0
        for i in range(1, len(out)):
            prev = out[i - 1][1]
            for p in prev:
                hits = [g for g in out[i][1] if over(g, p)]
                if not hits or (len(hits) == 1 and hd(p, hits[0]) <= jump):
                    continue
                # продолжение p не должно продолжать и соседнюю пустоту: слияние двух карманов — не шум
                if any(over(g, o) for g in hits for o in prev if o is not p):
                    continue
                ret = None
                for k in range(1, w + 1):
                    if i + k >= len(out):
                        break
                    qs = [q for q in out[i + k][1] if over(q, p)]
                    if len(qs) == 1 and hd(p, qs[0]) <= jump:
                        ret = k
                        break
                if ret is None:
                    continue
                events.append(p)
                if allowed is None or not any(over(p, a) for a in allowed):
                    continue
                for j in range(i, i + ret):
                    hs = [g for g in out[j][1] if over(g, p)]
                    out[j] = (out[j][0], [g for g in out[j][1] if not any(g is h for h in hs)] + [p])
                nfix += ret
        return out, events, nfix
    # проход 1: где мерцает; места — группы перекрывающихся пустот p, у которых случился возврат
    _, events, _ = run(None)
    zones = []
    for p in (q.convex_hull for q in events):  # место — выпуклая оболочка: сравнение быстрое, контур не нужен
        hit = [zn for zn in zones if over(zn["g"], p)]
        if hit:
            zn = hit[0]
            for o in hit[1:]:
                zn["g"] = zn["g"].union(o["g"]).convex_hull; zn["n"] += o["n"]; zones.remove(o)
            zn["g"] = zn["g"].union(p).convex_hull; zn["n"] += 1
        else:
            zones.append({"g": p, "n": 1})
    allowed = [zn["g"] for zn in zones if zn["n"] >= min_events]
    FLICKER_ZONES[:] = [a.buffer(0.5) for a in allowed]
    if not allowed:
        if events:
            print("void flicker below threshold", len(events), flush=True)
        return voids
    out, _, nfix = run(allowed)
    print("void deflicker levels", nfix, "zones", len(allowed), flush=True)
    return out


def _const_void(polys, jump, tol=float(os.environ.get("VOID_CONST", "0.06"))):
    """контур выреза, если он по высоте не меняется (сквозное отверстие, паз во всю длину), иначе None.
    Опорный контур — уровень, ближе всех к остальным (медоида по выборке уровней); постоянный, если у 90 %
    уровней средний зазор до него (площадь симметричной разности / периметр) меньше tol и ни один не
    дальше jump по Хаусдорфу. Такой вырез — прямое выдавливание, а не лофт по шумным кольцам"""
    P = [Polygon(g.exterior) for g in polys]
    S = P[::max(1, len(P) // 25)]
    ref = min(S, key=lambda c: float(np.median([c.symmetric_difference(q).area / c.length for q in S])))
    w = [ref.symmetric_difference(q).area / ref.length for q in P]
    rs = ref.simplify(0.02)
    if float(np.percentile(w, 90)) >= tol or max(rs.hausdorff_distance(q.simplify(0.02)) for q in P) > jump:
        return None
    return ref


def _spline_poly(r, m=1500):
    """замкнутый сплайн через точки кольца (как у prism/loft), снятый густо в многоугольник"""
    e = cq.Edge.makeSpline([cq.Vector(float(x), float(y), 0.0) for x, y in r], periodic=True)
    return Polygon([(q.x, q.y) for q in (e.positionAt(t) for t in np.linspace(0, 1, m, endpoint=False))])


def _const_ring_pts(ref, n):
    """кольцо прямого выреза по контуру ref. Острые углы (паз у отверстия) при n равных шагах дают петлю
    сплайна — тело выреза формально цело, а вычитание ломается (вместо отверстия тело прибавляет объём).
    Пробуем гуще и с точками по кривизне; берём самое редкое кольцо без петли, не хуже лучшего более чем
    на 20 % по среднему зазору до контура"""
    P = Polygon(ref.exterior); got = []
    for m, ad in ((n, 0.0), (n, 1.0), (int(1.5 * n), 1.0), (2 * n, 0.0), (2 * n, 1.0), (3 * n, 0.0)):
        try:
            r = ring_pts(ref.exterior.coords, m, 0.03, adapt=ad)
            S = _spline_poly(r)
        except Exception:  # noqa: BLE001
            continue
        if Polygon(r).is_valid and S.is_valid:
            got.append((S.symmetric_difference(P).area / P.length, m, r))
    if not got:
        return None
    best = min(g[0] for g in got)
    return min((g for g in got if g[0] <= 1.2 * best + 1e-3), key=lambda g: g[1])[2]


def void_chains(voids, dz, n, zb, jump=0.5):
    """каждое отделение кармана — своя цепочка контуров по всей высоте и один лофт.
    Отделение, которое выше сливается с большим (начались рёбра), продолжается
    вверх до конца того, с чем слилось: его стенки не обрываются ступенькой."""
    FLICKER_ZONES.clear()
    if os.environ.get("VOID_DEFLICKER", "1") == "1":
        voids = _deflicker_voids(voids, float(os.environ.get("VOID_JUMP", "2.0")), int(os.environ.get("VOID_FLICKER_W", "4")))
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
    taper = float(os.environ.get("VOID_JOIN_TAPER", "0"))
    joins = {}  # id(первого элемента сегмента) -> (есть вырез ниже, есть вырез выше)
    for k, c in enumerate(chains):
        # скачок формы внутри цепочки (верх ребра, уступ) — настоящая ступень: делим цепочку
        cur = [c["items"][0]]
        # сосед снизу: то, из чего отделение выросло (контуры уровнем ниже); сосед сверху: во что влилось
        below = [chains[j]["items"][-1][1] for j in c.get("from") or []]
        below = unary_union(below) if below else None
        for z, g in c["items"][1:]:
            if _hd(cur[-1][1], g) > float(os.environ.get("VOID_JUMP", "2.0")):
                segs.append((cur, cur[-1][0] + dz / 2)); joins[id(cur)] = (below, g)
                below = cur[-1][1]; cur = []
            cur.append((z, g))
        joins[id(cur)] = (below, chains[c["into"]]["items"][0][1] if c["into"] is not None else None)
        if taper and c["into"] is not None:
            # стык без уступа (см. ниже): отделение кончается в плоскости стыка, выше его ведёт общий вырез
            segs.append((cur, cur[-1][0] + dz / 2))
        elif os.environ.get("VOID_MERGE_TOP", "1") == "1":
            segs.append((cur, top_of(k) + dz / 2))
        else:
            # отделение, слившееся выше с большим, продолжается только на слой внахлёст: выше его стенки
            # ведёт общая цепочка, а прямая призма до верха давала рядом с её стенками тонкие полоски
            segs.append((cur, cur[-1][0] + dz / 2 + (dz if c["into"] is not None else 0)))
    segs = [(it, z1) + joins.get(id(it), (None, None)) for it, z1 in segs]
    if os.environ.get("VOID_DUP", "1") == "1":
        # однослойный вырез, целиком лежащий в другом вырезе того же уровня (тот же проём, вторая
        # копия контура от шума у плоскости верха), — двойник: его торцы давали узкие полоски
        by_z = {}
        for k, (it, *_r) in enumerate(segs):
            for z, g in it:
                by_z.setdefault(round(z, 3), []).append((k, g))
        keep_ = []
        for k, sg in enumerate(segs):
            it = sg[0]
            if len(it) == 1:
                z, g = it[0]
                if any(j != k and len(segs[j][0]) > 1 and h.buffer(0.15).intersection(g).area >= 0.9 * g.area
                       for j, h in by_z.get(round(z, 3), [])):
                    print("void chain duplicate skipped", round(z, 2), round(g.area, 1), flush=True)
                    continue
            keep_.append(sg)
        segs = keep_
    if os.environ.get("VOID_TIP", "1") == "1" and RAW_VOIDS:
        segs = [(it if (taper and pr is not None) else _extend_down(it, dz), z1, pr, su) for it, z1, pr, su in segs]
    def _taper_ok(g):
        q = g.buffer(-2 * taper)
        return taper > 0 and not q.is_empty and q.geom_type == "Polygon" and q.area > 0.5 * g.area
    def _big(g, ref):
        g = g.buffer(0.01).buffer(-0.02).buffer(0.01)
        if g.geom_type != "Polygon":
            g = max(getattr(g, "geoms", [ref]), key=lambda q: q.area, default=ref)
        return Polygon(g.exterior) if (not g.is_empty and g.area > 0.5 * ref.area) else ref
    def _fill(g):
        return unary_union([Polygon(q.exterior) for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon"])
    def _snap(a, b):
        """контур a, притянутый к контуру b там, где они расходятся меньше чем на taper; где расходятся
        сильнее (настоящий уступ), остаётся a"""
        a = _fill(a); b = _fill(b)
        far_out = b.difference(a.buffer(taper)).buffer(-0.03).buffer(0.03)   # b заметно шире a
        far_in = a.difference(b.buffer(taper)).buffer(-0.03).buffer(0.03)    # a заметно шире b
        g = a.union(b.intersection(a.buffer(taper)).difference(far_out.buffer(1.5 * taper)))
        g = g.intersection(b.union(far_in.buffer(1.5 * taper)))
        return _big(g, a)
    def _tucked(u, low):
        """низ верхнего выреза под стыком: сужен на taper + 0,03 там, где рядом стенка нижнего выреза,
        и не тронут над настоящим уступом"""
        u = _fill(u)
        ledge = u.difference(_fill(low).buffer(taper)).buffer(-0.03).buffer(0.03)
        return _big(u.buffer(-(taper + 0.03)).union(ledge), u)
    def _bulge(sol, lv, tol=0.25):
        """гладкий лофт выгнуло в плане за свои кольца (близкие уровни с разной расстановкой точек):
        такой вырез прорезает соседнюю стенку"""
        if not taper: return False  # прежнее поведение без стыков
        bb = sol.BoundingBox(); pts = np.vstack([r for r, _ in lv])
        return (bb.xmin < pts[:, 0].min() - tol or bb.xmax > pts[:, 0].max() + tol
                or bb.ymin < pts[:, 1].min() - tol or bb.ymax > pts[:, 1].max() + tol)
    cut = []
    for it, z1, pred, succ in segs:
        # Стык вырезов без уступа. Соседние по высоте вырезы одного отделения строятся каждый по своим
        # контурам, и на стыке их стенки расходятся на сотые доли мм: торцы вырезов выходили на стенку
        # узкими полосками-гранями (268 из 622 граней кнопки). Теперь нижний вырез к плоскости стыка
        # подходит к контуру верхнего (там, где они ближе taper), а выше стыка уходит внутрь его пустоты;
        # верхний ниже стыка уходит внутрь нижнего. Стенки пересекаются по одной кромке в плоскости стыка,
        # торцы спрятаны в пустоте соседа. Где контуры расходятся сильнее, это настоящий уступ: он остаётся.
        low, up = pred, succ
        pred = low is not None and _taper_ok(it[0][1])
        # вырез, уходящий выше верха детали, кончается в воздухе: стыка сверху у него нет
        succ = up is not None and z1 < zb - 1e-6 and _taper_ok(it[-1][1])
        if succ:
            g_top = _snap(it[-1][1], up)
            r_top = ring_pts(g_top.exterior.coords, n, 0.03)
        if pred:
            # кольцо низа под стыком; кольцо по сложному контуру может перекрутиться — тогда
            # берём сглаженное, а не выйдет — то же кольцо низа, равномерно сдвинутое внутрь
            tk = _tucked(it[0][1], low)
            r_tuck = None
            for sm in (0.03, 0.08):
                r = ring_pts(tk.exterior.coords, n, sm)
                if Polygon(r).is_valid: r_tuck = r; break
            if r_tuck is None:
                r = _inset(ring_pts(_fill(it[0][1]).exterior.coords, n, 0.03), taper + 0.03)
                r_tuck = r if Polygon(r).is_valid else None
            pred = r_tuck is not None
        if succ and not Polygon(r_top).is_valid:
            r_top = ring_pts(g_top.exterior.coords, n, 0.08)
            succ = Polygon(r_top).is_valid and Polygon(_inset(r_top, 0.1)).is_valid
        # шумный скан (вырезов больше VOID_SEQ_MAX): мелкий однослойный срез — шум, не карман
        noisy = len(segs) > int(os.environ.get("VOID_SEQ_MAX", "40"))
        if ((os.environ.get("CAD_KOMPAS_VOID_PERSISTENCE", "0") == "1" or noisy)
                and len(it)==1 and it[0][1].area<12):
            # A small, isolated slice can be caused by floor noise. It
            # must not inherit the height of a later, persistent pocket.
            continue
        if (noisy and len(it) <= int(os.environ.get("VOID_NOISE_LEVELS", "2")) and z1 < zb - 1e-6
                and not pred and not succ):
            # шумный скан: вырез высотой в 1–2 среза (0,2–0,4 мм), не выходящий наверх и ни с чем не
            # связанный — шум сечения, не карман (конструктор таких не делает)
            continue
        z0 = it[0][0] - dz / 2
        if z1 >= zb - 1e-6: z1 = TOPINFO['ztop'] + 0.5
        # попытки: стык с уступами по контуру соседа; стык с равномерно суженным низом (кольца точка
        # в точку, лофт не перекручивается); без стыка, как раньше
        tries = [(pred, succ, 0)] + ([(pred, succ, 1)] if pred else []) + ([(False, False, 0)] if (pred or succ) else [])
        for pred, succ, even in tries:
            try:
                zs_c = [z for z, _ in it]
                rings_c = [ring_pts(g.exterior.coords, n, 0.03) for _, g in it]
                cref = None
                if (os.environ.get("VOID_CONST_PRISM", "1") == "1" and not (pred or succ)
                        and len(it) >= int(os.environ.get("VOID_CONST_MIN", "10"))):
                    cref = _const_void([g for _, g in it], float(os.environ.get("VOID_JUMP", "2.0")))
                if cref is not None:
                    r = _const_ring_pts(cref, n)
                if cref is not None and r is not None:
                    # низ — как у лофта (на 0,1 ниже первого среза): сквозное отверстие проходит дно насквозь
                    sol = prism(r, z0 - 0.1, z1 + 0.02)
                    print("void chain const", round(z0, 2), round(z1, 2), len(it), round(cref.area, 1), flush=True)
                elif len(it) >= 3:
                    kk = pick_levels(zs_c, rings_c, jump=jump, dzmax=1.5, dzmin=0.4)
                    lv = [(rings_c[i], zs_c[i]) for i in kk]
                    lv[0] = (lv[0][0], z0 - 0.02)
                    if len(lv) < 2: lv.append((lv[0][0], z1 + 0.02))
                    else: lv[-1] = (lv[-1][0], z1 + 0.02) if z1 - zs_c[-1] < 0.5 else lv[-1]
                    if lv[-1][1] < z1: lv.append((rings_c[-1], z1 + 0.02))
                    lv[0] = (lv[0][0], z0 - 0.1)
                    if pred:
                        lv[0] = (lv[0][0], z0)
                        lv = [(_inset(lv[0][0], taper + 0.03) if even else r_tuck, z0 - 0.1)] + lv
                    if succ:
                        while len(lv) > 1 and lv[-1][1] > zs_c[-1] + 1e-6: lv.pop()
                        if lv[-1][1] < zs_c[-1] - 1e-6: lv.append((rings_c[-1], zs_c[-1]))
                        lv += [(r_top, z1), (_inset(r_top, 0.1), z1 + 0.1)]
                    tip = 0.0
                    a0, a2 = it[0][1].area, it[2][1].area
                    if os.environ.get("VOID_TIP", "1") == "1" and a0 < 0.85 * a2 and not pred:
                        # отделение сужается книзу (канавка клином, конец сверла): сечения мельче порога
                        # теряются, и дно выходит плоским выше настоящего. Продолжаем сужение вниз до острия
                        w0, w2 = np.sqrt(a0), np.sqrt(a2)
                        tip = min(1.0, 0.8 * w0 * 2 * dz / (w2 - w0))
                        r0 = lv[0][0]; c = r0.mean(0)
                        lv = [(c + 0.25 * (r0 - c), z0 - 0.1 - tip)] + lv
                    sol = loft(lv)
                    bb = sol.BoundingBox()
                    if sol.Volume() <= 0 or bb.zmin < z0 - 0.3 - tip or bb.zmax > lv[-1][1] + 0.3 or _bulge(sol, lv):
                        sol = loft(lv, ruled=True)  # гладкий лофт выгнуло за свои уровни — прямые переходы
                        if sol.Volume() <= 0: raise RuntimeError("negative")
                elif pred or succ:
                    r = rings_c[len(it) // 2]
                    lv = ([(_inset(r, taper + 0.03) if even else r_tuck, z0 - 0.1), (r, z0)] if pred else [(r, z0 - 0.02)]) \
                        + ([(r, z1 - 0.1), (r_top, z1), (_inset(r_top, 0.1), z1 + 0.1)] if succ else [(r, z1 + 0.02)])
                    try:
                        sol = loft(lv)
                        bb = sol.BoundingBox()
                        if sol.Volume() <= 0 or bb.zmin < lv[0][1] - 0.2 or bb.zmax > lv[-1][1] + 0.2 or _bulge(sol, lv):
                            raise RuntimeError("taper loft")
                    except RuntimeError:
                        sol = loft(lv, ruled=True)
                else:
                    sol = prism(rings_c[len(it) // 2], z0 - 0.02, z1 + 0.02)
                sol = sol if isinstance(sol, cq.Shape) else cq.Workplane().add(sol).val()
                vdeg = int(os.environ.get("VOID_DEG", "0"))
                if vdeg and len(it) >= 3 and cref is None:
                    # квадратичный лофт кармана: на его кромках строятся скругления; если вырез им не
                    # удастся, при вычитании берётся прежний лофт (_fallback)
                    try:
                        s2 = loft(lv, deg=vdeg); b2 = s2.BoundingBox()
                        if s2.isValid() and s2.Volume() > 0 and b2.zmin >= z0 - 0.3 and b2.zmax <= lv[-1][1] + 0.3:
                            s2._fallback = sol; sol = s2
                    except Exception:  # noqa: BLE001
                        pass
                if (pred or succ) and not sol.isValid():
                    raise RuntimeError("taper invalid")
                cut.append(sol)
                print("void chain", round(z0, 2), round(z1, 2), len(it), "join" if (pred or succ) else "", flush=True)
                break
            except Exception as e:
                if pred or succ:
                    # стык не построился: этот вырез — как раньше, без стыка
                    if os.environ.get("VOID_DEBUG"):
                        import pickle
                        with open(os.environ["VOID_DEBUG"] + f"_{len(cut)}.pkl", "wb") as fh:
                            pickle.dump(dict(it=it, z1=z1, low=low, up=up, lv=lv, pred=pred, succ=succ), fh)
                    print("void chain join fail", round(z0, 2), str(e)[:40], flush=True)
                    continue
                print("void chain fail", round(z0, 2), str(e)[:50], flush=True)
                try:
                    cut.append(prism(rings_c[len(it) // 2], z0 - 0.02, z1 + 0.02))
                except Exception:
                    pass
    return cut

def _island_flicker(ps, zset, dz, w=int(os.environ.get("VOID_FLICKER_W", "4"))):
    """«остров» в несколько срезов, висящий в пустоте: и под ним, и над ним (в пределах w уровней) на его месте пустота
    (сквозное отверстие, где сечение шумного скана замкнулось внутри крошкой). Настоящая бобышка стоит на
    полу кармана — под ней материал"""
    if len(zset) > w or not RAW_VOIDS:
        return False
    keys = sorted(RAW_VOIDS)
    foot = shapely.union_all([Polygon(q.exterior) for _, q in ps])
    def hollow(zl):
        k = min(keys, key=lambda q: abs(q - zl))
        return abs(k - zl) <= dz / 4 and RAW_VOIDS[k].intersection(foot).area >= 0.9 * foot.area
    # соседние срезы мерцают так же — достаточно пустоты хотя бы на одном из w уровней снизу и сверху
    return (any(hollow(zset[0] - j * dz) for j in range(1, w + 1))
            and any(hollow(zset[-1] + j * dz) for j in range(1, w + 1)))


def islands(secs, zb, dz=0.2, n=32, min_area=float(os.environ.get("ISL_MIN", "0.8")), *, exact=False):
    """бобышки и рёбра внутри кармана и над краем: отдельные многоугольники сечений,
    сгруппированные по перекрытию в плане; каждое — выдавливание своего контура с отверстиями"""
    items = []
    secs = [(z, ps) for z, ps in secs if ps]
    zbot = secs[0][0] - dz / 2 if secs else None
    floor_on = os.environ.get("CAD_KOMPAS_ISL_FLOOR", "1") == "1"
    for z, polys in secs:
        big = max(polys, key=lambda p: Polygon(p.exterior).area)
        for p in polys:
            if (p is big and z <= zb) or p.area < min_area:
                continue
            # крошки шума в самом нижнем срезе (касание дна) — не бобышки
            if floor_on and z == secs[0][0] and p.area < float(os.environ.get("ISL_BOTTOM_CRUMB", "5")):
                continue
            items.append((z, p))
    groups = []
    for z, p in items:
        f = Polygon(p.exterior).buffer(0.3)
        # бобышка — непрерывная по высоте: кусок через разрыв в срезах (шум у дна под втулкой)
        # с ней не сливается
        hit = [g for g in groups if g["foot"].intersects(f) and (not floor_on or z - max(t[0] for t in g["polys"]) <= 1.5 * dz)]
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
    segw = TOPINFO.get("seg_rings")
    for g in groups:
        ps = g["polys"]
        if segw:
            # стенка из участков: остров сравниваем со стенками участков на его высоте
            z0g, z1g = ps[0][0], ps[-1][0]
            rs = [r for a, b, r1, r0 in segw if a - dz < z1g and z0g < b + dz for r in (r0, r1)] or [segw[-1][2]]
            wall = shapely.union_all([Polygon(r).exterior for r in rs])
        if wall is not None and g["foot"].distance(wall) < 0.8:
            print("island at wall skipped", round(g["foot"].area, 1), flush=True); continue
        zset = sorted({z for z, _ in ps})
        if len(zset) < int(os.environ.get("ISL_MIN_LEVELS", "1")): continue
        if os.environ.get("ISL_FLICKER", "1") == "1" and _flicker_zone(g["foot"]) and _island_flicker(ps, zset, dz):
            print("island flicker skipped", round(zset[0], 2), len(zset), round(g["foot"].area, 1), flush=True); continue
        # низ бобышки утапливаем в пол кармана, чтобы она срослась с телом, а не висела отдельно
        za, zb2 = zset[0] - dz / 2 - float(os.environ.get("CAD_SECTIONS_EMBED", "0.6")), zset[-1] + dz / 2
        if floor_on and zbot is not None:
            za = max(za, zbot)  # утапливание в пол не выводит бобышку ниже дна детали
        with_holes = [q for _, q in ps if len(q.interiors)]
        ref = max(with_holes or [q for _, q in ps], key=lambda q: Polygon(q.exterior).area)
        single = len(zset) == len(ps)
        try:
            if exact == "hooked":
                from hooked_islands import hooked_island

                fitted,error = hooked_island(ps,za,zb2,dz)
                if fitted is not None:
                    sol,audit = fitted
                    sol._cylindrical_island = audit
                    solids.append(sol)
                    continue
                print(f"windowed islands: level={error['level']} z={error['z_mm']:.3f} mm: "
                      f"{error['reason']}",flush=True)
            elif exact == "windowed":
                from windowed_islands import windowed_island

                fitted = windowed_island(ps,za,zb2,dz)
                if fitted is not None:
                    sol,audit = fitted
                    sol._cylindrical_island = audit
                    solids.append(sol)
                    continue
            elif exact == "partial":
                from partial_islands import partial_island

                fitted = partial_island(ps,za,zb2,dz,loft,prism,ring_pts,n)
                if fitted is not None:
                    sol,audit = fitted
                    sol._cylindrical_island = audit
                    solids.append(sol)
                    continue
            elif exact:
                from cylindrical_islands import cylindrical_island

                fitted = cylindrical_island(ps,za,zb2)
                if fitted is not None:
                    sol,sol._cylindrical_island = fitted
                    solids.append(sol)
                    continue
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

def cut_above(solid, coef, bbox, margin=5.0, below=False, clip=None):
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
    tool = cq.Solid(hs)
    if clip is not None:
        # clip = (кольцо, z): срез только над (под) этим контуром и выше (ниже) уровня z — уступы
        # нижних участков стопки поверхность верха не трогает
        ring, zc_ = clip
        P = Polygon(ring).buffer(1.0).simplify(0.05)
        z0_, z1_ = (zc_ - 200.0, zc_ - 0.05) if below else (zc_ + 0.05, zc_ + 200.0)
        box = cq.Workplane("XY").workplane(offset=float(z0_)).polyline([tuple(map(float, p)) for p in list(P.exterior.coords)[:-1]]).close().extrude(float(z1_ - z0_)).val()
        try:
            nb = solid.cut(tool.intersect(box))
            if nb.isValid() and 0 < nb.Volume() <= solid.Volume() + 1e-6:
                return nb
        except Exception:  # noqa: BLE001
            pass
        print("clipped cut failed, full cut", flush=True)
    return solid.cut(tool)

T0 = time.time()
if __name__ == "__main__":
    t = time.time()
    m = trimesh.load(sys.argv[1], force="mesh")
    body, info = build(m)
    for i in info: print(i)
    print("valid", body.isValid(), "faces", len(body.Faces()), "solids", len(body.Solids()), round(time.time() - t, 1), "s")
    cq.exporters.export(body, sys.argv[2] + ".step")
    cq.exporters.export(body, sys.argv[2] + ".stl", tolerance=0.01, angularTolerance=0.1)


def correction_prisms(scan, model_mesh, dz=0.2, n=32, extra=False, min_area=float(os.environ.get("CORR_MIN", "0.2")), shrink=0.12,
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
    w = int(os.environ.get("VOID_FLICKER_W", "4"))
    flick = extra and os.environ.get("CORR_ADD_FLICKER", "1") == "1"
    cache = {}
    def hollow(z, g):
        # в скане на этом месте пустота (на уровне z)
        k = round(z, 3)
        if k not in cache: cache[k] = sec(scan, z) if z0 < z < z1 else None
        return cache[k] is not None and cache[k].intersection(g).area < 0.1 * g.area
    for L in layers_:
        if len(L["all"]) < 2: continue
        g = shapely.intersection_all(L["all"]) if not extra else L["all"][len(L["all"]) // 2]
        if g.is_empty or g.area < min_area: g = L["all"][len(L["all"]) // 2]
        if g.geom_type != "Polygon": g = max(g.geoms, key=lambda q: q.area)
        if (flick and _flicker_zone(g) and round((L["z1"] - L["z0"]) / dz) + 1 <= w
                and any(hollow(L["z0"] - j * dz, g) for j in range(1, w + 1))
                and any(hollow(L["z1"] + j * dz, g) for j in range(1, w + 1))):
            # материал скана в несколько срезов, а и ниже, и выше на том же месте пустота: сечение шумного
            # скана замкнуло отверстие или паз хордой (мерцание, см. _deflicker_voids) — не добавка
            print("correction add flicker skipped", round(L["z0"], 2), len(L["all"]), round(g.area, 1), flush=True)
            continue
        try:
            out.append(prism(ring_pts(g.exterior.coords, n, 0.02), L["z0"] - dz / 2 - 0.02, L["z1"] + dz / 2 + 0.02))
        except Exception:
            pass
    return out


def thin_rib_prisms(scan, model_mesh, dz=0.2, tmax=float(os.environ.get("THIN_RIB_T", "0.45")),
                    min_area=float(os.environ.get("THIN_RIB_MIN", "0.06")), cut=False):
    """тонкое ребро (язычок защёлки, перегородка): по каждому сечению — тонкая часть «скан минус модель»
    (толщина 2A/L < tmax), прилегающая к модели; выдавливание ровно на высоту слоя многоугольником с
    прямыми сторонами, без сплайна по 32 точкам (сплайн и сглаживание 0,12 мм съедали полоску 0,2 мм)"""
    z0 = max(scan.bounds[0, 2], model_mesh.bounds[0, 2]); z1 = min(scan.bounds[1, 2], model_mesh.bounds[1, 2])
    def sec(mesh, z):
        s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is None: return Polygon()
        P, _ = s.to_2D(to_2D=np.eye(4))
        return shapely.union_all([p.buffer(0) for p in P.polygons_full])
    out = []; found = []
    for z in np.arange(z0 + dz, z1 - dz / 2, dz):
        a, b = sec(model_mesh, z), sec(scan, z)
        if a.is_empty or b.is_empty: continue
        # cut=True — наоборот: тонкий клин материала модели там, где в скане пусто (острие V-канавки)
        D = (a.difference(b) if cut else b.difference(a)).buffer(-0.01).buffer(0.01)
        ext = max(getattr(a, "geoms", [a]), key=lambda q: q.area).exterior
        ref = b.boundary if cut else a.boundary
        for g in getattr(D, "geoms", [D]):
            if g.is_empty or g.geom_type != "Polygon" or g.area < min_area: continue
            if 2 * g.area / g.length > tmax: continue          # толстое — дело CORR_ADD
            if g.distance(ref) > 0.03: continue                  # висит в воздухе — шум скана
            if g.distance(ext) < 0.05: continue                  # корка по наружной стенке — не ребро
            from shapely.geometry import Point
            if cut:
                # дно выреза модели здесь, а скан уходит глубже: слоем выше у модели уже пусто
                if sec(model_mesh, z + dz).intersection(g).area > 0.3 * g.area: continue
                # и скан пуст ещё слоем ниже: канавка глубже дна модели больше чем на допуск
                if sec(model_mesh, z - dz).difference(sec(scan, z - dz)).intersection(g.buffer(0.05)).area < 0.02: continue
            elif max(ref.distance(Point(c)) for c in g.exterior.coords) < 0.3: continue  # и так в допуске
            L = g.length / 2
            if L < 0.5: continue                                 # короче 0,5 мм — не ребро
            # под полоской слоем ниже сплошной материал модели: возможно, ступенька пологого дна
            floor_ = sec(model_mesh, z - dz).intersection(g).area > 0.7 * g.area
            found.append((float(z), g.buffer(0.02, join_style=2), floor_))
    if os.environ.get("THIN_RIB_LOFT", "1") == "1" and not cut:
        # одно ребро на язычок: сечения соседних слоёв, которые перекрываются, — одна цепочка и один
        # лофт по ним (как операция «по сечениям» в Компасе), а не стопка брусков высотой в слой
        groups = []; floors = {}
        for z, g, fl in found:
            for G in groups:
                if abs(G[-1][0] - (z - dz)) < 1e-6 and G[-1][1].buffer(0.1).intersects(g):
                    G.append((z, g)); floors[id(G)] = False; break
            else:
                groups.append([(z, g)]); floors[id(groups[-1])] = fl
        if os.environ.get("THIN_RIB_FLOOR", "1") == "1":
            # одна полоска в один слой прямо на материале модели — ступенька пологого дна (скан чуть
            # выше гладкого дна), а не ребро: добавка рисовала на дне извилистые линии-террасы
            groups = [G for G in groups if not floors[id(G)]]
        if os.environ.get("CORR_ADD_FLICKER", "1") == "1":
            # полоска, висящая в пустоте скана (ниже и выше на её месте пусто в пределах w слоёв), —
            # мерцание сечения шумного скана у паза или отверстия, не ребро (см. _deflicker_voids)
            w = int(os.environ.get("VOID_FLICKER_W", "4"))
            def hollow(z, g):
                return z0 < z < z1 and sec(scan, z).intersection(g).area < 0.1 * g.area
            keep_ = []
            for G in groups:
                u = shapely.union_all([q for _, q in G])
                if (len(G) <= w and _flicker_zone(u) and any(hollow(G[0][0] - j * dz, u) for j in range(1, w + 1))
                        and any(hollow(G[-1][0] + j * dz, u) for j in range(1, w + 1))):
                    print("thin rib flicker skipped", round(G[0][0], 2), len(G), flush=True)
                    continue
                keep_.append(G)
            groups = keep_
        for G in groups:
            if os.environ.get("THIN_RIB_UNION", "1") == "1":
                # как конструктор: язычок — одно выдавливание по общему контуру его сечений на всю
                # высоту, одна боковая грань вместо лофта по слоям
                try:
                    u = shapely.union_all([q for _, q in G]).buffer(0.01).buffer(-0.01)
                    if u.geom_type != "Polygon":
                        u = max(u.geoms, key=lambda q: q.area)
                    # гладкий контур сплайном: одна боковая грань (многоугольник давал грань на каждую сторону)
                    sol = prism(ring_pts(u.exterior.coords, 24, 0.01), G[0][0] - dz / 2 - 0.01, G[-1][0] + dz / 2 + 0.01)
                    sol = sol if isinstance(sol, cq.Shape) else cq.Workplane().add(sol).val()
                    if sol.isValid() and sol.Volume() > 0:
                        out.append(sol); continue
                except Exception:  # noqa: BLE001
                    pass
            try:
                if len(G) < 2:
                    raise ValueError("one level")
                rings = [(ring_pts(q.exterior.coords, 24, 0.01), z) for z, q in G]
                lv = [(rings[0][0], G[0][0] - dz / 2 - 0.01)] + rings + [(rings[-1][0], G[-1][0] + dz / 2 + 0.01)]
                sol = loft(lv, ruled=True)
                if not sol.isValid() or sol.Volume() <= 0:
                    raise ValueError("bad loft")
                out.append(sol)
            except Exception:  # noqa: BLE001
                for z, q in G:
                    try:
                        out.append(prism_poly(q, z - dz / 2 - 0.01, z + dz / 2 + 0.01, tol=0.02))
                    except Exception:  # noqa: BLE001
                        pass
        return out
    for z, q, _ in found:
        try:
            out.append(prism_poly(q, z - dz / 2 - 0.01, z + dz / 2 + 0.01, tol=0.02))
        except Exception:  # noqa: BLE001
            pass
    return out


def _nz(f):
    try:
        return f.normalAt().z
    except Exception:  # noqa: BLE001
        return 0.0


def _edge_fillets(body, info, scan=None, *, fit_variable=False):
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
        if fit_variable and scan is not None and len(edges)==1:
            from variable_fillet import variable_fillet

            fitted = variable_fillet(scan,body,edges[0])
            if fitted is not None and _fillet_sane(fitted[0], body):
                body, law = fitted
                info[key.lower()] = round(float(np.median(law[:,1])),3)
                info[key.lower()+"_law"] = law.tolist()
                continue
        for r in (rf, rf * 0.7, rf * 0.5):
            try:
                # скругление на сложной кромке ядро может считать десятки минут — предел времени
                nb, ok = op_limited(lambda: body.fillet(r, edges), float(os.environ.get("FILLET_SEC", "90")))
                if ok and _fillet_sane(nb, body):
                    body = nb; info[key.lower()] = round(r, 3)
                    break
            except RuntimeError as e:
                if "timeout" in str(e):
                    info[key.lower() + "_timeout"] = True
                    break  # меньший радиус на той же кромке будет так же долго
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
    filled = 0
    if os.environ.get("CAD_KOMPAS_FILL_HOLES", "1") == "1" and not scan.is_watertight:
        from scan_fill import fill_loops

        scan, filled = fill_loops(scan)
        log(f"    дыры скана закрыты: {filled} треугольников, замкнут: {scan.is_watertight}")
    body,info = _assemble_candidates(scan,log)
    if filled: info["scan_holes_filled"] = filled
    if info.get("solo_build"):
        return body, info
    partial = os.environ.get("CAD_KOMPAS_PARTIAL_ISLANDS", "0") == "1"
    windowed = os.environ.get("CAD_KOMPAS_WINDOWED_ISLANDS", "0") == "1"
    hooked = os.environ.get("CAD_KOMPAS_HOOKED_ISLANDS", "1") == "1"
    if hooked or windowed or partial or os.environ.get("CAD_KOMPAS_EXACT_ISLANDS", "0") == "1":
        from cylindrical_islands import guarded_exact_islands

        mode = "hooked" if hooked else "windowed" if windowed else "partial" if partial else True
        body,info = guarded_exact_islands(scan,body,info,log,_assemble_stock,mode=mode)
    curved = os.environ.get("CAD_KOMPAS_CURVED_PADS", "0") == "1"
    if curved or os.environ.get("CAD_KOMPAS_LEVEL_PADS", "0") == "1":
        from level_pads import guarded_level_pads

        body,info = guarded_level_pads(scan,body,info,log,curved=curved)
    if os.environ.get("CAD_KOMPAS_POCKET_STEPS","0") == "1":
        from pocket_steps import clean_pocket_steps

        body,audit = clean_pocket_steps(body,scan=scan)
        info = {**info,"pocket_steps":audit}
        if audit["removed"]:
            info["faces"] = audit["faces"]
        log(f"    pocket steps: {audit}")
    return body,info


def _assemble_candidates(scan, log):
    """Build 0с and validate an opt-in candidate against the stock result."""
    candidate = None
    experiment = None
    if os.environ.get("CAD_KOMPAS_STABLE_POCKETS", "0") == "1":
        from stable_pockets import assemble_stable

        candidate = assemble_stable(scan, log=log)
        experiment = "stable_pockets"
    t_base = time.time()
    baseline = _assemble_stock(scan, log=log)
    # большая или сложная деталь: каждый кандидат — ещё одна полная сборка; если основная шла долго,
    # кандидатов не строим (иначе 3–4 сборки не укладываются в предел времени)
    if candidate is None and time.time() - t_base > float(os.environ.get("CAD_KOMPAS_SOLO_SEC", "200")):
        log(f"    сборка долгая ({time.time() - t_base:.0f} с): без проверки кандидатов")
        baseline[1]["solo_build"] = True
        return baseline
    upper = os.environ.get("CAD_KOMPAS_UPPER_WALL", "0") == "1"
    floor = os.environ.get("CAD_KOMPAS_POCKET_FLOOR", "1") == "1"
    variable = os.environ.get("CAD_KOMPAS_VARIABLE_FILLET", "0") == "1"
    if candidate is None and (upper or floor or variable):
        experiment = "local_surfaces"
        try:
            # вместе с густыми контурами кармана (48 точек), иначе опытный проход вытесняет кандидата fine_voids
            fine_n = int(os.environ.get("CAD_KOMPAS_FINE_VOIDS", "48")) or None
            trial = _assemble_stock(scan, log=log, fit_upper_wall=upper, fit_pocket_floor=floor, fit_variable=variable,
                                    void_n=fine_n)
            if any(trial[1].get(key) for key in ("upper_wall_fit","smooth_pocket_floors",
                                                 "top_fillet_law","bottom_fillet_law")):
                candidate = trial
        except (RuntimeError, ValueError) as error:
            log(f"    local surfaces rollback: {error}")
    if candidate is None and os.environ.get("CAD_KOMPAS_PERIODIC_WALL", "0") == "1":
        experiment = "periodic_wall"
        try:
            trial = _assemble_stock(scan, log=log, periodic_wall=True)
            if trial[1].get("periodic_wall"):
                candidate = trial
        except (RuntimeError, ValueError) as error:
            log(f"    periodic wall rollback: {error}")
    fine = int(os.environ.get("CAD_KOMPAS_FINE_VOIDS", "48"))
    if candidate is None and fine:
        # контуры кармана гуще (меньше «волны» в углах): построение капризнее, поэтому это кандидат,
        # который берётся, только если тело цело и ближе к скану, чем основное
        experiment = "fine_voids"
        try:
            candidate = _assemble_stock(scan, log=log, void_n=fine)
        except (RuntimeError, ValueError) as error:
            log(f"    fine voids rollback: {error}")
    if candidate is None:
        return baseline
    from candidate_guard import compare_candidate

    accepted, audit = compare_candidate(scan, baseline[0], candidate[0])
    if experiment == "local_surfaces" and baseline[1].get("top_fillet") and not candidate[1].get("top_fillet"):
        accepted = False
        audit["reason"] = "top_fillet_lost"
    if accepted and experiment == "local_surfaces" and variable and not (upper or floor):
        gains = np.subtract(audit["candidate"],audit["stock"])
        if min(gains)<-1e-9 or max(gains)<=.0003+1e-9:
            accepted = False
            audit["reason"] = "variable_fillet_gain_below_threshold"
    audit["accepted"] = accepted
    log(f"    {experiment} guard: {audit}")
    body, info = candidate if accepted else baseline
    return body, {**info, experiment+"_guard": audit}


def _assemble_stock(scan, log=print, *, periodic_wall=False, void_n=None,
                    fit_upper_wall=False, fit_pocket_floor=False, fit_variable=False,exact_islands=False):
    """Сборка «как в Компасе» вдоль оси z выровненного скана. Возвращает (тело, сведения).
    Порядок операций: наружная стенка (лофт по сечениям, сглаженным по высоте) → вырез кармана слоями →
    срез верха и низа гладкими поверхностями по скану → бобышки и площадка выдавливанием →
    поправка по сечениям (пазы и канавки вырезом). Если деталь не «стопка» вдоль z, тело выйдет негодным
    или далёким от скана — тогда попытку отбрасывает сравнение со сканом."""
    import tempfile
    TOPINFO.clear()
    info = {}
    t = time.time()
    pieces, cut, isl = build2(scan, periodic_wall=periodic_wall, void_n=void_n,
                             fit_upper_wall=fit_upper_wall,exact_islands=exact_islands)
    if void_n is not None:
        info["void_n"] = void_n
    for key in ("periodic_wall", "wall_end_recovery", "upper_wall_fit"):
        if TOPINFO.get(key):
            info[key] = True
    body = pieces[0]
    if not body.isValid() or body.Volume() <= 0:
        raise RuntimeError("наружная стенка не построена")
    info["voids"] = len(cut)
    if TOPINFO.get("wall_segments"):
        info["wall_segments"] = TOPINFO["wall_segments"]
    skipped = 0
    trace = os.environ.get("CAD_KOMPAS_TRACE", "0") == "1"
    void_checks = []
    if fit_pocket_floor:
        from local_surfaces import floor_samples, trim_pocket

        floor_points = floor_samples(scan)
    limited = len(cut) > int(os.environ.get("VOID_SEQ_MAX", "40"))
    lim_s = float(os.environ.get("VOID_CUT_SEC", "60"))
    cur_vol = [None]
    def _good(nb, body):
        if limited:
            return getattr(nb, "_ok", False)
        return nb.isValid() and nb.Volume() > 0.5 * body.Volume()
    def _cut_try(body, v):
        if limited:
            # большая шумная деталь: у каждого вычитания (с проверкой) предел времени
            if cur_vol[0] is None: cur_vol[0] = body.Volume()
            try:
                nb, ok, vol = cut_limited(body, [v], lim_s, cur_vol[0])
                if ok:
                    nb._ok = True; cur_vol[0] = vol
                    return nb
            except RuntimeError as e:
                print("void cut skipped:", e, flush=True)
            body._ok = True  # без изменений
            return body
        try:
            nb = body.cut(v)
            if nb.isValid() and nb.Volume() > 0.5 * body.Volume():
                return nb
        except Exception:  # noqa: BLE001
            pass
        fb = getattr(v, "_fallback", None)
        return body.cut(fb) if fb is not None else body.cut(v)

    order = sorted(cut, key=lambda c: -c.Volume())
    nseq = int(os.environ.get("VOID_HEAD", "20"))
    budget = time.time() + float(os.environ.get("VOID_BUDGET_SEC", "600"))
    if limited and len(order) > nseq and os.environ.get("VOID_BATCH", "1") == "1":
        # много мелких вырезов (шумный скан, рёбра): по одному каждый вычитается всё дольше (тело растёт
        # гранями) — 160 вырезов крышки не укладывались в 25 минут. Крупные — по одному с проверками,
        # мелкий хвост — одним вычитанием сразу; не вышло — по одному, как раньше
        head, tail = order[:nseq], order[nseq:]
        batch_tail = tail
    else:
        head, batch_tail = order, []
    for v in head:
        if limited and time.time() > budget:
            # предел времени на вырезы исчерпан: оставшиеся мелкие не вычитаем (иначе сборка не кончится)
            skipped += 1; continue
        t_cut = time.time()
        try:
            before = body.Volume() if trace else None
            nb = None
            if fit_pocket_floor:
                try:
                    smooth_tool = trim_pocket(scan,v,samples=floor_points)
                    if smooth_tool is not None:
                        trial = body.cut(smooth_tool)
                        if (trial.isValid() and len(trial.Solids())==1
                                and .5*body.Volume()<trial.Volume()<body.Volume()):
                            nb = trial
                            info["smooth_pocket_floors"] = info.get("smooth_pocket_floors",0)+1
                except (ValueError,RuntimeError):
                    pass
            if nb is None:
                nb = _cut_try(body, v)
            if _good(nb, body):
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
        if time.time() - t_cut > 5:
            print(f"slow cut {time.time() - t_cut:.0f} s, tool {v.Volume():.1f} mm3, z {v.BoundingBox().zmin:.1f}-{v.BoundingBox().zmax:.1f}", flush=True)
    if batch_tail:
        # кусками: один битый вырез не валит весь хвост
        size = int(os.environ.get("VOID_BATCH_SIZE", "12"))
        nb_ok = nb_fail = 0
        for k0 in range(0, len(batch_tail), size):
            if time.time() > budget:
                print(f"void budget over: {len(batch_tail) - k0} small cutters left out", flush=True)
                info["voids_over_budget"] = len(batch_tail) - k0
                break
            chunk = batch_tail[k0:k0 + size]
            t_b = time.time(); ok_b = False
            try:
                if cur_vol[0] is None: cur_vol[0] = body.Volume()
                nb, ok_c, vol = cut_limited(body, chunk, 2 * lim_s, cur_vol[0])
                if ok_c:
                    body = nb; ok_b = True; cur_vol[0] = vol
            except Exception:  # noqa: BLE001
                pass
            if ok_b:
                nb_ok += len(chunk)
                continue
            nb_fail += len(chunk)
            for v in chunk:
                try:
                    nb = _cut_try(body, v)
                    if _good(nb, body):
                        body = nb
                    else:
                        skipped += 1
                except Exception:  # noqa: BLE001
                    skipped += 1
        print(f"void batch {len(batch_tail)} tools: {nb_ok} together, {nb_fail} one by one", flush=True)
        info["void_batch"] = nb_ok
    info["voids_skipped"] = skipped
    if trace:
        info["void_checks"] = void_checks
    bb = pieces[0].BoundingBox()
    if os.environ.get("CAD_KOMPAS_DIAG"):
        print("diag after voids: valid", body.isValid(), "solids", len(body.Solids()), "faces", len(body.Faces()), flush=True)
    for key, below in (("coef", False), ("bcoef", True)):
        if TOPINFO.get(key) is not None:
            clip = None
            if TOPINFO.get("seg_rings"):
                # стенка из участков: верх режем только над верхним участком, дно — только под нижним
                a, b, r1, r0 = TOPINFO["seg_rings"][0 if below else -1]
                clip = (r0 if below else r1, b if below else a)
            nb = cut_above(body, TOPINFO[key], (bb.xmin, bb.ymin, bb.xmax, bb.ymax), below=below, clip=clip)
            if nb.isValid() and nb.Volume() > 0:
                body = nb
                info["bottom_surface" if below else "top_surface"] = True
            if trace:
                info[key+"_check"] = dict(valid=nb.isValid(), volume=nb.Volume(), solids=len(nb.Solids()))
    if os.environ.get("CAD_KOMPAS_DIAG"):
        print("diag after voids+caps: valid", body.isValid(), "solids", len(body.Solids()), "faces", len(body.Faces()), flush=True)
    try:
        body = body.clean()
    except Exception as e:  # noqa: BLE001  ядро не смогло объединить грани (шумный скан) — тело как есть
        print("clean failed:", str(e)[:60], flush=True)
    body = _edge_fillets(body, info, scan,fit_variable=fit_variable)
    added = 0
    for p in isl:
        try:
            try:
                nb = body.fuse(p)
            except Exception:  # noqa: BLE001  Null TopoDS_Shape: слияние целиком не сошлось
                if os.environ.get("ISLAND_FUSE_FIX", "1") != "1":
                    raise
                # сливаем остров двумя слоями по высоте: каждый слой ядро сливает, а clean() потом
                # склеивает разрезанные грани обратно в одну (у втулки с окнами помогло)
                pb = p.BoundingBox(); zm = (pb.zmin + pb.zmax) / 2
                nb = body
                for za, zb in ((pb.zmin - 0.1, zm), (zm, pb.zmax + 0.1)):
                    nb = nb.fuse(p.intersect(cq.Solid.makeBox(pb.xlen + 2, pb.ylen + 2, zb - za,
                                                               cq.Vector(pb.xmin - 1, pb.ymin - 1, za))))
                nb = nb.clean()
                if nb.Volume() - body.Volume() > p.Volume() * 1.02 + 1e-3:
                    raise ValueError("слойное слияние острова добавило лишний объём")
            if not nb.isValid() and os.environ.get("ISLAND_FUSE_FIX", "1") == "1":
                # ядро иногда отдаёт слитое тело с дефектом рёбер; ShapeFix чинит его без изменения объёма
                try:
                    fx = nb.fix()
                    if fx.isValid() and abs(fx.Volume() - nb.Volume()) < 1e-3:
                        nb = fx
                except Exception:  # noqa: BLE001
                    pass
            # пустое или «похудевшее» тело после слияния острова — отказ ядра, а не результат
            if nb.isValid() and len(nb.Solids()) <= len(body.Solids()) and nb.Volume() >= 0.99 * body.Volume():
                body = nb; added += 1
                if hasattr(p,"_cylindrical_island"):
                    info.setdefault("cylindrical_islands",[]).append(p._cylindrical_island)
        except Exception:  # noqa: BLE001
            pass
    info["bosses"] = added
    if os.environ.get("CAD_KOMPAS_DIAG"):
        print("diag after islands: valid", body.isValid(), "faces", len(body.Faces()), flush=True)
    body = body.clean()
    if os.environ.get("CAD_SECTIONS_CORRECT", "1") == "1":
        tf = os.path.join(tempfile.gettempdir(), "_kcorr%d.stl" % os.getpid())
        # сетка для поправки — с проверкой объёма: битая сетка давала ложные вырезы
        from stl_out import write_stl_checked
        mm = _corr_mesh(body, tf, info)
        nc = 0
        for v in sorted((correction_prisms(scan, mm) if mm is not None else []), key=lambda c: -c.Volume()):
            try:
                nb = body.cut(v)
                if len(nb.Faces()) - len(body.Faces()) > int(os.environ.get("CORR_ADD_MAXF", "40")):
                    # поправочный вырез «лесенкой» (сотни мелких граней) — не так строит конструктор
                    info["correction_cuts_stairs_skipped"] = info.get("correction_cuts_stairs_skipped", 0) + 1
                elif nb.isValid() and nb.Volume() > 0.9 * body.Volume():
                    body = nb; nc += 1
            except Exception:  # noqa: BLE001
                pass
        info["correction_cuts"] = nc
        if os.environ.get("CORR_ADD", "1") == "1":
            # обратная поправка: где в скане материал есть, а в модели нет (язычок защёлки, тонкая
            # перегородка) — выдавливание по сечениям скана
            mm = _corr_mesh(body, tf, info)
            na = 0
            outer = Polygon(TOPINFO["ring"]).buffer(0.05) if TOPINFO.get("ring") is not None else None
            if TOPINFO.get("seg_rings"):
                outer = shapely.union_all([Polygon(r) for s_ in TOPINFO["seg_rings"] for r in s_[2:]]).buffer(0.05)
            # наружная стенка в плане: у стопки участков — стенки всех участков (и уступов между ними)
            wall_line = (shapely.union_all([Polygon(r).exterior for s_ in TOPINFO["seg_rings"] for r in s_[2:]])
                         if TOPINFO.get("seg_rings") else outer.exterior if outer is not None else None)
            for v in sorted((correction_prisms(scan, mm, extra=True) if mm is not None else []), key=lambda c: -c.Volume()):
                # прибавка снаружи стенки — ступенчатый нарост из призм по слоям (видно глазом);
                # наружную стенку правит лофт, поправка только внутри
                if (outer is not None and os.environ.get("CORR_ADD_OUTER", "0") != "1"
                        and any(not outer.contains(shapely.geometry.Point(q.X, q.Y)) for q in v.Vertices())):
                    info["correction_adds_outer_skipped"] = info.get("correction_adds_outer_skipped", 0) + 1
                    continue
                # прибавка, прилипшая к наружной стенке изнутри, — тот же нарост ступеньками на стенке
                if (outer is not None and os.environ.get("CORR_ADD_WALL", "0") != "1"
                        and min(wall_line.distance(shapely.geometry.Point(q.X, q.Y)) for q in v.Vertices()) < 0.3):
                    info["correction_adds_wall_skipped"] = info.get("correction_adds_wall_skipped", 0) + 1
                    continue
                try:
                    nb = body.fuse(v)
                    # прибавка «лесенкой» (сотни мелких граней ради сотых мм) — не так строит конструктор
                    many = len(nb.Faces()) - len(body.Faces()) > int(os.environ.get("CORR_ADD_MAXF", "40"))
                    if many:
                        info["correction_adds_stairs_skipped"] = info.get("correction_adds_stairs_skipped", 0) + 1
                    elif nb.isValid() and len(nb.Solids()) == 1 and nb.Volume() < 1.05 * body.Volume():
                        body = nb; na += 1
                except Exception:  # noqa: BLE001
                    pass
            info["correction_adds"] = na
            body = body.clean()
        if os.environ.get("THIN_RIB", "1") == "1":
            # задача 66: язычок защёлки и тонкие перегородки — тоньше, чем ловит CORR_ADD
            mm = _corr_mesh(body, tf, info)
            nt = 0
            for v in (thin_rib_prisms(scan, mm) if mm is not None else []):
                try:
                    nb = body.fuse(v)
                    if len(nb.Faces()) - len(body.Faces()) > int(os.environ.get("CORR_ADD_MAXF", "40")):
                        info["thin_ribs_stairs_skipped"] = info.get("thin_ribs_stairs_skipped", 0) + 1
                    elif nb.isValid() and len(nb.Solids()) == 1 and nb.Volume() < 1.02 * body.Volume():
                        body = nb; nt += 1
                except Exception:  # noqa: BLE001
                    pass
            info["thin_ribs"] = nt
            body = body.clean()
        if os.environ.get("THIN_CUT", "0") == "1":
            # острие узкой канавки (клин у основания бобышки) — тоньше порога поправочных вырезов
            mm = _corr_mesh(body, tf, info)
            nt = 0
            for v in (thin_rib_prisms(scan, mm, cut=True) if mm is not None else []):
                try:
                    nb = body.cut(v)
                    if nb.isValid() and len(nb.Solids()) == 1 and nb.Volume() > 0.98 * body.Volume():
                        body = nb; nt += 1
                except Exception:  # noqa: BLE001
                    pass
            info["thin_cuts"] = nt
            body = body.clean()
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
        if len(sols) > 1 and os.environ.get("DROP_CAVITIES", "1") == "1":
            # оболочки внутри наружной — закрытые полости (вырез шумного скана, не вышедший наружу);
            # конструктор полостей внутри сплошного тела не делает — заполняем их
            big = max(sols, key=lambda x: abs(x.Volume()))
            inner = [x for x in sols if x is not big and big.isInside(x.Center())]
            # только мелкие полости (шум); большие — признак того, что метод не подходит детали
            if inner and sum(abs(x.Volume()) for x in inner) < 0.02 * abs(big.Volume()):
                info["cavities_filled"] = len(inner)
                rest = [x for x in sols if x is big or not any(x is y for y in inner)]
                if len(rest) == 1:
                    body = big
                sols = rest
    if len(sols) > 1:
        # мелкие отрезанные обрывки (тонкие перемычки между вырезами) — убираем, если их меньше 1% объёма
        sols = sorted(sols, key=lambda x: -x.Volume())
        rest = sum(x.Volume() for x in sols[1:])
        if rest < 0.01 * sols[0].Volume():
            body = sols[0]
            info["fragments_dropped"] = len(sols) - 1
        else:
            raise RuntimeError(f"сборка распалась на {len(sols)} тела: " + ", ".join(f"{x.Volume():.1f}" for x in sols[:5]))
    info["faces"] = len(body.Faces())
    info["seconds"] = round(time.time() - t, 1)
    log(f"    сборка по сечениям: стенка лофтом, вырезов кармана {len(cut) - skipped}, бобышек {added}, "
        f"пазов поправкой {info.get('correction_cuts', 0)}, граней {info['faces']}, {info['seconds']} с")
    return body, info
