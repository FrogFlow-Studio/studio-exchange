"""Этап Д, способ 2: скол на детали без симметрии.

Скол находится по шероховатому излому (остаток локальной квадрики на масштабе нескольких рёбер сетки
заметно выше, чем на остальной поверхности, и это область, а не полоса вдоль ребра). Закрывается
продолжением целой поверхности вокруг него:
1) стык гладких листов — каждый лист описывается плоскостью или общей квадрикой и продолжается через
   скол, тело под листами — пересечение (выпуклое ребро) или объединение (вогнутое);
2) выпуклое место — выпуклая оболочка целой поверхности вокруг скола;
3) иначе — поверхность наименьшего изгиба (бигармоническое уравнение) по краю отверстия.
Достройка принимается, только если излом лежит ниже продолженной поверхности (скол — впадина) и
объём сравним с площадь × глубину. Стенд: bench_chips/run_fill.py.
"""
import sys, time, numpy as np, trimesh
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def rough_cores(mesh, rho=None, n=40000, tol=0.3):
    """Точки ядра шероховатых областей и радиус rho. Шероховатость — СКО остатка локальной квадрики в
    круге rho по точкам той же стороны поверхности; ядро — где шероховаты и почти все соседи в 2·rho
    (полоса вдоль острого ребра так не выглядит: она узкая)."""
    step = float(np.sqrt(mesh.area_faces.mean() * 2))
    rho = rho or 6 * step
    C, fc = trimesh.sample.sample_surface(mesh, int(min(600_000, max(100_000, 30 * mesh.area / rho ** 2))), seed=1)
    CN = mesh.face_normals[fc]
    tree = cKDTree(C)
    n = int(min(n, 8 * mesh.area / rho ** 2))
    idx = np.random.default_rng(2).choice(len(C), n, replace=False)
    P, N = C[idx], CN[idx]
    q = np.full(n, np.nan)
    for i, j in enumerate(tree.query_ball_point(P, rho)):
        j = np.asarray(j)
        if len(j) < 12:
            continue
        nn = CN[j].mean(0); nn /= np.linalg.norm(nn) + 1e-12
        j = j[CN[j] @ nn > 0.85]        # только свой гладкий лист: за острым ребром нормаль другая
        if len(j) < 12:
            continue
        Q = C[j] - P[i]
        e1 = np.cross(nn, [1.0, 0, 0] if abs(nn[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nn, e1)
        x, y, z = Q @ e1, Q @ e2, Q @ nn
        A = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
        co, *_ = np.linalg.lstsq(A, z, rcond=None)
        q[i] = np.sqrt(np.mean((z - A @ co) ** 2))
    ok = np.isfinite(q)
    med = np.median(q[ok]); mad = 1.4826 * np.median(np.abs(q[ok] - med))
    # рельеф мельче десятой доли допуска — не излом: на очень чистом скане (порог 0,012 мм) «шероховатым»
    # оказывалось тиснение на пластике (Hyundai, полный скан: две ложные достройки глубиной 0,8 мм)
    thr = max(med + 6 * mad, tol / 10)
    rough = ok & (q > thr)
    tp = cKDTree(P)
    core = np.zeros(n, bool)
    for i in np.where(rough)[0]:
        j = tp.query_ball_point(P[i], 2 * rho)
        core[i] = rough[j].mean() >= 0.8
    return P[core], rho, {"median": float(med), "threshold": float(thr), "rough_share": float(rough.mean()), "core": int(core.sum())}


def regions(mesh, cores, rho, min_share=0.002):
    """Грани сетки в 2·rho от ядра, по связным кускам (крупные первыми). Островки целой поверхности
    внутри области (ядро там не набралось) входят в неё: иначе у дыры два края."""
    if not len(cores):
        return []
    d = cKDTree(cores).query(mesh.triangles_center, distance_upper_bound=2 * rho)[0]
    sel = np.isfinite(d)
    adj_all = mesh.face_adjacency
    rest = adj_all[~sel[adj_all[:, 0]] & ~sel[adj_all[:, 1]]]
    isl = [np.asarray(c) for c in trimesh.graph.connected_components(rest, nodes=np.where(~sel)[0], min_len=1)]
    if len(isl) > 1:
        isl.sort(key=lambda c: -mesh.area_faces[c].sum())
        for c in isl[1:]:
            if mesh.area_faces[c].sum() < mesh.area_faces[sel].sum():
                sel[c] = True
    adj = adj_all[sel[adj_all[:, 0]] & sel[adj_all[:, 1]]]
    out = [np.asarray(c) for c in trimesh.graph.connected_components(adj, nodes=np.where(sel)[0], min_len=3)]
    out = [c for c in out if mesh.area_faces[c].sum() >= min_share * mesh.area]
    return sorted(out, key=lambda c: -mesh.area_faces[c].sum())


def _ring_patch(idx, P, nv, edge):
    """Заплата на петле: кольца к центру. idx — номера вершин петли, P — их координаты."""
    m = len(idx)
    c = P.mean(0)
    K = max(2, int(round(np.linalg.norm(P - c, axis=1).mean() / edge)))
    V, F, rings = [], [], [np.asarray(idx)]
    for k in range(1, K):
        V.append(c + (1 - k / K) * (P - c)); rings.append(np.arange(nv, nv + m)); nv += m
    V.append(c[None]); center = nv; nv += 1
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(m):
            j = (i + 1) % m
            F += [(a[i], a[j], b[j]), (a[i], b[j], b[i])]
    last = rings[-1]
    F += [(last[i], last[(i + 1) % m], center) for i in range(m)]
    return np.vstack(V), F, nv


def fill(mesh, faces, sharp_deg=25.0, info=None):
    """Грани faces убираются, отверстие затягивается поверхностью наименьшего изгиба, продолжающей
    соседнюю (закреплены кольца соседей). Острое ребро, проходящее через скол, продолжается: его линия
    проводится через отверстие (кубическая кривая по касательным ребра на входе и выходе), и изгиб
    поперёк неё не штрафуется — плоскость продолжается плоскостью по обе стороны. None — край не простой."""
    drop = np.zeros(len(mesh.faces), bool); drop[faces] = True
    for _ in range(8):
        keep = mesh.faces[~drop]
        e = np.vstack([keep[:, [0, 1]], keep[:, [1, 2]], keep[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        be = e[cnt[inv.ravel()] == 1]
        starts, many = np.unique(be[:, 0], return_counts=True)
        pinch = starts[many > 1]
        if not len(pinch):
            break
        drop |= np.isin(mesh.faces, pinch).any(axis=1)
    else:
        return None
    nxt = {int(b): int(a) for a, b in be}      # обход края в направлении новых граней
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop, cur = [], s
        while cur not in seen:
            seen.add(cur); loop.append(cur); cur = nxt.get(cur)
            if cur is None:
                return None
        if cur != s:
            return None
        loops.append(loop)
    if len(loops) != 1:
        return None
    loop = np.array(loops[0]); m = len(loop)
    n0 = len(mesh.vertices)
    L = mesh.vertices[loop]
    edge = float(np.linalg.norm(L - np.roll(L, 1, axis=0), axis=1).mean())
    # нормали вершин по оставшимся граням и острота: наибольший угол между нормалью грани в двух
    # кольцах и средней нормалью вершины
    fn = mesh.face_normals[~drop]
    vn = np.zeros((n0, 3))
    for k in range(3):
        np.add.at(vn, keep[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1)[:, None] + 1e-12
    A0 = sp.coo_matrix((np.ones(len(keep) * 6), (keep[:, [0, 1, 2, 1, 2, 0]].T.ravel(), keep[:, [1, 2, 0, 0, 1, 2]].T.ravel())),
                       shape=(n0, n0)).tocsr()
    near = np.zeros(n0, bool); near[loop] = True
    for _ in range(4):
        near = near | (A0 @ near.astype(float) > 0)
    cosmin = np.ones(n0)
    fnear = near[keep].any(axis=1)
    for k in range(3):
        np.minimum.at(cosmin, keep[fnear, k], np.einsum("ij,ij->i", fn[fnear], vn[keep[fnear, k]]))
    c1 = cosmin.copy()
    rows, cols = A0.nonzero()
    np.minimum.at(c1, rows, cosmin[cols])          # второе кольцо
    sharp = np.degrees(np.arccos(np.clip(c1, -1, 1))) > sharp_deg
    # куски края на остром ребре
    sl = sharp[loop]
    runs = []
    if sl.any() and not sl.all():
        start = int(np.where(~sl)[0][0])
        i = 0
        while i < m:
            k = (start + i) % m
            if sl[k]:
                j = i
                while j < m and sl[(start + j) % m]:
                    j += 1
                runs.append(((start + i) % m, j - i)); i = j
            else:
                i += 1
        merged = []
        for a, ln in runs:                          # слить куски через разрыв ≤ 2 вершин
            if merged and (a - (merged[-1][0] + merged[-1][1])) % m <= 2:
                merged[-1] = (merged[-1][0], (a + ln - merged[-1][0]) % m)
            else:
                merged.append((a, ln))
        if len(merged) > 1 and (merged[0][0] - (merged[-1][0] + merged[-1][1])) % m <= 2:
            merged[0] = (merged[-1][0], (merged[0][0] + merged[0][1] - merged[-1][0]) % m); merged.pop()
        runs = merged
    V = [mesh.vertices]; nv = n0; F = []; crease = np.array([], int)
    if info is not None:
        info["creases"] = len(runs)
    if len(runs) == 2:
        ia = (runs[0][0] + runs[0][1] // 2) % m; ib = (runs[1][0] + runs[1][1] // 2) % m
        pa, pb = L[ia], L[ib]
        chord = float(np.linalg.norm(pb - pa))

        def tangent(i, ln, toward):
            k = ln // 2 + 3
            t = np.cross(vn[loop[(i - k) % m]], vn[loop[(i + k) % m]])
            if np.linalg.norm(t) < 0.2:
                return toward / np.linalg.norm(toward)
            t /= np.linalg.norm(t)
            return t if t @ toward > 0 else -t

        ta = tangent(ia, runs[0][1], pb - pa) * chord
        tb = tangent(ib, runs[1][1], pa - pb) * -chord
        nc = max(1, int(round(chord / edge)) - 1)
        u = (np.arange(1, nc + 1) / (nc + 1))[:, None]
        h = (2 * u ** 3 - 3 * u ** 2 + 1) * pa + (u ** 3 - 2 * u ** 2 + u) * ta + (-2 * u ** 3 + 3 * u ** 2) * pb + (u ** 3 - u ** 2) * tb
        crease = np.arange(nv, nv + nc); V.append(h); nv += nc
        seq = lambda a, b: [loop[(a + t) % m] for t in range(((b - a) % m) + 1)]
        for idx in (seq(ia, ib) + list(crease[::-1]), seq(ib, ia) + list(crease)):
            idx = np.array(idx)
            Pl = np.vstack([mesh.vertices[i][None] if i < n0 else h[i - n0][None] for i in idx])
            Vn, Fn, nv = _ring_patch(idx, Pl, nv, edge)
            V.append(Vn); F += Fn
    else:
        Vn, Fn, nv = _ring_patch(loop, L, nv, edge)
        V.append(Vn); F += Fn
    V = np.vstack(V); faces_new = np.vstack([keep, np.array(F, int)])
    fixed_new = np.zeros(nv, bool); fixed_new[crease] = True
    free = np.where((np.arange(nv) >= n0) & ~fixed_new)[0]
    A = sp.coo_matrix((np.ones(len(faces_new) * 6), (faces_new[:, [0, 1, 2, 1, 2, 0]].T.ravel(),
                                                       faces_new[:, [1, 2, 0, 0, 1, 2]].T.ravel())), shape=(nv, nv)).tocsr()
    A.data[:] = 1.0
    act = np.zeros(nv, bool); act[free] = True
    for _ in range(3):
        act = act | (A @ act.astype(float) > 0)
    ids = np.where(act)[0]
    sub = A[ids][:, ids]
    deg = np.asarray(sub.sum(1)).ravel()
    Lp = (sp.eye(len(ids)) - sp.diags(1.0 / np.maximum(deg, 1)) @ sub).tocsr()
    w = np.ones(len(ids))
    w[ids >= n0] = np.where(fixed_new[ids[ids >= n0]], 0.0, 1.0)     # на линии ребра изгиб не штрафуется
    old = ids < n0
    w[old] = np.where(sharp[ids[old]], 0.0, 1.0)
    B = (Lp.T @ sp.diags(w) @ Lp).tocsr()
    pos = {v: i for i, v in enumerate(ids)}
    fi = np.array([pos[v] for v in free])
    isfree = np.zeros(len(ids), bool); isfree[fi] = True
    ci = np.where(~isfree)[0]
    X = V[ids].copy()
    rhs = -(B[fi][:, ci] @ X[ci])
    X[fi] = np.c_[tuple(spsolve(B[fi][:, fi].tocsc(), rhs[:, k]) for k in range(3))]
    V[ids] = X
    out = trimesh.Trimesh(V, faces_new, process=False)
    out.remove_unreferenced_vertices()
    return out if out.is_watertight else None


def hull_fill(mesh, faces, rho, tol=0.3):
    """Скол на выпуклом месте (кромка, угол, выпуклая стенка — там сколы и бывают): целая поверхность
    вокруг скола лежит на своей выпуклой оболочке, и оболочка продолжает её через скол точно (плоскость
    плоскостью, прямое ребро ребром). Возвращает сетку или None, если место не выпуклое."""
    import manifold3d as m3

    def man(t):
        return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))

    drop = np.zeros(len(mesh.faces), bool); drop[faces] = True
    cen = mesh.triangles_center
    reg_v = mesh.vertices[np.unique(mesh.faces[faces])]
    d = cKDTree(reg_v).query(cen, distance_upper_bound=2 * rho)[0]
    band = np.isfinite(d) & ~drop
    if band.sum() < 20:
        return None
    bv = mesh.vertices[np.unique(mesh.faces[band])]
    try:
        H = trimesh.convex.convex_hull(bv)
    except Exception:  # noqa: BLE001
        return None
    dist = trimesh.proximity.closest_point(H, bv)[1]
    if (dist < tol).mean() < 0.97:
        return None
    R = _region_hull(reg_v, rho)   # недостающий материал выпуклого места лежит снаружи оболочки излома
    try:
        u = man(mesh) + (man(H) ^ man(R))
        if u.is_empty():
            return None
        parts = u.decompose()
        if len(parts) > 1:
            u = max(parts, key=lambda q: q.volume())
        out = _tri(u)
    except Exception:  # noqa: BLE001
        return None
    return out if out.volume > mesh.volume else None


def _man(t):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))


def _tri(u):
    """Булева оставляет вдоль линии пересечения иглы длиной в сотые доли мм; после склейки вершин
    появлялось ребро с четырьмя гранями, и упрощение сетки для гладкого тела теряло пятую часть объёма
    (ручка: 19 680 → 15 480 мм³). Иглы убираются внутри manifold, вершины после этого не склеиваются."""
    o = u.simplify(0.01).to_mesh()
    out = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=False)
    if not out.is_watertight:
        o = u.to_mesh()
        out = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    return out


def _region_hull(reg_v, grow=1.0):
    R = trimesh.convex.convex_hull(reg_v)
    c = R.centroid
    R.vertices = c + (R.vertices - c) * (1 + grow / max(1e-6, np.linalg.norm(R.vertices - c, axis=1).mean()))
    return R


def _implicit(P, N, tol):
    """Неявная поверхность листа по точкам P с наружными нормалями N: плоскость, а если она не годится —
    общая квадрика (метод Таубина; точна для цилиндра, конуса, сферы). Возвращает (функция расстояния
    со знаком, плюс снаружи; СКО по точкам листа; вид)."""
    c = P.mean(0)
    n = N.mean(0); n /= np.linalg.norm(n)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    pn = v[:, 0] if v[:, 0] @ n > 0 else -v[:, 0]
    rms_p = float(np.sqrt(np.mean(((P - c) @ pn) ** 2)))
    if rms_p < tol / 3:
        return (lambda X: (X - c) @ pn), rms_p, "плоскость"
    sc = float(np.linalg.norm(P - c, axis=1).max())
    Q = (P - c) / sc
    x, y, z = Q.T
    o, zz = np.ones_like(x), np.zeros_like(x)
    M = np.c_[o, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z]
    Dx = np.c_[zz, o, zz, zz, 2 * x, zz, zz, y, z, zz]
    Dy = np.c_[zz, zz, o, zz, zz, 2 * y, zz, x, zz, z]
    Dz = np.c_[zz, zz, zz, o, zz, zz, 2 * z, zz, x, y]
    A = M.T @ M
    Bm = Dx.T @ Dx + Dy.T @ Dy + Dz.T @ Dz
    from scipy.linalg import eig
    lam, vec = eig(A, Bm + 1e-12 * np.eye(10))
    lam = np.real(lam); good = np.isfinite(lam) & (lam > -1e-9)
    if not good.any():
        return None, 1e9, "нет"
    co = np.real(vec[:, np.where(good)[0][np.argmin(lam[good])]])

    def dist(X):
        q = (X - c) / sc
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        F = co[0] + co[1] * x + co[2] * y + co[3] * z + co[4] * x * x + co[5] * y * y + co[6] * z * z + co[7] * x * y + co[8] * x * z + co[9] * y * z
        gx = co[1] + 2 * co[4] * x + co[7] * y + co[8] * z
        gy = co[2] + 2 * co[5] * y + co[7] * x + co[9] * z
        gz = co[3] + 2 * co[6] * z + co[8] * x + co[9] * y
        return sc * F / np.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)

    step = 0.2 * tol
    if np.median(dist(P + step * N) - dist(P - step * N)) < 0:      # знак: плюс снаружи материала
        co = -co
    return dist, float(np.sqrt(np.mean(dist(P) ** 2))), "квадрика"


def _trees(items):
    """Все способы собрать тело из листов пересечениями (min) и объединениями (max)."""
    if len(items) == 1:
        yield items[0]
        return
    first, rest = items[0], items[1:]
    for mask in range(1 << len(rest)):
        left = [first] + [r for k, r in enumerate(rest) if mask >> k & 1]
        right = [r for k, r in enumerate(rest) if not mask >> k & 1]
        if not right:
            continue
        for a in _trees(left):
            for b in _trees(right):
                yield ("min", a, b)
                yield ("max", a, b)


def _ev(tree, vals):
    if isinstance(tree, int):
        return vals[tree]
    op = np.minimum if tree[0] == "min" else np.maximum
    return op(_ev(tree[1], vals), _ev(tree[2], vals))


def _pure(tree):
    """True — одни пересечения (выпуклое место), False — одни объединения (вогнутое), None — смешанное."""
    ops = set()

    def walk(t):
        if not isinstance(t, int):
            ops.add(t[0]); walk(t[1]); walk(t[2])
    walk(tree)
    return None if len(ops) > 1 else ops == {"min"}


def _csg(sheets, cen, fn, tol, lo, hi, info=None):
    """Как листы складываются в тело: на выпуклом ребре материал — пересечение полупространств под
    листами, на вогнутом — объединение, на стыке того и другого — их сочетание. Берётся сочетание,
    при котором целая поверхность вокруг скола лежит на границе тела; если данным одинаково отвечают
    сочетания, по-разному заполняющие сам скол, ответа нет (None)."""
    idx = np.concatenate([q["faces"] for q in sheets])
    P = cen[idx]
    on = [-q["dist"](P) for q in sheets]
    rng = np.random.default_rng(0)
    G = lo + rng.random((4000, 3)) * (hi - lo)
    grid = [-q["dist"](G) for q in sheets]
    rows, seen = [], set()
    for t in _trees(list(range(len(sheets)))):
        sign = _ev(t, grid) > 0
        key = sign.tobytes()
        if key in seen:
            continue
        seen.add(key)
        fit = float((np.abs(_ev(t, on)) < tol).mean())
        rows.append((fit, _pure(t) is not None, t, sign))
    rows.sort(key=lambda r: (-r[0], not r[1]))
    best = rows[0]
    if info is not None:
        info["csg_fit"] = round(best[0], 3)
    if best[0] < 0.95:
        if info is not None:
            info["mixed"] = True
        return None
    near = [r for r in rows if r[0] > best[0] - 0.02]
    pure = [r for r in near if r[1]]
    if pure:
        best = pure[0]
    elif any((r[3] != best[3]).mean() > 0.01 for r in near[1:]):
        if info is not None:
            info["ambiguous"] = len(near)
        return None
    if info is not None and _pure(best[2]) is None:
        info["csg"] = str(best[2])
    return best[2]


def sheet_fill(mesh, faces, rho, tol=0.3, info=None):
    """Скол на стыке гладких листов (ребро между стенками): каждый лист вокруг скола описывается своей
    поверхностью (плоскость или квадрика) и продолжается через скол; тело под листами — их пересечение
    на выпуклом ребре и объединение на вогнутом. None, если листов меньше двух или они описаны плохо."""
    import manifold3d as m3

    drop = np.zeros(len(mesh.faces), bool); drop[faces] = True
    reg_v = mesh.vertices[np.unique(mesh.faces[faces])]
    d = cKDTree(reg_v).query(mesh.triangles_center, distance_upper_bound=3 * rho)[0]
    band = np.where(np.isfinite(d) & ~drop)[0]
    if len(band) < 30:
        return None
    inband = np.zeros(len(mesh.faces), bool); inband[band] = True
    adj, ang = mesh.face_adjacency, mesh.face_adjacency_angles
    ok = inband[adj[:, 0]] & inband[adj[:, 1]]
    left = inband.copy()
    area = mesh.area_faces
    total = area[band].sum()
    fn, cen = mesh.face_normals, mesh.triangles_center
    sheets = []
    for _ in range(6):
        if area[left].sum() < 0.08 * total:
            break
        e = adj[ok & left[adj[:, 0]] & left[adj[:, 1]] & (ang < np.radians(12))]
        comps = trimesh.graph.connected_components(e, nodes=np.where(left)[0], min_len=5)
        if not len(comps):
            break
        seed = np.asarray(max(comps, key=lambda c: area[c].sum()))
        dist, rms, kind = _implicit(cen[seed], fn[seed], tol)
        if dist is None:
            left[seed] = False
            continue
        cand = np.where(left)[0]
        g = 0.2 * tol
        grad = (dist(cen[cand] + g * fn[cand]) - dist(cen[cand] - g * fn[cand])) / (2 * g)     # ≈ косинус между нормалями
        mem = cand[(np.abs(dist(cen[cand])) < tol) & (grad > 0.7)]
        if area[mem].sum() < area[seed].sum():
            mem = seed
        dist, rms, kind = _implicit(cen[mem], fn[mem], tol)
        left[mem] = False
        if dist is not None and area[mem].sum() >= 0.08 * total:
            sheets.append({"dist": dist, "rms": rms, "faces": mem, "kind": kind})
    if info is not None:
        info["sheets"] = [(q["kind"], round(q["rms"], 3)) for q in sheets]
    if len(sheets) < 2 or any(q["rms"] > tol for q in sheets):
        return None
    if area[left].sum() > 0.15 * total:        # вокруг скола есть форма, не описанная листами (паз, ребро)
        if info is not None:
            info["unexplained"] = round(float(area[left].sum() / total), 2)
        return None
    R = _region_hull(reg_v, 3 * rho)      # листы описаны в полосе 3·rho: дальше продолжению не верим
    lo, hi = R.bounds[0] - 0.5 * rho, R.bounds[1] + 0.5 * rho
    fs = [q["dist"] for q in sheets]
    tree = _csg(sheets, cen, fn, tol, lo, hi, info)
    if tree is None:
        return None
    convex = _pure(tree)

    def sdf(x, y, z):            # плюс внутри материала
        return float(_ev(tree, [-f(np.array([[x, y, z]])) for f in fs])[0])

    try:
        cap = m3.Manifold.level_set(sdf, [*lo, *hi], rho / 3)
        u = _man(mesh) + (cap ^ _man(R))
        if u.is_empty():
            return None
        parts = u.decompose()
        if len(parts) > 1:
            u = max(parts, key=lambda p_: p_.volume())
        out = _tri(u)
    except Exception as ex:  # noqa: BLE001
        if info is not None:
            info["error"] = repr(ex)[:80]
        return None
    if info is not None:
        info["convex"] = convex
    return out if out.volume > mesh.volume else None


def restore_all(scan, clean, tol=0.3):
    """Сколы находятся по сырому скану, закрываются на очищенной поверхности. Возвращает (сетка, журнал)."""
    cores, rho, info = rough_cores(scan, tol=tol)
    log, cur = [], clean
    for _ in range(6):
        done = False
        for reg in regions(cur, cores, rho):
            fi_ = {}
            r, how = sheet_fill(cur, reg, rho, tol, info=fi_), "продолжение листов"
            if r is None:
                r, how = hull_fill(cur, reg, rho, tol), "выпуклая оболочка"
            if r is None:
                r, how = fill(cur, reg, info=fi_), "гладкое продолжение"
            if r is None:
                continue
            gain = float(r.volume - cur.volume)
            area = float(cur.area_faces[reg].sum())
            pts = cur.vertices[np.unique(cur.faces[reg])]
            sd = trimesh.proximity.signed_distance(r, pts)      # > 0 внутри достроенного тела
            below = float((sd > -tol).mean()); depth = float(sd.max())
            # скол — впадина: излом целиком ниже продолженной поверхности, и объём сравним с площадь × глубину
            if below < 0.85 or depth < 2 * tol or gain < 0.08 * area * depth:
                continue
            c = pts.mean(0)
            log.append({"center": c, "radius": float(np.linalg.norm(pts - c, axis=1).max()) + 1.0, "area": area,
                        "depth": depth, "volume": gain, "how": how, "detail": fi_})
            cur, done = r, True
            cores = cores[np.linalg.norm(cores - c, axis=1) > log[-1]["radius"]]
            break
        if not done:
            break
    return cur, log, info


def describe(entry: dict) -> str:
    """Строка для отчёта «Достроено (этого нет в скане)»."""
    c = np.round(entry["center"], 1)
    return (f"скол у точки ({c[0]:g}, {c[1]:g}, {c[2]:g}) мм найден по шероховатому излому и закрыт "
            f"({entry['how']}): площадь {entry['area']:.0f} мм², глубина {entry['depth']:.1f} мм, "
            f"добавлено {entry['volume']:.0f} мм³")


def zone(entry: dict) -> dict:
    return {"center": [float(x) for x in entry["center"]], "radius": float(entry["radius"])}
