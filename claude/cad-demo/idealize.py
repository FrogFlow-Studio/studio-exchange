"""Выпрямление скана: поверхность приводится к виду «как построено в CAD».

Обтяжка скана (тела 0в и 0к) повторяет его шум и вмятины: плоскость выходит мятой, ребро — волнистым.
Конструктор строит иначе: плоскость ровная, ребро — прямая линия пересечения двух плоскостей, угол —
точка пересечения трёх. Здесь то же делается с сеткой скана:

1. разметка (regions.segment) находит плоскости и цилиндры по сырому скану;
2. направления плоскостей выравниваются: почти параллельные и почти перпендикулярные главным
   направлениям детали становятся точно такими (в пределах шума);
3. каждая вершина садится на свои поверхности: одна плоскость — проекция на неё, две — на линию
   пересечения, три — в точку; цилиндр — по радиусу. Сдвиг ограничен допуском: что не ложится, не трогается;
4. участки свободной формы сглаживаются при закреплённых границах.

Топология сетки не меняется (те же треугольники), поэтому результат можно отдавать прежним сборщикам тела.
Способ общий: все пороги — через допуск, шум и шаг сетки.

python idealize.py скан.stl выход.stl   — печатает доли и сдвиги
"""
from __future__ import annotations

import sys

import numpy as np
import trimesh


def _axes(planes, angle_deg: float):
    """Главные взаимно перпендикулярные направления детали по площадям плоскостей (или None)."""
    if not planes:
        return None
    n = np.array([p["normal"] for p in planes], float)
    a = np.array([p["area"] for p in planes], float)
    cos = np.cos(np.radians(angle_deg))

    def dominant(mask, perp=()):
        best, score = None, 0.0
        for i in np.flatnonzero(mask):
            d = n[i]
            for q in perp:
                d = d - q * (d @ q)
            ln = np.linalg.norm(d)
            if ln < 0.9:      # не перпендикулярно уже выбранным направлениям
                continue
            d = d / ln
            near = np.abs(n @ d) > cos
            s = a[near].sum()
            if s > score:
                # направление — среднее по своим плоскостям, взвешенное площадью
                m = (n[near] * (np.sign(n[near] @ d) * a[near])[:, None]).sum(0)
                for q in perp:
                    m = m - q * (m @ q)
                best, score = m / np.linalg.norm(m), s
        return best, score

    x, sx = dominant(np.ones(len(n), bool))
    if x is None:
        return None
    y, sy = dominant(np.abs(n @ x) < np.sin(np.radians(angle_deg)), (x,))
    if y is None:
        return None
    z = np.cross(x, y)
    return np.array([x, y, z / np.linalg.norm(z)])


def regularize(regions, vertices, faces, label, angle_deg: float = 3.0, tol: float = 0.3):
    """Плоскости, почти параллельные главным направлениям, становятся точно параллельными: нормаль
    заменяется осью, если при этом точки области остаются в допуске. Возвращает число выровненных."""
    planes = [r for r in regions if r["kind"] == "plane"]
    axes = _axes(planes, angle_deg)
    if axes is None:
        return 0, None
    cos = np.cos(np.radians(angle_deg))
    done = 0
    for i, r in enumerate(regions):
        if r["kind"] != "plane":
            continue
        dots = axes @ r["normal"]
        k = int(np.argmax(np.abs(dots)))
        if abs(dots[k]) < cos:
            continue
        n = axes[k] * np.sign(dots[k])
        pts = vertices[np.unique(faces[label == i])]
        h = pts @ n
        d = float(np.median(h))
        if np.percentile(np.abs(h - d), 95) > tol:
            continue      # выравнивание уводит плоскость от скана: остаётся как измерена
        r["normal"], r["origin"], r["aligned"] = n, n * d, True
        done += 1
    return done, axes


def merge_coplanar(regions, vertices, faces, label, angle_deg: float = 3.0, tol: float = 0.3):
    """Соседние области одной и той же плоскости (разметка разделила её шумом или узкой перемычкой)
    сливаются в одну: иначе между ними остаётся ломаная граница посреди ровной грани.
    Возвращает (новые метки, число слияний). Слитая область остаётся в списке с kind == "merged"."""
    label = np.asarray(label).copy()
    cos = np.cos(np.radians(angle_deg))
    touch = set()
    seen = {}
    for fi, li in enumerate(label):
        if li < 0 or regions[li]["kind"] != "plane":
            continue
        for v in faces[fi]:
            for lj in seen.setdefault(int(v), set()):
                if lj != li:
                    touch.add((min(li, lj), max(li, lj)))
            seen[int(v)].add(int(li))
    pts = {}

    def points(i):
        if i not in pts:
            pts[i] = vertices[np.unique(faces[label == i])]
        return pts[i]

    done = 0
    for i, j in sorted(touch, key=lambda ij: -(regions[ij[0]]["area"] + regions[ij[1]]["area"])):
        while regions[i]["kind"] == "merged":
            i = regions[i]["into"]
        while regions[j]["kind"] == "merged":
            j = regions[j]["into"]
        if i == j:
            continue
        ri, rj = regions[i], regions[j]
        ni, nj = np.asarray(ri["normal"], float), np.asarray(rj["normal"], float)
        if float(ni @ nj) < cos:
            continue
        if ri["area"] < rj["area"]:
            i, j, ri, rj, ni, nj = j, i, rj, ri, nj, ni
        # плоскость большей области, если меньшая на неё ложится; иначе общая по площади
        cand = [(ni, float(ni @ np.asarray(ri["origin"], float)))]
        n = ni * ri["area"] + nj * rj["area"]
        n = n / np.linalg.norm(n)
        both = np.vstack([points(i), points(j)])
        cand.append((n, float(np.median(both @ n))))
        for k, (n, d) in enumerate(cand):
            if ri.get("aligned") and k == 1:
                break
            if np.percentile(np.abs(both @ n - d), 95) <= tol:
                ri["normal"], ri["origin"], ri["area"] = n, n * d, ri["area"] + rj["area"]
                rj["kind"], rj["into"] = "merged", i
                label[label == j] = i
                pts.pop(i, None)
                done += 1
                break
    return label, done


def regularize_cylinders(regions, vertices, faces, label, angle_deg: float = 3.0, tol: float = 0.3):
    """Ось цилиндра, почти совпадающая с нормалью найденной плоскости или с главным направлением, ставится
    точно по нему (отверстие сверлят перпендикулярно грани); центр и радиус подбираются заново при этой оси.
    Замена принимается, если точки области остаются в допуске. Возвращает число выровненных."""
    dirs = [np.asarray(r["normal"], float) for r in regions if r["kind"] == "plane"]
    area = [r["area"] for r in regions if r["kind"] == "plane"]
    if not dirs:
        return 0
    dirs, area = np.array(dirs), np.array(area)
    cos = np.cos(np.radians(angle_deg))
    done = 0
    for i, r in enumerate(regions):
        if r["kind"] != "cylinder":
            continue
        dots = dirs @ np.asarray(r["axis"], float)
        ok = np.abs(dots) >= cos
        if not ok.any():
            continue
        k = int(np.flatnonzero(ok)[np.argmax(area[ok])])      # из подходящих — нормаль самой большой плоскости
        ax = dirs[k] * np.sign(dots[k])
        pts = vertices[np.unique(faces[label == i])]
        if len(pts) < 6:
            continue
        u = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        x, y = pts @ u, pts @ v
        A = np.c_[2 * x, 2 * y, np.ones(len(x))]
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        rad2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
        if rad2 <= 0:
            continue
        rad = float(np.sqrt(rad2))
        res = np.abs(np.hypot(x - sol[0], y - sol[1]) - rad)
        if np.percentile(res, 95) > tol or abs(rad - r["r"]) > max(tol, 0.2 * r["r"]):
            continue
        h = pts @ ax
        r["axis"], r["r"], r["aligned"] = ax, rad, True
        r["origin"] = u * sol[0] + v * sol[1] + ax * float(h.mean())
        done += 1
    return done


def _project(x0, cons, iters: int = 6):
    """Ближайшая к x0 точка на пересечении поверхностей вершины: плоскости решаются разом
    (псевдообратная гасит почти параллельные), цилиндры — чередованием проекций."""
    x = x0.copy()
    P = [(c[1], c[2]) for c in cons if c[0] == "plane"]
    C = [c[1:] for c in cons if c[0] == "cylinder"]
    A = np.array([p[0] for p in P]) if P else None
    d = np.array([p[1] for p in P]) if P else None
    for _ in range(iters if C else 1):
        if A is not None:
            x = x - np.linalg.pinv(A, rcond=0.15) @ (A @ x - d)
        for axis, origin, r in C:
            v = x - origin
            radial = v - axis * (v @ axis)
            ln = np.linalg.norm(radial)
            if ln > 1e-9:
                x = origin + axis * (v @ axis) + radial * (r / ln)
    if A is not None and C:
        x = x - np.linalg.pinv(A, rcond=0.15) @ (A @ x - d)
    return x


def fit_tol(tol: float, noise: float) -> float:
    """Допуск посадки на поверхности: два шума, но не больше двух допусков. Оценка шума на сглаженных
    сканах студии бывает 0,6–1,4 мм (задача 22), и посадка с таким допуском срезала бы детали на миллиметры."""
    return max(tol, min(2.0 * noise, 2.0 * tol))


def idealize(scan: trimesh.Trimesh, tol: float = 0.3, noise: float = 0.0, *, seg: dict | None = None,
             angle_deg: float = 3.0, fair_iters: int = 10) -> tuple[trimesh.Trimesh, dict]:
    """Возвращает (выпрямленная сетка с теми же треугольниками, сведения)."""
    import regions

    fit = fit_tol(tol, noise)
    if seg is None:
        step = float(np.median(scan.edges_unique_length))
        # область меньше нескольких треугольников — не поверхность, а случайность
        seg = regions.segment(scan, tol=fit, min_area=max(6.0, 4 * step * step * 0.43), scan=scan)
    label = np.asarray(seg["label"])
    regs = [dict(r) for r in seg["regions"]]
    V = np.asarray(scan.vertices, float).copy()
    F = np.asarray(scan.faces)
    aligned, axes = regularize(regs, V, F, label, angle_deg, fit)
    label, merged = merge_coplanar(regs, V, F, label, angle_deg, fit)
    aligned_cyl = regularize_cylinders(regs, V, F, label, angle_deg, fit)

    # поверхности каждой вершины — области её треугольников
    cons = [set() for _ in range(len(V))]
    for fi, li in enumerate(label):
        if li >= 0 and regs[li]["kind"] in ("plane", "cylinder"):
            for v in F[fi]:
                cons[v].add(int(li))

    def con(i):
        r = regs[i]
        if r["kind"] == "plane":
            n = np.asarray(r["normal"], float)
            return ("plane", n, float(n @ np.asarray(r["origin"], float)))
        return ("cylinder", np.asarray(r["axis"], float), np.asarray(r["origin"], float), float(r["r"]))

    built = {i: con(i) for i in range(len(regs)) if regs[i]["kind"] in ("plane", "cylinder")}

    def dist(i, x):
        c = built[i]
        if c[0] == "plane":
            return abs(float(c[1] @ x) - c[2])
        v = x - c[2]
        return abs(float(np.linalg.norm(v - c[1] * (v @ c[1]))) - c[3])

    def direction(i, x):
        c = built[i]
        if c[0] == "plane":
            return c[1]
        v = x - c[2]
        radial = v - c[1] * (v @ c[1])
        ln = np.linalg.norm(radial)
        return radial / ln if ln > 1e-9 else c[1]

    def place(x0, ids):
        """Точка на поверхностях ids и те из них, на которые она действительно легла: почти параллельные
        плоскости вместе не решаются точно, тогда последняя снимается (метка грани требует точной посадки)."""
        ids = tuple(ids)
        while ids:
            x = _project(x0, [built[i] for i in ids])
            if all(dist(i, x) <= (1e-7 if built[i][0] == "plane" else 1e-3) for i in ids):
                return x, ids
            ids = ids[:-1]
        return x0.copy(), ()

    # Сосед по сетке подсказывает поверхности: вершина на полоске между двумя плоскостями (шум на ребре,
    # фаска уже допуска) лежит в допуске от обеих — она садится на линию их пересечения, и ребро
    # становится острым и прямым. Скругление шире допуска так не схлопнется: его середина дальше.
    ring = [set(c) for c in cons]
    e = scan.edges_unique
    for _ in range(2):
        nxt = [set(c) for c in ring]
        for a_, b_ in e:
            nxt[a_] |= ring[b_]
            nxt[b_] |= ring[a_]
        ring = nxt
    limit = 3.0 * fit
    moved = np.zeros(len(V))
    snapped = np.zeros(len(V), bool)
    used = [()] * len(V)
    kept = 0
    for v in range(len(V)):
        x0 = V[v]
        own = sorted(cons[v], key=lambda i: -regs[i]["area"])
        near = sorted((i for i in ring[v] - cons[v] if dist(i, x0) <= fit), key=lambda i: dist(i, x0))
        ids = []
        for i in own + near:
            # почти параллельная уже взятой (другая сторона тонкого ребра) точку не уточняет, а схлопывает стенку
            if all(abs(float(direction(i, x0) @ direction(j, x0))) < 0.9 for j in ids):
                ids.append(i)
            if len(ids) == 3:
                break
        if not ids:
            continue
        x, ids = place(x0, ids)
        shift = float(np.linalg.norm(x - x0))
        while shift > limit and len(ids) > 1:
            # вершина не ложится на пересечение (острый клин): отбрасываем последнюю поверхность
            x, ids = place(x0, ids[:-1])
            shift = float(np.linalg.norm(x - x0))
        if shift > limit or not ids:
            kept += 1
            continue
        V[v], moved[v], snapped[v], used[v] = x, shift, True, tuple(ids)

    # свободная форма: сглаживание по соседям при закреплённых вершинах поверхностей (без усадки: шаг
    # вперёд и шаг назад, как у Таубина); уход от скана не больше допуска
    free = ~snapped
    V0 = np.asarray(scan.vertices, float)
    if fair_iters and free.any():
        deg = np.bincount(e.ravel(), minlength=len(V)).astype(float)
        deg[deg == 0] = 1
        for it in range(2 * fair_iters):
            acc = np.zeros_like(V)
            for k in range(3):
                acc[:, k] = (np.bincount(e[:, 0], V[e[:, 1], k], minlength=len(V))
                             + np.bincount(e[:, 1], V[e[:, 0], k], minlength=len(V)))
            lap = acc / deg[:, None] - V
            lam = 0.5 if it % 2 == 0 else -0.53
            V[free] += lam * lap[free]
            off = V[free] - V0[free]
            ln = np.linalg.norm(off, axis=1)
            V[free] = V0[free] + off * np.minimum(1.0, fit / np.maximum(ln, 1e-12))[:, None]

    # Ребро сетки поперёк ребра детали: один конец на плоскости A, другой на B, общей поверхности нет.
    # Такой треугольник висит «фаской» через ребро. Если линия пересечения A и B проходит у самого конца
    # ребра, этот конец садится на неё; иначе на ребре ставится новая вершина — на линии пересечения.
    # После этого по обе стороны от линии треугольники лежат каждый в своей плоскости.
    def crease(a_, b_):
        """Лучшая пара плоскостей концов ребра: (зазор, точка на линии, точка на ребре, (i, j), доля вдоль ребра)."""
        pa, pb = V[a_], V[b_]
        best = None
        for i in used[a_]:
            if built[i][0] != "plane":
                continue
            for j in used[b_]:
                if built[j][0] != "plane":
                    continue
                ni, di, nj, dj = built[i][1], built[i][2], built[j][1], built[j][2]
                t = np.cross(ni, nj)
                ln = np.linalg.norm(t)
                if ln < 0.3:
                    continue
                t = t / ln
                p0 = np.linalg.solve(np.array([ni, nj, t]), np.array([di, dj, float(t @ pa)]))
                w = (pa - p0) - t * float((pa - p0) @ t)
                d = (pb - pa) - t * float((pb - pa) @ t)
                dd = float(d @ d)
                s_ = float(np.clip(-(w @ d) / dd, 0.0, 1.0)) if dd > 1e-18 else 0.5
                q = pa + s_ * (pb - pa)
                x = p0 + t * float((q - p0) @ t)
                gap = float(np.linalg.norm(q - x))
                if gap <= 1.5 * fit and (best is None or gap < best[0]):
                    best = (gap, x, q, (i, j), s_)
        return best

    V0 = np.asarray(scan.vertices, float)
    end = 0.2      # ближе этой доли ребра к концу новая вершина дала бы треугольник-иглу
    pulled = 0
    for _ in range(2):
        for a_, b_ in e:
            if not used[a_] or not used[b_] or set(used[a_]) & set(used[b_]):
                continue
            c = crease(a_, b_)
            if c is None or end < c[4] < 1 - end:
                continue
            v, add = (a_, c[3][1]) if c[4] <= end else (b_, c[3][0])
            if len(used[v]) >= 3:
                continue
            ids = used[v] + (add,)
            x, got = place(V0[v], ids)
            if len(got) == len(ids) and np.linalg.norm(x - V0[v]) <= limit:
                V[v], used[v] = x, ids
                pulled += 1

    ref_n = np.asarray(scan.face_normals).copy()
    old_label = label.copy()
    V = list(V)
    V0 = list(V0)
    used = list(used)
    snapped = list(snapped)
    split = {}
    for a_, b_ in e:
        if not used[a_] or not used[b_] or set(used[a_]) & set(used[b_]):
            continue
        best = crease(a_, b_)
        if best is not None and end < best[4] < 1 - end:
            split[(int(a_), int(b_))] = len(V)
            V.append(best[1])
            V0.append(best[2])
            used.append(best[3])
            snapped.append(True)
    V, V0, snapped = np.array(V), np.array(V0), np.array(snapped, bool)
    if split:
        def mid(u, v):
            return split.get((min(u, v), max(u, v)))

        nf, nl, nn_ = [], [], []
        for fi, (p_, q_, r_) in enumerate(F):
            p_, q_, r_ = int(p_), int(q_), int(r_)
            m = [mid(p_, q_), mid(q_, r_), mid(r_, p_)]
            k = sum(x is not None for x in m)
            if k == 0:
                tris = [(p_, q_, r_)]
            elif k == 3:
                tris = [(p_, m[0], m[2]), (m[0], q_, m[1]), (m[2], m[1], r_), (m[0], m[1], m[2])]
            else:
                # поворот так, чтобы первое ребро было разрезано
                while m[0] is None:
                    p_, q_, r_ = q_, r_, p_
                    m = [m[1], m[2], m[0]]
                if k == 1:
                    tris = [(p_, m[0], r_), (m[0], q_, r_)]
                elif m[1] is not None:      # разрезаны pq и qr
                    tris = [(m[0], q_, m[1]), (p_, m[0], m[1]), (p_, m[1], r_)]
                else:                       # разрезаны pq и rp
                    tris = [(p_, m[0], m[2]), (m[0], q_, r_), (m[0], r_, m[2])]
            nf.extend(tris)
            nl.extend([old_label[fi]] * len(tris))
            nn_.extend([ref_n[fi]] * len(tris))
        F, old_label, ref_n = np.array(nf), np.array(nl), np.array(nn_)

    # перевёрнутые треугольники (вершина перескочила через соседа): у самой сдвинутой вершины снимается
    # последняя поверхность — она остаётся на своей плоскости, но не скользит вдоль неё к линии пересечения
    adj = trimesh.Trimesh(V, F, process=False).face_adjacency

    def flips():
        tri = V[F]
        nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        bad = np.einsum("ij,ij->i", nn, ref_n) < 0
        # складка: соседние треугольники легли друг на друга (стенка тоньше допуска схлопнулась)
        ln = np.linalg.norm(nn, axis=1)
        ok = ln > 1e-6
        un = nn / np.maximum(ln, 1e-300)[:, None]
        fold = (np.einsum("ij,ij->i", un[adj[:, 0]], un[adj[:, 1]]) < -0.5) & ok[adj[:, 0]] & ok[adj[:, 1]]
        bad[adj[fold].ravel()] = True
        return bad

    relaxed = 0
    for _ in range(30):
        bad = np.flatnonzero(flips())
        if not len(bad):
            break
        sh = np.linalg.norm(V - V0, axis=1)
        todo = {int(F[f][int(np.argmax(sh[F[f]]))]) for f in bad}
        todo = [v for v in todo if sh[v] > 1e-9]
        if not todo:
            break
        for v in todo:
            x, ids = place(V0[v], used[v][:-1])
            if ids:
                V[v] = x
            else:
                V[v] = V0[v]
                snapped[v] = False
            used[v] = ids
            relaxed += 1

    # область треугольника после посадки: поверхность, общая для всех трёх его вершин
    label = old_label.copy()
    for fi in range(len(F)):
        common = set(used[F[fi, 0]]) & set(used[F[fi, 1]]) & set(used[F[fi, 2]])
        if old_label[fi] >= 0 and old_label[fi] in common:
            continue
        label[fi] = max(common, key=lambda i: regs[i]["area"]) if common else -1

    out = trimesh.Trimesh(V, F, process=False)
    area = out.area_faces
    kinds = np.array([regs[i]["kind"] if i >= 0 else "free" for i in label])
    was = np.array([regs[i]["kind"] if i >= 0 else "free" for i in np.asarray(seg["label"])])
    info = {
        "fit_tol": fit, "planes": int(sum(r["kind"] == "plane" for r in regs)),
        "cylinders": int(sum(r["kind"] == "cylinder" for r in regs)), "aligned_planes": int(aligned), "merged_planes": int(merged), "aligned_cylinders": int(aligned_cyl),
        "plane_share": float(area[kinds == "plane"].sum() / area.sum()),
        "cylinder_share": float(area[kinds == "cylinder"].sum() / area.sum()),
        "plane_share_before": float(scan.area_faces[was == "plane"].sum() / scan.area),
        "snapped_vertices": int(snapped.sum()), "kept_vertices": int(kept), "split_edges": len(split),
        "shift_p95": float(np.percentile(moved[moved > 0], 95)) if (moved > 0).any() else 0.0,
        "shift_max": float(moved.max()) if len(moved) else 0.0,
        "pulled_to_crease": int(pulled), "relaxed": int(relaxed), "flipped": int(flips().sum()),
        "degenerate": int((area < 1e-9).sum()),
    }
    return out, {**info, "label": label, "regions": regs, "axes": axes}


if __name__ == "__main__":
    import time

    scan = trimesh.load(sys.argv[1], force="mesh")
    t = time.time()
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    mesh, info = idealize(scan, noise=noise)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items() if k not in ("label", "regions", "axes")},
          "с", round(time.time() - t, 1))
    mesh.export(sys.argv[2])
