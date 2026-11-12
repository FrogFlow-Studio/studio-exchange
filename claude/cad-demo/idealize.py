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


def idealize(scan: trimesh.Trimesh, tol: float = 0.3, noise: float = 0.0, *, seg: dict | None = None,
             angle_deg: float = 3.0, fair_iters: int = 10) -> tuple[trimesh.Trimesh, dict]:
    """Возвращает (выпрямленная сетка с теми же треугольниками, сведения)."""
    import regions

    fit = max(tol, 2.0 * noise)
    if seg is None:
        step = float(np.median(scan.edges_unique_length))
        # область меньше нескольких треугольников — не поверхность, а случайность
        seg = regions.segment(scan, tol=fit, min_area=max(6.0, 4 * step * step * 0.43), scan=scan)
    label = np.asarray(seg["label"])
    regs = [dict(r) for r in seg["regions"]]
    V = np.asarray(scan.vertices, float).copy()
    F = np.asarray(scan.faces)
    aligned, axes = regularize(regs, V, F, label, angle_deg, fit)

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
        x = _project(x0, [built[i] for i in ids])
        shift = float(np.linalg.norm(x - x0))
        while shift > limit and len(ids) > 1:
            # вершина не ложится на пересечение (острый клин): отбрасываем последнюю поверхность
            ids = ids[:-1]
            x = _project(x0, [built[i] for i in ids])
            shift = float(np.linalg.norm(x - x0))
        if shift > limit:
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

    # перевёрнутые треугольники (вершина перескочила через соседа): её сдвиг отменяется
    for _ in range(5):
        tri = V[F]
        nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        bad = np.einsum("ij,ij->i", nn, scan.face_normals) < 0
        if not bad.any():
            break
        vs = np.unique(F[bad])
        worst = vs[np.linalg.norm(V[vs] - V0[vs], axis=1) > 1e-9]
        if not len(worst):
            break
        V[worst] = V0[worst]
        snapped[worst] = False
        for v in worst:
            used[v] = ()

    # область треугольника после посадки: поверхность, общая для всех трёх его вершин
    new_label = label.copy()
    for fi in range(len(F)):
        common = set(used[F[fi, 0]]) & set(used[F[fi, 1]]) & set(used[F[fi, 2]])
        if label[fi] >= 0 and label[fi] in common:
            continue
        new_label[fi] = max(common, key=lambda i: regs[i]["area"]) if common else -1
    old_label, label = label, new_label

    out = trimesh.Trimesh(V, F, process=False)
    area = scan.area_faces
    kinds = np.array([regs[i]["kind"] if i >= 0 else "free" for i in label])
    info = {
        "fit_tol": fit, "planes": int(sum(r["kind"] == "plane" for r in regs)),
        "cylinders": int(sum(r["kind"] == "cylinder" for r in regs)), "aligned_planes": int(aligned),
        "plane_share": float(area[kinds == "plane"].sum() / area.sum()),
        "cylinder_share": float(area[kinds == "cylinder"].sum() / area.sum()),
        "snapped_vertices": int(snapped.sum()), "kept_vertices": int(kept),
        "shift_p95": float(np.percentile(moved[snapped], 95)) if snapped.any() else 0.0,
        "shift_max": float(moved.max()) if len(moved) else 0.0,
        "flipped": int((np.einsum("ij,ij->i", np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]),
                                  scan.face_normals) < 0).sum()),
        "degenerate": int((out.area_faces < 1e-9).sum()),
        "plane_share_before": float(area[np.array([regs[i]["kind"] == "plane" if i >= 0 else False for i in old_label])].sum()
                                    / area.sum()),
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
