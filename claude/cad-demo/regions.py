"""Разметка поверхности на области: плоскость, цилиндр, свободная форма (этап Б плана, шаг 1).

Каждый треугольник чистой поверхности получает номер области или -1 (свободная форма).
Плоскости берутся ростом области от крупных треугольников (walls.planar_patches), цилиндры —
ростом от затравки: по соседям оценивается ось (направление, которому перпендикулярны все
нормали) и окружность в поперечной плоскости, затем область растёт, пока треугольники лежат
на этом цилиндре. Способ общий: размеры и положение берутся только из самой сетки.

python regions.py скан.stl [папка]   — печатает доли площади по видам областей
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import trimesh


def _neighbours(mesh: trimesh.Trimesh) -> list[list[int]]:
    nb = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nb[a].append(b)
        nb[b].append(a)
    return nb


def _circle(q: np.ndarray):
    """Окружность по точкам на плоскости (алгебраическая подгонка): центр, радиус."""
    A = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q**2).sum(1), rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def _cylinder(P: np.ndarray, N: np.ndarray, W: np.ndarray):
    """Цилиндр по точкам и нормалям: (ось, точка на оси, радиус, знак) или None.
    Знак +1 — нормали наружу от оси (вал, скругление снаружи), -1 — к оси (отверстие, галтель)."""
    if len(P) < 6:
        return None
    _, s, vt = np.linalg.svd(N * np.sqrt(W)[:, None], full_matrices=False)
    # нормали цилиндра лежат в одной плоскости: третье сингулярное число мало, второе — нет
    if s[1] < 0.12 * s[0] or s[2] > 0.5 * s[1]:
        return None
    ax = vt[2]
    u = vt[0] - (vt[0] @ ax) * ax
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    q = np.c_[P @ u, P @ v]
    c2, r = _circle(q)
    if not np.isfinite(r) or r < 1e-3:
        return None
    c = c2[0] * u + c2[1] * v
    rad = P - c
    rad -= np.outer(rad @ ax, ax)
    d = np.linalg.norm(rad, axis=1)
    sign = 1.0 if float(np.average(((rad / np.maximum(d, 1e-9)[:, None]) * N).sum(1), weights=W)) > 0 else -1.0
    return ax, c, r, sign


def _on_cylinder(P, N, cyl, tol: float, cos_n: float) -> np.ndarray:
    ax, c, r, sign = cyl
    rad = P - c
    rad -= np.outer(rad @ ax, ax)
    d = np.linalg.norm(rad, axis=1)
    rn = rad / np.maximum(d, 1e-9)[:, None]
    return (np.abs(d - r) < tol) & ((rn * N).sum(1) * sign > cos_n)


def segment(mesh: trimesh.Trimesh, tol: float = 0.3, min_area: float | None = None, max_angle: float = 12.0) -> dict:
    """Возвращает {"label": номер области у каждого треугольника (-1 свободная форма),
    "regions": [{"kind": "plane"|"cylinder", ...параметры..., "area", "faces"}]}."""
    import walls

    area = mesh.area_faces
    total = float(area.sum())
    if min_area is None:  # область мельче этого не стоит точной грани: доля всей поверхности
        min_area = max(4.0, 0.0015 * total)
    fn, fc = mesh.face_normals, mesh.triangles_center
    nb = _neighbours(mesh)
    label = -np.ones(len(mesh.faces), int)
    regions: list[dict] = []

    cos_n = float(np.cos(np.radians(2 * max_angle)))
    cos_c = float(np.cos(np.radians(25.0)))
    tried = np.zeros(len(mesh.faces), bool)
    for seed in np.argsort(-area):
        if label[seed] >= 0 or tried[seed]:
            continue
        # затравка: кольца свободных соседей; у крупного цилиндра малое кольцо почти плоское,
        # поэтому кольцо растёт, пока нормали не разойдутся достаточно для оценки оси
        tried[seed] = True
        ring, q, seen, cyl = [seed], deque([seed]), {seed}, None
        for size in (60, 250, 1000, 4000):
            while q and len(ring) < size:
                f = q.popleft()
                for g in nb[f]:
                    # через излом кольцо не идёт: за ребром уже другая поверхность
                    if g not in seen and label[g] < 0 and fn[f] @ fn[g] > cos_c:
                        seen.add(g)
                        ring.append(g)
                        q.append(g)
            if len(ring) < 12:
                break
            r_ = np.asarray(ring)
            cyl = _cylinder(fc[r_], fn[r_], area[r_])
            if cyl is not None and _on_cylinder(fc[r_], fn[r_], cyl, tol, cos_n).mean() >= 0.8:
                break
            cyl = None
            if not q:
                break
        if cyl is None:
            tried[np.asarray(ring[:60])] = True  # рядом с неудачной затравкой заново не пробуем
            continue
        members: list[int] = []
        for _ in range(4):  # рост и уточнение цилиндра по набранной области
            ok = _on_cylinder(fc, fn, cyl, tol, cos_n) & (label < 0)
            if not ok[seed]:
                break
            members, q, got = [seed], deque([seed]), {seed}
            while q:
                f = q.popleft()
                for g in nb[f]:
                    if g not in got and ok[g]:
                        got.add(g)
                        members.append(g)
                        q.append(g)
            m = np.asarray(members)
            new = _cylinder(fc[m], fn[m], area[m])
            if new is None:
                members = []
                break
            if abs(new[2] - cyl[2]) < 0.02 * cyl[2] and abs(float(new[0] @ cyl[0])) > 0.9995:
                cyl = new
                break
            cyl = new
        if not members:
            continue
        m = np.asarray(members)
        ax, c, r, sign = cyl
        rad = fc[m] - c
        h = rad @ ax
        rad -= np.outer(h, ax)
        e1 = rad[0] / np.linalg.norm(rad[0])
        ang = np.arctan2(rad @ np.cross(ax, e1), rad @ e1)
        # угловой охват: наибольший разрыв между соседними углами вычитается из полного круга
        s = np.sort(ang)
        gap = float(np.max(np.diff(np.r_[s, s[0] + 2 * np.pi])))
        span = float(np.degrees(2 * np.pi - gap))
        A = float(area[m].sum())
        # узкая дуга неотличима от плоскости или свободной формы: нужен заметный охват
        v = mesh.vertices[np.unique(mesh.faces[m])] - c
        v -= np.outer(v @ ax, ax)
        rms = float(np.sqrt(((np.linalg.norm(v, axis=1) - r) ** 2).mean()))
        sag = r * (1 - np.cos(np.radians(min(span, 180.0)) / 2))  # стрела дуги: насколько область выгнута
        # дуга должна быть выгнута заметно больше допуска и больше собственного разброса точек,
        # иначе это участок свободной формы, случайно похожий на цилиндр
        if A < min_area or span < 25 or sag < 3 * tol or sag < 8 * rms:
            tried[m] = True
            continue
        label[m] = len(regions)
        regions.append({"kind": "cylinder", "axis": ax, "origin": c + ax * float(h.mean()), "r": float(r),
                        "inner": sign < 0, "span_deg": round(span, 1), "length": float(np.ptp(h)),
                        "area": A, "faces": m})
    for p in walls.planar_patches(mesh, max_angle=max_angle, max_dist=2 * tol, min_area=min_area):
        f = np.asarray(p["faces"])
        # в плоскость идут только треугольники, действительно лежащие на ней: рост области допускает больше
        n, o = np.asarray(p["normal"]), np.asarray(p["point"])
        f = f[(np.abs((fc[f] - o) @ n) < tol) & (label[f] < 0)]  # цилиндры уже размечены: узкие полосы на них плоскостями не считаются
        if area[f].sum() < min_area:
            continue
        # плоскость на кривой оболочке — кусок в пределах допуска, но не плоскость: у настоящей
        # вершины лежат заметно ближе допуска
        v = mesh.vertices[np.unique(mesh.faces[f])]
        if float(np.sqrt((((v - o) @ n) ** 2).mean())) > 0.5 * tol:
            continue
        label[f] = len(regions)
        regions.append({"kind": "plane", "normal": n, "origin": o, "area": float(area[f].sum()), "faces": f})

    return {"label": label, "regions": regions, "area": total}


def summary(seg: dict) -> dict:
    out = {"plane": 0.0, "cylinder": 0.0}
    for r in seg["regions"]:
        out[r["kind"]] += r["area"]
    share = {k: round(v / seg["area"], 4) for k, v in out.items()}
    share["free"] = round(1 - sum(share.values()), 4)
    share["planes"] = sum(r["kind"] == "plane" for r in seg["regions"])
    share["cylinders"] = sum(r["kind"] == "cylinder" for r in seg["regions"])
    return share


if __name__ == "__main__":
    import time

    import surface

    scan = trimesh.load(sys.argv[1], force="mesh")
    t = time.time()
    clean = surface.surface_mesh(scan)
    seg = segment(clean)
    print(summary(seg), "с", round(time.time() - t, 1))
    for r in sorted(seg["regions"], key=lambda r: -r["area"]):
        if r["kind"] == "cylinder":
            print(f"  цилиндр R={r['r']:.2f} {'внутр.' if r['inner'] else 'наруж.'} охват {r['span_deg']}° длина {r['length']:.1f} площадь {r['area']:.0f}")
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        col = np.full((len(clean.faces), 4), (190, 190, 190, 255), np.uint8)
        for r in seg["regions"]:
            col[r["faces"]] = (70, 130, 220, 255) if r["kind"] == "plane" else ((230, 90, 60, 255) if r["inner"] else (60, 180, 90, 255))
        clean.visual.face_colors = col
        clean.export(out / "regions.ply")
