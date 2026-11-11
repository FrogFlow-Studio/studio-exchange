"""Скругления между плоскими гранями: поиск по скану и радиус (без построения грани).

Две плоскости с нормалями n1, n2 наружу пересекаются по прямой через точку p. Скругление радиуса R,
касательное к обеим, — цилиндр с осью вдоль n1 × n2 через c = p − R·m, m = (n1 + n2) / (1 + n1·n2):
расстояние от оси до каждой плоскости тогда ровно R. Неизвестное одно. Для каждой точки поверхности
у линии пересечения, не лежащей ни на одной из плоскостей, R находится из |x + R·m|² = R² (x — точка
в сечении, перпендикулярном оси, от p). Скругление принимается, если радиусы по точкам согласны.
У вогнутого угла то же с обратным знаком. Способ и границы применимости: FEWER_FACES_PLAN.md, шаг 4.

python fillets.py скан.stl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def find_fillets(mesh: trimesh.Trimesh, reach: float = 12.0, min_len: float = 3.0, samples: int = 300_000) -> list[dict]:
    """mesh — чистая поверхность (surface.surface_mesh). Возвращает список скруглений:
    radius_mm, kind (выпуклое/вогнутое), length_mm, spread (разброс радиуса, доля), points."""
    import walls

    pats = walls.planar_patches(mesh)
    if len(pats) < 2:
        return []
    label = -np.ones(len(mesh.faces), int)
    for k, p in enumerate(pats):
        label[p["faces"]] = k
    pts, fi = trimesh.sample.sample_surface(mesh, samples, seed=7)
    nrm, lab = mesh.face_normals[fi], label[fi]
    planes = []
    for k, p in enumerate(pats):  # плоскость участка по его точкам; нормаль наружу
        X = pts[lab == k]
        if len(X) < 50:
            planes.append(None)
            continue
        c0 = X.mean(0)
        n = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
        if n @ p["normal"] < 0:
            n = -n
        res = float(np.median(np.abs((X - c0) @ n)))
        planes.append((n, float(c0 @ n), res))
    cos20 = np.cos(np.radians(20))
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    box = {k: (pts[lab == k].min(0) - reach, pts[lab == k].max(0) + reach) for k in range(len(pats)) if planes[k]}
    all_pts, all_nrm, all_lab = pts, nrm, lab
    out = []
    for i in range(len(pats)):
        for j in range(i + 1, len(pats)):
            if planes[i] is None or planes[j] is None:
                continue
            (n1, d1, r1), (n2, d2, r2) = planes[i], planes[j]
            c = float(n1 @ n2)
            if abs(c) > cos20:
                continue
            ax = np.cross(n1, n2)
            ax /= np.linalg.norm(ax)
            # только точки там, где участки рядом: общая часть их габаритов с запасом
            b_lo, b_hi = np.maximum(box[i][0], box[j][0]), np.minimum(box[i][1], box[j][1])
            if (b_hi <= b_lo).any():
                continue
            sel = np.array(tree.query_ball_point((b_lo + b_hi) / 2, float(np.linalg.norm(b_hi - b_lo)) / 2), int)
            if len(sel) < 100:
                continue
            pts, nrm, lab = all_pts[sel], all_nrm[sel], all_lab[sel]
            p = np.linalg.solve(np.array([n1, n2, ax]), np.array([d1, d2, 0.0]))
            rel = pts - p
            t = rel @ ax
            xs = rel - t[:, None] * ax
            rad = np.linalg.norm(xs, axis=1)
            near = rad < reach
            # участок линии, у которого есть обе плоскости
            ti, tj = t[near & (lab == i)], t[near & (lab == j)]
            if len(ti) < 20 or len(tj) < 20:
                continue
            lo, hi = max(np.percentile(ti, 2), np.percentile(tj, 2)), min(np.percentile(ti, 98), np.percentile(tj, 98))
            if hi - lo < min_len:
                continue
            delta = max(0.06, 3 * max(r1, r2))
            e1, e2 = pts @ n1 - d1, pts @ n2 - d2
            cand = (near & (t > lo) & (t < hi) & (np.abs(e1) > delta) & (np.abs(e2) > delta)
                    & (nrm @ n1 > c - 0.1) & (nrm @ n2 > c - 0.1))
            if cand.sum() < 40:
                continue
            x = xs[cand]
            m = (n1 + n2) / (1 + c)
            side = np.median(x @ m)  # выпуклый угол: поверхность по ту же сторону, что и ось (x·m < 0)
            if side > 0:
                x = -x
            a = (1 - c) / (1 + c)
            b = x @ m
            disc = b * b - a * (x * x).sum(1)
            ok = disc >= 0
            if ok.sum() < 40 or ok.mean() < 0.6:
                continue
            R = (-b[ok] + np.sqrt(disc[ok])) / a
            Rm = float(np.median(R))
            spread = float(np.median(np.abs(R - Rm)) / max(Rm, 1e-9))
            fit = float(np.median(np.abs(np.linalg.norm(x[ok] + Rm * m, axis=1) - Rm)))
            if not (0.4 < Rm < reach / 2) or spread > 0.15 or fit > max(0.1, 0.1 * Rm):
                continue
            tt = t[cand][ok]
            # участки вдоль оси, где скругление подтверждено точками (перерыв — вырез, ребро, отверстие)
            bins = np.bincount(np.floor(tt - lo).astype(int), minlength=int(hi - lo) + 1)
            good = bins >= max(3, 0.25 * np.median(bins[bins > 0]))
            runs, k0 = [], None
            for k, g in enumerate(np.r_[good, False]):
                if g and k0 is None:
                    k0 = k
                elif not g and k0 is not None:
                    if k - k0 >= min_len:
                        runs.append((float(lo + k0), float(lo + k)))
                    k0 = None
            if not runs:
                continue
            out.append({"radius_mm": round(Rm, 1), "kind": "вогнутое" if side > 0 else "выпуклое",
                        "length_mm": round(sum(b_ - a_ for a_, b_ in runs), 1),
                        "spread": round(spread, 3), "points": int(ok.sum()), "planes": (i, j),
                        "geo": {"R": Rm, "n1": n1, "n2": n2, "p": p, "ax": ax, "concave": bool(side > 0), "runs": runs}})
    return sorted(out, key=lambda f: -f["length_mm"])


MIN_PIECE = 6.0   # мм: короче этого участок скругления пополам не делим


def _sector(cc, e1, _unused, ax, r0, r1, phi, t0, t1):
    """Призма вдоль оси: кольцевой сектор r0..r1, углы 0..phi от e1 к e2, от t0 до t1 по оси."""
    import cadquery as cq

    pl = cq.Plane(origin=tuple(cc + t0 * ax), xDir=tuple(e1), normal=tuple(ax))
    P = lambda r, a: (r * np.cos(a), r * np.sin(a))  # noqa: E731
    wp = (cq.Workplane(pl).moveTo(*P(r1, 0)).threePointArc(P(r1, phi / 2), P(r1, phi))
          .lineTo(*P(r0, phi)).threePointArc(P(r0, phi / 2), P(r0, 0)).close().extrude(t1 - t0))
    return wp.val()


def apply(body, fl: list[dict], budget: float = 240.0, trim_deg: float = 4.0, log: list | None = None,
          mesh=None, *, volumes=None, timings=None):
    """Заменяет в теле приближённые (сплайновые) скругления точными цилиндрическими гранями.

    У каждого участка скругления узкая полоса у дуги сначала вырезается (инструмент режет тело
    поперёк, без касаний), потом на её место ставится точный кольцевой сектор радиуса R. Сектор
    чуть короче дуги (на trim_deg с каждого конца): у самой линии касания цилиндр и плоскость
    расходятся на микроны, и булева операция там ненадёжна. Участок, после которого тело распалось
    или стало невалидным, делится пополам и пробуется по частям; что не вышло — пропускается.
    mesh — чистая поверхность скана: по ней быстро проверяется, что под вставкой есть материал
    (без неё — по самому телу, это около 1,5 с на точку).
    Возвращает (тело, сколько кусков построено, сколько участков пропущено)."""
    import os
    import time

    from cad_checks import InsideMemo, VolumeMemo, add_count, boolean_op, disjoint_tool_boxes, measured

    if volumes is None:
        volumes = VolumeMemo()
    t_start = time.time()
    grouped = os.environ.get("CAD_FILLET_GROUPS", "0") == "1"
    add_count(timings, "fillet_ops", 0)
    add_count(timings, "fillet_rolled_back_ops", 0)
    if mesh is not None:
        from scipy.spatial import cKDTree

        with measured(timings, "fillet_checks"):
            cen, nrm = mesh.triangles_center, mesh.face_normals
            tree = cKDTree(cen)

        def inside(q):  # точка под ближайшим куском поверхности
            with measured(timings, "fillet_checks"):
                k = tree.query(q)[1]
                return float((q - cen[k]) @ nrm[k]) < 0
    else:
        classifier = InsideMemo()

        def inside(q):
            with measured(timings, "fillet_checks"):
                return classifier.contains(body, q)
    jobs = []  # (вырез, вставка) по участкам
    skipped = 0
    for group, f in enumerate(fl):
        g = f["geo"]
        R, n1, n2, p, ax = g["R"], g["n1"], g["n2"], g["p"], g["ax"]
        c = float(n1 @ n2)
        m = (n1 + n2) / (1 + c)
        sgn = -1.0 if g["concave"] else 1.0
        cc = p - sgn * R * m            # ось: внутри материала у выпуклого, в воздухе у вогнутого
        r1v, r2v = sgn * n1, sgn * n2   # от оси к линиям касания
        phi_full = float(np.arccos(np.clip(c, -1, 1)))
        a = np.radians(trim_deg)
        phi = phi_full - 2 * a
        if phi < np.radians(20):
            skipped += len(g["runs"])
            continue
        w = np.cross(ax, r1v)
        ax_, flip = ax, 1.0
        if w @ r2v < 0:
            ax_, w, flip = -ax, -w, -1.0
        e1 = r1v * np.cos(a) + w * np.sin(a)
        mid = r1v * np.cos(phi_full / 2) + w * np.sin(phi_full / 2)
        h = min(0.5, R / 3)             # глубина точной вставки в материал
        hc = h / 2                      # глубина выреза (мельче вставки)
        far = R * float(np.linalg.norm(m)) + 0.5   # дальше острого угла
        def make(a0, a1, g=g, cc=cc, e1=e1, ax_=ax_, phi=phi, R=R, h=h, hc=hc, far=far, mid=mid):
            """(вырез, вставка) для куска участка a0..a1 вдоль оси; None — под вставкой нет материала."""
            depth = (R + h) if g["concave"] else (R - h)
            probes = [cc + ax_ * s_ + mid * depth for s_ in np.linspace(a0, a1, 5)]
            # материал под вставкой должен быть: иначе у тонкой стенки вставка вылезет внутрь
            if not all(inside(q) for q in probes):
                return None
            if g["concave"]:
                return (_sector(cc, e1, None, ax_, max(R - 1.0, 0.3 * R), R + hc, phi, a0, a1),
                        _sector(cc, e1, None, ax_, R, R + h, phi, a0, a1))
            return (_sector(cc, e1, None, ax_, R - hc, far, phi, a0, a1),
                    _sector(cc, e1, None, ax_, R - h, R, phi, a0, a1))

        for t0, t1 in g["runs"]:
            a0, a1 = (t0 + 0.3, t1 - 0.3) if flip > 0 else (-(t1 - 0.3), -(t0 + 0.3))
            try:
                pair = make(a0, a1)
            except Exception:  # noqa: BLE001
                pair = None
            jobs.append((pair, make, a0, a1, group))

    def note(*a):
        if log is not None:
            log.append(" ".join(str(x) for x in a))

    accepted_ops = 0

    def attempt(base, part, tag="", *, trial=False):
        """Вырез и вставка для набора участков; None, если тело распалось или стало невалидным.
        Объём здесь не считаем (у тела из тысяч сплайновых граней это ~10 с): булевы операции
        не могут изменить его больше, чем на объём инструментов, общая сверка — в конце."""
        nonlocal accepted_ops
        t0_ = time.time()
        ops = 0
        try:
            with measured(timings, "fillet_boolean"):
                ops += 1
                cut = boolean_op(base, [c for c, _ in part], preserve_inputs=trial)
                ops += 1
                new = boolean_op(cut, [a for _, a in part], fuse=True, preserve_inputs=trial)
            with measured(timings, "fillet_checks"):
                sols = new.Solids()
                why = ("тел " + str(len(sols))) if len(sols) != 1 else ("" if sols[0].isValid() else "невалидно")
        except Exception as e:  # noqa: BLE001  скругление — улучшение, без него тело то же
            sols, why = [], "ошибка " + type(e).__name__
        add_count(timings, "fillet_ops", ops)
        if why:
            add_count(timings, "fillet_rolled_back_ops", ops)
        else:
            accepted_ops += ops
        note(tag, why or "ок", round(time.time() - t0_, 1), "с")
        return None if why else sols[0]

    def tool_volume(pair):
        with measured(timings, "fillet_checks"):
            return pair[0].Volume() + pair[1].Volume()

    start = body
    ready = [j for j in jobs if j[0] is not None]
    note("участков", len(jobs), "без материала под вставкой", len(jobs) - len(ready))
    if not ready:
        return body, 0, skipped + len(jobs)
    done, tools = 0, 0.0
    new = attempt(body, [j[0] for j in ready], "все") if len(ready) > 1 else None
    if new is not None:  # все участки двумя операциями
        body, done = new, len(ready)
        tools = sum(tool_volume(j[0]) for j in ready)
    else:  # по одному; неудачный участок делим пополам: вдоль него может мешать одно место
        queue = list(ready)
        tried_groups = set()
        while queue and time.time() - t_start < budget:
            pair, make, a0, a1, group = queue.pop(0)
            if grouped and group not in tried_groups:
                tried_groups.add(group)
                # Original jobs of one fillet are contiguous; split pieces are appended later.
                end = next((i for i, job in enumerate(queue) if job[4] != group), len(queue))
                if 0 < end < len(ready) - 1:
                    group_pairs = [pair, *[job[0] for job in queue[:end]]]
                    if disjoint_tool_boxes(group_pairs):
                        new = attempt(body, group_pairs, f"скругление {group + 1}", trial=True)
                    else:
                        new = None
                        note(f"скругление {group + 1}", "коробки инструментов пересекаются")
                    if new is not None:
                        body, done = new, done + len(group_pairs)
                        tools += sum(tool_volume(p) for p in group_pairs)
                        del queue[:end]
                        continue
            new = attempt(body, [pair], f"{a0:.0f}..{a1:.0f}")
            if new is not None:
                body, done = new, done + 1
                tools += tool_volume(pair)
            elif a1 - a0 >= 2 * MIN_PIECE:
                am = (a0 + a1) / 2
                for b0, b1 in ((a0, am), (am, a1)):
                    try:
                        sub = make(b0, b1)
                    except Exception:  # noqa: BLE001
                        sub = None
                    if sub is not None:
                        queue.append((sub, make, b0, b1, group))
    if done:
        try:  # общая сверка: тело не могло измениться больше, чем на объём инструментов
            with measured(timings, "fillet_checks"):
                after = body.Volume()
                if abs(after - volumes.get(start)) > tools + 1.0:
                    note("объём изменился больше объёма инструментов, скругления отменены")
                    add_count(timings, "fillet_rolled_back_ops", accepted_ops)
                    return start, 0, skipped + len(jobs)
                volumes.remember(body, after)
        except Exception:  # noqa: BLE001
            add_count(timings, "fillet_rolled_back_ops", accepted_ops)
            return start, 0, skipped + len(jobs)
    return body, done, skipped + max(0, len(jobs) - done)


def summary(fl: list[dict]) -> str:
    """Строка для отчёта: скругления, сгруппированные по радиусу."""
    if not fl:
        return ""
    groups: dict[float, list[dict]] = {}
    for f in fl:
        groups.setdefault(round(f["radius_mm"] * 2) / 2, []).append(f)
    parts = [f"R ≈ {str(r).replace('.', ',')} мм × {len(g)}" for r, g in sorted(groups.items())]
    return "скругления между плоскими гранями (по скану, радиус по точкам скана): " + ", ".join(parts)


if __name__ == "__main__":
    import surface

    scan = trimesh.load(sys.argv[1], force="mesh")
    fl = find_fillets(surface.surface_mesh(scan) if "--raw" not in sys.argv else scan)
    for f in fl:
        print(f)
    print(summary(fl) or "скруглений не найдено")
