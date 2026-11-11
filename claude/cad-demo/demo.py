"""Демо: скан повреждённой детали → CAD-модель целой детали (STEP) с проверкой.

    python demo.py samples/bracket_scan.stl --notes "угол плиты отколот"

Цикл: замеры скана → ИИ пишет CadQuery → запуск → сравнение со сканом →
ИИ правит → ... → лучшая попытка в STEP + отчёт report.html.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import shutil
import subprocess
import sys

# Windows: вывод в канал по умолчанию в cp1251, а страница и проверки читают UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
import time
from pathlib import Path

import logging

import numpy as np
import trimesh

logging.getLogger("trimesh").setLevel(logging.ERROR)

import compare as cmpmod
import holes as holesmod
import gates
import geometric
import llm
import scan_analysis

HERE = Path(__file__).parent


_STEP = {"t": 0.0, "last": 0.0, "limit": None}


def log(msg: str, limit: float | None = None) -> None:
    """limit — предел шага в секундах: пока шаг идёт, раз в минуту печатается, сколько прошло."""
    now = time.time()
    if msg.lstrip()[:3] in ("1/4", "2/4", "3/4", "4/4"):
        _STEP.update(t=now, limit=limit)
    _STEP["last"] = now
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _ticker() -> None:
    # длинный шаг без новых строк выглядит как зависание: напоминаем, что работа идёт
    while True:
        time.sleep(5)
        now = time.time()
        if _STEP["t"] and now - _STEP["last"] >= 60:
            m = int((now - _STEP["t"]) // 60)
            lim = f" из предела {round(_STEP['limit'] / 60)} мин" if _STEP["limit"] else ""
            log(f"    …идёт {m} мин{lim}, работа продолжается")


import threading  # noqa: E402

threading.Thread(target=_ticker, daemon=True).start()


DRAFT: dict = {}  # путь к STEP черновика, который ИИ может взять за основу (переменная draft)


def run_code(code: str, it_dir: Path, timeout: int = 90) -> dict:
    (it_dir / "model.py").write_text(code, encoding="utf-8")
    try:
        # код от ИИ не должен видеть ключи и прочие секреты окружения
        env = {k: v for k, v in os.environ.items()
               if not any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))}
        if DRAFT.get("step"):
            env["CAD_DRAFT"] = str(DRAFT["step"])
        subprocess.run([sys.executable, str(HERE / "cad_runner.py"), str(it_dir / "model.py"), str(it_dir)],
                       timeout=timeout, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"код считался дольше {timeout} с и был остановлен"}
    p = it_dir / "run.json"
    if not p.exists():
        return {"ok": False, "error": "процесс построения упал без отчёта"}
    return json.loads(p.read_text(encoding="utf-8"))


def want_model(mode: str, name: str) -> bool:
    """Запускать ли ИИ-модель-кандидата (stepcad, cadena): auto — при видеокарте NVIDIA и найденном коде."""
    if mode == "off":
        return False
    try:
        mod = __import__(f"{name}_worker")
        if getattr(mod, f"{name}_dir")() is None:
            return False
        if mode == "on":
            return True
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def score(run: dict, cmp: dict | None) -> float:
    if not run.get("ok") or not run.get("valid") or cmp is None:
        return -1.0
    return cmp["within_tol_share"] - 3 * cmp["outside_share"]


def render_views(mesh: trimesh.Trimesh, path: Path) -> None:
    """Скан с четырёх сторон (затенённые грани, оси в мм) — чтобы ИИ видел форму и скол."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = mesh
    if len(m.faces) > 40000:  # для картинки хватит части граней
        idx = np.random.default_rng(0).choice(len(m.faces), 40000, replace=False)
        tri, nrm = m.triangles[idx], m.face_normals[idx]
    else:
        tri, nrm = m.triangles, m.face_normals
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.6 * np.clip(nrm @ light, 0, 1)
    fig = plt.figure(figsize=(10, 9), dpi=90)
    lo, hi = m.bounds
    for i, (el, az, name) in enumerate(((28, -60, "общий вид"), (90, -90, "сверху (X вправо, Y вверх)"),
                                        (0, -90, "спереди (X вправо, Z вверх)"), (0, 0, "сбоку (Y вправо, Z вверх)"))):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[shade, shade, shade * 1.05].clip(0, 1),
                                             edgecolor="none"))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo + 1e-6)
        ax.view_init(el, az)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=6)
    fig.suptitle("Скан (выровнен, мм)", fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render(cmp: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, dev = cmp["_scan_pts"], cmp["_scan_dev"]
    tol = cmp["tolerance_mm"]
    col = np.full((len(pts), 3), 0.72)
    out, ins = dev < -tol, dev > tol
    col[out] = [0.85, 0.15, 0.1]
    col[ins] = [0.15, 0.35, 0.9]
    mp, md = cmp["_model_pts"], cmp["_model_dev"]
    add = md > tol
    fig = plt.figure(figsize=(11, 4.6), dpi=110)
    for i, (el, az) in enumerate(((28, -60), (88, -90))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(*pts[~(out | ins)].T, s=0.6, c=col[~(out | ins)], depthshade=False)
        ax.scatter(*mp[add].T, s=1.2, c=[[0.1, 0.65, 0.3]], depthshade=False)
        ax.scatter(*pts[ins].T, s=3, c=col[ins], depthshade=False)
        ax.scatter(*pts[out].T, s=4, c=col[out], depthshade=False)
        ax.view_init(el, az)
        ax.set_box_aspect(np.ptp(pts, axis=0) + 1e-6)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.02, "серый: совпало (в допуске)   красный: модели не хватает материала   "
             "синий: модель больше скана (скол)   зелёный: достроено, в скане этого нет",
             ha="center", fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def importmap() -> str:
    """three.js вшит в отчёт (data: URL), чтобы 3D работало без интернета."""
    v = HERE / "vendor"
    url = lambda f: "data:text/javascript;base64," + b64(v / f)
    return json.dumps({"imports": {"three": url("three.module.min.js"),
                                   "three/addons/controls/OrbitControls.js": url("OrbitControls.js"),
                                   "three/addons/loaders/STLLoader.js": url("STLLoader.js")}})


def _holes_row(rows):
    """Отверстия: сколько круглых отверстий нашлось в скане и все ли они в модели того же размера."""
    title = "Отверстия на месте и того же диаметра, что в скане"
    if rows is None:
        return (None, title, "не проверялось — сверить по таблице размеров ниже или по детали")
    if not rows:
        return (None, title, "круглых отверстий в уцелевшей части скана не нашлось — сверить по детали")
    bad = [r for r in rows if not r["ok"]]
    note = f"найдено {len(rows)}, совпали {len(rows) - len(bad)}"
    if bad:
        note += "; не совпали: " + "; ".join(
            f"Ø{r['d_scan']} в скане → " + (f"Ø{r['d_model']} в модели" if r["d_model"] else "в модели нет") for r in bad)
    return (not bad, title, note)



def check_step(step: Path) -> tuple[bool, str]:
    """Готовый STEP читается заново, как его прочтёт SolidWorks или КОМПАС: тело одно и валидное,
    единицы длины — миллиметры. Это замена открытия в CAD, а не само открытие."""
    import cadquery as cq

    t = time.time()
    try:
        shape = cq.importers.importStep(str(step)).val()
    except Exception as ex:  # noqa: BLE001
        return False, f"не читается: {type(ex).__name__}"
    with open(step, encoding="latin-1") as fh:
        head = fh.read(200_000)
    mm = "SI_UNIT(.MILLI.,.METRE.)" in head.replace(" ", "")
    solids, valid = len(shape.Solids()), bool(shape.isValid())
    note = (f"тел {solids}, валидно: {'да' if valid else 'нет'}, единицы: {'мм' if mm else 'не мм'}, "
            f"граней {len(shape.Faces())}, файл {step.stat().st_size / 1e6:.1f} МБ, чтение {time.time() - t:.0f} с")
    return solids == 1 and valid and mm, note

def print_ready_html(best: dict | None, ctx: dict) -> str:
    """Готово ли к печати: что проверено автоматически, а что — только инженером."""
    e = html.escape
    if not best or not best["run"].get("ok"):
        return ""
    run, cmp = best["run"], best["cmp"] or {}
    restored = (best.get("meta") or {}).get("restored") or []
    one = run.get("solids") == 1 and not run.get("voids")
    fit = cmp.get("within_tol_share", 0) >= ctx["target"] and cmp.get("outside_share", 1) < 0.02
    rows = [
        (one, "Одно цельное тело без закрытых пустот внутри",
         f"тел: {run.get('solids')}, пустот: {run.get('voids', '—')}"),
        (bool(run.get("valid")), "Тело проходит проверку геометрии CAD-ядра (OpenCascade)", ""),
        (fit, f"Совпадает с уцелевшей частью скана: не меньше {ctx['target']*100:.0f}% в допуске {ctx['tol']} мм, "
              "нехватка материала меньше 2%",
         f"{cmp.get('within_tol_share', 0)*100:.1f}%, нехватка {cmp.get('outside_share', 0)*100:.2f}%"),
        (None, "Скол восстановлен по замыслу",
         ("достроено: " + "; ".join(map(str, restored))) if restored else "достроенных мест модель не указала"),
        _holes_row(ctx.get("holes")),
    ]
    if ctx.get("step_check"):
        rows.append((ctx["step_check"][0], "Файл STEP открывается заново: одно валидное тело, единицы — миллиметры",
                     ctx["step_check"][1]))
    mark = {True: "<b style='color:var(--ok)'>да</b>", False: "<b class=err>нет</b>", None: "инженер"}
    auto = [r for r in rows if r[0] is not None]
    done = sum(1 for r in auto if r[0])
    body = "".join(f"<tr><td>{mark[ok]}</td><td>{e(t)}</td><td class=small>{e(n)}</td></tr>" for ok, t, n in rows)
    return (f"<section><h2>Готово к печати?</h2><p>Автоматические проверки: <b>{done} из {len(auto)}</b>. "
            "Пункты с отметкой «инженер» проверяет человек: машина не знает замысла детали наверняка.</p>"
            f"<table><tr><th class=chk>Итог</th><th>Проверка</th><th>Что получилось</th></tr>{body}</table></section>")


def model_dims(step: Path | None) -> dict:
    """Что в итоговой модели: габарит и цилиндры с осью вдоль Z (центр, Ø, отверстие или нет)."""
    if step is None or not step.exists():
        return {}
    try:
        import cadquery as cq
        from hints import cylinders

        bb = cq.importers.importStep(str(step)).val().BoundingBox()
        cyl = [c for c in cylinders(step, holes_only=False) if abs(abs(c["d"][2]) - 1) < 1e-3]
        return {"box": (bb.xlen, bb.ylen, bb.zlen), "cyl": cyl}
    except Exception:  # noqa: BLE001 — таблица без столбца модели лучше, чем отчёт без таблицы
        return {}


def _model_d(md: dict, d: float, hole: bool, c=None) -> str:
    """Диаметр той же поверхности в модели: ближайший по размеру цилиндр (у отверстия — с тем же центром)."""
    best = None
    for x in md.get("cyl", []):
        if x["hole"] != hole or abs(2 * x["r"] - d) > 1.0:
            continue
        if c is not None and ((x["o"][0] - c[0]) ** 2 + (x["o"][1] - c[1]) ** 2) ** 0.5 > 1.0:
            continue
        if best is None or abs(2 * x["r"] - d) < abs(2 * best - d):
            best = x["r"]
    return "" if best is None else f"{2 * best:.2f}"


def dims_table(sheet: dict, model_step: Path | None = None) -> str:
    """Таблица размеров для сверки штангенциркулем (можно распечатать): по скану и в итоговой модели."""
    e = html.escape
    md = model_dims(model_step)
    rows, seen = [], set()
    lo, hi = sheet["bounds_mm"]
    box = md.get("box")
    rows.append(("Габарит X × Y × Z", f"{hi[0]-lo[0]:.2f} × {hi[1]-lo[1]:.2f} × {hi[2]-lo[2]:.2f}",
                 " × ".join(f"{v:.2f}" for v in box) if box else "", "по скану — с учётом скола"))
    for p in sheet["horizontal_planes"]:
        if p["facing"] == "смотрит вверх":
            rows.append(("Высота площадки от опорной плоскости", f"{p['z_mm']:.2f}", "", ""))
    for sec in sheet["sections"]:
        for reg in sec["regions"]:
            o = reg["outer"]
            if o.get("shape") == "окружность":
                k = ("o", round(o["diameter"]))
                if k not in seen:
                    seen.add(k)
                    rows.append(("Наружный Ø", f"{o['diameter']:.2f}", _model_d(md, o["diameter"], False), f"на высоте {sec['z_mm']}"))
            elif o.get("partial_circle"):
                k = ("o", round(o["partial_circle"]["diameter"]))
                if k not in seen:
                    seen.add(k)
                    rows.append(("Наружный Ø (по уцелевшей дуге)", f"{o['partial_circle']['diameter']:.2f}",
                                 _model_d(md, o["partial_circle"]["diameter"], False), f"на высоте {sec['z_mm']}"))
            for h in reg["inner"]:
                d = h.get("diameter") or (h.get("partial_circle") or {}).get("diameter")
                c = h.get("center_xy") or (h.get("partial_circle") or {}).get("center_xy")
                if not d:
                    continue
                k = ("h", round(d), round(c[0]), round(c[1]))
                if k not in seen:
                    seen.add(k)
                    rows.append(("Отверстие Ø" + (" (с пазом)" if h.get("partial_circle") else ""), f"{d:.2f}",
                                 _model_d(md, d, True, c), f"центр {c}"))
            for pat in reg.get("hole_patterns", []):
                k = ("p", pat["pattern"], round(pat["hole_diameter"]))
                if k in seen:
                    continue
                seen.add(k)
                if pat.get("pcd_diameter"):
                    rows.append(("Диаметр окружности центров отверстий", f"{pat['pcd_diameter']:.2f}",
                                 "", f"{pat['likely_total']} отв., шаг {360 / pat['likely_total']:.0f}°"))
                if pat.get("sides_mm"):
                    rows.append(("Межосевые отверстий (прямоугольник)", " × ".join(f"{v:.2f}" for v in pat["sides_mm"]), "", ""))
    body = "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td><b>{e(m)}</b></td><td>{e(c)}</td><td class=chk></td></tr>"
                   for a, b, m, c in rows)
    return (f"<section><h2>Размеры для сверки штангенциркулем</h2><p class=small>Мм. «По скану» — грубый замер "
            f"по сечениям скана (шум ~{sheet['diagnostics'].get('noise_mm', 0)} мм), «в модели» — что в итоговом STEP "
            f"после уточнения; пусто — в модели такой поверхности нет или размер не круглый. Последний столбец — для "
            f"отметки инженера (страницу можно распечатать).</p><table><tr><th>Что</th><th>По скану</th><th>В модели</th>"
            f"<th>Где</th><th>Замер</th></tr>"
            f"{body}</table></section>")




_EXACT = ("PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS")


def _faces_cell(run: dict) -> str:
    """Сколько граней и сколько из них точные (плоскость, цилиндр, конус, сфера, тор)."""
    if not run.get("ok"):
        return ""
    ft = run.get("face_types") or {}
    exact = sum(v for k, v in ft.items() if k in _EXACT)
    return f"{run.get('faces', '')} ({exact})" if ft else str(run.get("faces", ""))


def _editable(code: str) -> str:
    # тело, загруженное из готового STEP, — это поверхность без размеров; программа с размерами
    # (выдавливания, отверстия) правится числом
    if not code:
        return ""
    return "только форма (STEP)" if "importStep" in code else "с размерами"

def _viewer_mesh(m, n):
    """Лёгкая копия модели для 3D-просмотра: вершины сливаются по сетке (кластеризация).
    Грани STL гладкого тела разбиты каждая отдельно и не сшиты, поэтому обычное упрощение
    почти не работает; кластеризации топология не важна. Ячейка растёт, пока граней не станет ≤ n."""
    cell = float(np.linalg.norm(m.extents)) / 1000
    while True:
        _, inv = np.unique(np.round(m.vertices / cell).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        f = inv[m.faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        _, u = np.unique(np.sort(f, 1), axis=0, return_index=True)
        if len(u) <= n:
            break
        cell *= 1.3
    v = np.zeros((inv.max() + 1, 3))
    np.add.at(v, inv, m.vertices)
    v /= np.bincount(inv)[:, None]
    return trimesh.Trimesh(v, f[np.sort(u)], process=False)

def write_report(out: Path, ctx: dict) -> Path:
    e = html.escape
    its = ctx["iterations"]
    best = ctx["best"]
    rows = []
    for it in its:
        c = it.get("cmp")
        status = ("ошибка кода" if not it["run"].get("ok") else
                  ("не прошла обязательные проверки: " + "; ".join(it["gates"]) if it.get("gates") else "построено"))
        rows.append(f"<tr{' class=best' if best and it['n'] == best['n'] else ''}><td>{it['n']}</td><td>{e(it.get('by', ''))}: {status}</td>"
                    f"<td>{'' if not c else format(c['within_tol_share'] * 100, '.1f') + '%'}</td>"
                    f"<td>{'' if not c else format(c['outside_share'] * 100, '.2f') + '%'}</td>"
                    f"<td>{'' if not c else format(c['median_abs_dev_mm']) + ' / ' + format(c['mean_abs_dev_mm'])}</td>"
                    f"<td>{'' if not c else c['p95_abs_dev_mm']}</td>"
                    f"<td>{_faces_cell(it['run'])}</td><td>{_editable(it.get('code') or '')}</td>"
                    f"<td>{it['seconds']:.0f} с</td></tr>")
    blocks = []
    for it in its:
        c, r, meta = it.get("cmp"), it["run"], it.get("meta", {})
        img = it.get("img")
        blocks.append(f"""
<section class=it><h3>Попытка {it['n']} · {e(it.get('by', ''))}{' — лучшая' if best and it['n'] == best['n'] else ''}</h3>
<p><b>Замысел:</b> {e(str(meta.get('plan', '')))}</p>
{'<p class=err><b>Ошибка:</b> ' + e(r.get('error', '')) + '</p>' if not r.get('ok') else ''}
{f'<p>Тело валидно: <b>{"да" if r.get("valid") else "нет"}</b>, объём {r.get("volume_mm3")} мм³, граней {r.get("faces")}</p>' if r.get('ok') else ''}
{f'<img src="data:image/png;base64,{b64(img)}" alt="карта отклонений">' if img else ''}
<details><summary>Код модели</summary><pre>{e(it['code'])}</pre></details>
<details><summary>Что вернулось на доработку (отчёт проверки)</summary><pre>{e(it.get('feedback', ''))}</pre></details>
</section>""")
    meta = best.get("meta", {}) if best else {}
    lst = lambda xs: "".join(f"<li>{e(str(x))}</li>" for x in (xs or [])) or "<li>—</li>"
    d = ctx["sheet"]["diagnostics"]
    truth_html = ""
    if ctx.get("truth"):
        t = ctx["truth"]
        truth_html = (f"<section><h2>Проверка по эталону (только для тестовой детали)</h2><p>Целая деталь до поломки "
                      f"известна, потому что мы её сами сломали. Совпадение модели с эталоном: "
                      f"<b>{t['within_tol_share']*100:.1f}%</b> точек эталона в допуске {t['tolerance_mm']} мм, "
                      f"95% ближе {t['p95_abs_dev_mm']} мм, включая восстановленный угол.</p>"
                      + ("<table><tr><th>Попытка</th><th>С эталоном</th><th>Отверстия</th><th>Ошибка Ø, средняя</th>"
                         "<th>Ошибка Ø, наибольшая</th></tr>" + "".join(
                             f"<tr{' class=bestrow' if r['best'] else ''}><td>{e(str(r['n']))} — {e(r['by'])}</td>"
                             f"<td>{r['fit']*100:.1f}%</td><td>{r['holes'] - r['missing']} из {r['holes']}</td>"
                             f"<td>{'—' if r['d_err'] is None else str(r['d_err']) + ' мм'}</td>"
                             f"<td>{'—' if r['d_max'] is None else str(r['d_max']) + ' мм'}</td></tr>"
                             for r in t.get("by_attempt", [])) + "</table>" if t.get("by_attempt") else "")
                      + "</section>")
    viewer = ""
    if best and (out / "model.stl").exists():
        pts = best["cmp"]["_scan_pts"][::2]
        dev = best["cmp"]["_scan_dev"][::2]
        tol = ctx["tol"]
        mdl = trimesh.load(out / "model.stl", force="mesh")
        if len(mdl.faces) > 100000:  # гладкое тело даёт сотни тысяч треугольников: отчёт весил бы 70+ МБ
            mdl = _viewer_mesh(mdl, 100000)
        scan_m = ctx["scan_mesh"]
        _, dist, _ = trimesh.proximity.closest_point(scan_m, mdl.triangles_center)
        tri = mdl.triangles.reshape(-1, 3).astype(np.float32)
        fc = np.where(dist[:, None] > tol, [[0.13, 0.72, 0.36]], [[0.50, 0.64, 0.84]]).astype(np.float32)
        col = np.repeat(fc, 3, axis=0)
        pc = np.where(dev[:, None] < -tol, [[0.85, 0.1, 0.1]], np.where(dev[:, None] > tol, [[0.1, 0.3, 0.95]], [[0.3, 0.3, 0.3]])).astype(np.float32)
        f32 = lambda x: base64.b64encode(np.ascontiguousarray(x, dtype=np.float32).tobytes()).decode()
        added_pct = float((mdl.area_faces[dist > tol]).sum() / mdl.area * 100)
        viewer = f"""<section><h2>3D: что было в скане и что достроено</h2>
<div class=vbtns><button data-v=both class=on>Обе</button><button data-v=scan>Только скан</button><button data-v=model>Только модель</button></div>
<div id=v></div>
<p class=small>Крутить мышью, колесо — приблизить. <b style="color:#21b85c">Зелёным</b> — поверхности модели, которых в скане нет
(достроено, {added_pct:.0f}% площади). Точки — скан: красные — модели там не хватает материала, синие — скол.</p>
<script type=importmap>{importmap()}</script>
<script type=module>
import * as THREE from 'three';import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
const f32=b=>new Float32Array(Uint8Array.from(atob(b),c=>c.charCodeAt(0)).buffer);
const el=document.getElementById('v');const W=()=>el.clientWidth;const r=new THREE.WebGLRenderer({{antialias:true}});r.setPixelRatio(window.devicePixelRatio||1);r.setSize(W(),480);el.appendChild(r.domElement);
const s=new THREE.Scene();s.background=new THREE.Color(0xf4f5f7);const cam=new THREE.PerspectiveCamera(40,W()/480,0.1,5000);
s.add(new THREE.HemisphereLight(0xffffff,0x8899aa,2.0));const dl=new THREE.DirectionalLight(0xffffff,1.6);dl.position.set(50,-80,120);s.add(dl);
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(f32("{f32(tri)}"),3));g.setAttribute('color',new THREE.BufferAttribute(f32("{f32(col)}"),3));g.computeVertexNormals();
const mm=new THREE.MeshStandardMaterial({{vertexColors:true,transparent:true,opacity:0.6,metalness:0.1,roughness:0.6,side:THREE.DoubleSide}});const m=new THREE.Mesh(g,mm);s.add(m);
const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(f32("{f32(pts)}"),3));pg.setAttribute('color',new THREE.BufferAttribute(f32("{f32(pc)}"),3));
const p=new THREE.Points(pg,new THREE.PointsMaterial({{size:0.7,vertexColors:true}}));s.add(p);
g.computeBoundingSphere();const c=g.boundingSphere.center,R=g.boundingSphere.radius;cam.up.set(0,0,1);cam.position.set(c.x+R*1.6,c.y-R*2.0,c.z+R*1.4);
const oc=new OrbitControls(cam,r.domElement);oc.target.copy(c);oc.update();(function a(){{requestAnimationFrame(a);r.render(s,cam);}})();
document.querySelectorAll('.vbtns button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.vbtns button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
const v=b.dataset.v;m.visible=v!=='scan';p.visible=v!=='model';mm.opacity=v==='model'?1:0.6;mm.transparent=v!=='model';mm.needsUpdate=true;}});
addEventListener('resize',()=>{{r.setSize(W(),480);cam.aspect=W()/480;cam.updateProjectionMatrix();}});
</script></section>"""
    dims = dims_table(ctx["sheet"], out / "model.step" if best else None)
    page = f"""<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Скан → CAD: отчёт</title><style>
:root{{--bg:#fff;--fg:#1d2330;--mut:#5b6474;--line:#e3e6eb;--acc:#2f6fd6;--warn:#b3261e;--ok:#1e7d3c}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1060px;padding:20px 16px 60px}}
h1{{font-size:24px;margin:.2em 0}}h2{{font-size:19px;margin-top:1.6em;border-bottom:1px solid var(--line);padding-bottom:4px}}
.mode{{display:inline-block;padding:3px 10px;border-radius:12px;background:#fff3cd;font-size:13px}}
.mode.live{{background:#dff3e4}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;font-size:14px}}
tr.best td{{background:#eef5ff;font-weight:600}}img{{max-width:100%}}pre{{background:#f6f7f9;padding:10px;overflow:auto;font-size:12.5px}}
.err{{color:var(--warn)}}.small{{color:var(--mut);font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:10px}}.vbtns{{margin:6px 0}}.vbtns button{{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 12px;margin-right:6px;cursor:pointer;font:inherit}}.vbtns button.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
.chk{{width:70px}}@media print{{#v,.vbtns,details,a.btn{{display:none}}}}.card b{{font-size:20px;display:block}}#v{{width:100%;height:480px;border:1px solid var(--line)}}
a.btn{{display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;border-radius:6px;background:var(--acc);color:#fff;text-decoration:none}}
</style></head><body>
<h1>Скан повреждённой детали → CAD-модель</h1>
<p class=small>{e(ctx['started'])} · файл скана: {e(ctx['scan_name'])}</p>
<p><span class="mode{' live' if ctx['mode'] in ('anthropic', 'openai') else ''}">Режим: {e(ctx['mode_label'])}</span></p>
{'<p class=err><b>' + e(ctx['test_label']) + '</b></p>' if ctx.get('test_label') else ''}
<section><h2>Итог</h2><div class=cards>
<div class=card>Попыток<b>{len(its)}</b></div>
<div class=card>Уцелевшей части скана в допуске {ctx['tol']} мм<b>{(best['cmp']['within_tol_share']*100 if best else 0):.1f}%</b></div>
<div class=card>Модели не хватает материала<b>{(best['cmp']['outside_share']*100 if best else 0):.2f}%</b></div>
<div class=card>Время<b>{ctx['total_s']:.0f} с</b></div></div>
<p>{'<b>Модель сошлась с уцелевшей частью скана.</b> Это не доказывает, что достроенное место верно: его проверяет инженер.' if ctx['converged'] else '<b class=err>Не сошлась до цели</b>: смотреть лучшую попытку и замечания.' if best else '<b class=err>Ни одна попытка не дала тела.</b>'}
Цель: не менее {ctx['target']*100:.0f}% точек скана в допуске и почти нет мест, где модели не хватает материала. Точки на сколе в допуск не попадают, так и должно быть.</p>
{('<a class=btn href="model.step" download>Скачать STEP (КОМПАС-3D, SolidWorks и др.)</a><a class=btn href="model.stl" download>STL (для печати)</a><a class=btn href="model.py" download>Код модели</a>'
  '<p class=filehint hidden>Отчёт открыт как файл, поэтому браузер может показать STEP вместо скачивания. '
  'Все файлы лежат в папке: <b>' + e(str(out.resolve())) + '</b> (или запускайте через start.bat — там кнопки скачивают).</p>'
  '<script>if(location.protocol==="file:")document.querySelectorAll(".filehint").forEach(function(x){x.hidden=false})</script>') if best else ''}
</section>
{print_ready_html(best, ctx)}
{truth_html}
<section><h2>Что восстановлено и что должен проверить инженер</h2>
<p><b>Как строил:</b> {e(str(meta.get('plan', '—')))}</p>
<h3>Достроено (этого нет в скане)</h3><ul>{lst(meta.get('restored'))}</ul>
<h3>Допущения</h3><ul>{lst(meta.get('assumptions'))}</ul>
<h3>Проверить инженеру</h3><ul>{lst(meta.get('check_by_engineer'))}</ul>
<p class=small>Модель — черновик для инженера. Производственное решение принимает человек, проверив размеры по детали (при желании в КОМПАС-3D или SolidWorks).</p></section>
{viewer}
{dims}
<section><h2>Журнал попыток</h2><table><tr><th>#</th><th>Результат</th><th>В допуске</th><th>Не хватает</th><th>Отклонение, мм: медиана / среднее</th><th>95% точек ближе, мм</th><th>Граней всего (точных)</th><th>Программа построения</th><th>Время</th></tr>{''.join(rows)}</table>
{''.join(blocks)}</section>
<section><h2>Скан</h2><p>Граней {d['faces']}, замкнут: {'да' if d['watertight'] else 'нет'}, кусков: {d['components']}, габарит {d['extents_mm']} мм.
{' '.join(e(w) for w in d['warnings'])}</p>
<h3>Сводка замеров (геометрия, без ИИ)</h3><ul>{''.join('<li>' + e(x) + '</li>' for x in ctx['sheet'].get('summary', []))}</ul>
<details><summary>Лист замеров, который получил ИИ</summary><pre>{e(json.dumps(ctx['sheet'], ensure_ascii=False, indent=1))}</pre></details></section>
<section><h2>Ограничения</h2><ul>
<li>STEP — точная геометрия (B-rep), но не дерево построения SolidWorks. Открывается в любой CAD (КОМПАС-3D, SolidWorks, Fusion, FreeCAD) как готовое тело.</li>
<li>Работает для деталей из плоскостей, отверстий, цилиндров, рёбер. Свободные поверхности (литьё, органика) этим способом не восстанавливаются.</li>
<li>Точность ограничена сканом: единицы и масштаб берутся из файла как есть.</li></ul></section>
</body></html>"""
    p = out / "report.html"
    p.write_text(page, encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Скан повреждённой детали → CAD (STEP) с проверкой")
    ap.add_argument("scan", help="STL/OBJ/PLY скана")
    ap.add_argument("--notes", default="", help="заметки мастера: что сломано, что важно")
    ap.add_argument("--image", help="фото или эскиз детали (png/jpg), необязательно")
    ap.add_argument("--mode", choices=["auto", "anthropic", "openai", "replay", "scripted", "geometric"], default="auto")
    ap.add_argument("--units", choices=["mm", "cm", "m", "in"], default="mm",
                    help="в чём записан скан: STL единиц не хранит (по умолчанию мм)")
    ap.add_argument("--no-restore", action="store_true",
                    help="не достраивать ничего (деталь целая); то же, если в заметке «деталь целая» или «не достраивать»")
    ap.add_argument("--geo-timeout", type=int, default=120, help="сколько секунд ждать попытку 0 (алгоритм без ИИ)")
    ap.add_argument("--no-geometric", action="store_true", help="не строить черновик без ИИ (попытка 0)")
    ap.add_argument("--stepcad", choices=["auto", "on", "off"], default="off",
                    help="кандидат StepCAD (экспериментально; на сканах студии пока грубый, ~15 мин): по умолчанию выключен; "
                         "auto — только при видеокарте NVIDIA "
                         "и найденном StepCAD (STEPCAD_DIR); on — и на процессоре (медленно, ~15 мин)")
    ap.add_argument("--cadena", choices=["auto", "on", "off"], default="off",
                    help="кандидат CADENA (экспериментально: редактируемая программа построения по снимкам скана): "
                         "по умолчанию выключен; auto — только при видеокарте NVIDIA и найденной CADENA (CADENA_DIR); "
                         "on — и на процессоре (~2–10 мин)")
    ap.add_argument("--smooth", choices=["auto", "off"], default="auto",
                    help="гладкое тело из B-сплайн граней для кривых деталей (попытка 0к)")
    ap.add_argument("--p2s", choices=["auto", "on", "off"], default="auto",
                    help="уточнить грани лучшей модели по скану через Point2STEP; auto — если он установлен "
                         "(POINT2STEP_DIR или папка point2step рядом / в домашней)")
    ap.add_argument("--model", help="модель ИИ (по умолчанию: claude-sonnet-5-5 / gpt-4.1)")
    ap.add_argument("--replay", help="llm_log.json для режимов replay/scripted")
    ap.add_argument("--ai-input", choices=["tools", "sheet"], default="tools",
                    help="tools: ИИ сам опрашивает сетку скана инструментами (без картинок); "
                         "sheet: только лист замеров и картинки видов скана")
    ap.add_argument("--max-tool-calls", type=int, default=30, help="вызовов инструментов на одну попытку ИИ")
    ap.add_argument("--budget-usd", type=float, default=1.0, help="остановить ИИ, если оценка расхода выше")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.3, help="допуск сравнения, мм")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--truth", help="эталон (STEP/STL) для тестовой детали — проверка в конце")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    sp = Path(a.scan).resolve()
    if not a.truth and sp.parent == (HERE / "samples").resolve() and sp.stem.endswith("_scan"):
        t = sp.with_name(sp.stem.replace("_scan", "_truth") + ".step")
        if t.exists():
            a.truth = str(t)  # эталон есть только у тестовых деталей
    mode = llm.auto_mode() if a.mode == "auto" else a.mode
    if mode == "scripted" and not a.replay:
        stem = Path(a.scan).stem.split("-")[-1].replace("_scan", "")
        cand = HERE / "samples" / f"scripted_{stem}.json"
        a.replay = str(cand if cand.exists() else HERE / "samples" / "scripted_bracket.json")
    if mode == "geometric" and a.no_geometric:
        sys.exit("Ключ ИИ не найден (переменная ANTHROPIC_API_KEY пуста), а попытка без ИИ выключена (--no-geometric): "
                 "строить нечем. В PowerShell ключ задаётся так: $env:ANTHROPIC_API_KEY=\"...\"; в cmd: set ANTHROPIC_API_KEY=...")
    if mode == "geometric" and a.mode == "auto":
        print("ВНИМАНИЕ: ключ ИИ не найден (ANTHROPIC_API_KEY пуст), работаю без ИИ. "
              "PowerShell: $env:ANTHROPIC_API_KEY=\"...\"; cmd: set ANTHROPIC_API_KEY=...", flush=True)
    prov = None if mode == "geometric" else llm.make(mode, a.model, a.replay)
    mode_label = prov.label if prov else "БЕЗ ИИ: геометрический алгоритм (настоящий расчёт по скану, не заготовка)"
    started = dt.datetime.now()
    out = Path(a.out) / started.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log(f"режим: {mode_label}")
    log(f"1/4 читаю скан {a.scan}")
    sheet, scan = scan_analysis.measurement_sheet(a.scan, units=a.units)
    log(f"    единицы файла: {a.units} (STL их не хранит; если габарит неверный, укажите --units)")
    scan.export(out / "scan_aligned.stl")
    d = sheet["diagnostics"]
    log(f"    граней {d['faces']}, габарит {d['extents_mm']} мм, плоскостей {len(sheet['horizontal_planes'])}, "
        f"сечений {len(sheet['sections'])}")
    for w in d["warnings"]:
        log("    внимание: " + w)
    if d.get("noise_mm", 0) * 3 > a.tol:
        a.tol = round(3 * d["noise_mm"], 2)
        log(f"    допуск сравнения поднят до {a.tol} мм: меньше шума скана мерить нельзя")
    (out / "sheet.json").write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")

    iterations, llm_log, best, converged = [], [], None, False

    def evaluate(n: int, code: str, meta: dict, t0: float, by: str) -> dict:
        it_dir = out / f"iter{n}"
        it_dir.mkdir(exist_ok=True)
        log("3/4 строю модель…")
        # черновик из стенок собирается из сотен пластин: на крупной детали это минуты
        limit = 420 if n in ("0б", "0в", "0к") else 180
        run = run_code(code, it_dir, timeout=limit) if code else {"ok": False, "error": "в ответе нет блока python"}
        cmp = None
        model = None
        if run.get("ok"):
            model = trimesh.load(it_dir / "model.stl", force="mesh")
            log(f"    тело валидно: {run.get('valid')}, объём {run.get('volume_mm3')} мм³")
            log("4/4 сравниваю со сканом…")
            cmp = cmpmod.compare(scan, model, tol=a.tol)
            log(f"    в допуске {cmp['within_tol_share']*100:.1f}%, не хватает материала {cmp['outside_share']*100:.2f}%, "
                f"скол/лишнее {cmp['inside_share']*100:.2f}%, 95% ближе {cmp['p95_abs_dev_mm']} мм")
            render(cmp, it_dir / "deviation.png", f"Попытка {n} ({by})")
        else:
            log(f"    ошибка: {run.get('error', '')[:200]}")
        gate_fail = gates.check(run, model, sheet, scan)
        for g in gate_fail:
            log("    обязательная проверка не пройдена: " + g)
        fb = llm.feedback_prompt(run, cmp, a.target)
        if gate_fail and run.get("ok"):
            fb += "\n\nОбязательные проверки НЕ пройдены (без них модель не принимается):\n- " + "\n- ".join(gate_fail)
        return {"n": n, "by": by, "code": code, "meta": meta, "run": run, "cmp": cmp, "feedback": fb, "gates": gate_fail,
                "seconds": time.time() - t0, "img": (it_dir / "deviation.png") if cmp else None, "dir": it_dir}

    def good(it: dict) -> bool:
        c = it["cmp"]
        return bool(c and not it["gates"] and c["within_tol_share"] >= a.target and c["outside_share"] < 0.005)

    def rank(it: dict) -> tuple:
        # сначала — прошла ли обязательные проверки, потом цель, потом счёт: «лучшая»
        # и «сошлось» всегда относятся к одной и той же попытке
        return (not it["gates"], good(it), score(it["run"], it["cmp"]))

    # Попытка 0: геометрический алгоритм без ИИ. Всегда есть, работает без ключа и сети.
    if not a.no_geometric:
        t0 = time.time()
        # черновик из стенок и тело по поверхности считаются в фоне с самого начала, параллельно
        # с попыткой 0: ждать её, чтобы потом начинать их с нуля, на крупной детали долго
        def launch(kind: str, path: Path):
            try:
                return subprocess.Popen([sys.executable, str(HERE / "geo_worker.py"), str(out / "scan_aligned.stl"),
                                         str(out / "sheet.json"), "1", str(path), kind],
                                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            except Exception:  # noqa: BLE001
                return None

        sj, wj = out / "surface.json", out / "walls.json"
        bg_limit, bg_started = a.geo_timeout + 300, time.time()
        # тело по поверхности на огромном скане (миллионы треугольников) считается дольше: даём больше
        surf_limit = bg_limit + (480 if sheet["diagnostics"]["faces"] > 1_000_000 else 0)
        surf_proc, walls_proc = launch("surface", sj), launch("walls", wj)
        models = {}  # ИИ-модели-кандидаты: имя -> (процесс, файл ответа)
        for name, mode in (("stepcad", a.stepcad), ("cadena", a.cadena)):
            if want_model(mode, name):
                try:
                    models[name] = (subprocess.Popen([sys.executable, str(HERE / f"{name}_worker.py"),
                                                      str(out / "scan_aligned.stl"), str(out / name),
                                                      str(out / f"{name}.json"), "900"],
                                                     env=dict(os.environ, PYTHONIOENCODING="utf-8")),
                                    out / f"{name}.json")
                except Exception:  # noqa: BLE001
                    pass

        def finish(proc, limit: float) -> None:
            if proc is None:
                raise RuntimeError("процесс не запустился")
            try:
                code = proc.wait(timeout=max(10.0, limit - (time.time() - bg_started)))
            except subprocess.TimeoutExpired:
                proc.kill()  # иначе он продолжит считать и съест память следующим попыткам
                raise
            if code != 0:
                raise RuntimeError("процесс завершился с ошибкой")

        log("2/4 попытка 0: алгоритм без ИИ строит черновик по сечениям…", a.geo_timeout)
        low = (a.notes or "").lower()
        restore = not a.no_restore and not any(w in low for w in ("деталь целая", "не достраивать", "ничего не восстанавливать"))
        if not restore:
            log("    достройка выключена (заметка или --no-restore): строю только то, что есть в скане")
        gj = out / "geometric.json"
        try:
            subprocess.run([sys.executable, str(HERE / "geo_worker.py"), str(out / "scan_aligned.stl"),
                            str(out / "sheet.json"), "1" if restore else "0", str(gj)],
                           timeout=a.geo_timeout, check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            g = json.loads(gj.read_text(encoding="utf-8"))
            g_code, g_meta = g["code"], g["meta"]
        except subprocess.TimeoutExpired:
            g_code = None
            log(f"    деталь слишком сложная для алгоритма без ИИ: не уложился в {a.geo_timeout} с, дальше работает ИИ")
        except Exception as ex:  # noqa: BLE001
            g_code = None
            log(f"    алгоритм без ИИ не справился ({type(ex).__name__}), дальше работает ИИ")
        if g_code is not None:
            for r in g_meta["restored"]:
                log("    " + r)
            it0 = evaluate(0, g_code, g_meta, t0, "алгоритм, без ИИ")
            iterations.append(it0)
            if it0["run"].get("ok"):
                best = it0
        # Тонкостенная деталь (пластик, лист): слои по сечениям не справляются — пробуем
        # черновик из стенок (каждая пластина целиком по паре параллельных сторон).
        if best is None or best["cmp"]["within_tol_share"] < 0.9:
            t0 = time.time()
            w_limit = bg_limit
            if surf_proc is not None and surf_proc.poll() == 0 and walls_proc is not None and walls_proc.poll() is None:
                # тело по поверхности уже готово, а стенки ещё считаются: на тонкостенной детали они
                # обычно успевают за минуту, на кривой оболочке не успевают совсем — долго не ждём
                w_limit = min(bg_limit, time.time() - bg_started + 60)
            log("2/4 попытка 0б: черновик из стенок (для тонкостенных деталей), без ИИ…", max(10.0, w_limit - (time.time() - bg_started)))
            try:
                finish(walls_proc, w_limit)
                w = json.loads(wj.read_text(encoding="utf-8"))
                log("    " + w["meta"]["plan"])
                itw = evaluate("0б", w["code"], w["meta"], t0, "стенки, без ИИ")
                iterations.append(itw)
                if itw["run"].get("ok") and (best is None or rank(itw) > rank(best)):
                    best = itw
            except subprocess.TimeoutExpired:
                log("    черновик из стенок не уложился во время, дальше работает ИИ")
            except Exception as ex:  # noqa: BLE001
                log(f"    черновик из стенок не получился ({type(ex).__name__})")

        # Любая форма (кривые оболочки, линзы, корпуса): тело по поверхности скана. Считается
        # параллельно с черновиками выше (запущено в начале), берётся, если они не дошли до
        # цели; из всех попыток выбирается лучшая по сравнению.
        if best is None or not good(best):
            t0 = time.time()
            log("2/4 попытка 0в: тело по поверхности скана (для деталей любой формы), без ИИ…", max(10.0, surf_limit - (time.time() - bg_started)))
            try:
                finish(surf_proc, surf_limit)
                g = json.loads(sj.read_text(encoding="utf-8"))
                log("    " + g["meta"]["plan"])
                its = evaluate("0в", g["code"], g["meta"], t0, "поверхность, без ИИ")
                iterations.append(its)
                if its["run"].get("ok") and (best is None or rank(its) > rank(best)):
                    best = its
            except subprocess.TimeoutExpired:
                log("    тело по поверхности не уложилось во время")
            except Exception as ex:  # noqa: BLE001
                log(f"    тело по поверхности не получилось ({type(ex).__name__})")
            # Те же формы гладкими гранями (B-сплайн), чтобы модель можно было править в CAD, а не
            # тысячами треугольников. Берётся вместо треугольного тела, если почти не уступает ему.
            if a.smooth != "off" and (best is None or best["n"] == "0в"):
                t0 = time.time()
                log("2/4 попытка 0к: гладкое тело из B-сплайн граней по скану, без ИИ…", 1200)
                kj = out / "smooth.json"
                try:
                    subprocess.run([sys.executable, str(HERE / "geo_worker.py"), str(out / "scan_aligned.stl"),
                                    str(out / "sheet.json"), "1", str(kj), "smooth"],
                                   timeout=1200, check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                    g = json.loads(kj.read_text(encoding="utf-8"))
                    log("    " + g["meta"]["plan"])
                    itk = evaluate("0к", g["code"], g["meta"], t0, "гладкие грани, без ИИ")
                    iterations.append(itk)
                    if itk["run"].get("ok") and not itk["gates"] and itk["cmp"] and (
                            best is None or best["cmp"] is None
                            or itk["cmp"]["within_tol_share"] >= best["cmp"]["within_tol_share"] - 0.01):
                        best = itk
                except subprocess.TimeoutExpired:
                    log("    гладкое тело не уложилось во время")
                except Exception as ex:  # noqa: BLE001
                    log(f"    гладкое тело не получилось ({type(ex).__name__})")
        # ИИ-модели-кандидаты пишут программу построения; берутся, только если по сравнению лучше
        titles = {"stepcad": ("0г", "StepCAD"), "cadena": ("0д", "CADENA")}
        for name, (proc, ans) in models.items():
            n, title = titles[name]
            t0 = time.time()
            log(f"2/4 попытка {n}: {title} (ИИ-модель, локально, бесплатно)…", 1500)
            try:
                if proc.wait(timeout=1500) != 0:
                    raise RuntimeError("процесс завершился с ошибкой")
                g = json.loads(ans.read_text(encoding="utf-8"))
                log("    " + g["meta"]["plan"])
                itc = evaluate(n, g["code"], g["meta"], t0, title)
                iterations.append(itc)
                if itc["run"].get("ok") and (best is None or rank(itc) > rank(best)):
                    best = itc
            except subprocess.TimeoutExpired:
                log(f"    {title} не уложилась во время")
            except Exception as ex:  # noqa: BLE001
                log(f"    {title} не дала рабочей модели ({type(ex).__name__})")
        # Point2STEP: грани лучшей модели уточняются по скану; утраченное берётся из неё же
        # Уточнение по скану: каждая грань нашей модели заново подгоняется по своим точкам скана,
        # точные размеры пишутся обратно в редактируемую программу (fitfaces.py + merge.py)
        if best is not None and best["run"].get("ok") and "importStep" not in best["code"]:
            t0 = time.time()
            log("2/4 попытка 0ж: уточняю размеры граней по точкам скана…", 300)
            try:
                mj = out / "fit.json"
                subprocess.run([sys.executable, str(HERE / "merge.py"), str(best["dir"] / "model.py"),
                                str(best["dir"] / "model.step"), str(out / "scan_aligned.stl"), str(out / "fitted.py"),
                                str(mj), "--fit"], check=True, timeout=300, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                changes = json.loads(mj.read_text(encoding="utf-8"))["changes"]
                for c in changes:
                    log("    " + c)
                if changes:
                    meta = dict(best["meta"])
                    meta["plan"] = (str(meta.get("plan") or "") + f" Размеры {len(changes)} элементов уточнены по точкам "
                                    "скана: " + "; ".join(changes))
                    itf = evaluate("0ж", (out / "fitted.py").read_text(encoding="utf-8"), meta, t0,
                                   "наш алгоритм, размеры уточнены по скану")
                    iterations.append(itf)
                    if (itf["run"].get("ok") and rank(itf)[:2] >= rank(best)[:2]
                            and itf["cmp"]["within_tol_share"] >= best["cmp"]["within_tol_share"] - 0.003
                            and itf["cmp"]["outside_share"] <= best["cmp"]["outside_share"] + 0.001):
                        best = itf
                else:
                    log("    уточнять нечего: размеры и так ближе всего к скану")
            except Exception as ex:  # noqa: BLE001
                log(f"    уточнение по скану не удалось ({type(ex).__name__})")
        if a.p2s == "auto":
            from p2s_worker import p2s_dir

            a.p2s = "on" if p2s_dir() else "off"
            if a.p2s == "on":
                log(f"    Point2STEP найден ({p2s_dir()}), включаю уточнение граней")
        # тело по поверхности (0в, 0к) Point2STEP не улучшает: на кривых корпусах он проверен
        # и рассыпается (накладка Hyundai: 39 мин, покрытие 14%), а на тысячах граней идёт часами
        if a.p2s == "on" and best is not None and best["run"].get("ok") and best["n"] not in ("0в", "0к"):
            t0 = time.time()
            log("2/4 попытка 0е: Point2STEP уточняет грани лучшей модели по скану…", 700)
            try:
                pj = out / "p2s.json"
                subprocess.run([sys.executable, str(HERE / "p2s_worker.py"), str(out / "scan_aligned.stl"),
                                str(best["dir"] / "model.step"), str(out / "p2s"), str(pj), "600"],
                               check=True, timeout=700, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                g = json.loads(pj.read_text(encoding="utf-8"))
                base = best  # модель, которую уточняем: ниже в её программу запишутся точные размеры
                g["meta"]["restored"] = list(best["meta"].get("restored") or []) + g["meta"]["restored"]
                log("    " + g["meta"]["plan"])
                itp = evaluate("0е", g["code"], g["meta"], t0, "Point2STEP")
                iterations.append(itp)
                # совпадение со сканом почти то же, выигрыш — в точности размеров: берём, если не хуже
                if (itp["run"].get("ok") and rank(itp)[:2] >= rank(best)[:2]
                        and itp["cmp"]["within_tol_share"] >= best["cmp"]["within_tol_share"] - 0.003
                        and itp["cmp"]["outside_share"] <= best["cmp"]["outside_share"] + 0.001):
                    best = itp
                # Объединение: точные размеры Point2STEP — обратно в нашу редактируемую программу
                # (только там, где новая поверхность ближе к скану, чем старая)
                if "importStep" not in base["code"]:
                    t0 = time.time()
                    log("2/4 попытка 0и: объединяю — размеры от Point2STEP в нашей программе построения…", 300)
                    mj = out / "merge.json"
                    subprocess.run([sys.executable, str(HERE / "merge.py"), str(base["dir"] / "model.py"),
                                    str(out / "p2s.step"), str(out / "scan_aligned.stl"), str(out / "merged.py"), str(mj)],
                                   check=True, timeout=300, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                    changes = json.loads(mj.read_text(encoding="utf-8"))["changes"]
                    for c in changes:
                        log("    " + c)
                    if changes:
                        meta = dict(base["meta"])
                        meta["plan"] = (str(meta.get("plan") or "") + f" Размеры {len(changes)} элементов уточнены по скану "
                                        "(Point2STEP): " + "; ".join(changes))
                        itm = evaluate("0и", (out / "merged.py").read_text(encoding="utf-8"), meta, t0,
                                       "наш алгоритм + Point2STEP")
                        iterations.append(itm)
                        if (itm["run"].get("ok") and rank(itm)[:2] >= rank(best)[:2]
                                and itm["cmp"]["within_tol_share"] >= best["cmp"]["within_tol_share"] - 0.003
                                and itm["cmp"]["outside_share"] <= best["cmp"]["outside_share"] + 0.001):
                            best = itm
                    else:
                        log("    уточнять нечего: размеры Point2STEP не ближе к скану")
            except subprocess.TimeoutExpired:
                log("    Point2STEP не уложился во время")
            except Exception as ex:  # noqa: BLE001
                log(f"    Point2STEP не дал рабочей модели ({type(ex).__name__})")
        # Подсказки CADENA: наш алгоритм проверяет её отверстия по скану и достраивает недостающее сам
        cad_step = out / "cadena.step"
        if cad_step.exists() and best is not None and best["run"].get("ok") and "importStep" not in best["code"]:
            t0 = time.time()
            log("2/4 попытка 0з: проверяю подсказки CADENA по скану…", 300)
            try:
                hj = out / "hints.json"
                subprocess.run([sys.executable, str(HERE / "hints.py"), str(best["dir"] / "model.py"),
                                str(best["dir"] / "model.step"), str(cad_step), str(out / "scan_aligned.stl"),
                                str(out / "hinted.py"), str(hj)],
                               check=True, timeout=300, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                h = json.loads(hj.read_text(encoding="utf-8"))
                log(f"    CADENA предложила отверстий: {h['hints']}; уже есть у нас: {h['confirmed']}; "
                    f"достроено по скану: {len(h['added'])}; не подтверждено сканом: {len(h['unconfirmed'])}")
                for s in h["added"] + h["unconfirmed"]:
                    log("    " + s)
                meta = dict(best["meta"])
                meta["assumptions"] = list(meta.get("assumptions") or []) + [s + " — проверить" for s in h["unconfirmed"]]
                if h["added"]:
                    meta["restored"] = list(meta.get("restored") or []) + h["added"]
                    ith = evaluate("0з", (out / "hinted.py").read_text(encoding="utf-8"), meta, t0,
                                   "наш алгоритм + подсказки CADENA")
                    iterations.append(ith)
                    if ith["run"].get("ok") and rank(ith) >= rank(best):
                        best = ith
                else:
                    best["meta"] = meta
            except Exception as ex:  # noqa: BLE001
                log(f"    подсказки CADENA проверить не удалось ({type(ex).__name__})")
        for proc in [surf_proc, walls_proc] + [p for p, _ in models.values()]:
            if proc is not None and proc.poll() is None:
                proc.kill()  # не понадобился или не уложился во время

    hint = ""
    if best is not None and best["n"] in (0, "0б", "0в", "0к") and best["cmp"]:
        c0 = best["cmp"]
        hint = ("\n\nДля справки: черновик без ИИ совпал со сканом на "
                f"{c0['within_tol_share']*100:.1f}%. Он копирует форму ступеньками и не знает замысла; "
                "построй деталь по замыслу. Что черновик восстановил: " + "; ".join(best["meta"].get("restored", [])))
    if prov and best is not None and best["n"] in ("0б", "0в", "0к"):
        DRAFT["step"] = best["dir"] / "model.step"
        hint += ("\n\nВ коде доступна готовая переменная `draft` (cq.Shape): это черновик "
                 + {"0б": "из стенок, ", "0к": "по скану гладкими B-сплайн гранями, "}.get(
                     best["n"], "по поверхности скана (много мелких плоских граней), ")
                 + f"он уже совпал со сканом на {best['cmp']['within_tol_share']*100:.1f}%. Не строй деталь "
                 "заново: начни с него (например, result = draft.fuse(...) или draft.cut(...)) и исправь "
                 "только места, где не хватает материала или есть лишнее: стыки стенок, скругления, "
                 "наклонные и кривые участки. Где именно — смотри сравнение ниже и инструменты.")
        if best["feedback"]:
            hint += "\n\nСравнение черновика со сканом:\n" + best["feedback"]
    use_tools = bool(prov and a.ai_input == "tools" and prov.supports_tools)
    tools = None
    if use_tools:
        import scan_tools

        tools = scan_tools.ScanTools(scan)
        tools.verbose = True
        wp = out / "walls.pkl"
        if wp.exists():  # стенки уже посчитаны для черновика: инструмент walls отвечает сразу
            import pickle

            try:
                tools._walls = pickle.loads(wp.read_bytes())
            except Exception:  # noqa: BLE001
                pass
        prov.max_tool_calls = a.max_tool_calls
        prov.budget_usd = a.budget_usd
        mode_label += " · ИИ сам смотрит сетку скана инструментами, без картинок"
        log(f"    ИИ смотрит сетку скана сам: {len(scan_tools.TOOLS)} инструментов, до {a.max_tool_calls} вызовов на попытку")
    views = out / "scan_views.png"
    if prov and not use_tools:
        render_views(scan, views)
    messages = [{"role": "user", "content": llm.first_prompt(sheet, a.notes) + hint,
                 "images": [p for p in (a.image, views if views.exists() else None) if p]}]
    for n in range(1, a.iters + 1 if prov else 1):
        t0 = time.time()
        log(f"2/4 попытка {n}: ИИ пишет модель…")
        try:
            answer = prov.chat(messages, tools=tools)
        except Exception as ex:  # noqa: BLE001
            log(f"    ИИ не ответил: {type(ex).__name__}: {ex}")
            break
        entry = {"provider": prov.label, "answer": answer}
        if getattr(prov, "trace", None):
            entry["tool_calls"] = prov.trace
            log(f"    ИИ сделал запросов к сетке: {len(prov.trace)} ("
                + ", ".join(sorted({t['tool'] for t in prov.trace})) + ")")
        if hasattr(prov, "cost_usd"):
            entry["usage"] = dict(prov.usage, cost_usd=prov.cost_usd())
            log(f"    расход пока: ${prov.cost_usd():.3f} ({prov.usage['requests']} запросов)")
        llm_log.append(entry)
        code, meta = llm.parse(answer)
        messages.append({"role": "assistant", "content": answer})
        log(f"    замысел: {str(meta.get('plan', ''))[:140]}")
        it = evaluate(n, code, meta, t0, "ИИ" if mode in ("anthropic", "openai") else
                      ("повтор записи ИИ" if mode == "replay" else "заготовка, не ИИ"))
        iterations.append(it)
        if it["run"].get("ok") and (best is None or rank(it) > rank(best)):
            best = it
        if good(it):
            log(f"    сошлось на попытке {n}")
            break
        spent = prov.cost_usd() if hasattr(prov, "cost_usd") else 0.0
        if spent and spent + spent / n > a.budget_usd:  # следующая попытка в среднем столько же
            log(f"    стоп: потрачено ${spent:.2f}, ещё одна попытка выйдет за бюджет ${a.budget_usd}")
            break
        if hasattr(prov, "cost_usd") and prov.cost_usd() > a.budget_usd:
            log(f"    стоп: расход ${prov.cost_usd():.2f} больше бюджета ${a.budget_usd}")
            break
        messages.append({"role": "user", "content": it["feedback"],  # в режиме tools без картинок
                         "images": [it["img"]] if it.get("img") and not use_tools else []})
    converged = bool(best and good(best))

    (out / "llm_log.json").write_text(json.dumps(llm_log, ensure_ascii=False, indent=1), encoding="utf-8")
    truth = None
    if best:
        for f in ("model.step", "model.stl", "model.py"):
            shutil.copy(best["dir"] / f, out / f)
        if a.truth:
            tp = Path(a.truth)
            if tp.suffix.lower() in (".step", ".stp"):
                import cadquery as cq

                cq.exporters.export(cq.importers.importStep(str(tp)), str(out / "_truth.stl"), tolerance=0.02)
                tm = trimesh.load(out / "_truth.stl", force="mesh")
            else:
                tm = trimesh.load(tp, force="mesh")
            # эталон в координатах исходного скана → переводим в выровненные
            tm.apply_transform(np.array(sheet["transform_from_scan_4x4"]))
            truth = cmpmod.compare(tm, trimesh.load(out / "model.stl", force="mesh"), tol=a.tol)
            log(f"эталон: в допуске {truth['within_tol_share']*100:.1f}% (тестовая деталь)")
            # каждая попытка против эталона: совпадение и точность диаметров отверстий
            try:
                th = [(h, holesmod._measure(tm, h, 0.6)) for h in holesmod.find_holes(tm)]
                th = [(h, m) for h, m in th if m]
            except Exception:  # noqa: BLE001
                th = []
            truth["by_attempt"] = []
            for it in iterations:
                if not it["run"].get("ok") or not (it["dir"] / "model.stl").exists():
                    continue
                mm = trimesh.load(it["dir"] / "model.stl", force="mesh")
                errs, miss = [], 0
                for h, m in th:
                    r = holesmod._measure(mm, h, 0.6)
                    if r is None:
                        miss += 1
                    else:
                        errs.append(abs(r["d"] - m["d"]))
                row = {"n": it["n"], "by": it["by"], "fit": cmpmod.compare(tm, mm, tol=a.tol)["within_tol_share"],
                       "holes": len(th), "missing": miss,
                       "d_err": round(float(np.mean(errs)), 3) if errs else None,
                       "d_max": round(float(np.max(errs)), 3) if errs else None, "best": it is best}
                truth["by_attempt"].append(row)
                log(f"    эталон, попытка {it['n']} ({it['by']}): {row['fit']*100:.1f}%, отверстия {len(th) - miss}/{len(th)}"
                    + (f", ошибка Ø в среднем {row['d_err']} мм, наибольшая {row['d_max']} мм" if errs else ""))
    holes_rows = None
    if best and (out / "model.stl").exists():
        log("4/4 ищу отверстия скана и сверяю их с моделью…", 600)
        try:  # отверстия скана: на месте ли они в модели и того же ли размера
            holes_rows = holesmod.check(scan, trimesh.load(out / "model.stl", force="mesh"), tol=max(0.2, a.tol / 2))
            log(f"отверстия: найдено {len(holes_rows)}, совпали {sum(r['ok'] for r in holes_rows)}")
        except Exception as ex:  # noqa: BLE001
            log(f"проверка отверстий не удалась: {ex}")
    step_check = None
    if best and (out / "model.step").exists():
        step_check = check_step(out / "model.step")
        log("файл STEP перечитан: " + step_check[1])
    test_label = ""
    if "samples" in Path(a.scan).resolve().parts:
        test_label = "Тестовая деталь, сломанная нами специально. Это не деталь студии."
    rep = write_report(out, {
        "iterations": iterations, "best": best, "sheet": sheet, "mode": mode, "mode_label": mode_label,
        "started": started.strftime("%d.%m.%Y %H:%M"), "scan_name": Path(a.scan).name, "tol": a.tol,
        "target": a.target, "converged": converged, "total_s": (dt.datetime.now() - started).total_seconds(),  # весь запуск, с замерами
        "truth": truth, "holes": holes_rows, "step_check": step_check,
        "test_label": test_label, "scan_mesh": scan})
    log(f"готово: {rep}")
    if not a.no_open:
        import webbrowser

        webbrowser.open(rep.resolve().as_uri())


if __name__ == "__main__":
    main()
