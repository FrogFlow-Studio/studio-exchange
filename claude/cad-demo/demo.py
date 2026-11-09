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
import gates
import geometric
import llm
import scan_analysis

HERE = Path(__file__).parent


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_code(code: str, it_dir: Path, timeout: int = 90) -> dict:
    (it_dir / "model.py").write_text(code, encoding="utf-8")
    try:
        # код от ИИ не должен видеть ключи и прочие секреты окружения
        env = {k: v for k, v in os.environ.items()
               if not any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))}
        subprocess.run([sys.executable, str(HERE / "cad_runner.py"), str(it_dir / "model.py"), str(it_dir)],
                       timeout=timeout, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"код считался дольше {timeout} с и был остановлен"}
    p = it_dir / "run.json"
    if not p.exists():
        return {"ok": False, "error": "процесс построения упал без отчёта"}
    return json.loads(p.read_text(encoding="utf-8"))


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


def dims_table(sheet: dict) -> str:
    """Таблица размеров по скану для сверки штангенциркулем (можно распечатать)."""
    e = html.escape
    rows, seen = [], set()
    lo, hi = sheet["bounds_mm"]
    rows.append(("Габарит X × Y × Z", f"{hi[0]-lo[0]:.2f} × {hi[1]-lo[1]:.2f} × {hi[2]-lo[2]:.2f}", "с учётом скола"))
    for p in sheet["horizontal_planes"]:
        if p["facing"] == "смотрит вверх":
            rows.append(("Высота площадки от опорной плоскости", f"{p['z_mm']:.2f}", ""))
    for sec in sheet["sections"]:
        for reg in sec["regions"]:
            o = reg["outer"]
            if o.get("shape") == "окружность":
                k = ("o", round(o["diameter"]))
                if k not in seen:
                    seen.add(k)
                    rows.append(("Наружный Ø", f"{o['diameter']:.2f}", f"на высоте {sec['z_mm']}"))
            elif o.get("partial_circle"):
                k = ("o", round(o["partial_circle"]["diameter"]))
                if k not in seen:
                    seen.add(k)
                    rows.append(("Наружный Ø (по уцелевшей дуге)", f"{o['partial_circle']['diameter']:.2f}", f"на высоте {sec['z_mm']}"))
            for h in reg["inner"]:
                d = h.get("diameter") or (h.get("partial_circle") or {}).get("diameter")
                c = h.get("center_xy") or (h.get("partial_circle") or {}).get("center_xy")
                if not d:
                    continue
                k = ("h", round(d), round(c[0]), round(c[1]))
                if k not in seen:
                    seen.add(k)
                    rows.append(("Отверстие Ø" + (" (с пазом)" if h.get("partial_circle") else ""), f"{d:.2f}", f"центр {c}"))
            for pat in reg.get("hole_patterns", []):
                k = ("p", pat["pattern"], round(pat["hole_diameter"]))
                if k in seen:
                    continue
                seen.add(k)
                if pat.get("pcd_diameter"):
                    rows.append(("Диаметр окружности центров отверстий", f"{pat['pcd_diameter']:.2f}",
                                 f"{pat['likely_total']} отв., шаг {360 / pat['likely_total']:.0f}°"))
                if pat.get("sides_mm"):
                    rows.append(("Межосевые отверстий (прямоугольник)", " × ".join(f"{v:.2f}" for v in pat["sides_mm"]), ""))
    body = "".join(f"<tr><td>{e(a)}</td><td><b>{e(b)}</b></td><td>{e(c)}</td><td class=chk></td></tr>" for a, b, c in rows)
    return (f"<section><h2>Размеры для сверки штангенциркулем</h2><p class=small>Посчитаны по скану, мм. "
            f"Шум скана ~{sheet['diagnostics'].get('noise_mm', 0)} мм. Последний столбец — для отметки инженера "
            f"(страницу можно распечатать).</p><table><tr><th>Что</th><th>По скану</th><th>Где</th><th>Замер</th></tr>"
            f"{body}</table></section>")


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
                    f"<td>{'' if not c else c['p95_abs_dev_mm']}</td><td>{it['seconds']:.0f} с</td></tr>")
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
                      f"95% ближе {t['p95_abs_dev_mm']} мм, включая восстановленный угол.</p></section>")
    viewer = ""
    if best and (out / "model.stl").exists():
        pts = best["cmp"]["_scan_pts"][::2]
        dev = best["cmp"]["_scan_dev"][::2]
        tol = ctx["tol"]
        mdl = trimesh.load(out / "model.stl", force="mesh")
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
    dims = dims_table(ctx["sheet"])
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
{('<a class=btn href="model.step" download>Скачать STEP (КОМПАС-3D, SolidWorks и др.)</a><a class=btn href="model.stl" download>STL (для печати)</a><a class=btn href="model.py" download>Код модели</a>') if best else ''}
</section>
{truth_html}
<section><h2>Что восстановлено и что должен проверить инженер</h2>
<p><b>Как строил:</b> {e(str(meta.get('plan', '—')))}</p>
<h3>Достроено (этого нет в скане)</h3><ul>{lst(meta.get('restored'))}</ul>
<h3>Допущения</h3><ul>{lst(meta.get('assumptions'))}</ul>
<h3>Проверить инженеру</h3><ul>{lst(meta.get('check_by_engineer'))}</ul>
<p class=small>Модель — черновик для инженера. Производственное решение принимает человек, проверив размеры по детали (при желании в КОМПАС-3D или SolidWorks).</p></section>
{viewer}
{dims}
<section><h2>Журнал попыток</h2><table><tr><th>#</th><th>Результат</th><th>В допуске</th><th>Не хватает</th><th>95% точек ближе, мм</th><th>Время</th></tr>{''.join(rows)}</table>
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
        run = run_code(code, it_dir) if code else {"ok": False, "error": "в ответе нет блока python"}
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
        log("2/4 попытка 0: алгоритм без ИИ строит черновик по сечениям…")
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

    hint = ""
    if iterations and iterations[0]["cmp"]:
        c0 = iterations[0]["cmp"]
        hint = ("\n\nДля справки: черновик без ИИ (слои по сечениям) совпал со сканом на "
                f"{c0['within_tol_share']*100:.1f}%. Он копирует форму ступеньками и не знает замысла; "
                "построй деталь по замыслу. Что черновик восстановил: " + "; ".join(iterations[0]["meta"]["restored"]))
    use_tools = bool(prov and a.ai_input == "tools" and prov.supports_tools)
    tools = None
    if use_tools:
        import scan_tools

        tools = scan_tools.ScanTools(scan)
        tools.verbose = True
        prov.max_tool_calls = a.max_tool_calls
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
    test_label = ""
    if "samples" in Path(a.scan).resolve().parts:
        test_label = "Тестовая деталь, сломанная нами специально. Это не деталь студии."
    rep = write_report(out, {
        "iterations": iterations, "best": best, "sheet": sheet, "mode": mode, "mode_label": mode_label,
        "started": started.strftime("%d.%m.%Y %H:%M"), "scan_name": Path(a.scan).name, "tol": a.tol,
        "target": a.target, "converged": converged, "total_s": (dt.datetime.now() - started).total_seconds(),  # весь запуск, с замерами "truth": truth,
        "test_label": test_label, "scan_mesh": scan})
    log(f"готово: {rep}")
    if not a.no_open:
        import webbrowser

        webbrowser.open(rep.resolve().as_uri())


if __name__ == "__main__":
    main()
