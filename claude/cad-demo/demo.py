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
import shutil
import subprocess
import sys
import time
from pathlib import Path

import logging

import numpy as np
import trimesh

logging.getLogger("trimesh").setLevel(logging.ERROR)

import compare as cmpmod
import llm
import scan_analysis

HERE = Path(__file__).parent


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_code(code: str, it_dir: Path, timeout: int = 90) -> dict:
    (it_dir / "model.py").write_text(code, encoding="utf-8")
    try:
        subprocess.run([sys.executable, str(HERE / "cad_runner.py"), str(it_dir / "model.py"), str(it_dir)],
                       timeout=timeout, capture_output=True, text=True)
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


def write_report(out: Path, ctx: dict) -> Path:
    e = html.escape
    its = ctx["iterations"]
    best = ctx["best"]
    rows = []
    for it in its:
        c = it.get("cmp")
        status = ("ошибка кода" if not it["run"].get("ok") else
                  ("тело невалидно" if not it["run"].get("valid") else "построено"))
        rows.append(f"<tr{' class=best' if best and it['n'] == best['n'] else ''}><td>{it['n']}</td><td>{status}</td>"
                    f"<td>{'' if not c else f'{c['within_tol_share']*100:.1f}%'}</td>"
                    f"<td>{'' if not c else f'{c['outside_share']*100:.2f}%'}</td>"
                    f"<td>{'' if not c else c['p95_abs_dev_mm']}</td><td>{it['seconds']:.0f} с</td></tr>")
    blocks = []
    for it in its:
        c, r, meta = it.get("cmp"), it["run"], it.get("meta", {})
        img = it.get("img")
        blocks.append(f"""
<section class=it><h3>Попытка {it['n']}{' — лучшая' if best and it['n'] == best['n'] else ''}</h3>
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
        viewer = f"""<section><h2>3D: модель и точки скана</h2><div id=v></div>
<p class=small>Крутить мышью. Полупрозрачная: модель ИИ. Точки: скан (красные — модели не хватает материала, синие — скол).</p>
<script type=importmap>{importmap()}</script>
<script type=module>
import * as THREE from 'three';import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';import {{STLLoader}} from 'three/addons/loaders/STLLoader.js';
const el=document.getElementById('v');const r=new THREE.WebGLRenderer({{antialias:true}});r.setSize(el.clientWidth,480);el.appendChild(r.domElement);
const s=new THREE.Scene();s.background=new THREE.Color(0xf4f5f7);const cam=new THREE.PerspectiveCamera(40,el.clientWidth/480,0.1,5000);
s.add(new THREE.HemisphereLight(0xffffff,0x8899aa,2.2));const dl=new THREE.DirectionalLight(0xffffff,1.5);dl.position.set(50,-80,120);s.add(dl);
const bin=Uint8Array.from(atob("{b64(out / 'model.stl')}"),c=>c.charCodeAt(0)).buffer;
const g=new STLLoader().parse(bin);const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({{color:0x7fa6d6,transparent:true,opacity:0.55,metalness:0.1,roughness:0.6}}));s.add(m);
const P={json.dumps(np.round(pts, 2).tolist())};const D={json.dumps(np.round(best['cmp']['_scan_dev'][::2], 3).tolist())};const tol={ctx['tol']};
const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.Float32BufferAttribute(P.flat(),3));
pg.setAttribute('color',new THREE.Float32BufferAttribute(D.flatMap(d=>d<-tol?[0.85,0.1,0.1]:d>tol?[0.1,0.3,0.95]:[0.25,0.25,0.25]),3));
s.add(new THREE.Points(pg,new THREE.PointsMaterial({{size:0.6,vertexColors:true}})));
g.computeBoundingSphere();const c=g.boundingSphere.center,R=g.boundingSphere.radius;cam.up.set(0,0,1);cam.position.set(c.x+R*1.6,c.y-R*2.0,c.z+R*1.4);
const oc=new OrbitControls(cam,r.domElement);oc.target.copy(c);oc.update();(function a(){{requestAnimationFrame(a);r.render(s,cam);}})();
</script></section>"""
    page = f"""<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Скан → CAD: отчёт</title><style>
:root{{--bg:#fff;--fg:#1d2330;--mut:#5b6474;--line:#e3e6eb;--acc:#2f6fd6;--warn:#b3261e;--ok:#1e7d3c}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1060px;padding:20px 16px 60px}}
h1{{font-size:24px;margin:.2em 0}}h2{{font-size:19px;margin-top:1.6em;border-bottom:1px solid var(--line);padding-bottom:4px}}
.mode{{display:inline-block;padding:3px 10px;border-radius:12px;background:#fff3cd;font-size:13px}}
.mode.live{{background:#dff3e4}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;font-size:14px}}
tr.best td{{background:#eef5ff;font-weight:600}}img{{max-width:100%}}pre{{background:#f6f7f9;padding:10px;overflow:auto;font-size:12.5px}}
.err{{color:var(--warn)}}.small{{color:var(--mut);font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:10px}}.card b{{font-size:20px;display:block}}#v{{width:100%;height:480px;border:1px solid var(--line)}}
a.btn{{display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;border-radius:6px;background:var(--acc);color:#fff;text-decoration:none}}
</style></head><body>
<h1>Скан повреждённой детали → CAD-модель</h1>
<p class=small>{e(ctx['started'])} · файл скана: {e(ctx['scan_name'])}</p>
<p><span class="mode{' live' if ctx['mode'] in ('anthropic', 'openai') else ''}">Режим: {e(ctx['mode_label'])}</span></p>
{'<p class=err><b>' + e(ctx['test_label']) + '</b></p>' if ctx.get('test_label') else ''}
<section><h2>Итог</h2><div class=cards>
<div class=card>Попыток<b>{len(its)}</b></div>
<div class=card>Точек скана в допуске {ctx['tol']} мм<b>{(best['cmp']['within_tol_share']*100 if best else 0):.1f}%</b></div>
<div class=card>Модели не хватает материала<b>{(best['cmp']['outside_share']*100 if best else 0):.2f}%</b></div>
<div class=card>Время<b>{ctx['total_s']:.0f} с</b></div></div>
<p>{'<b>Модель сошлась со сканом.</b>' if ctx['converged'] else '<b class=err>Не сошлась до цели</b>: смотреть лучшую попытку и замечания.' if best else '<b class=err>Ни одна попытка не дала тела.</b>'}
Цель: не менее {ctx['target']*100:.0f}% точек в допуске и почти нет мест, где модели не хватает материала. Точки на сколе в допуск не попадают, так и должно быть.</p>
{('<a class=btn href="model.step" download>Скачать STEP (КОМПАС-3D, SolidWorks и др.)</a><a class=btn href="model.stl" download>STL (для печати)</a><a class=btn href="model.py" download>Код модели</a>') if best else ''}
</section>
{truth_html}
<section><h2>Что ИИ восстановил и что должен проверить инженер</h2>
<p><b>Как строил:</b> {e(str(meta.get('plan', '—')))}</p>
<h3>Достроено (этого нет в скане)</h3><ul>{lst(meta.get('restored'))}</ul>
<h3>Допущения</h3><ul>{lst(meta.get('assumptions'))}</ul>
<h3>Проверить инженеру</h3><ul>{lst(meta.get('check_by_engineer'))}</ul>
<p class=small>Модель — черновик для инженера. Производственное решение принимает человек, проверив размеры по детали (при желании в КОМПАС-3D или SolidWorks).</p></section>
{viewer}
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
    ap.add_argument("--mode", choices=["auto", "anthropic", "openai", "replay", "scripted"], default="auto")
    ap.add_argument("--model", help="модель ИИ (по умолчанию: claude-sonnet-5-5 / gpt-4.1)")
    ap.add_argument("--replay", help="llm_log.json для режимов replay/scripted")
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
    prov = llm.make(mode, a.model, a.replay)
    started = dt.datetime.now()
    out = Path(a.out) / started.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log(f"режим: {prov.label}")
    log(f"1/4 читаю скан {a.scan}")
    sheet, scan = scan_analysis.measurement_sheet(a.scan)
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

    messages = [{"role": "user", "content": llm.first_prompt(sheet, a.notes), "image": a.image}]
    iterations, llm_log, best, converged = [], [], None, False
    t_all = time.time()
    for n in range(1, a.iters + 1):
        t0 = time.time()
        it_dir = out / f"iter{n}"
        it_dir.mkdir()
        log(f"2/4 попытка {n}: ИИ пишет модель…")
        try:
            answer = prov.chat(messages)
        except Exception as ex:  # noqa: BLE001
            log(f"    ИИ не ответил: {type(ex).__name__}: {ex}")
            break
        llm_log.append({"provider": prov.label, "answer": answer})
        code, meta = llm.parse(answer)
        messages.append({"role": "assistant", "content": answer})
        log(f"    замысел: {str(meta.get('plan', ''))[:140]}")
        log("3/4 строю модель…")
        run = run_code(code, it_dir) if code else {"ok": False, "error": "в ответе нет блока python"}
        cmp = None
        if run.get("ok"):
            model = trimesh.load(it_dir / "model.stl", force="mesh")
            log(f"    тело валидно: {run.get('valid')}, объём {run.get('volume_mm3')} мм³")
            log("4/4 сравниваю со сканом…")
            cmp = cmpmod.compare(scan, model, tol=a.tol)
            log(f"    в допуске {cmp['within_tol_share']*100:.1f}%, не хватает материала {cmp['outside_share']*100:.2f}%, "
                f"скол/лишнее {cmp['inside_share']*100:.2f}%, 95% ближе {cmp['p95_abs_dev_mm']} мм")
            render(cmp, it_dir / "deviation.png", f"Попытка {n}")
        else:
            log(f"    ошибка: {run.get('error', '')[:200]}")
        fb = llm.feedback_prompt(run, cmp, a.target)
        it = {"n": n, "code": code, "meta": meta, "run": run, "cmp": cmp, "feedback": fb,
              "seconds": time.time() - t0, "img": (it_dir / "deviation.png") if cmp else None, "dir": it_dir}
        iterations.append(it)
        if score(run, cmp) > (score(best["run"], best["cmp"]) if best else -0.5):
            best = it
        if cmp and run.get("valid") and cmp["within_tol_share"] >= a.target and cmp["outside_share"] < 0.005:
            converged = True
            log(f"    сошлось на попытке {n}")
            break
        messages.append({"role": "user", "content": fb})

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
        "iterations": iterations, "best": best, "sheet": sheet, "mode": mode, "mode_label": prov.label,
        "started": started.strftime("%d.%m.%Y %H:%M"), "scan_name": Path(a.scan).name, "tol": a.tol,
        "target": a.target, "converged": converged, "total_s": time.time() - t_all, "truth": truth,
        "test_label": test_label})
    log(f"готово: {rep}")
    if not a.no_open:
        import webbrowser

        webbrowser.open(rep.resolve().as_uri())


if __name__ == "__main__":
    main()
