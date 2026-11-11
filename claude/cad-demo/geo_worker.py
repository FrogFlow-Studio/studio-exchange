"""Попытка 0 в отдельном процессе, чтобы на сложной детали её можно было остановить по времени.

python geo_worker.py скан_выровненный.stl sheet.json 1|0 ответ.json [walls|surface|smooth]
"""
import json
import sys
from pathlib import Path

import trimesh

import geometric



def _memory_guard(share: float = 0.45) -> None:
    """Попытки считаются параллельно. Если одна разрастается (черновик из стенок на огромной
    кривой детали брал больше 13 ГБ), она не должна уронить соседние: выходит сама."""
    try:
        import psutil
    except ImportError:
        return
    import os
    import threading
    import time

    limit = share * psutil.virtual_memory().total
    me = psutil.Process()

    def watch() -> None:
        while True:
            if me.memory_info().rss > limit:
                sys.stderr.write(f"попытка заняла больше {limit / 2**30:.1f} ГБ памяти и остановлена\n")
                sys.stderr.flush()
                os._exit(3)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()


_memory_guard()
scan = trimesh.load(sys.argv[1], force="mesh")
KIND = sys.argv[5] if len(sys.argv) > 5 else "geometric"
if KIND in ("walls", "geometric") and len(scan.faces) > 300_000:
    # стенкам и сечениям хватает 300 тыс. треугольников (сдвиг поверхности сотые доли мм), а на
    # миллионе они не успевают за отведённое время
    from holes import _coarse

    scan = _coarse(scan)
sheet = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if len(sys.argv) > 5 and sys.argv[5] == "walls":
    import walls

    import pickle

    ws = walls.walls(scan)
    code = walls.draft_code(ws)
    # те же стенки отдаст инструмент walls для ИИ, чтобы не считать их второй раз
    Path(sys.argv[4]).with_suffix(".pkl").write_bytes(pickle.dumps(ws))
    meta = {"plan": f"черновик из стенок: найдено {len(ws)} стенок, каждая выдавлена по контуру",
            "restored": [], "assumptions": ["деталь тонкостенная: собрана из пластин"]}
elif len(sys.argv) > 5 and sys.argv[5] == "surface":
    import cadquery as cq

    import surface

    body, info = surface.surface_body(scan)
    step = Path(sys.argv[4]).with_suffix(".step")
    cq.exporters.export(body, str(step))
    code = f"import cadquery as cq\nresult = cq.importers.importStep({str(step)!r}).val()\n"
    meta = {"plan": f"тело по поверхности скана: {info['faces']} треугольников, отверстий вырезано точно {info['holes_cut']}",
            "restored": [], "assumptions": ["форма любая: тело повторяет поверхность скана, плоские участки выпрямлены"]}
elif len(sys.argv) > 5 and sys.argv[5] == "smooth":
    import cadquery as cq

    import nurbs
    import surface

    # Приёмка гладкого тела меряется по очищенной поверхности, а не по сырому скану: шум там уже снят,
    # и допуск сравнения (3×шум) здесь слишком широк — пропускает крупные грани, срезающие рёбра.
    # Поэтому берётся сам шум, не меньше 0,3 мм (CAD_FIT_TOL=wide — прежнее правило, для сравнения).
    noise = float(sheet.get("diagnostics", {}).get("noise_mm", 0) or 0)
    wide = __import__("os").environ.get("CAD_FIT_TOL") == "wide"
    tol = max(0.3, (3 if wide else 1) * noise)
    import time

    t_smooth = time.time()
    clean = surface.surface_mesh(scan)
    body, info = nurbs.smooth_body(surface.flatten_planes(clean), scan, tol=tol)
    if body is None:
        raise SystemExit("гладкое тело не собралось")
    planes = sum(f.geomType() == "PLANE" for f in body.Faces())
    fil_note = []
    try:  # скругления между плоскими гранями: точные цилиндры вместо сплайнов; не вышло — тело прежнее
        import fillets

        left = 1000 - (time.time() - t_smooth)   # у попытки 0к в demo.py предел 1200 с
        on = __import__("os").environ.get("CAD_FILLETS", "1") != "0"   # CAD_FILLETS=0 — без точных скруглений (для сравнения)
        fl = fillets.find_fillets(clean) if on and left > 120 else []
        if fl:
            body, built, _ = fillets.apply(body, fl, budget=min(240.0, left - 90), mesh=clean)
            fil_note = [fillets.summary(fl) + (f"; точными цилиндрами построено кусков: {built}" if built else "")]
    except Exception:  # noqa: BLE001
        fil_note = []
    body, cut = surface.cut_holes(body, scan)
    step = Path(sys.argv[4]).with_suffix(".step")
    cq.exporters.export(body, str(step))
    code = f"import cadquery as cq\nresult = cq.importers.importStep({str(step)!r}).val()\n"
    meta = {"plan": f"гладкое тело: {info['faces']} граней по скану (из них {planes} точных плоскостей, остальные B-сплайн), {round(info['near'] * 100, 1)}% поверхности "
                    f"в пределах 0,2 мм, отверстий вырезано точно {cut}",
            "restored": [], "assumptions": ["форма любая: гладкие грани повторяют поверхность скана"] + fil_note}
else:
    code, meta = geometric.build(scan, sheet, restore=sys.argv[3] == "1")
Path(sys.argv[4]).write_text(json.dumps({"code": code, "meta": meta}, ensure_ascii=False), encoding="utf-8")
