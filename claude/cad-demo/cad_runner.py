"""Исполняет код CadQuery, который написал ИИ, в отдельном процессе.

Запуск: python cad_runner.py code.py outdir [STEP от геометрического воркера]
Код должен положить итоговое тело в переменную `result`.
Пишет outdir/model.step, outdir/model.stl, outdir/run.json (валидность, объём,
габарит) или outdir/run.json с ошибкой и трассировкой.
"""
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from cad_checks import measured

code_path, out = Path(sys.argv[1]), Path(sys.argv[2])
step_source = Path(sys.argv[3]) if len(sys.argv) > 3 else None
# Ядро CAD на неудачной геометрии иногда съедает всю память; ограничиваем процесс
# (3 ГБ), чтобы упала одна попытка, а не ноутбук. На Windows такого ограничения нет.
try:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 ** 3, 3 * 1024 ** 3))
except (ImportError, ValueError, OSError):
    pass
timings = {}
info = {"ok": False, "timings": timings}
try:
    import cadquery as cq

    if step_source is not None:
        with measured(timings, "step_import"):
            res = cq.importers.importStep(str(step_source)).val()
    else:
        ns: dict = {"cq": cq, "__name__": "__ai_model__"}
        if os.environ.get("CAD_DRAFT") and Path(os.environ["CAD_DRAFT"]).exists():
            ns["draft"] = cq.importers.importStep(os.environ["CAD_DRAFT"]).val()
        exec(compile(code_path.read_text(encoding="utf-8"), str(code_path), "exec"), ns)
        res = ns.get("result")
    if res is None:
        raise RuntimeError("код не задал переменную result")
    shape = res.val() if isinstance(res, cq.Workplane) else res
    if isinstance(res, cq.Workplane) and len(res.vals()) > 1:
        shape = cq.Compound.makeCompound(res.vals())
    solids = shape.Solids()
    if not solids:
        raise RuntimeError("result не содержит ни одного твёрдого тела (solid)")
    bb = shape.BoundingBox()
    info.update({
        "valid": bool(shape.isValid()),
        "solids": len(solids),
        # оболочек больше, чем тел: внутри закрытые пустоты (печать их не простит)
        "voids": sum(len(x.Shells()) - 1 for x in solids),
        "volume_mm3": round(shape.Volume(), 1),
        "bbox_mm": [[round(bb.xmin, 2), round(bb.ymin, 2), round(bb.zmin, 2)],
                    [round(bb.xmax, 2), round(bb.ymax, 2), round(bb.zmax, 2)]],
        "faces": len(shape.Faces()),
        # типы граней: точные (плоскость, цилиндр…) инженер правит размером, сплайн — только формой
        "face_types": {t: n for t, n in __import__("collections").Counter(f.geomType() for f in shape.Faces()).items()},
    })
    if step_source is None:
        with measured(timings, "step_write"):
            cq.exporters.export(shape, str(out / "model.step"))
    else:
        with measured(timings, "step_copy"):
            shutil.copyfile(step_source, out / "model.step")
    # гладкое тело из тысяч B-сплайн граней при мелком угловом шаге даёт миллионы треугольников
    # (сотни мегабайт и падение); точность STL задаёт линейный допуск, угол можно взять крупнее
    # Линейный допуск здесь относительный (доля размера грани): при 0,05 сетка крупной сплайновой
    # грани уходила от неё до 0,8 мм, и сравнение со сканом штрафовало модель за погрешность самой
    # сетки (Э11). Поэтому у гладкого тела допуск 0,01; признак — число сплайновых граней, а не всех.
    many = sum(f.geomType() not in ("PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS") for f in shape.Faces()) > 200
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    # CadQuery exportStl already uses relative deflection and parallel meshing.
    with measured(timings, "stl_tessellation"):
        BRepMesh_IncrementalMesh(shape.wrapped, 0.01 if many else 0.02, True, 1.0 if many else 0.1, True)
    with measured(timings, "stl_write"):
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape.wrapped, str(out / "model.stl")):
            raise RuntimeError("не удалось записать STL")
    # Относительный допуск на тысячах мелких сплайновых граней даёт миллионы треугольников: крышка
    # студии, 4287 граней — 2,4 млн, сравнение со сканом съело 13 ГБ и было убито. Более грубая
    # тесселяция OCC мало помогает (абсолютный допуск 0,1 мм — 4 млн), поэтому готовая сетка
    # упрощается до 1 млн треугольников: на гладких гранях это сотые доли миллиметра.
    stl_tri = ((out / "model.stl").stat().st_size - 84) // 50
    if stl_tri > 1_200_000:
        import subprocess

        # отдельный процесс: здесь память уже занята ядром CAD (см. stl_simplify.py); если
        # упрощение не удалось, остаётся полная сетка — модель не теряется
        with measured(timings, "stl_simplify"):
            try:
                done = subprocess.run([sys.executable, str(Path(__file__).with_name("stl_simplify.py")),
                                       str(out / "model.stl")], capture_output=True, text=True, timeout=600)
                if done.returncode == 0 and done.stdout.strip().isdigit():
                    info["stl_simplified_from"] = int(done.stdout.strip())
                elif done.returncode != 0:
                    info["stl_simplify_error"] = done.stderr.strip()[-300:]
            except subprocess.TimeoutExpired:
                info["stl_simplify_error"] = "время вышло"
    info["ok"] = True
except Exception as e:  # noqa: BLE001 — вся ошибка уходит обратно ИИ
    tb = traceback.format_exc()
    info["error"] = f"{type(e).__name__}: {e}"
    info["traceback"] = tb[-3000:]
(out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
