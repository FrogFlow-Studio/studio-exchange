"""Исполняет код CadQuery, который написал ИИ, в отдельном процессе.

Запуск: python cad_runner.py code.py outdir
Код должен положить итоговое тело в переменную `result`.
Пишет outdir/model.step, outdir/model.stl, outdir/run.json (валидность, объём,
габарит) или outdir/run.json с ошибкой и трассировкой.
"""
import json
import sys
import traceback
from pathlib import Path

code_path, out = Path(sys.argv[1]), Path(sys.argv[2])
# Ядро CAD на неудачной геометрии иногда съедает всю память; ограничиваем процесс
# (3 ГБ), чтобы упала одна попытка, а не ноутбук. На Windows такого ограничения нет.
try:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 ** 3, 3 * 1024 ** 3))
except (ImportError, ValueError, OSError):
    pass
info = {"ok": False}
try:
    import cadquery as cq

    ns: dict = {"cq": cq, "__name__": "__ai_model__"}
    import os

    if os.environ.get("CAD_DRAFT") and Path(os.environ["CAD_DRAFT"]).exists():
        ns["draft"] = cq.importers.importStep(os.environ["CAD_DRAFT"]).val()  # черновик без ИИ
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
    cq.exporters.export(shape, str(out / "model.step"))
    # гладкое тело из тысяч B-сплайн граней при мелком угловом шаге даёт миллионы треугольников
    # (сотни мегабайт и падение); точность STL задаёт линейный допуск, угол можно взять крупнее
    # Линейный допуск здесь относительный (доля размера грани): при 0,05 сетка крупной сплайновой
    # грани уходила от неё до 0,8 мм, и сравнение со сканом штрафовало модель за погрешность самой
    # сетки (Э11). Поэтому у гладкого тела допуск 0,01; признак — число сплайновых граней, а не всех.
    many = sum(f.geomType() not in ("PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS") for f in shape.Faces()) > 200
    cq.exporters.export(shape, str(out / "model.stl"), tolerance=0.01 if many else 0.02,
                        angularTolerance=1.0 if many else 0.1)
    info["ok"] = True
except Exception as e:  # noqa: BLE001 — вся ошибка уходит обратно ИИ
    tb = traceback.format_exc()
    info["error"] = f"{type(e).__name__}: {e}"
    info["traceback"] = tb[-3000:]
(out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
