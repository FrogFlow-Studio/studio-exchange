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
        "volume_mm3": round(shape.Volume(), 1),
        "bbox_mm": [[round(bb.xmin, 2), round(bb.ymin, 2), round(bb.zmin, 2)],
                    [round(bb.xmax, 2), round(bb.ymax, 2), round(bb.zmax, 2)]],
        "faces": len(shape.Faces()),
    })
    cq.exporters.export(shape, str(out / "model.step"))
    cq.exporters.export(shape, str(out / "model.stl"), tolerance=0.02, angularTolerance=0.1)
    info["ok"] = True
except Exception as e:  # noqa: BLE001 — вся ошибка уходит обратно ИИ
    tb = traceback.format_exc()
    info["error"] = f"{type(e).__name__}: {e}"
    info["traceback"] = tb[-3000:]
(out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
