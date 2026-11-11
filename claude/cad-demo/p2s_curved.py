"""Эксперимент: Point2STEP на кривой детали (линза, заглушка Hyundai) — по плану Вектора от 11.11.

Скан размечается на участки (segment.py), Point2STEP подгоняет к каждому участку поверхность
(плоскость, цилиндр, конус, сфера или свободную поверхность нейросетью INR) и собирает STEP.
Здесь же результат проверяется: сшивается ли в тело, какую часть скана покрывает, насколько
близко. Свободные поверхности на процессоре идут больше часа на участок, поэтому запускать на
видеокарте (ПК с RTX).

python p2s_curved.py скан.stl папка [R мм = 5] [порог = 0.005]
python p2s_curved.py скан.stl готовый.step   (только проверка результата)
→ папка/p2s_curved.json (итог) и папка/work/out/part/unified/unified.step
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import p2s_worker as pw  # noqa: E402
import segment  # noqa: E402


def check(step: Path, scan: trimesh.Trimesh) -> dict:
    """Что дал Point2STEP: грани, покрытие скана, сшивается ли в тело и насколько близко."""
    res: dict = {}
    import cadquery as cq

    import compare as cmpmod

    shape = cq.importers.importStep(str(step)).val()
    res["faces"] = len(shape.Faces())
    res["face_types"] = {}
    for f in shape.Faces():
        res["face_types"][f.geomType()] = res["face_types"].get(f.geomType(), 0) + 1
    # покрытие: доля точек скана, у которых рядом (0,5 мм) есть поверхность Point2STEP
    v, tri = shape.tessellate(0.05, 0.3)
    m = trimesh.Trimesh([(q.x, q.y, q.z) for q in v], tri)
    sp, _ = trimesh.sample.sample_surface(scan, 20000, seed=1)
    _, d, _ = trimesh.proximity.closest_point(m, sp)
    res["coverage_0_5mm"] = round(float((d < 0.5).mean()), 4)
    res["median_mm"] = round(float(np.median(d)), 3)
    try:
        solid = pw.sew_solid(step)
        res["solid_valid"] = bool(solid.isValid())
        mv, mt = solid.tessellate(0.05, 0.3)
        c = cmpmod.compare(scan, trimesh.Trimesh([(q.x, q.y, q.z) for q in mv], mt), tol=0.2)
        res["within_0_2mm"] = c["within_tol_share"]
    except SystemExit as ex:
        res["solid_valid"] = False
        res["solid_error"] = str(ex)
    return res


def main() -> None:
    scan_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.suffix.lower() == ".step":  # только проверка готового результата
        print(json.dumps(check(out, trimesh.load(scan_path, force="mesh")), ensure_ascii=False, indent=1))
        return
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
    src = pw.p2s_dir()
    if src is None:
        sys.exit("Point2STEP не найден: склонируйте github.com/lukau2357/point2step и задайте POINT2STEP_DIR")
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    res = {"scan": scan_path.name, "R": R, "thr": thr}
    try:
        import torch

        res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        res["gpu"] = None
    print("видеокарта:", res["gpu"] or "нет (на процессоре будет очень долго)", flush=True)
    run = pw.prepare(src, work)
    scan = trimesh.load(scan_path, force="mesh")
    pts, lab = segment.segment(scan, R, thr)
    inp = work / "in" / "part"
    inp.mkdir(parents=True, exist_ok=True)
    np.savetxt(inp / "0.xyzc", np.column_stack([pts, lab]))
    res["segments"] = int(lab.max() + 1)
    print(f"участков: {res['segments']}, точек: {len(pts)}", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([str(HERE / "occ_shim"), os.environ.get("PYTHONPATH", "")]))
    t = time.time()
    p = subprocess.run([sys.executable, "step_pipeline.py", "--input_dir", str(work / "in"), "--model_id", "part",
                        "--output_dir", str(work / "out")], cwd=run, env=env)
    res["p2s_sec"] = round(time.time() - t)
    res["p2s_exit"] = p.returncode
    step = work / "out" / "part" / "unified" / "unified.step"
    res["step"] = str(step) if step.exists() else None
    if step.exists():
        res.update(check(step, scan))
    (out / "p2s_curved.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
