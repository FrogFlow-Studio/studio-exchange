"""Уточнение граней через Point2STEP: грани готовой модели размечают точки скана, Point2STEP
подгоняет к каждой метке точную поверхность (плоскость, цилиндр, конус, сфера или свободная
поверхность) и собирает из них B-rep. Утраченное (отверстия в сколе), которого в скане нет,
возвращается из исходной модели.

Код Point2STEP (MIT) клонируют рядом: github.com/lukau2357/point2step, путь — в POINT2STEP_DIR.
Ему нужен pythonocc (только conda); вместо него подставляется occ_shim поверх OCP из CadQuery.
Зависимости: torch, open3d, libigl (часть CGAL в libigl под GPL — для продажи разобраться отдельно).

python p2s_worker.py скан.stl модель.step папка ответ.json [секунд]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MAX_FACES = 300  # тело по поверхности из тысяч треугольников так не уточнить: нужна своя разметка


def p2s_dir() -> Path | None:
    for p in (os.environ.get("POINT2STEP_DIR"), HERE / "ext" / "point2step", HERE.parent / "point2step",
              Path.home() / "point2step"):
        if p and (Path(p) / "step_pipeline.py").exists():
            return Path(p)
    return None


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    t = path.read_text(encoding="utf-8")
    for a, b in pairs:
        if a not in t:
            print(f"Point2STEP: правка не применилась в {path.name}: {a[:60]}", flush=True)
        t = t.replace(a, b)
    path.write_text(t, encoding="utf-8")


def prepare(src: Path, work: Path) -> Path:
    """Копия Point2STEP с правками для процессора (оригинал местами жёстко требует CUDA)."""
    run = work / "p2s_run"
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(src, run, ignore=shutil.ignore_patterns(".git", "__pycache__", "output_step", "docs",
                                                             "sample_clouds", "point2cad_original", "cadrecode"))
    patch(run / "point2step" / "inr_fitting.py", [("\n    torch.cuda.synchronize()\n",
                                                   "\n    if torch.cuda.is_available(): torch.cuda.synchronize()\n")])
    patch(run / "point2step" / "primitive_fitting_utils.py", [
        ('cluster_points, "cuda:0")', 'cluster_points, "cuda:0" if torch.cuda.is_available() else "cpu")')])
    return run


def label_scan(model, scan_path: Path, n: int = 20000):
    """Точки скана и номер ближайшей грани модели; грани без точек — утраченные (в скане их нет)."""
    import trimesh
    from scipy.spatial import cKDTree

    faces = model.Faces()
    pts, lab = [], []
    total = model.Area()
    for i, f in enumerate(faces):
        v, t = f.tessellate(0.05, 0.2)
        if not t:
            continue
        m = trimesh.Trimesh([(p.x, p.y, p.z) for p in v], t)
        s, _ = trimesh.sample.sample_surface(m, max(50, int(30000 * m.area / total)), seed=0)
        pts.append(s)
        lab.append(np.full(len(s), i))
    pts, lab = np.vstack(pts), np.concatenate(lab)
    scan = trimesh.load(scan_path, force="mesh")
    sp, _ = trimesh.sample.sample_surface(scan, n, seed=1)
    _, j = cKDTree(pts).query(sp)
    L = lab[j]
    u, c = np.unique(L, return_counts=True)
    keep = u[c >= 0.004 * len(L)]
    lost = [i for i in range(len(faces)) if i not in set(keep.tolist())]
    mask = np.isin(L, keep)
    remap = {o: k for k, o in enumerate(keep)}
    return sp[mask], np.array([remap[x] for x in L[mask]]), lost


def sew_solid(path: Path):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS

    s = cq.importers.importStep(str(path)).val()
    sw = BRepBuilderAPI_Sewing(0.05)
    for f in s.Faces():
        sw.Add(f.wrapped)
    sw.Perform()
    sewn = sw.SewedShape()
    if sewn.ShapeType() != TopAbs_SHELL:
        raise SystemExit("Point2STEP: грани не сшились в одну оболочку")
    fs = ShapeFix_Shell(TopoDS.Shell_s(sewn))
    fs.Perform()
    shell = cq.Shape.cast(fs.Shell())
    if not shell.Closed():
        raise SystemExit("Point2STEP: оболочка не замкнута")
    solid = cq.Solid.makeSolid(shell)
    fx = ShapeFix_Solid(solid.wrapped)
    fx.Perform()
    solid = cq.Shape.cast(fx.Solid())
    if not solid.isValid() or solid.Volume() <= 0:
        raise SystemExit("Point2STEP: тело невалидно")
    return solid


def restore_lost(solid, model, lost: list[int]):
    """Утраченные грани исходной модели (их нет в скане): цилиндрические отверстия вырезаем заново."""
    import cadquery as cq

    faces = model.Faces()
    back = []
    for i in lost:
        f = faces[i]
        if f.geomType() != "CYLINDER":
            raise SystemExit(f"Point2STEP: утраченную грань типа {f.geomType()} вернуть не умею")
        cyl = f._geomAdaptor().Cylinder()
        ax, r = cyl.Axis(), cyl.Radius()
        o = cq.Vector(ax.Location().X(), ax.Location().Y(), ax.Location().Z())
        d = cq.Vector(ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())
        c = f.positionAt(0.5, 0.5)  # точка на самой грани (центр масс цилиндра лежит на оси)
        # отверстие: нормаль поверхности смотрит к оси; иначе это выступ, его не трогаем
        foot = o + d * (c - o).dot(d)
        if f.normalAt(c).dot(foot - c) <= 0:
            raise SystemExit("Point2STEP: утраченный выступ вернуть не умею")
        bb = model.BoundingBox()
        length = 2 * bb.DiagonalLength
        tool = cq.Solid.makeCylinder(r, length, foot - d * (length / 2), d)
        solid = solid.cut(tool)
        back.append(f"отверстие Ø{2 * r:.2f} мм — из исходной модели (в скане его нет)")
    return solid, back


def main() -> None:
    scan, model_step, work, answer = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    budget = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    t0 = time.time()
    src = p2s_dir()
    if src is None:
        raise SystemExit("Point2STEP не найден: склонируйте github.com/lukau2357/point2step и задайте POINT2STEP_DIR")
    import cadquery as cq

    model = cq.importers.importStep(str(model_step)).val()
    if len(model.Faces()) > MAX_FACES:
        raise SystemExit(f"Point2STEP: у модели {len(model.Faces())} граней, уточнять нечего (тело по поверхности)")
    work.mkdir(parents=True, exist_ok=True)
    run = prepare(src, work)
    pts, lab, lost = label_scan(model, scan)
    inp = work / "in" / "part"
    inp.mkdir(parents=True, exist_ok=True)
    np.savetxt(inp / "0.xyzc", np.column_stack([pts, lab]))
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([str(HERE / "occ_shim"), os.environ.get("PYTHONPATH", "")]))
    out = work / "out"
    subprocess.run([sys.executable, "step_pipeline.py", "--input_dir", str(work / "in"), "--model_id", "part",
                    "--output_dir", str(out)], cwd=run, env=env, check=True,
                   timeout=max(30, budget - (time.time() - t0)))
    step = out / "part" / "unified" / "unified.step"
    if not step.exists():
        raise SystemExit("Point2STEP не собрал STEP")
    solid = sew_solid(step)
    solid, back = restore_lost(solid, model, lost)
    if not solid.isValid():
        raise SystemExit("Point2STEP: после возврата утраченного тело невалидно")
    res = answer.with_suffix(".step")
    cq.exporters.export(cq.Workplane().add(solid), str(res))
    kinds: dict[str, int] = {}
    for f in solid.Faces():
        kinds[f.geomType()] = kinds.get(f.geomType(), 0) + 1
    meta = {"plan": f"Point2STEP уточнил по скану {int(lab.max()) + 1} граней исходной модели: "
                    + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())),
            "restored": back,
            "assumptions": ["Point2STEP (код MIT; libigl/CGAL — GPL, для продажи проверить)",
                            "грани размечены по лучшей модели; утраченное взято из неё"]}
    code = f"import cadquery as cq\nresult = cq.importers.importStep({str(res)!r}).val()\n"
    answer.write_text(json.dumps({"code": code, "meta": meta}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
