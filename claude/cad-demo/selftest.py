"""Самопроверка: python selftest.py  — всё ли работает после правок (без ИИ, ~1 мин).

Проверяет замеры на трёх тестовых деталях и на испорченных копиях скана
(повёрнут, в метрах, дыры, мусор), затем полный цикл в режиме без ИИ.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

import scan_analysis as sa

HERE = Path(__file__).parent
S = HERE / "samples"
fails = []


def check(name, cond, info=""):
    print(("  ок   " if cond else "  СБОЙ ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        fails.append(name)


def pattern(sheet):
    for sec in sheet["sections"]:
        for r in sec["regions"]:
            for p in r.get("hole_patterns", []):
                return p
    return None


def near(a, b, tol):
    return a is not None and abs(a - b) <= tol


print("Совместимость с Python 3.10–3.12 (README обещает их):")
import shutil  # noqa: E402

for v in ("3.10", "3.11", "3.12"):
    exe = shutil.which(f"python{v}")
    if not exe:
        print(f"  --   Python {v} не установлен здесь, пропуск")
        continue
    bad = [f.name for f in HERE.glob("*.py")
           if subprocess.run([exe, "-m", "py_compile", str(f)], capture_output=True).returncode != 0]
    check(f"Python {v}: все файлы компилируются", not bad, bad)

print("Замеры:")
sh, _ = sa.measurement_sheet(str(S / "bracket_scan.stl"))
p = pattern(sh)
check("кронштейн: найдено 3 из 4 отверстий", p and p["found"] == 3 and p["likely_total"] == 4)
check("кронштейн: место отломанного отверстия", p and np.allclose(np.abs(p["missing_positions_xy"][0]), [32, 17], atol=0.5), p)
lv = [x["z_mm"] for x in sh["horizontal_planes"]]
check("кронштейн: высоты 0 / 8 / 26", all(any(abs(z - t) < 0.2 for z in lv) for t in (0, 8, 26)), lv)
sh, _ = sa.measurement_sheet(str(S / "flange_scan.stl"))
p = pattern(sh)
check("фланец: 5 из 6 по окружности, PCD 68", p and p["found"] == 5 and p["likely_total"] == 6 and near(p.get("pcd_diameter"), 68, 0.3), p)
sh, _ = sa.measurement_sheet(str(S / "bushing_scan.stl"))
check("втулка: найден паз у отверстия", any("паз" in line for line in sh["summary"]))

print("Испорченные сканы:")
base = trimesh.load(S / "bracket_scan.stl")
rng = np.random.default_rng(1)
tmp = Path(tempfile.mkdtemp())
variants = {}
m = base.copy()
T = trimesh.transformations.random_rotation_matrix(rng.random(3))
T[:3, 3] = [100, -50, 30]
m.apply_transform(T)
variants["повёрнут"] = m
m = base.copy()
m.apply_scale(0.001)
variants["в метрах"] = m
m = base.copy()
keep = np.ones(len(m.faces), bool)
for _ in range(6):
    keep &= np.linalg.norm(m.triangles_center - m.vertices[rng.integers(len(m.vertices))], axis=1) > 3
m.update_faces(keep)
variants["с дырами"] = m
variants["с мусором"] = trimesh.util.concatenate([base.copy(), trimesh.creation.icosphere(radius=2).apply_translation([60, 60, 0])])
for name, m in variants.items():
    f = tmp / "v.stl"
    m.export(f)
    try:
        sh, _ = sa.measurement_sheet(str(f), units="m" if name == "в метрах" else "mm")
        p = pattern(sh)
        ext = sorted(sh["diagnostics"]["extents_mm"])
        lo, hi = sh["bounds_mm"]
        ok = p is not None and p["likely_total"] == 4 and abs((hi[0] - lo[0]) - 80) < 1.5
        check(f"кронштейн {name}: габарит и отверстия", ok, f"{p}, {lo}, {hi}")
    except Exception as e:  # noqa: BLE001
        check(f"кронштейн {name}", False, f"{type(e).__name__}: {e}")

tiny = trimesh.creation.box(extents=[1.0, 0.8, 0.5])
tiny.export(tmp / "tiny.stl")
sh, _ = sa.measurement_sheet(str(tmp / "tiny.stl"))
check("деталь 1×0.8×0.5 мм не увеличивается молча в 1000 раз", max(sh["diagnostics"]["extents_mm"]) < 2,
      sh["diagnostics"]["extents_mm"])

print("Отрицательные проверки (неправильная модель не должна приниматься):")
import cadquery as cq  # noqa: E402

import gates  # noqa: E402


def _mesh(wp):
    f = tmp / "neg.stl"
    cq.exporters.export(wp, str(f), tolerance=0.02)
    return trimesh.load(f, force="mesh")


plate = (cq.Workplane("XY").box(80, 50, 8, centered=(True, True, False)).faces(">Z").workplane()
         .rect(60, 30, forConstruction=True).vertices().hole(2))
pf = tmp / "plate.stl"
cq.exporters.export(plate, str(pf), tolerance=0.02)
psheet, pscan = sa.measurement_sheet(str(pf))
Tm = np.array(psheet["transform_from_scan_4x4"])
ok_run = {"ok": True, "valid": True, "solids": 1}
m_ok = _mesh(plate)
m_ok.apply_transform(Tm)
check("целая плита с отверстиями Ø2 проходит проверки", not gates.check(ok_run, m_ok, psheet, pscan),
      gates.check(ok_run, m_ok, psheet, pscan))
plug = plate.union(cq.Workplane("XY").center(30, 15).circle(1.2).extrude(8))
m_plug = _mesh(plug)
m_plug.apply_transform(Tm)
check("заделанное отверстие Ø2 не принимается", bool(gates.check(ok_run, m_plug, psheet, pscan)))
tower = plate.union(cq.Workplane("XY").center(0, 0).rect(2, 2).extrude(100))
m_tow = _mesh(tower)
m_tow.apply_transform(Tm)
check("лишняя «башня» 2×2×100 не принимается", bool(gates.check(ok_run, m_tow, psheet, pscan)))
three = (cq.Workplane("XY").box(80, 50, 8, centered=(True, True, False)).faces(">Z").workplane()
         .pushPoints([(-30, -15), (30, -15), (-30, 15)]).hole(4))
tf = tmp / "three.stl"
cq.exporters.export(three, str(tf), tolerance=0.02)
tsheet, tscan = sa.measurement_sheet(str(tf))
import geometric  # noqa: E402

tcode, tmeta = geometric.build(tscan, tsheet)
check("целая плита с тремя отверстиями: четвёртое не сверлится", tcode.count(".circle(2.0") <= 3
      and not any("добавлено" in r for r in tmeta["restored"]), tmeta["restored"])
check("два тела не принимаются", bool(gates.check(dict(ok_run, solids=2), m_ok, psheet, pscan)))

print("Полный цикл: алгоритм без ИИ (попытка 0), затем заготовленные ответы:")
import re

for part in ("bracket", "flange", "bushing"):
    r = subprocess.run([sys.executable, str(HERE / "demo.py"), str(S / f"{part}_scan.stl"), "--no-open",
                        "--mode", "geometric", "--out", str(tmp / "out")], capture_output=True, text=True, encoding="utf-8")
    m = re.search(r"эталон: в допуске ([\d.]+)%", r.stdout)
    t = float(m.group(1)) if m else 0.0
    check(f"{part}: алгоритм без ИИ построил модель, совпадение с целым эталоном ≥ 95% ({t}%)",
          r.returncode == 0 and "готово:" in r.stdout and t >= 95, r.stdout[-400:] + r.stderr[-400:])
    r = subprocess.run([sys.executable, str(HERE / "demo.py"), str(S / f"{part}_scan.stl"), "--no-open",
                        "--mode", "scripted", "--out", str(tmp / "out")], capture_output=True, text=True, encoding="utf-8")
    ok = r.returncode == 0 and "сошлось" in r.stdout and "готово:" in r.stdout
    check(f"{part}: цикл с исправлениями сошёлся, отчёт есть", ok, r.stdout[-400:] + r.stderr[-400:])

print("\nИТОГ: " + ("всё в порядке" if not fails else f"сбоев {len(fails)}: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
