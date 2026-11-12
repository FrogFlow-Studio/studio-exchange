"""Макрос SolidWorks с деревом построения для призматической модели (пути 0 и 0ж).

Программа модели CadQuery из путей 0 и 0ж — это эскизы и операции: слой вытягивается по контуру
(отрезки и дуги), круглые слои — тело вращения по профилю, пазы и отверстия — вырезы. Здесь эти
операции разбираются и записываются макросом VBA для SolidWorks. Инженер получает модель с деревом:
каждый слой, вырез и отверстие — отдельная операция со своим эскизом, размеры правятся в эскизе.

Запуск: python sw_macro.py model.py model_sw.bas [model.step]
  model.step (необязательно) — сверить объём: операции повторяются в CadQuery и сравниваются
  с исходной моделью. Это проверяет разбор, но не сам SolidWorks: макрос надо прогнать в нём.

Что не переносится (пишется в отчёт и в начало макроса): наклонные грани, отверстия в стенках,
вторая ось вращения. Для таких деталей макрос даёт основу, остальное — STEP.

Оси: в CadQuery Z вверх, в SolidWorks Y вверх. Эскизы слоёв — на плоскости «Сверху» (XZ
SolidWorks) и параллельных ей. Точка (x, y) эскиза на этой плоскости — это (x, y) CadQuery,
выдавливание идёт вверх. Тело вращения — на плоскости «Спереди», ось — вертикальная линия
через начало координат: модель сдвинута так, чтобы ось первого тела вращения прошла через него.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"
PT = rf"\(\s*({NUM})\s*,\s*({NUM})\s*\)"


def _chain(s: str):
    """Контур: список отрезков и дуг [("line", p0, p1) | ("arc", p0, pm, p1)] или None."""
    m = re.match(rf"polyline\((\[\[.*?\]\])\)\.close\(\)", s)
    if m:
        pts = [tuple(map(float, p)) for p in json.loads(m.group(1))]
        return [("line", pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))], s[m.end():]
    m = re.match(rf"moveTo{PT}", s)
    if not m:
        return None, s
    cur = start = (float(m.group(1)), float(m.group(2)))
    s = s[m.end():]
    segs = []
    while True:
        if (m := re.match(rf"\.lineTo{PT}", s)):
            p = (float(m.group(1)), float(m.group(2)))
            segs.append(("line", cur, p))
            cur = p
        elif (m := re.match(rf"\.threePointArc\({PT}\s*,\s*{PT}\)", s)):
            pm = (float(m.group(1)), float(m.group(2)))
            p = (float(m.group(3)), float(m.group(4)))
            segs.append(("arc", cur, pm, p))
            cur = p
        elif (m := re.match(r"\.close\(\)", s)):
            if abs(cur[0] - start[0]) + abs(cur[1] - start[1]) > 1e-6:
                segs.append(("line", cur, start))
            return segs, s[m.end():]
        else:
            return None, s
        s = s[m.end():]


def parse(code: str) -> tuple[list[dict], list[str]]:
    """Операции модели по порядку и список строк, которые не перенесены."""
    ops, skipped = [], []
    for ln in code.splitlines():
        t = ln.strip()
        if not t or t.startswith("#") or not re.search(r"cq\.(Workplane|Plane)", t):
            continue
        body = re.sub(r"\s*#.*$", "", t)
        m = re.match(rf"s = cq\.Workplane\('XY'\)\.workplane\(offset=({NUM})\)\.(.*)\.extrude\(({NUM})\)$", body)
        if m:
            segs, rest = _chain(m.group(2))
            if segs is not None and not rest:
                ops.append({"op": "boss", "z": float(m.group(1)), "h": float(m.group(3)), "segs": segs})
                continue
        m = re.match(rf"s = s\.cut\(cq\.Workplane\('XY'\)\.workplane\(offset=({NUM})\)\.(.*)\.extrude\(({NUM})\)\)$", body)
        if m:
            segs, rest = _chain(m.group(2))
            if segs is not None and not rest:
                ops.append({"op": "cut", "z": float(m.group(1)), "h": float(m.group(3)), "segs": segs})
                continue
        m = re.match(rf"s = cq\.Workplane\('XZ'\)\.polyline\((\[\[.*?\]\])\)\.close\(\)\.revolve\(360, \(0, 0, 0\), \(0, 1, 0\)\)"
                     rf"\.translate\(\(({NUM}), ({NUM}), 0\)\)$", body)
        if m:
            ops.append({"op": "revolve", "prof": [tuple(map(float, p)) for p in json.loads(m.group(1))],
                        "c": (float(m.group(2)), float(m.group(3)))})
            continue
        m = re.match(rf"result = result\.cut\(cq\.Workplane\('XY'\)\.workplane\(offset=({NUM})\)\.center\(({NUM}), ({NUM})\)"
                     rf"\.circle\(({NUM})\)\.extrude\(({NUM})\)\)$", body)
        if m:
            z, x, y, r, h = map(float, m.groups())
            ops.append({"op": "hole", "z": z, "h": h, "c": (x, y), "r": r})
            continue
        skipped.append(t[:160])
    revs = [o for o in ops if o["op"] == "revolve"]
    for o in revs[1:]:
        if max(abs(o["c"][0] - revs[0]["c"][0]), abs(o["c"][1] - revs[0]["c"][1])) > 1e-3:
            skipped.append(f"тело вращения на второй оси ({o['c'][0]:.3f}, {o['c'][1]:.3f})")
            ops.remove(o)
    return ops, skipped


def replay(ops: list[dict]):
    """Те же операции в CadQuery: проверка разбора (объём должен совпасть с исходной моделью)."""
    import cadquery as cq

    def wire(wp, segs):
        wp = wp.moveTo(*segs[0][1])
        for s in segs:
            wp = wp.lineTo(*s[2]) if s[0] == "line" else wp.threePointArc(s[2], s[3])
        return wp.close()

    result = None
    for o in ops:
        if o["op"] == "boss":
            s = wire(cq.Workplane("XY").workplane(offset=o["z"]), o["segs"]).extrude(o["h"])
        elif o["op"] == "revolve":
            s = (cq.Workplane("XZ").polyline([list(p) for p in o["prof"]]).close()
                 .revolve(360, (0, 0, 0), (0, 1, 0)).translate((*o["c"], 0)))
        elif o["op"] == "cut" and result is not None:
            result = result.cut(wire(cq.Workplane("XY").workplane(offset=o["z"]), o["segs"]).extrude(o["h"]))
            continue
        elif o["op"] == "hole" and result is not None:
            result = result.cut(cq.Workplane("XY").workplane(offset=o["z"]).center(*o["c"]).circle(o["r"]).extrude(o["h"]))
            continue
        else:
            continue
        result = s if result is None else result.union(s)
    return result


def vba(ops: list[dict], skipped: list[str], name: str) -> str:
    revs = [o for o in ops if o["op"] == "revolve"]
    gx, gy = revs[0]["c"] if revs else (0.0, 0.0)

    def m(v):  # мм → м (SolidWorks API работает в метрах)
        return f"{v / 1000:.7f}"

    def xy(p):
        return m(p[0] - gx), m(p[1] - gy)

    L = ["' Макрос SolidWorks: модель с деревом построения (FrogFlow Studio, CAD Intelligence)",
         f"' Источник: {name}. Единицы: мм. Ось Y SolidWorks = ось Z модели.",
         f"' Модель сдвинута на ({-gx:.3f}; {-gy:.3f}) мм по X и Z, чтобы ось вращения прошла через начало координат."
         if revs else "' Сдвига нет: тел вращения в модели нет.",
         "' Запуск: Инструменты → Макрос → Создать, вставить этот текст в модуль, запустить main.",
         "' ВНИМАНИЕ: сгенерировано автоматически, перед показом проверить в SolidWorks."]
    for s in skipped:
        L.append(f"' НЕ ПЕРЕНЕСЕНО (есть в STEP): {s}")
    L += ["Option Explicit", "Dim swApp As Object, Part As Object, SM As Object, FM As Object", "",
          "Function BasePlane(idx As Integer) As Object",
          "    ' 1 = Спереди, 2 = Сверху, 3 = Справа: по порядку в дереве, имена зависят от языка",
          "    Dim f As Object, n As Integer",
          "    Set f = Part.FirstFeature",
          "    Do While Not f Is Nothing",
          "        If f.GetTypeName2 = \"RefPlane\" Then",
          "            n = n + 1",
          "            If n = idx Then Set BasePlane = f: Exit Function",
          "        End If",
          "        Set f = f.GetNextFeature",
          "    Loop",
          "End Function", "",
          "Function TopAt(z As Double) As Object",
          "    ' плоскость, параллельная «Сверху», на высоте z (м); на нуле — сама «Сверху»",
          "    Part.ClearSelection2 True",
          "    BasePlane(2).Select2 False, 0",
          "    If Abs(z) < 0.0000001 Then",
          "        Set TopAt = BasePlane(2)",
          "    ElseIf z > 0 Then",
          "        Set TopAt = FM.InsertRefPlane(8, z, 0, 0, 0, 0)",
          "    Else",
          "        Set TopAt = FM.InsertRefPlane(8 + 256, -z, 0, 0, 0, 0)",
          "    End If",
          "End Function", "",
          "Function OpenSketch(pl As Object) As Object",
          "    Part.ClearSelection2 True",
          "    pl.Select2 False, 0",
          "    SM.InsertSketch True",
          "    Set OpenSketch = SM.ActiveSketch",
          "End Function", "",
          "Function CloseSketch(sk As Object) As Object",
          "    Dim f As Object",
          "    SM.InsertSketch True",
          "    Part.ClearSelection2 True",
          "    Set f = Part.FeatureByPositionReverse(0)  ' только что закрытый эскиз",
          "    f.Select2 False, 0",
          "    Set CloseSketch = f",
          "End Function", "",
          "Sub Boss(h As Double)",
          "    ' на середину: направление вытягивания не важно, слой стоит на высоте z..z+h",
          "    FM.FeatureExtrusion2 True, False, False, 6, 0, h, 0, False, False, False, False, 0, 0, _",
          "        False, False, False, False, True, True, True, 0, 0, False",
          "End Sub", "",
          "Sub Cut(h As Double)",
          "    FM.FeatureCut4 True, False, False, 6, 0, h, 0, False, False, False, False, 0, 0, _",
          "        False, False, False, False, False, True, True, True, True, False, 0, 0, False, False",
          "End Sub", "",
          "Sub main()",
          "    Set swApp = Application.SldWorks",
          "    Set Part = swApp.NewDocument(swApp.GetUserPreferenceStringValue(8), 0, 0, 0)",
          "    Set SM = Part.SketchManager",
          "    Set FM = Part.FeatureManager",
          "    SM.AddToDB = True",
          "    SM.DisplayWhenAdded = False",
          "    Dim sk As Object"]

    def contour(segs):
        out = []
        for s in segs:
            if s[0] == "line":
                (x0, y0), (x1, y1) = xy(s[1]), xy(s[2])
                out.append(f"    SM.CreateLine {x0}, {y0}, 0, {x1}, {y1}, 0")
            else:
                (x0, y0), (xm, ym), (x1, y1) = xy(s[1]), xy(s[2]), xy(s[3])
                out.append(f"    SM.Create3PointArc {x0}, {y0}, 0, {x1}, {y1}, 0, {xm}, {ym}, 0")
        return out

    n = {"boss": 0, "cut": 0, "revolve": 0, "hole": 0}
    for o in ops:
        n[o["op"]] += 1
        if o["op"] in ("boss", "cut"):
            what = "Слой" if o["op"] == "boss" else "Вырез (паз)"
            L.append(f"    ' {what} {n[o['op']]}: высота {o['z']:.3f}…{o['z'] + o['h']:.3f} мм")
            L.append(f"    Set sk = OpenSketch(TopAt({m(o['z'] + o['h'] / 2)}))")
            L += contour(o["segs"])
            L.append("    CloseSketch sk")
            L.append(f"    {'Boss' if o['op'] == 'boss' else 'Cut'} {m(o['h'])}")
        elif o["op"] == "hole":
            x, y = xy(o["c"])
            L.append(f"    ' Отверстие {n['hole']}: Ø{2 * o['r']:.3f} мм, высота {o['z']:.3f}…{o['z'] + o['h']:.3f} мм")
            L.append(f"    Set sk = OpenSketch(TopAt({m(o['z'] + o['h'] / 2)}))")
            L.append(f"    SM.CreateCircleByRadius {x}, {y}, 0, {m(o['r'])}")
            L.append("    CloseSketch sk")
            L.append(f"    Cut {m(o['h'])}")
        else:
            prof = o["prof"]
            L.append(f"    ' Тело вращения {n['revolve']}: профиль радиус/высота, ось — вертикаль через начало")
            L.append("    Set sk = OpenSketch(BasePlane(1))")
            for i in range(len(prof)):
                (r0, z0), (r1, z1) = prof[i], prof[(i + 1) % len(prof)]
                if abs(r0 - r1) + abs(z0 - z1) < 1e-6:
                    continue
                L.append(f"    SM.CreateLine {m(r0)}, {m(z0)}, 0, {m(r1)}, {m(z1)}, 0")
            zs = [p[1] for p in prof]
            L.append(f"    SM.CreateCenterLine 0, {m(min(zs) - 1)}, 0, 0, {m(max(zs) + 1)}, 0")
            L.append("    CloseSketch sk")
            L.append("    FM.FeatureRevolve2 True, True, False, False, False, False, 0, 0, 6.283185307, 0, _")
            L.append("        False, False, 0.01, 0.01, 0, 0, 0, True, True, True")
    L += ["    SM.AddToDB = False", "    SM.DisplayWhenAdded = True", "    Part.ViewZoomtofit2",
          f"    MsgBox \"Построено: слоёв {n['boss']}, тел вращения {n['revolve']}, вырезов {n['cut']}, отверстий {n['hole']}\"",
          "End Sub"]
    return "\r\n".join(L) + "\r\n"


def export(model_py: Path, step: Path, dst: Path) -> dict | None:
    """Макрос рядом с моделью, только если перенесено всё и объём повтора совпал (0,1%)."""
    try:
        ops, skipped = parse(Path(model_py).read_text(encoding="utf-8"))
        if not ops or skipped:
            return None
        import cadquery as cq

        v0 = cq.importers.importStep(str(step)).val().Volume()
        v1 = replay(ops).val().Volume()
        if v0 <= 0 or abs(v1 - v0) / v0 > 0.001:
            return None
        dst.write_text(vba(ops, skipped, Path(model_py).name), encoding="utf-8-sig")
        return {k: sum(o["op"] == k for o in ops) for k in ("boss", "revolve", "cut", "hole")}
    except Exception:  # noqa: BLE001 — макрос необязателен, ошибка не ломает выдачу
        return None


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ops, skipped = parse(src.read_text(encoding="utf-8"))
    if not ops:
        raise SystemExit("в программе модели нет операций, которые можно перенести")
    dst.write_text(vba(ops, skipped, src.name), encoding="utf-8-sig")
    kinds = {k: sum(o["op"] == k for o in ops) for k in ("boss", "revolve", "cut", "hole")}
    print("операции:", kinds, "не перенесено:", len(skipped))
    if len(sys.argv) > 3:
        import cadquery as cq

        ref = cq.importers.importStep(sys.argv[3]).val()
        rep = replay(ops)
        v0, v1 = ref.Volume(), rep.val().Volume()
        print(f"объём: модель {v0:.1f}, повтор операций {v1:.1f} мм³, расхождение {abs(v1 - v0) / v0 * 100:.3f}%")


if __name__ == "__main__":
    main()
