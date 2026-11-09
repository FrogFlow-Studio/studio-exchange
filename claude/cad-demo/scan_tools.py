"""Инструменты, которыми ИИ сам опрашивает сетку скана (режим --tools).

Вместо готовых картинок ИИ получает доступ к самой сетке: делает сечения где хочет,
пускает лучи, смотрит точки и нормали в любой области, вписывает окружности и
плоскости. Всё в выровненной системе координат скана (мм), той же, что в листе замеров.
Ответ каждого инструмента — короткий JSON (сетку целиком в запрос не положить:
десятки тысяч треугольников не поместятся и не будут прочитаны).
"""
from __future__ import annotations

import builtins
import json

import numpy as np
import trimesh

from scan_analysis import describe_ring, fit_circle

MAX_CHARS = 6000

TOOLS = [
    {"name": "mesh_overview",
     "description": "Общие сведения о сетке скана: число вершин и треугольников, габарит, площадь, "
                    "замкнута ли сетка, открытые края (дыры в скане), крупные плоские участки "
                    "(нормаль, положение, площадь).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "section",
     "description": "Сечение сетки плоскостью, перпендикулярной оси X, Y или Z, на заданной координате. "
                    "Возвращает замкнутые контуры: внешний и внутренние (отверстия) с вписанной "
                    "окружностью, если контур круглый, иначе упрощённые вершины. Для axis=z координаты "
                    "контура (x, y), для axis=x — (y, z), для axis=y — (x, z).",
     "input_schema": {"type": "object", "properties": {
         "axis": {"type": "string", "enum": ["x", "y", "z"]},
         "value": {"type": "number", "description": "координата плоскости, мм"}},
         "required": ["axis", "value"]}},
    {"name": "ray",
     "description": "Луч из точки в заданном направлении: все пересечения с поверхностью скана "
                    "(расстояние, точка, нормаль). Удобно мерить толщину стенки, глубину отверстия, "
                    "есть ли материал в месте.",
     "input_schema": {"type": "object", "properties": {
         "origin": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
         "direction": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}},
         "required": ["origin", "direction"]}},
    {"name": "points_in_box",
     "description": "Вершины сетки внутри прямоугольной области (до limit штук, равномерная выборка) "
                    "с нормалями и общим числом. Пусто — в этом месте скана нет поверхности "
                    "(скол или пропуск).",
     "input_schema": {"type": "object", "properties": {
         "min": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
         "max": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
         "limit": {"type": "integer", "default": 60}},
         "required": ["min", "max"]}},
    {"name": "fit_plane",
     "description": "Вписать плоскость в вершины внутри области: нормаль, смещение, СКО, число точек.",
     "input_schema": {"type": "object", "properties": {
         "min": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
         "max": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}},
         "required": ["min", "max"]}},
    {"name": "fit_circle",
     "description": "Вписать окружность в точки сечения (axis, value) рядом с заданным центром "
                    "(в пределах radius_max от него). Работает и на дуге сломанного края: даёт центр, "
                    "диаметр и долю окружности, покрытую сканом.",
     "input_schema": {"type": "object", "properties": {
         "axis": {"type": "string", "enum": ["x", "y", "z"]},
         "value": {"type": "number"},
         "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
         "radius_max": {"type": "number"}},
         "required": ["axis", "value", "center", "radius_max"]}},
    {"name": "nearest",
     "description": "Для каждой точки (до 50): ближайшая точка поверхности скана, расстояние и нормаль. "
                    "Удобно проверить, лежит ли задуманная грань на скане.",
     "input_schema": {"type": "object", "properties": {
         "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                               "minItems": 3, "maxItems": 3}, "maxItems": 50}},
         "required": ["points"]}},
]

_AX = {"x": 0, "y": 1, "z": 2}


def _r(v, n=2):
    return np.round(np.asarray(v, dtype=float), n).tolist()


def _to2d(axis: str, value: float) -> np.ndarray:
    rows = {"z": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -value]],
            "x": [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -value]],
            "y": [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, -value]]}[axis]
    return np.array(rows + [[0, 0, 0, 1]], dtype=float)


class ScanTools:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.calls: list[dict] = []

    # ---- инструменты ----
    def mesh_overview(self) -> dict:
        m = self.mesh
        edges = m.edges_sorted
        uniq, cnt = np.unique(edges, axis=0, return_counts=True)
        open_edges = uniq[cnt == 1]
        n = m.face_normals
        planes = []
        for ax, name in ((2, "Z"), (0, "X"), (1, "Y")):
            for sgn in (1, -1):
                sel = n[:, ax] * sgn > 0.98
                if not sel.any():
                    continue
                c = m.triangles_center[sel][:, ax]
                a = m.area_faces[sel]
                for lv in np.unique(np.round(c, 0)):
                    k = np.abs(c - lv) < 0.6
                    if a[k].sum() > 0.01 * m.area:
                        planes.append({"normal": f"{'+' if sgn > 0 else '-'}{name}",
                                       "at_mm": round(float(np.average(c[k], weights=a[k])), 2),
                                       "area_mm2": round(float(a[k].sum()), 1)})
        out = {"vertices": int(len(m.vertices)), "triangles": int(len(m.faces)),
               "bbox_mm": _r(m.bounds), "area_mm2": round(float(m.area), 1),
               "watertight": bool(m.is_watertight), "open_edges": int(len(open_edges)),
               "big_flat_areas": sorted(planes, key=lambda p: -p["area_mm2"])[:16]}
        if len(open_edges):
            pts = m.vertices[np.unique(open_edges)]
            out["open_edges_bbox_mm"] = _r([pts.min(axis=0), pts.max(axis=0)])
        return out

    def section(self, axis: str, value: float) -> dict:
        sec = self.mesh.section(plane_origin=np.eye(3)[_AX[axis]] * value, plane_normal=np.eye(3)[_AX[axis]])
        if sec is None:
            return {"contours": [], "note": "плоскость не пересекает скан"}
        planar, _ = sec.to_2D(to_2D=_to2d(axis, value))
        polys = [p.buffer(0) for p in planar.polygons_full if p.area > 0.5]
        out = []
        for p in sorted(polys, key=lambda q: -q.area)[:8]:
            if p.geom_type != "Polygon":
                continue
            d = {"outer": describe_ring(np.asarray(p.exterior.coords)[:-1]),
                 "holes": [describe_ring(np.asarray(r.coords)[:-1]) for r in p.interiors][:12]}
            out.append(d)
        open_len = float(sec.length) - sum(p.exterior.length + sum(r.length for r in p.interiors) for p in polys)
        res = {"axis": axis, "value": value, "coords": {"z": "(x,y)", "x": "(y,z)", "y": "(x,z)"}[axis],
               "contours": out}
        if open_len > 1:
            res["unclosed_length_mm"] = round(open_len, 1)
        return res

    def ray(self, origin, direction) -> dict:
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) or 1)
        o = np.asarray(origin, float)
        loc, _, tri = self.mesh.ray.intersects_location([o], [d], multiple_hits=True)
        hits = sorted(({"distance_mm": round(float(np.dot(p - o, d)), 2), "point": _r(p),
                        "normal": _r(self.mesh.face_normals[t], 3)} for p, t in zip(loc, tri)),
                      key=lambda h: h["distance_mm"])
        return {"hits": [h for h in hits if h["distance_mm"] >= -1e-6][:20]}

    def _in_box(self, lo, hi):
        v = self.mesh.vertices
        k = np.all((v >= np.asarray(lo)) & (v <= np.asarray(hi)), axis=1)
        return np.nonzero(k)[0]

    def points_in_box(self, min, max, limit: int = 60) -> dict:  # noqa: A002
        idx = self._in_box(min, max)
        limit = int(np.clip(limit, 1, 150))
        pick = idx[np.linspace(0, len(idx) - 1, builtins.min(limit, len(idx))).astype(int)] if len(idx) else idx
        vn = self.mesh.vertex_normals
        return {"count": int(len(idx)),
                "points": [{"p": _r(self.mesh.vertices[i]), "n": _r(vn[i], 2)} for i in pick]}

    def fit_plane(self, min, max) -> dict:  # noqa: A002
        idx = self._in_box(min, max)
        if len(idx) < 3:
            return {"count": int(len(idx)), "note": "мало точек"}
        p = self.mesh.vertices[idx]
        c = p.mean(axis=0)
        _, s, vt = np.linalg.svd(p - c, full_matrices=False)
        nrm = vt[2] * (1 if vt[2][np.argmax(np.abs(vt[2]))] > 0 else -1)
        dist = (p - c) @ nrm
        return {"count": int(len(idx)), "normal": _r(nrm, 4), "point": _r(c),
                "offset_mm": round(float(np.dot(nrm, c)), 3), "rms_mm": round(float(dist.std()), 3)}

    def fit_circle(self, axis: str, value: float, center, radius_max: float) -> dict:
        sec = self.mesh.section(plane_origin=np.eye(3)[_AX[axis]] * value, plane_normal=np.eye(3)[_AX[axis]])
        if sec is None:
            return {"note": "плоскость не пересекает скан"}
        planar, _ = sec.to_2D(to_2D=_to2d(axis, value))
        pts = np.vstack([d for d in planar.discrete]) if planar.discrete else np.zeros((0, 2))
        pts = pts[np.linalg.norm(pts - np.asarray(center), axis=1) <= radius_max]
        if len(pts) < 5:
            return {"count": int(len(pts)), "note": "мало точек рядом с центром"}
        c, r, _ = fit_circle(pts)
        ang = np.sort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        covered = 1 - float(gaps[gaps > 0.15].sum()) / (2 * np.pi)
        rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - r) ** 2)))
        return {"count": int(len(pts)), "center": _r(c), "diameter_mm": round(2 * r, 3),
                "rms_mm": round(rms, 3), "share_of_circle_covered": round(covered, 2)}

    def nearest(self, points) -> dict:
        p = np.asarray(points, float)[:50]
        if not len(p):
            return {"results": []}
        cp, dist, tri = trimesh.proximity.closest_point(self.mesh, p)
        return {"results": [{"query": _r(q), "closest": _r(c), "distance_mm": round(float(d), 3),
                             "normal": _r(self.mesh.face_normals[t], 3)} for q, c, d, t in zip(p, cp, dist, tri)]}

    # ---- вызов по имени ----
    def call(self, name: str, args: dict) -> str:
        try:
            fn = getattr(self, name) if name in {t["name"] for t in TOOLS} else None
            if fn is None:
                res = {"error": f"нет инструмента {name}"}
            else:
                res = fn(**(args or {}))
        except Exception as ex:  # noqa: BLE001 — ошибку отдаём ИИ, а не роняем прогон
            res = {"error": f"{type(ex).__name__}: {ex}"}
        s = json.dumps(res, ensure_ascii=False)
        if len(s) > MAX_CHARS:
            s = s[:MAX_CHARS] + ' …(обрезано, сузьте запрос)"'
        self.calls.append({"tool": name, "args": args, "chars": len(s)})
        return s
