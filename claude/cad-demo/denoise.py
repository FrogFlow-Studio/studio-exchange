"""Очистка шумного скана с сохранением рёбер.

Обычное сглаживание убирает шум вместе с рёбрами: излом скругляется, мелкая ступенька тает.
Здесь сначала выравниваются направления треугольников: сосед учитывается тем слабее, чем сильнее
его направление отличается (по ту сторону ребра он почти не влияет). Потом вершины подтягиваются
под новые направления. Ровные участки становятся гладкими, излом остаётся изломом.
Способ общий (двустороннее сглаживание нормалей), под конкретную деталь ничего не настраивается.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _pass(m: trimesh.Trimesh, guide: trimesh.Trimesh, normal_iters: int, vertex_iters: int,
          angle_deg: float | None) -> trimesh.Trimesh:
    F = m.faces
    adj = m.face_adjacency
    a = np.r_[adj[:, 0], adj[:, 1]]
    b = np.r_[adj[:, 1], adj[:, 0]]
    n = m.face_normals.copy()
    area = m.area_faces
    cen = m.triangles_center
    d2 = ((cen[a] - cen[b]) ** 2).sum(1)
    sc2 = 2 * float(np.mean(d2))                      # расстояние между соседями — в их среднем шаге
    gn = guide.face_normals
    dg2 = ((gn[a] - gn[b]) ** 2).sum(1)
    if angle_deg is None:
        # порог — по самой сетке: вдвое больше типичного расхождения соседей (это остаток шума),
        # но в разумных пределах, чтобы настоящие рёбра (десятки градусов) оставались рёбрами
        typical = np.degrees(2 * np.arcsin(np.sqrt(np.median(dg2)) / 2))
        angle_deg = float(np.clip(2.0 * typical, 30.0, 40.0))
    ss2 = 2 * (2 * np.sin(np.radians(angle_deg) / 2)) ** 2
    w = area[b] * np.exp(-d2 / sc2) * np.exp(-dg2 / ss2)
    for _ in range(normal_iters):
        acc = n * area[:, None] * 0.5
        for k in range(3):
            acc[:, k] += np.bincount(a, w * n[b, k], minlength=len(F))
        ln = np.linalg.norm(acc, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, acc / np.maximum(ln, 1e-12), n)
    V = m.vertices.copy()
    deg = np.bincount(F.ravel(), minlength=len(V)).astype(float)
    deg[deg == 0] = 1
    fi = np.repeat(np.arange(len(F)), 3)
    vi = F.ravel()
    for _ in range(vertex_iters):
        c = V[F].mean(1)
        step = n[fi] * ((c[fi] - V[vi]) * n[fi]).sum(1, keepdims=True)
        dv = np.zeros_like(V)
        for k in range(3):
            dv[:, k] = np.bincount(vi, step[:, k], minlength=len(V))
        V += dv / deg[:, None]
    return trimesh.Trimesh(V, F, process=False)


def bilateral(mesh: trimesh.Trimesh, normal_iters: int | None = None, vertex_iters: int = 20,
              angle_deg: float | None = None, guide_iters: int = 10, passes: int = 1) -> trimesh.Trimesh:
    """Возвращает копию сетки без шума. angle_deg — разница направлений соседних треугольников,
    с которой сосед считается «за ребром» и перестаёт влиять (None — подбирается по самой сетке).

    При сильном шуме направления соседних треугольников сами расходятся на десятки градусов, и
    ребро от шума по ним не отличить. Поэтому «за ребром ли сосед» решается по направлениям
    слегка сглаженной копии (guide_iters шагов обычного сглаживания): шум в ней уже мал, а грани
    по разные стороны ребра по-прежнему смотрят в разные стороны. Второй проход (образец —
    результат первого) на проверке ничего не дал, по умолчанию проход один."""
    if len(mesh.face_adjacency) == 0:
        return mesh.copy()
    if normal_iters is None:
        # Чем сильнее шум, тем шире надо усреднять; но широкое усреднение спрямляет плавные дуги
        # (скругление R4 при 100 шагах выходило с ошибкой 0,06 мм и заниженным радиусом). Поэтому
        # число шагов — по самому шуму: по типичному углу между соседними треугольниками.
        typical = float(np.degrees(np.median(mesh.face_adjacency_angles)))
        normal_iters = int(np.clip(typical ** 2 / 20, 8, 100))
    guide = mesh.copy()
    if guide_iters:
        trimesh.smoothing.filter_taubin(guide, iterations=guide_iters)
    out = mesh
    for _ in range(passes):
        out = _pass(mesh, guide, normal_iters, vertex_iters, angle_deg)
        guide = out
    return out
