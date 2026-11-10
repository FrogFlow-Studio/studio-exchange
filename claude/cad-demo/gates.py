"""Обязательные проверки модели (сверх процента совпадения со сканом).

Процент точек в допуске почти не замечает маленькие, но важные ошибки: заделанное
отверстие Ø2 или лишний тонкий выступ дают 99%+ (контрпримеры Севера, 09.11). Поэтому
модель принимается, только если прошла все проверки ниже; процент — дополнительная цифра.
"""
from __future__ import annotations

import numpy as np
import trimesh


def _holes_at(model: trimesh.Trimesh, z: float) -> list[tuple[np.ndarray, float]]:
    sec = model.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return []
    planar, T = sec.to_2D(to_2D=np.eye(4))
    out = []
    for p in planar.polygons_full:
        for ring in p.interiors:
            pts = np.asarray(ring.coords)[:-1]
            if len(pts) < 6:
                continue
            c = pts.mean(axis=0)
            r = float(np.linalg.norm(pts - c, axis=1).mean())
            out.append((c, 2 * r))
    return out


def check(run: dict, model: trimesh.Trimesh | None, sheet: dict, scan: trimesh.Trimesh,
          restored_extent: float = 0.0) -> list[str]:
    """Список нарушений (пустой — всё в порядке)."""
    bad = []
    if not run.get("ok"):
        return ["код не выполнился"]
    if not run.get("valid"):
        bad.append("тело невалидно")
    if run.get("solids", 1) != 1:
        bad.append(f"тел {run.get('solids')}, а деталь одна")
    if model is None:
        return bad
    # 1) каждое круглое отверстие, найденное в скане, есть в модели (центр ±0.5, Ø ±0.4)
    # отверстие мельче трёх шумов скана может оказаться просто шумом: его не требуем
    min_d = 3 * float(sheet.get("diagnostics", {}).get("noise_mm") or 0)
    for sec in sheet.get("sections", []):
        z = sec["z_mm"]
        have = None
        for reg in sec["regions"]:
            for h in reg["inner"]:
                if h.get("shape") != "окружность" or h["diameter"] < min_d:
                    continue
                if have is None:
                    have = _holes_at(model, z)
                c, d = np.asarray(h["center_xy"]), h["diameter"]
                if not any(np.linalg.norm(hc - c) < 0.5 and abs(hd - d) < 0.4 + 0.02 * d for hc, hd in have):
                    bad.append(f"в модели нет отверстия Ø{d} в {list(h['center_xy'])} на высоте {z} (в скане оно есть)")
    # 2) модель не выходит за габарит скана больше, чем можно объяснить достройкой скола
    lo_s, hi_s = scan.bounds
    lo_m, hi_m = model.bounds
    ext = hi_s - lo_s
    allow = np.maximum(2.0, 0.2 * ext) + restored_extent
    over = np.maximum(lo_s - lo_m, hi_m - hi_s)
    for ax, o, al in zip("XYZ", over, allow):
        if o > al:
            bad.append(f"модель выходит за габарит скана по {ax} на {o:.1f} мм")
    return sorted(set(bad))
