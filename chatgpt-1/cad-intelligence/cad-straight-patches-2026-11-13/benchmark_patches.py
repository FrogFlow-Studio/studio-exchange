"""Sew smooth replacements to closed, independently built planar neighborhoods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid,
                                BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS

from patch_faces import SEW_TOL, _edges, _loops, patch, topology


def fixture(kind, noise, seed=43, *, boundary_noise=False):
    x, y = np.meshgrid(np.linspace(-6, 6, 11), np.linspace(-5, 5, 11), indexing="ij")
    if kind == "fillet":
        angle = (y + 5) * np.pi / 20
        y, z = 5 * np.sin(angle), 5 * np.cos(angle)
    elif kind == "saddle":
        z = .055 * (x**2 - y**2)
    else:
        z = 2.1 * (1 - (x / 6)**2) * (1 - (y / 5)**2)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    triangles = []
    for i in range(10):
        for j in range(10):
            if kind == "hole" and 3 <= i < 7 and 3 <= j < 7:
                continue
            a = 11 * i + j
            b, c, d = a + 11, a + 12, a + 1
            triangles.extend(((a, b, c), (a, c, d)) if (i + j) % 2 else ((a, b, d), (b, c, d)))
    triangles = np.array(triangles)
    loops = _loops(topology(triangles))
    ids = np.unique(triangles)
    if not boundary_noise:
        ids = np.setdiff1d(ids, np.concatenate(loops))
    points[ids, 2] += np.random.default_rng(seed).normal(0, noise, len(ids))
    return points, triangles, loops


def _wire(points):
    wire = BRepBuilderAPI_MakeWire()
    for edge in _edges(points):
        wire.Add(edge)
    return wire.Wire()


def sew_neighbors(face, vertices, loops):
    sew = BRepBuilderAPI_Sewing(SEW_TOL)
    sew.Add(face)
    loops = sorted(loops, key=len, reverse=True)
    base_wires = []
    for loop in loops:
        top = vertices[loop]
        base = top.copy()
        base[:, 2] = -5
        base_wires.append(_wire(base[::-1]))
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            side = _wire(np.array([top[j], top[i], base[i], base[j]]))
            sew.Add(BRepBuilderAPI_MakeFace(side, True).Face())
    bottom = BRepBuilderAPI_MakeFace(base_wires[0], True)
    for wire in base_wires[1:]:
        bottom.Add(wire)
    sew.Add(bottom.Face())
    sew.Perform()
    shape = sew.SewedShape()
    report = {"free_edges": sew.NbFreeEdges(), "multiple_edges": sew.NbMultipleEdges(), "valid_solid": False}
    if report["free_edges"] or report["multiple_edges"] or shape.ShapeType() != TopAbs_SHELL:
        return report
    solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shape)).Solid()
    report["valid_solid"] = bool(BRepLib.OrientClosedSolid_s(solid) and BRepCheck_Analyzer(solid).IsValid())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--boundary-noise", action="store_true")
    args = parser.parse_args()
    rows = []
    for kind in ("fillet", "saddle", "dome", "hole"):
        for noise in (0., .1):
            vertices, faces, loops = fixture(kind, noise, args.seed, boundary_noise=args.boundary_noise)
            start = perf_counter()
            face, audit = patch(vertices, faces, np.arange(len(faces)), .3)
            row = {"kind": kind, "noise_mm": noise, "triangles": len(faces), "seconds": perf_counter() - start, **audit}
            if face is not None:
                row.update(sew_neighbors(face, vertices, loops))
            rows.append(row)
            print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if all(r.get("valid_solid") and r["seconds"] <= 2 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
