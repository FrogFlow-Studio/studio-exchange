"""Поиск видимых дефектов модели относительно скана: выступы, вырезы, смятые места, потерянные элементы.

Взято из задачи 35 Севера (chatgpt-1/cad-intelligence/cad-fillet-defects-2026-11-13)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh


def _load(path):
    if Path(path).suffix.lower() in (".step", ".stp"):
        import cadquery as cq
        from nurbs import body_mesh

        mesh = body_mesh(cq.importers.importStep(str(path)).val(), .05)
        mesh.merge_vertices()
    else:
        mesh = trimesh.load(path, force="mesh")
    if not len(mesh.faces):
        raise ValueError(f"empty mesh: {path}")
    if not mesh.is_winding_consistent or (mesh.is_watertight and mesh.volume < 0):
        mesh = mesh.copy()
        trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def _distance(source, target):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints, reference
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator

    poly = vtkPolyData()
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(target.vertices), deep=True))
    poly.SetPoints(points)
    faces = vtkCellArray()
    faces.SetData(3, numpy_to_vtkIdTypeArray(np.ascontiguousarray(target.faces, dtype=np.int64).ravel(), deep=True))
    poly.SetPolys(faces)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    if poly.GetNumberOfCells() != len(target.faces):
        raise RuntimeError("VTK triangle connectivity mismatch")
    closest = np.empty_like(source.triangles_center)
    face_ids = np.empty(len(closest), dtype=np.int64)
    out = [0., 0., 0.]
    cell, sub, squared = reference(0), reference(0), reference(0.)
    for i, point in enumerate(source.triangles_center):
        locator.FindClosestPoint(point, out, cell, sub, squared)
        closest[i] = out
        face_ids[i] = int(cell)
    delta = source.triangles_center - closest
    distance = np.linalg.norm(delta, axis=1)
    normal = target.face_normals[face_ids]
    signed = np.einsum("ij,ij->i", delta, normal)
    return distance, signed, face_ids


def _clusters(mesh, mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return
    lookup = np.full(len(mask), -1, dtype=int)
    lookup[ids] = np.arange(len(ids))
    adjacency = mesh.face_adjacency
    pairs = lookup[adjacency[np.all(mask[adjacency], axis=1)]]
    links = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(ids), len(ids))).tocsr()
    _, labels = connected_components(links, directed=False)
    order = np.argsort(labels, kind="stable")
    for part in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
        yield ids[part]


def _folds(model, scan, nearest, distance, tol):
    a, b = model.face_adjacency.T
    model_dot = np.einsum("ij,ij->i", model.face_normals[a], model.face_normals[b])
    scan_dot = np.einsum("ij,ij->i", scan.face_normals[nearest[a]], scan.face_normals[nearest[b]])
    unpaired = (model_dot < np.cos(np.radians(40))) & (scan_dot > np.cos(np.radians(22)))
    unpaired &= np.maximum(distance[a], distance[b]) > .28 * tol
    mask = np.zeros(len(model.faces), dtype=bool)
    mask[np.r_[a[unpaired], b[unpaired]]] = True
    return mask


def analyze(model, scan, tol=.3):
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("positive finite tolerance required")
    step = float(np.median(scan.edges_unique_length))
    # Bound by tolerance scale as dense CAD tessellations have very small edge lengths.
    min_area = max(2., 2 * step**2, 8 * tol**2)
    forward, side, nearest = _distance(model, scan)
    reverse, reverse_side, _ = _distance(scan, model)
    folds = _folds(model, scan, nearest, forward, tol)
    kinds = (
        ("смятая поверхность", model, folds, forward, "model", "Резкая смена нормалей без соответствующего сгиба на скане"),
        ("выступ над сканом", model, (forward > 2*tol) & (side > 0), forward, "model", "Модель выступает за поверхность скана"),
        ("вырез в материале", model, (forward > 2*tol) & (side < 0), forward, "model", "Поверхность модели углублена в материал скана"),
        ("потерянный элемент", scan, (reverse > 2*tol) & (reverse_side > 0), reverse, "scan", "На модели отсутствует участок поверхности скана"),
    )
    results = []
    for kind, mesh, mask, distances, source, description in kinds:
        for ids in _clusters(mesh, mask):
            weights = mesh.area_faces[ids]
            area = float(weights.sum())
            if area < min_area:
                continue
            center = np.average(mesh.triangles_center[ids], weights=weights, axis=0)
            results.append({"type": kind, "center": center.tolist(), "area_mm2": area,
                            "max_deviation_mm": float(distances[ids].max()), "description": description,
                            "source": source, "_faces": ids})
    # Prefer a cut where the model has a connected patch of actual inner
    # wall. A flat truncation under a missing boss is reported on the scan.
    cuts = [row for row in results if row["type"] == "вырез в материале"]
    for row in cuts:
        normals = model.face_normals[row["_faces"]]
        row["_wall"] = bool(np.linalg.norm(np.average(normals, axis=0, weights=model.area_faces[row["_faces"]])) < .88)
    kept = []
    for row in results:
        center = np.array(row["center"])
        if row["type"] == "потерянный элемент" and any(
                other["_wall"] and np.linalg.norm(center - other["center"]) <=
                np.hypot(other["max_deviation_mm"], np.sqrt(other["area_mm2"] / np.pi))
                for other in cuts):
            continue
        if row["type"] == "вырез в материале" and not row["_wall"] and any(
                other["type"] == "потерянный элемент" and
                np.linalg.norm(center - other["center"]) <=
                np.hypot(other["max_deviation_mm"], np.sqrt(other["area_mm2"] / np.pi))
                for other in results):
            continue
        kept.append(row)
    return sorted(kept, key=lambda r: -r["area_mm2"])


def _public(rows):
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def find(model_step_or_stl, scan_stl, tol=.3):
    """Report defects in mm for an already aligned model and scan."""
    return _public(analyze(_load(model_step_or_stl), _load(scan_stl), tol))


def render(model, scan, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = {"выступ над сканом": "#ff694f", "вырез в материале": "#be70ff",
              "смятая поверхность": "#f2bd3e", "потерянный элемент": "#2dbbc7"}
    view = np.array([[.866, -.5, 0.], [.25, .433, .866], [.433, .75, -.5]])
    for ax, rotation in zip(axes, (view, view * [-1, -1, 1])):
        geometry = model.triangles @ rotation.T
        centers = geometry.mean(axis=1)
        order = np.argsort(-centers[:, 2])
        shade = .52 + .38 * np.abs(model.face_normals @ rotation[2])
        facecolors = np.clip(shade[:, None] * np.array([.78, .85, .95]), 0, 1)
        ax.add_collection(PolyCollection(geometry[order, :, :2], facecolors=facecolors[order],
                                         edgecolors="none"))
        for i, row in enumerate(rows, 1):
            source = model if row["source"] == "model" else scan
            part = source.triangles[row["_faces"]] @ rotation.T
            ax.add_collection(PolyCollection(part[:, :, :2], facecolors=colors[row["type"]],
                                             edgecolors="none", alpha=.82))
            x, y = (np.asarray(row["center"]) @ rotation.T)[:2]
            ax.annotate(str(i), (x, y), fontsize=9, weight="bold", color="white", ha="center", va="center",
                        bbox={"boxstyle": "circle,pad=.24", "fc": "#202b3c", "ec": "white"})
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
    if rows:
        from matplotlib.patches import Patch

        fig.legend(handles=[Patch(color=color, label=kind) for kind, color in colors.items()],
                   loc="outside lower center", ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("scan")
    parser.add_argument("--tol", type=float, default=.3)
    parser.add_argument("--image")
    args = parser.parse_args()
    model, scan = _load(args.model), _load(args.scan)
    rows = analyze(model, scan, args.tol)
    if args.image:
        render(model, scan, rows, args.image)
    print(json.dumps(_public(rows), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
