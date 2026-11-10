"""Разметка скана кривой детали на грани для Point2STEP: пологие панели и скругления —
разные участки. Нормали сглаживаются в шаре R (шум скана), участок растёт, пока сглаженная
нормаль меняется плавно (≤ 8° между соседями) и класс «пологий/изогнутый» тот же.

python segment.py скан.stl папка [R мм = 5] [порог = 0.005]
  → папка/part/0.xyzc (x y z метка) — вход step_pipeline.py Point2STEP (--model_id part)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def segment(mesh, R: float = 5.0, thr: float = 0.005, n: int = 40000, min_share: float = 0.004):
    import networkx as nx
    import trimesh
    from scipy.spatial import cKDTree

    p, fi = trimesh.sample.sample_surface(mesh, n, seed=1)
    nrm = mesh.face_normals[fi]
    tree = cKDTree(p)
    sn = np.zeros_like(nrm)
    var = np.zeros(len(p))
    for i, ix in enumerate(tree.query_ball_point(p, R)):
        q = nrm[ix]
        q = q * np.sign(q @ nrm[i])[:, None]
        mu = q.mean(0)
        var[i] = 1 - np.linalg.norm(mu)
        sn[i] = mu / max(np.linalg.norm(mu), 1e-9)
    flat = var < thr
    pairs = tree.query_pairs(R / 2, output_type="ndarray")
    ok = (flat[pairs[:, 0]] == flat[pairs[:, 1]]) & (np.einsum("ij,ij->i", sn[pairs[:, 0]], sn[pairs[:, 1]])
                                                     > np.cos(np.radians(8)))
    g = nx.Graph()
    g.add_nodes_from(range(len(p)))
    g.add_edges_from(pairs[ok])
    lab = np.full(len(p), -1)
    k = 0
    for c in sorted(nx.connected_components(g), key=len, reverse=True):
        if len(c) >= min_share * len(p):
            lab[np.fromiter(c, int)] = k
            k += 1
    if k == 0:
        return p, np.zeros(len(p), int)
    got = lab >= 0
    if (~got).any():  # мелкие клочки — к ближайшему участку
        _, j = cKDTree(p[got]).query(p[~got])
        lab[~got] = lab[got][j]
    return p, lab


def main() -> None:
    import trimesh

    mesh = trimesh.load(sys.argv[1], force="mesh")
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
    p, lab = segment(mesh, R, thr)
    out = Path(sys.argv[2]) / "part"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "0.xyzc", np.column_stack([p, lab]))
    print(f"участков: {lab.max() + 1}, точек: {len(p)} → {out / '0.xyzc'}")


if __name__ == "__main__":
    main()
