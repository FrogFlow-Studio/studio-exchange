"""Выглаживание свободных зон выпрямленной сетки перед сборкой прямого тела.

Скан мелкий и шумный, поэтому в свободных зонах (не плоскость и не цилиндр) сетка идёт кочками, а границы
между областями — зубцами по рёбрам треугольников. Гладкая грань обязана пройти точно по такой границе и
идёт рябью, треугольники дают видимую «мятость». Здесь:

1. границы между областями выпрямляются вдоль себя (сглаживание Таубина по линии, без усадки); точка
   границы с плоскостью остаётся на этой плоскости, точки на цилиндрах и в углах трёх областей не двигаются;
2. внутри свободных зон нормали граней сглаживаются двусторонним фильтром (острые рёбра сохраняются,
   кочки уходят), и вершины подгоняются под новые нормали (Sun и др., 2007).

Ни одна вершина не уходит от выпрямленной сетки дальше limit. Плоские и цилиндрические грани не меняются.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp


def _boundary_lines(V, F, label, regions, limit, iters=30, lam=0.5, mu=-0.53):
    V0 = V.copy()
    V = V.copy()
    e2f = defaultdict(list)
    for fi, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e2f[(min(a, b), max(a, b))].append(fi)
    nb = defaultdict(set)
    plane_of = defaultdict(set)
    pinned = set()
    labs = defaultdict(set)
    for fi, t in enumerate(F):
        lab = int(label[fi])
        kind = regions[lab]["kind"] if lab >= 0 else "free"
        for v in t:
            labs[v].add(lab)
            if kind == "plane":
                plane_of[v].add(lab)
            elif kind != "free":
                pinned.add(v)
    for (a, b), fs in e2f.items():
        if len(fs) == 2 and label[fs[0]] != label[fs[1]]:
            nb[a].add(b)
            nb[b].add(a)
    verts = np.array([v for v in nb if len(nb[v]) == 2 and len(labs[v]) == 2 and v not in pinned
                      and len(plane_of[v]) <= 1], dtype=int)
    if not len(verts):
        return V, 0
    nbr = np.array([sorted(nb[v]) for v in verts])
    on_plane = [(i, regions[next(iter(plane_of[v]))]) for i, v in enumerate(verts) if plane_of[v]]
    idx = np.array([i for i, _ in on_plane], dtype=int)
    normals = np.array([np.asarray(r["normal"], float) for _, r in on_plane]).reshape(-1, 3)
    origins = np.array([np.asarray(r["origin"], float) for _, r in on_plane]).reshape(-1, 3)
    for _ in range(iters):
        for f in (lam, mu):
            V[verts] += f * (V[nbr].mean(axis=1) - V[verts])
        if len(idx):
            p = V[verts[idx]]
            V[verts[idx]] = p - np.sum((p - origins) * normals, axis=1)[:, None] * normals
        off = V[verts] - V0[verts]
        length = np.linalg.norm(off, axis=1)
        big = length > limit
        V[verts[big]] = V0[verts[big]] + off[big] * (limit / length[big])[:, None]
    return V, len(verts)


def _normal_filter(V, F, movable, V0, limit, sigma_n=0.35, n_iter=20, v_iter=10):
    V = V.copy()
    nf = len(F)
    edges = defaultdict(list)
    for fi, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(min(a, b), max(a, b))].append(fi)
    pairs = np.array([fs for fs in edges.values() if len(fs) == 2], dtype=int).reshape(-1, 2)
    A = sp.coo_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                      (nf, nf)).tocsr()
    R = (A + A @ A + sp.eye(nf)).tocoo()          # соседи через две грани
    I, J = R.row, R.col
    counts = np.bincount(F.ravel(), minlength=len(V))[:, None]
    for _ in range(n_iter):
        tri = V[F]
        c = tri.mean(axis=1)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1) / 2
        n = cross / (2 * area[:, None] + 1e-15)
        sig_c = max(np.linalg.norm(c[I] - c[J], axis=1).mean(), 1e-9)
        w = area[J] * np.exp(-np.sum((c[I] - c[J]) ** 2, axis=1) / (2 * sig_c ** 2)) \
            * np.exp(-np.sum((n[I] - n[J]) ** 2, axis=1) / (2 * sigma_n ** 2))
        nn = np.zeros_like(n)
        np.add.at(nn, I, w[:, None] * n[J])
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-15
        for _ in range(v_iter):
            tri = V[F]
            c = tri.mean(axis=1)
            d = np.zeros_like(V)
            for k in range(3):
                np.add.at(d, F[:, k], np.sum(nn * (c - tri[:, k]), axis=1)[:, None] * nn)
            V[movable] += (d / np.maximum(counts, 1))[movable]
            off = V - V0
            length = np.linalg.norm(off, axis=1)
            big = movable & (length > limit)
            V[big] = V0[big] + off[big] * (limit / length[big])[:, None]
    return V


def fair(V, F, label, regions, limit: float):
    """Вернуть (новые вершины, сводка). Топология и метки не меняются."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    label = np.asarray(label)
    fixed_face = np.array([lab >= 0 and regions[lab]["kind"] in ("plane", "cylinder")
                           for lab in label])
    if fixed_face.all():
        return V, {"fair_boundary_vertices": 0, "fair_free_vertices": 0}
    V1, nb = _boundary_lines(V, F, label, regions, limit)
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    movable = used.copy()
    movable[F[fixed_face].ravel()] = False
    V2 = _normal_filter(V1, F, movable, V, limit)
    moved = np.linalg.norm(V2 - V, axis=1)
    return V2, {"fair_boundary_vertices": int(nb), "fair_free_vertices": int(movable.sum()),
                "fair_moved_max_mm": float(moved.max()) if len(moved) else 0.0}
