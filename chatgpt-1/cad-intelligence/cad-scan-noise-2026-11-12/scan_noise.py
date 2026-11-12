"""Estimate high-frequency scan noise in millimetres on an aligned triangle mesh."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Noise:
    noise_mm: float | None
    method: str
    vertex_sigma_mm: float | None
    patches_used: int
    support_area_share: float
    support_curvature_gain: float | None
    support_patches: int
    support_flat_share: float | None


def _mad(values):
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _design(x, y):
    return np.stack((np.ones_like(x), x, y, x*x, x*y, y*y), axis=-1)


def _support(mesh, rng):
    centers = mesh.triangles_center
    mask = (mesh.face_normals[:, 2] < -.97) & (np.abs(centers[:, 2]) < 2.)
    areas = mesh.area_faces
    total = float(areas.sum())
    selected = np.flatnonzero(mask)
    share = float(areas[mask].sum() / total) if total > 0 else 0.
    if len(selected) < 30 or share < .025:
        return None, share, None

    points = centers[rng.choice(selected, min(12000, len(selected)), replace=False)]
    z = points[:, 2]
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, spread, frame = np.linalg.svd(xy, full_matrices=False)
    if spread[1] < .08 * spread[0] or spread[1] <= 1e-10:
        return None, share, None
    x, y = (xy @ frame.T / (spread / np.sqrt(len(xy)))).T
    design = _design(x, y)
    keep = np.ones(len(z), dtype=bool)
    for _ in range(2):
        model = np.linalg.lstsq(design[keep], z[keep], rcond=None)[0]
        residual = z - design @ model
        keep = np.abs(residual - np.median(residual)) <= max(1e-7, 3.5 * _mad(residual))
    if keep.sum() < .75 * len(z):
        return None, share, None
    flat = design[:, :3]
    plane = np.linalg.lstsq(flat[keep], z[keep], rcond=None)[0]
    flat_noise = _mad((z - flat @ plane)[keep])
    quad_noise = _mad((z - design @ model)[keep])
    gain = max(0., 1 - quad_noise / max(flat_noise, 1e-7)) if flat_noise > 1e-7 else 0.
    if gain > .2:
        return None, share, gain
    return _mad(centers[mask, 2]), share, gain


def _patch_scales(patches):
    # Normal coordinates remove rigid pose. Normalize tangential coordinates for conditioning.
    centered = patches - patches.mean(axis=1, keepdims=True)
    cov = centered.transpose(0, 2, 1) @ centered
    values, frames = np.linalg.eigh(cov)
    usable = (values[:, 1] > .12 * values[:, 2]) & (values[:, 0] < .25 * values[:, 1])
    usable &= values[:, 1] > 1e-20
    xyz = centered[usable] @ frames[usable]
    if not len(xyz):
        return np.empty((0, 3))
    near_support = (np.abs(frames[usable, 2, 0]) > .97) & (np.abs(patches[usable, :, 2].mean(axis=1)) < 2.)
    tangential = xyz[:, :, 1:]
    tangential /= np.sqrt(np.mean(tangential**2, axis=1, keepdims=True))
    x, y = tangential.transpose(2, 0, 1)
    design = _design(x, y)
    q, r = np.linalg.qr(design, mode="reduced")
    diag = np.abs(np.diagonal(r, axis1=1, axis2=2))
    well_conditioned = diag.min(axis=1) > .04 * diag.max(axis=1)
    q, z = q[well_conditioned], xyz[well_conditioned, :, 0:1]
    projection = q.transpose(0, 2, 1) @ z
    residual = z - q @ projection
    residual_dof = design.shape[1] - design.shape[2]
    sigma = np.sqrt(np.sum(residual**2, axis=(1, 2)) / residual_dof)
    curvature = np.linalg.norm(projection[:, 3:, 0], axis=1) / np.sqrt(design.shape[1])
    return np.column_stack((sigma, curvature, near_support[well_conditioned]))


def _local(mesh, rng, samples):
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 32 or not np.isfinite(vertices).all():
        return None, 0, 0, None
    # Restrict neighbours to the anchor's connected sheet within the query ball.
    faces = np.asarray(mesh.faces)
    a, b, c = faces.T
    rows, cols = np.r_[a, b, c], np.r_[b, c, a]
    graph = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                       shape=(len(vertices), len(vertices)))
    graph = graph.maximum(graph.T)
    tree = cKDTree(vertices)
    anchors = rng.choice(len(vertices), min(samples, len(vertices)), replace=False)
    _, nearest = tree.query(vertices[anchors], k=min(160, len(vertices)))
    patches = []
    for anchor, neighbors in zip(anchors, nearest):
        anchor_index = np.flatnonzero(neighbors == anchor)
        if not len(anchor_index):
            continue
        _, labels = connected_components(graph[neighbors][:, neighbors], directed=False)
        connected = neighbors[labels == labels[anchor_index[0]]]
        if len(connected) >= 32:
            patches.append(vertices[connected[:32]])
    scores = _patch_scales(np.asarray(patches)) if patches else np.empty((0, 3))
    if len(scores) < max(12, len(anchors) // 8):
        return None, len(scores), 0, None
    # Lower central quantile reduces contamination from edges and sharp features.
    sigma = float(np.quantile(scores[:, 0], .4))
    supporting = scores[scores[:, 2] != 0]
    flat = supporting[:, 1] <= np.maximum(.008, .7 * np.minimum(supporting[:, 0], 1.5 * sigma))
    flat_share = float(flat.mean()) if len(flat) else None
    return sigma, len(scores), len(supporting), flat_share


def estimate(mesh, *, samples=2048) -> Noise:
    """Return noise, estimator used, and supporting diagnostics for an aligned mesh.

    Legacy scale is kept for a verified Z≈0 support. The local scale represents
    RMS vertex displacement along a fitted normal, after removing local curvature.
    """
    if samples < 32:
        raise ValueError("samples must be at least 32")
    rng = np.random.default_rng(181112)
    support, share, gain = _support(mesh, rng)
    sigma, count, near_count, flat_share = _local(mesh, np.random.default_rng(181113), samples)
    local_support = near_count >= 12 and flat_share >= .5
    # Face centroids average vertex jitter. A far larger spread at the apparent
    # support means that the selected faces span curvature or several levels.
    diagnostics = (sigma, count, share, gain, near_count, flat_share)
    if support is not None and (sigma is None or (local_support and support <= max(.01, 1.5 * sigma))):
        return Noise(support, "support_plane", *diagnostics)
    return Noise(sigma, "local_quadratic" if sigma is not None else "insufficient", *diagnostics)


def noise_estimate(mesh) -> float | None:
    """Return millimetres, or None when geometric evidence is insufficient."""
    return estimate(mesh).noise_mm
