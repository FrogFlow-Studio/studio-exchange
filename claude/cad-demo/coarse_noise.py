"""Noise scale from local smooth sheets of a triangle scan, in millimetres."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import gammaincinv

from scan_noise import _mad, _patch_scales


PATCH_VERTICES = 12
QUANTILE = .30
# Least-squares residual norm has n-6 degrees of freedom for a six-term graph.
DOF = PATCH_VERTICES - 6
NORMAL_QUANTILE = np.sqrt(2 * gammaincinv(DOF / 2, QUANTILE) / DOF)


@dataclass(frozen=True)
class Estimate:
    noise_mm: float | None
    method: str
    vertex_sigma_mm: float | None
    patches_used: int
    low_curvature_patches: int
    support_faces: int
    support_area_share: float
    support_status: str
    support_sigma_mm: float | None
    support_detrended_mm: float | None


class SmoothPatches:
    def __init__(self, mesh):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.faces)
        self.normals = np.asarray(mesh.face_normals)
        self.centers = np.asarray(mesh.triangles_center)
        pairs = np.asarray(mesh.face_adjacency)
        left, right = pairs.T
        normal_dot = np.einsum("ij,ij->i", self.normals[left], self.normals[right])
        left, right = pairs[normal_dot >= .78].T
        self.graph = csr_matrix((np.ones(2 * len(left), dtype=bool),
                                 (np.r_[left, right], np.r_[right, left])),
                                shape=(len(self.faces), len(self.faces)))
        self.tree = cKDTree(self.centers)

    def score(self, anchors):
        _, nearest = self.tree.query(self.centers[anchors], k=min(56, len(self.faces)))
        patches = []
        for anchor, faces in zip(anchors, nearest):
            faces = faces[self.normals[faces] @ self.normals[anchor] > .60]
            location = np.flatnonzero(faces == anchor)
            if not len(location):
                continue
            _, labels = connected_components(self.graph[faces][:, faces], directed=False)
            component = faces[labels == labels[location[0]]]
            vertices = self.vertices[np.unique(self.faces[component])]
            if len(vertices) < PATCH_VERTICES:
                continue
            distance = np.sum((vertices - self.centers[anchor])**2, axis=1)
            patches.append(vertices[np.argsort(distance)[:PATCH_VERTICES]])
        return _patch_scales(np.asarray(patches)) if patches else np.empty((0, 3))


def _scale(scores):
    return float(np.quantile(scores[:, 0], QUANTILE) / NORMAL_QUANTILE)


def _flat(scores, scale):
    return scores[:, 1] <= max(.008, .8 * scale)


def _support(patches, selected, rng):
    if len(selected) < 30:
        return "too_few_legacy_faces", None, None
    ids = rng.choice(selected, min(384, len(selected)), replace=False)
    scores = patches.score(ids)
    if len(scores) < 12:
        return "too_few_smooth_patches", None, None
    scale = _scale(scores)
    flat = scores[_flat(scores, scale)]
    if len(flat) < max(12, len(scores) // 2):
        return "curvature_or_mixed_surfaces", None, None
    sigma = _scale(flat)

    points = patches.centers[ids]
    x = points[:, :2] - points[:, :2].mean(axis=0)
    spread = np.linalg.svd(x, compute_uv=False)
    if spread[1] < .08 * spread[0] or spread[1] <= 1e-10:
        return "narrow_support", sigma, None
    design = np.column_stack((np.ones(len(x)), x))
    z = points[:, 2]
    keep = np.ones(len(z), dtype=bool)
    for _ in range(2):
        plane = np.linalg.lstsq(design[keep], z[keep], rcond=None)[0]
        residual = z - design @ plane
        detrended = _mad(residual)
        keep = np.abs(residual - np.median(residual)) <= max(1e-8, 3.5 * detrended)
    if keep.sum() < .75 * len(z):
        return "mixed_support_levels", sigma, detrended
    legacy = _mad(patches.centers[selected, 2])
    if legacy > max(.015, 1.35 * sigma) or legacy > max(.015, 1.35 * detrended):
        return "height_spread_exceeds_local_noise", sigma, detrended
    return "verified", sigma, detrended


def estimate(mesh, *, samples=2048) -> Estimate:
    """Measure vertex jitter and retain the legacy value on a verified Z≈0 support."""
    if samples < 32:
        raise ValueError("samples must be at least 32")
    faces = len(mesh.faces)
    if faces < 12 or not np.isfinite(mesh.vertices).all():
        return Estimate(None, "insufficient", None, 0, 0, 0, 0., "insufficient_mesh", None, None)
    rng = np.random.default_rng(181113)
    patches = SmoothPatches(mesh)
    anchors = rng.choice(faces, min(samples, faces), replace=False)
    scores = patches.score(anchors)
    selected = np.flatnonzero((patches.normals[:, 2] < -.97) & (np.abs(patches.centers[:, 2]) < 2.))
    total = float(mesh.area_faces.sum())
    share = float(mesh.area_faces[selected].sum() / total) if total > 0 else 0.
    status, support_sigma, detrended = _support(patches, selected, rng)

    scale = None
    flat_count = 0
    if len(scores) >= max(24, len(anchors) // 16):
        scale = _scale(scores)
        flat = scores[_flat(scores, scale)]
        flat_count = len(flat)
        if flat_count >= max(12, len(scores) // 4):
            scale = _scale(flat)
    fields = (scale, len(scores), flat_count, len(selected), share, status, support_sigma, detrended)
    if status == "verified":
        return Estimate(_mad(patches.centers[selected, 2]), "support_plane", *fields)
    return Estimate(scale, "smooth_quadratic" if scale is not None else "insufficient", *fields)


def noise_estimate(mesh) -> float | None:
    return estimate(mesh).noise_mm
