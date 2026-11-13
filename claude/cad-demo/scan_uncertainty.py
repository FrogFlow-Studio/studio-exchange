"""Separate observed vertex jitter from a working geometry uncertainty in mm."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coarse_noise import estimate as estimate_noise


@dataclass(frozen=True)
class Uncertainty:
    measured_mm: float | None
    vertex_sigma_mm: float | None
    working_mm: float | None
    floor_mm: float | None
    edge_step_mm: float | None
    sampling_reserve_mm: float | None
    binary_stl_roundoff_mm: float | None
    chord_estimate_mm: float | None
    chord_edges: int
    noise_method: str
    limiting_component: str


def binary_stl_roundoff(mesh) -> float:
    """Scale of half an ULP per axis for this binary STL's coordinate range."""
    maxima = np.max(np.abs(mesh.vertices), axis=0).astype(np.float32)
    if not np.isfinite(maxima).all():
        raise ValueError("coordinates outside float32 range")
    return .5 * float(np.linalg.norm(np.spacing(maxima).astype(float)))


def _chord(mesh, step):
    adjacent = np.asarray(mesh.face_adjacency)
    if len(adjacent) < 24:
        return None, 0
    if len(adjacent) > 32768:
        sample = np.random.default_rng(220112).choice(len(adjacent), 32768, replace=False)
        adjacent = adjacent[sample]
    left, right = adjacent.T
    normal = np.asarray(mesh.face_normals)
    dot = np.einsum("ij,ij->i", normal[left], normal[right])
    angle = np.arccos(np.clip(dot, -1., 1.))
    centers = np.asarray(mesh.triangles_center)
    span = np.linalg.norm(centers[left] - centers[right], axis=1)
    size = np.sqrt(2 * np.asarray(mesh.area_faces))
    supported = ((angle <= np.deg2rad(25)) & (span >= .25 * step) & (span <= 3 * step)
                 & (np.minimum(size[left], size[right]) >= .25 * step))
    if supported.sum() < 24:
        return None, int(supported.sum())
    h = np.minimum(2 * step, .5 * (size[left] + size[right]))
    # A smooth normal section gives kappa ~= angle/span and sagitta ~= kappa*h**2/8.
    sag = angle[supported] * h[supported]**2 / (8 * span[supported])
    return float(min(np.quantile(sag, .75), .10 * step)), int(supported.sum())


def combine(mesh, noise, *, roundoff_mm=None) -> Uncertainty:
    """Keep the noise measurement and calculate a separate operational reserve."""
    if roundoff_mm is not None and (not np.isfinite(roundoff_mm) or roundoff_mm < 0):
        raise ValueError("roundoff_mm must be nonnegative and finite")
    edges = np.asarray(mesh.edges_unique_length)
    edges = edges[(edges > 0) & np.isfinite(edges)]
    if not len(edges) or not len(mesh.faces):
        return Uncertainty(noise.noise_mm, noise.vertex_sigma_mm, None, None, None, None,
                           roundoff_mm, None, 0, noise.method, "insufficient_mesh")
    step = float(np.median(edges))
    reserve = .01 * step
    # At the noise floor, normal changes can be used as a proxy for curvature.
    # With measurable jitter they mix curvature and noise and are not a floor.
    smooth = noise.vertex_sigma_mm is not None and noise.vertex_sigma_mm <= max(reserve, roundoff_mm or 0.)
    chord, support = _chord(mesh, step) if smooth else (None, 0)
    floors = {"sampling": reserve, "coordinate_roundoff": roundoff_mm or 0., "chord": chord or 0.}
    floor_source = max(floors, key=floors.get)
    floor = floors[floor_source]
    measured = noise.noise_mm
    working = max(floor, measured or 0.)
    source = "measured" if measured is not None and measured >= floor else floor_source
    return Uncertainty(measured, noise.vertex_sigma_mm, working, floor, step, reserve, roundoff_mm,
                       chord, support, noise.method, source)


def estimate(mesh, *, samples=2048, roundoff_mm=None) -> Uncertainty:
    return combine(mesh, estimate_noise(mesh, samples=samples), roundoff_mm=roundoff_mm)
