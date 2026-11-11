"""Independent support for analytic surface proposals on a scanned mesh."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import trimesh


def _normal_field(mesh, rounds):
    """Area-weighted one-ring normals reduce triangulation noise without moving vertices."""
    adj = mesh.face_adjacency
    a, b = np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]]
    fn = mesh.face_normals
    area = np.sqrt(mesh.area_faces)
    cosine = np.einsum("ij,ij->i", fn[a], fn[b])
    weight = area[b] * np.exp(-3 * (1 - cosine))
    field = fn.copy()
    for _ in range(rounds):
        normal = field * area[:, None]
        for j in range(3):
            normal[:, j] += np.bincount(a, weight * field[b, j], minlength=len(fn))
        field = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return field


class Evidence:
    def __init__(self, mesh, scan, tol):
        self.mesh = mesh
        self.tol = tol
        self.area = mesh.area_faces
        self.fc = mesh.triangles_center
        self.adj = mesh.face_adjacency
        self.length = np.linalg.norm(
            np.diff(mesh.vertices[mesh.face_adjacency_edges], axis=1)[:, 0], axis=1
        )
        self.guide = mesh.copy()
        trimesh.smoothing.filter_taubin(self.guide, iterations=12)
        self.normals = _normal_field(self.guide, 3)
        self.pitch = float(np.median(self.length)) if len(self.length) else tol
        delta = mesh.vertices - self.guide.vertices
        movement = np.einsum("ij,ij->i", delta, self.guide.vertex_normals)
        self.noise = float(1.4826 * np.median(np.abs(movement)))
        self.raw_points = None
        self.raw_face = None
        if scan is not None and len(scan.vertices) and len(self.fc):
            self._sample_scan(scan)
        a, b = self.adj.T
        cos = np.einsum("ij,ij->i", self.normals[a], self.normals[b])
        rough = float(np.median(np.arccos(np.clip(cos, -1, 1)))) if len(cos) else 0.
        sharp = cos < np.cos(np.clip(4 * rough, np.radians(28), np.radians(50)))
        distances = np.maximum(np.linalg.norm(self.fc[a] - self.fc[b], axis=1), tol / 10)
        graph = csr_matrix((np.r_[distances, distances], (np.r_[a, b], np.r_[b, a])),
                           shape=(len(self.area), len(self.area)))
        self.graph = graph
        if np.any(sharp):
            self.seam_distance = dijkstra(graph, indices=np.unique(self.adj[sharp]), min_only=True)
        else:
            self.seam_distance = np.full(len(self.area), np.inf)
        # A single noisy hinge is weaker evidence when noise exceeds the fit
        # tolerance. Demand a larger supported fraction in that case.
        self.boundary_minimum = float(np.clip(1 / 3 + 0.4 * max(0., self.noise / tol - 0.75),
                                              1 / 3, 0.75))
        self.seam_band = max(3 * tol, 0.9 * self.pitch, min(3 * self.noise, 3 * self.pitch))

    def _sample_scan(self, scan):
        points = np.asarray(scan.vertices)
        step = max(1, (len(points) + 79999) // 80000)
        points = points[::step]
        tree = cKDTree(self.fc)
        k = min(8, len(self.fc))
        ids, offsets = [], []
        for start in range(0, len(points), 4000):
            block = points[start:start + 4000]
            _, candidates = tree.query(block, k=np.arange(1, k + 1))
            triangles = self.mesh.triangles[candidates]
            closest = trimesh.triangles.closest_point(triangles.reshape(-1, 3, 3),
                                                     np.repeat(block, k, axis=0))
            distance = np.linalg.norm(closest.reshape(-1, k, 3) - block[:, None, :], axis=2)
            best = distance.argmin(axis=1)
            ids.extend(candidates[np.arange(len(block)), best])
            offsets.extend(distance[np.arange(len(block)), best])
        self.raw_points = points
        self.raw_face = np.asarray(ids)
        self.noise = max(self.noise, 1.4826 * float(np.median(offsets)))

    def boundary_share(self, members):
        inside = np.zeros(len(self.area), dtype=bool)
        inside[members] = True
        a, b = self.adj.T
        boundary = inside[a] != inside[b]
        if not np.any(boundary):
            return float(self.mesh.is_watertight)
        u, v = a[boundary], b[boundary]
        lengths = self.length[boundary]
        both = np.maximum(self.seam_distance[u], self.seam_distance[v]) < self.seam_band
        return float(lengths[both].sum() / max(lengths.sum(), 1e-12))

    def neighboring_support(self, members, witnesses, *, fraction=0.15):
        """Require another independently supported surface along part of the perimeter."""
        if not len(witnesses):
            return False
        inside = np.zeros(len(self.area), dtype=bool)
        inside[members] = True
        a, b = self.adj.T
        border = inside[a] != inside[b]
        if not np.any(border):
            return False
        a, b = a[border], b[border]
        outside = np.where(inside[a], b, a)
        sources = witnesses[~inside[witnesses]]
        if not len(sources):
            return False
        reach = 2 * self.seam_band
        dist = dijkstra(self.graph, indices=sources, min_only=True, limit=reach)
        lengths = self.length[border]
        return float(lengths[dist[outside] <= reach].sum()) >= fraction * lengths.sum()

    def conical_rim(self, faces, model, tolerance):
        """A measured rim is a section curve on both the plane and the cone."""
        inside = np.zeros(len(self.area), dtype=bool)
        inside[faces] = True
        a, b = self.adj.T
        edge = inside[a] != inside[b]
        if not np.any(edge):
            return False
        segments = self.mesh.vertices[self.mesh.face_adjacency_edges[edge]]
        apex, axis, slope = model
        delta = segments - apex
        height = delta @ axis
        radius = np.linalg.norm(delta - height[..., None] * axis, axis=-1)
        distance = (radius - height * slope) / np.hypot(1, slope)
        match = (np.abs(distance) < tolerance).all(axis=1)
        lengths = self.length[edge]
        return bool(lengths[match].sum() >= max(0.15 * lengths.sum(), 6 * tolerance))

    def opposing_plane(self, region, other):
        n, v = region["normal"], other["normal"]
        if n @ v > -np.cos(np.radians(8)):
            return False
        gap = abs((region["origin"] - other["origin"]) @ n)
        if gap < max(4 * self.tol, 3 * self.noise):
            return False
        direction = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
        direction /= np.linalg.norm(direction)
        frame = np.stack([direction, np.cross(n, direction)], axis=1)
        a = self.fc[region["faces"]] @ frame
        b = self.fc[other["faces"]] @ frame
        width = np.minimum(a.max(0), b.max(0)) - np.maximum(a.min(0), b.min(0))
        areas = np.prod(np.ptp(a, axis=0)), np.prod(np.ptp(b, axis=0))
        return bool(np.min(width) > 0 and gap < np.max(width)
                    and np.prod(width) >= 0.5 * min(areas) and min(areas) >= 0.15 * max(areas))

    def scan_points(self, faces):
        if self.raw_points is None:
            return None
        return self.raw_points[np.isin(self.raw_face, faces)]

    def stable_plane(self, faces, normal, origin):
        points = self.scan_points(faces)
        if points is None:
            points = self.fc[faces]
        if len(points) < 20:
            return False
        delta = points - origin
        residual = delta @ normal
        scatter = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        u /= np.linalg.norm(u)
        for axis in (u, np.cross(normal, u)):
            coordinate = delta @ axis
            groups = np.array_split(np.argsort(coordinate), 5)[1:-1]
            levels = [np.median(residual[part]) for part in groups]
            limit = max(self.tol / 4, 3 * scatter / np.sqrt(min(map(len, groups))))
            if np.ptp(levels) > limit:
                return False
        return scatter < max(self.tol / 2, 1.4 * self.noise)

    def fit_plane(self, faces, normal, origin):
        # Localize the fit to the interior of the proposal before using raw points.
        good = faces[np.abs((self.fc[faces] - origin) @ normal) < max(self.tol, 2 * self.noise)]
        if len(good) < 3:
            return normal, origin
        points = self.fc[good]
        weights = self.area[good]
        if self.raw_points is not None:
            # Select observations from well-aligned facets and away from the noisy
            # tails. This avoids fitting a bevel or nearby wall into the plane.
            core = good[self.normals[good] @ normal > np.cos(np.radians(18))]
            raw = self.scan_points(core)
            valid = np.abs((raw - origin) @ normal) < max(self.tol, 3 * self.noise)
            if valid.sum() >= 24:
                points = raw[valid]
                weights = np.ones(len(points))
        for _ in range(3):
            origin = np.average(points, axis=0, weights=weights)
            delta = points - origin
            _, vectors = np.linalg.eigh((delta * weights[:, None]).T @ delta)
            n = vectors[:, 0]
            normal = n if n @ normal > 0 else -n
            residual = np.abs(delta @ normal)
            scale = max(self.tol / 4, 1.4826 * float(np.median(residual)))
            weights *= np.minimum(1., 2 * scale / np.maximum(residual, 1e-12))
        return normal, origin
