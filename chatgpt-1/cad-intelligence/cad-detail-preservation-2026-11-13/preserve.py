"""Release planar assignments contradicted by the original mesh."""
import numpy as np
from defects import _clusters
from scipy.spatial import cKDTree


def relief(scan, segmentation, noise=0.):
    """Release coherent shallow relief on otherwise supported planar regions."""
    labels = np.array(segmentation["label"], copy=True)
    regions = [dict(r) for r in segmentation["regions"]]
    step = np.median(scan.edges_unique_length)
    points = scan.triangles_center
    patches = []
    for i, row in enumerate(regions):
        ids = np.flatnonzero(labels == i)
        if row["kind"] != "plane" or len(ids) < 12:
            continue
        n = np.asarray(row["normal"])
        h = (points[ids] - row["origin"]) @ n
        h -= np.median(h)
        # Seed with significant tilt and a coherent height departure.
        tilt = np.abs(scan.face_normals[ids] @ n) < np.cos(np.radians(8))
        for side in (1,-1):
            mask = np.zeros(len(labels),bool)
            mask[ids] = (side*h > max(.075,1.3*noise)) & (side*h < 1.1)
            for zone in _clusters(scan,mask):
                own = np.isin(ids,zone)
                area = scan.area_faces[zone].sum()
                if area < max(2.,3*step**2) or area > .25*scan.area_faces[ids].sum() or not tilt[own].any():
                    continue
                labels[zone] = -1
                patches.append(dict(region=i,faces=zone,area_mm2=float(area)))
    return {**segmentation,"label":labels,"regions":regions},patches


def curvature(model, scan, segmentation, findings):
    """Release neighboring strips intersecting confirmed faceted patches."""
    label = np.array(segmentation["label"], copy=True)
    regs = [dict(r) for r in segmentation["regions"]]
    step = np.median(scan.edges_unique_length)
    tree = cKDTree(model.triangles_center)
    distance, near = tree.query(scan.triangles_center)
    support = np.zeros(len(model.faces),bool)
    released = np.zeros(len(label),bool)
    for row in findings:
        if row["type"] != "гранёная кривизна":
            continue
        ids = row.get("_faces")
        if ids is None:
            continue
        support[ids] = True
    candidate = support[near] & (distance < 1.8*step) & (label >= 0)
    for k in np.unique(label[candidate]):
        own = label == k
        if regs[k]["kind"] == "plane" and scan.area_faces[own & candidate].sum() >= .12*scan.area_faces[own].sum():
            released |= own
    label[released] = -1
    return {**segmentation,"label":label,"regions":regs}, np.flatnonzero(released)
