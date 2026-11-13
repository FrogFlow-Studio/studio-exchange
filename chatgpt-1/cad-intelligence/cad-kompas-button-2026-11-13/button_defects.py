"""Signed, spatially coherent differences in the eight comparison-sheet views."""
import numpy as np
from defects import _clusters, _distance


VIEWS = (
    ("top",(0,.001,1)), ("bottom",(0,.001,-1)),
    ("front_right",(1,-1.2,1)), ("back_left",(-1,1.2,1)),
    ("right",(1,0,.15)), ("left",(-1,0,.15)),
    ("front",(0,-1,.15)), ("back",(0,1,.15)),
)


def defects(scan, model, tol=.3):
    """Return coherent mismatch patches for two meshes in the same coordinates."""
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("positive finite tolerance required")
    result = []
    for name, source, target in (("scan",scan,model),("model",model,scan)):
        distance, signed, _ = _distance(source,target)
        step = np.median(source.edges_unique_length)
        area_min = max(1.5,12*tol**2,2*step**2)
        for sign, mask in (("above",signed>=0),("below",signed<0)):
            for part in _clusters(source,mask & (distance>tol)):
                area = float(source.area_faces[part].sum())
                if area < area_min or np.quantile(distance[part],.85) < 1.25*tol:
                    continue
                weights = source.area_faces[part]
                normals = source.face_normals[part]
                views = []
                for view, direction in VIEWS:
                    unit = np.array(direction)/np.linalg.norm(direction)
                    if np.sum(weights[normals@unit>.35]) >= .15*area:
                        views.append(view)
                result.append(dict(source=name, sign=sign, views=views, area_mm2=area,
                                   center_mm=np.average(source.triangles_center[part],axis=0,weights=weights).tolist(),
                                   max_distance_mm=float(distance[part].max())))
    return sorted(result,key=lambda patch:-patch["area_mm2"])
