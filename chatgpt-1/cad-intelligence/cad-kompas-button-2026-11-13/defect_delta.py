"""Task 55. Compare localized signed differences across model revisions."""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from button_defects import defects


def compare_defects(scan, before, after, tol=.3):
    """Match compatible patches one to one, then report appearances and removals."""
    old = defects(scan,before,tol)
    new = defects(scan,after,tol)
    cost = np.full((len(old),len(new)),1e6)
    for i,a in enumerate(old):
        for j,b in enumerate(new):
            if (a["source"],a["sign"]) != (b["source"],b["sign"]):
                continue
            distance = np.linalg.norm(np.subtract(a["center_mm"],b["center_mm"]))
            limit = max(3*tol,np.sqrt(max(a["area_mm2"],b["area_mm2"])/np.pi))
            if distance<=limit:
                cost[i,j] = distance/limit
    matches = [(i,j) for i,j in zip(*linear_sum_assignment(cost)) if cost[i,j]<1e6]
    old_matched = {i for i,_ in matches}
    new_matched = {j for _,j in matches}
    rows = []
    for i,j in matches:
        a,b = old[i],new[j]
        changed = (abs(a["area_mm2"]-b["area_mm2"])>max(.25,.1*a["area_mm2"])
                   or abs(a["max_distance_mm"]-b["max_distance_mm"])>.25*tol
                   or cost[i,j]>.3)
        rows.append(dict(status="changed" if changed else "unchanged",before=a,after=b))
    rows.extend(dict(status="removed",before=a,after=None) for i,a in enumerate(old) if i not in old_matched)
    rows.extend(dict(status="added",before=None,after=b) for j,b in enumerate(new) if j not in new_matched)
    counts = Counter(row["status"] for row in rows)
    return dict(patches=rows,summary={key:counts[key] for key in ("removed","added","changed","unchanged")})
