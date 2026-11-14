"""Tapered sleeve additions and rejected pocket step offsets."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cadquery as cq
import numpy as np
from OCP.StdFail import StdFail_NotDone
from shapely.geometry import Polygon

from benchmark_71_72 import noise
from benchmark_cleanliness import round_trip
from candidate_guard import compare_candidate
from cad_cleanliness import cad_cleanliness
from hooked_islands import _add_features, _track_tools
from pocket_steps import _rebuild


def tapered_additions():
    theta = np.arange(48)*2*np.pi/48
    tracks = [[(i,Polygon(np.column_stack((sign*(2.28+.21*np.cos(theta)-depth*(1-np.cos(theta))/2),
                                          .37*np.sin(theta)))))
               for i,depth in enumerate((.14,.33,.53,.67,.50,.24))] for sign in (-1,1)]
    bounds = np.linspace(1.,2.2,7)
    sleeve = cq.Solid.makeCylinder(3.75,3.).cut(cq.Solid.makeCylinder(2.225,3.))
    straight,_ = _add_features(sleeve,tracks,[],bounds,mode="legacy")
    smooth,audit = _add_features(sleeve,tracks,[],bounds)
    body = round_trip(smooth)
    assert audit["addition_methods"]=={"smooth_loft":2},audit
    step_scan = noise(straight,78)
    accepted,coverage = compare_candidate(step_scan,straight,body,allowance=0.)
    assert accepted,coverage
    before,after = (cad_cleanliness(b) for b in (straight,body))
    assert after["faces"]<before["faces"] and after["narrow_faces"]<=before["narrow_faces"],(before,after)

    from hooked_islands import _transition_loft

    def fail_smooth(track,bounds,ruled):
        return _transition_loft(track,bounds,ruled) if ruled else None

    with patch("hooked_islands._transition_loft",side_effect=fail_smooth):
        _,method = _track_tools(tracks[0],bounds,mode="smooth")
    assert method=="ruled_loft",method
    with patch("hooked_islands.cq.Solid.makeLoft",side_effect=StdFail_NotDone):
        _,fallback = _add_features(sleeve,tracks,[],bounds)
    assert fallback["addition_methods"]=={"stacked_prisms":2},fallback
    return dict(faces=[before["faces"],after["faces"]],
                narrow_faces=[before["narrow_faces"],after["narrow_faces"]],
                coverage=coverage,methods=audit["addition_methods"],ruled_fallback=method,
                prism_fallback=fallback["addition_methods"])


def rejected_offset():
    v = cq.Vector
    box = cq.Solid.makeBox(12,12,8,v(-6,-6,0))
    lower = cq.Solid.makeBox(6,6,3,v(-3,-3,1))
    upper = cq.Workplane("XY").workplane(offset=4).polyline(
        [(-3,-3),(3.02,-3),(3.12,3),(-3,3)]).close().extrude(4.1).val()
    body = box.cut(lower,upper).clean()
    unchanged,audit = _rebuild(body)
    assert unchanged is body and audit["reasons"]=={"offset_above_limit":1},audit
    assert audit["rejected_offset_mm"]==dict(count=1,min=.12,median=.12,max=.12),audit
    return audit


if __name__=="__main__":
    report = dict(additions=tapered_additions(),pocket=rejected_offset())
    rendered = json.dumps(report,indent=2)
    print(rendered,flush=True)
    Path(sys.argv[1]).write_text(rendered+"\n",encoding="utf8")
