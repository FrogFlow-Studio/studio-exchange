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
from cylindrical_islands import guarded_exact_islands
from hooked_islands import _add_features, _radial_loft, _track_tools, _union_spline, _valid_addition
from pocket_steps import _rebuild


def tapered_additions():
    with patch.dict("os.environ", {"HOOKED_UNION_SPLINE":"0"}):
        return _tapered_additions()


def _tapered_additions():
    theta = np.arange(48)*2*np.pi/48
    tracks = [[(i,Polygon(np.column_stack((sign*(2.28+.21*np.cos(theta)-depth*(1-np.cos(theta))/2),
                                          .37*np.sin(theta)))))
               for i,depth in enumerate((.14,.33,.53,.67,.50,.24))] for sign in (-1,1)]
    bounds = np.linspace(1.,2.2,7)
    sleeve = cq.Solid.makeCylinder(3.75,3.).cut(cq.Solid.makeCylinder(2.225,3.))
    assert not _valid_addition(sleeve,cq.Solid.makeCylinder(.2,.3),1.)
    radial,status = _radial_loft(tracks[0],bounds,False)
    assert radial is not None and status=="accepted",status
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

    def fail_smooth(track,bounds,ruled,detail):
        return _transition_loft(track,bounds,ruled,detail) if ruled else None

    with patch("hooked_islands._transition_loft",side_effect=fail_smooth):
        _,method = _track_tools(tracks[0],bounds,mode="smooth")
    assert method=="ruled_loft",method
    with patch("hooked_islands.cq.Solid.makeLoft",side_effect=StdFail_NotDone):
        diagnostics = {}
        tools,compact = _track_tools(tracks[0],bounds,mode="smooth",diagnostics=diagnostics)
    assert len(tools)==1 and compact=="union_prism",compact
    assert {entry["status"] for entry in diagnostics["lofts"].values()}=={"StdFail_NotDone"}
    with (patch("hooked_islands.cq.Solid.makeLoft",side_effect=StdFail_NotDone),
          patch("hooked_islands._union_prism",return_value=None)):
        _,fallback = _add_features(sleeve,tracks,[],bounds)
    assert fallback["addition_methods"]=={"stacked_prisms":2},fallback
    with patch.dict("os.environ", {"HOOKED_UNION_SPLINE":"1"}):
        spline,union_audit = _add_features(sleeve,tracks,[],bounds)
    assert union_audit["addition_methods"]=={"union_spline":2},union_audit
    spline = round_trip(spline)
    union_accepted,union_coverage = compare_candidate(step_scan,straight,spline,allowance=0.)
    assert not union_accepted and union_coverage["reason"]=="coverage_regression",union_coverage
    with patch.dict("os.environ", {"CAD_KOMPAS_ISLANDS_ALLOWANCE":"0.0003"}):
        kept,guard = guarded_exact_islands(step_scan,straight,{},lambda _:None,
            lambda *args,**kwargs: (spline,{"cylindrical_islands":[union_audit]}),mode="hooked")
    assert kept is straight and not guard["exact_islands_guard"]["accepted"],guard
    union_faces = cad_cleanliness(spline)
    assert union_faces["faces"] < before["faces"],union_faces
    assert union_faces["narrow_faces"] <= before["narrow_faces"],union_faces
    assert _union_spline([(0,Polygon()),(1,Polygon())],bounds)==(None,"empty_profile")
    return dict(faces=[before["faces"],after["faces"]],
                union_spline=dict(faces=union_faces["faces"],narrow_faces=union_faces["narrow_faces"],
                                  coverage=union_coverage,methods=union_audit["addition_methods"]),
                narrow_faces=[before["narrow_faces"],after["narrow_faces"]],
                coverage=coverage,methods=audit["addition_methods"],ruled_fallback=method,
                compact_fallback=compact,
                prism_fallback=fallback["addition_methods"])


def rejected_offset():
    v = cq.Vector
    box = cq.Solid.makeBox(12,12,8,v(-6,-6,0))
    lower = cq.Solid.makeBox(6,6,3,v(-3,-3,1))
    upper = cq.Workplane("XY").workplane(offset=4).polyline(
        [(-3,-3),(3.002,-3),(3.152,3),(-3,3)]).close().extrude(4.1).val()
    body = box.cut(lower,upper).clean()
    unchanged,audit = _rebuild(body)
    assert unchanged is body and audit["reasons"]=={"offset_above_limit":1},audit
    assert audit["rejected_offset_mm"]==dict(count=1,min=.152,median=.152,max=.152),audit
    return audit


if __name__=="__main__":
    report = dict(additions=tapered_additions(),pocket=rejected_offset())
    rendered = json.dumps(report,indent=2)
    print(rendered,flush=True)
    Path(sys.argv[1]).write_text(rendered+"\n",encoding="utf8")
