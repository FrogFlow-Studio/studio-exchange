"""STEP acceptance for a noisy sleeve with two inward hooks and upper gaps."""
import json
from pathlib import Path
import sys

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from shapely.geometry import Polygon

from benchmark_71_72 import noise
from benchmark_73 import fixture
from benchmark_cleanliness import round_trip
from candidate_guard import compare_candidate
from hooked_islands import hooked_island
from kompas_build import islands, sections


def wedge(angle,width,low,high):
    theta = np.deg2rad([angle-width/2,angle+width/2])
    points = [(0,0)]+[(8*np.cos(a),8*np.sin(a)) for a in theta]
    return cq.Workplane("XY").workplane(offset=low).polyline(points).close().extrude(high-low).val()


def run():
    floor,truth = fixture(False,windows=((0,60),(100,30),(180,60),(280,30)))
    for angle in (132.5,312.5):
        truth = truth.cut(wedge(angle,50,5.4,5.8))
        hook = cq.Solid.makeCylinder(2.30,.8,cq.Vector(0,0,4.6)).cut(
            cq.Solid.makeCylinder(1.525,.82,cq.Vector(0,0,4.59)))
        truth = truth.fuse(hook.intersect(wedge(angle,22,4.59,5.41))).clean()
    assert truth.isValid() and len(truth.Solids())==1
    scan = noise(truth,74)
    layers = [(z,p) for z,p in sections(scan,.2) if p and 1.13<z<6.96]
    sleeve = [(z,p) for z,polygons in layers for p in polygons if Polygon(p.exterior).area<60]
    missing_z = sorted({z for z,_ in sleeve})[4]
    rejected,error = hooked_island([(z,p) for z,p in sleeve if z!=missing_z],1.05,7.,.2)
    assert rejected is None and error["reason"]=="missing section" and error["level"]==5,error
    old = islands(layers,8.)
    new = islands(layers,8.,exact="hooked")
    matches = [p for p in new if hasattr(p,"_cylindrical_island")]
    assert len(matches)==1
    audit = matches[0]._cylindrical_island
    assert (audit["windows"],audit["gaps"],audit["additions"])==(4,2,2),audit
    before = floor.fuse(*old).clean()
    after = round_trip(floor.fuse(*new).clean())
    diameters = sorted({round(2*BRepAdaptor_Surface(f.wrapped).Cylinder().Radius(),4)
                        for f in after.Faces() if f.geomType()=="CYLINDER"})
    assert after.isValid() and len(after.Solids())==1
    assert 4.45 in diameters and 7.5 in diameters,diameters
    for angle in (0,100,180,280):
        t = np.deg2rad(angle)
        assert not after.isInside(cq.Vector(3*np.cos(t),3*np.sin(t),4.))
    for angle in (132.5,312.5):
        t = np.deg2rad(angle)
        assert after.isInside(cq.Vector(1.8*np.cos(t),1.8*np.sin(t),5.))
        assert not after.isInside(cq.Vector(3*np.cos(t),3*np.sin(t),5.6))
        assert after.isInside(cq.Vector(3*np.cos(t),3*np.sin(t),6.2))
    accepted,coverage = compare_candidate(scan,before,after,allowance=0.)
    assert accepted,coverage
    assert len(after.Faces())<len(before.Faces())
    return dict(diameters_mm=diameters,coverage=coverage,accepted=accepted,
                faces=[len(before.Faces()),len(after.Faces())],
                geometry=audit,missing_section=error)


if __name__=="__main__":
    report = run()
    print(json.dumps(report),flush=True)
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
