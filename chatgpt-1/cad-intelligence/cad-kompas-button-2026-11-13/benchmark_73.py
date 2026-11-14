"""Synthetic tests for a four-window sleeve extracted from mesh sections."""
import json
from pathlib import Path
import sys

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface

from benchmark_71_72 import noise,pad_case
from benchmark_cleanliness import round_trip
from candidate_guard import compare_candidate
from kompas_build import islands, sections
from level_pads import guarded_level_pads
from windowed_islands import windowed_island


def fixture(stepped, windows=((0,60),(90,30),(180,60),(270,30))):
    vec = cq.Vector
    base = cq.Workplane().box(24,20,1).translate((0,0,.5)).val()
    rim = cq.Workplane().box(24,20,7).translate((0,0,4.5)).val().cut(
        cq.Workplane().box(22,18,8).translate((0,0,5)).val())
    floor = base.fuse(rim).clean()
    ring = cq.Solid.makeCylinder(3.75,6.1,vec(0,0,.9))
    ring = ring.cut(cq.Solid.makeCylinder(2.225,6.2,vec(0,0,.89)))
    if stepped:
        for height in (.89,5.8):
            ring = ring.cut(cq.Solid.makeCylinder(2.5,1.22,vec(0,0,height)))
    # Four independent planar sectors, alternating between 60 and 30 degrees.
    for center,width in windows:
        angles = np.deg2rad([center-width/2,center+width/2])
        points = [(0,0)]+[(8*np.cos(a),8*np.sin(a)) for a in angles]
        cutter = cq.Workplane("XY").workplane(offset=2.2).polyline(points).close().extrude(3.6).val()
        ring = ring.cut(cutter)
    return floor, floor.fuse(ring).clean()


def sleeve_case(stepped=False):
    floor,truth = fixture(stepped)
    scan = noise(truth,73)
    layers = [(z,p) for z,p in sections(scan,.2) if p and 1.13<z<6.96]
    stock = islands(layers,8.)
    trial = islands(layers,8.,exact="windowed")
    matches = [p for p in trial if hasattr(p,"_cylindrical_island")]
    assert len(matches)==1, ("windowed not recognized",stepped)
    audit = matches[0]._cylindrical_island
    assert audit["windows"]==4
    before = floor.fuse(*stock).clean()
    after = round_trip(floor.fuse(*trial).clean())
    diameters = sorted({round(2*BRepAdaptor_Surface(f.wrapped).Cylinder().Radius(),5)
                        for f in after.Faces() if f.geomType()=="CYLINDER"})
    expected = [4.45,5.,7.5] if stepped else [4.45,7.5]
    assert len(diameters)==len(expected) and np.allclose(diameters,expected,atol=.05),diameters
    # Check material in each angular segment and absence at all four windows.
    for angle in (0,90,180,270):
        t = np.deg2rad(angle)
        assert not after.isInside(cq.Vector(3*np.cos(t),3*np.sin(t),4.))
        assert after.isInside(cq.Vector(3*np.cos(t),3*np.sin(t),6.4))
        t = np.deg2rad(angle+45)
        assert after.isInside(cq.Vector(3*np.cos(t),3*np.sin(t),4.))
    planes = sum(f.geomType()=="PLANE" and abs(f.normalAt().z)<1e-5 for f in after.Faces())
    assert planes>=16,planes
    passed,coverage = compare_candidate(scan,before,after,allowance=0.)
    assert passed,coverage
    if not stepped:
        assert len(after.Faces())<len(before.Faces())
        from shapely.affinity import scale
        from shapely.geometry import Polygon

        stretched = [(z,scale(p,xfact=1.14,origin=(0,0))) for z,polys in layers for p in polys
                     if Polygon(p.exterior).area<60]
        assert windowed_island(stretched,.5,7.,.2) is None
    return dict(stepped=stepped,diameters_mm=diameters,analytic=audit,coverage=coverage,
                coverage_ok=passed,faces=[len(before.Faces()),len(after.Faces())],vertical_planes=planes)


if __name__=="__main__":
    rows = []
    for stepped in (False,True):
        row = sleeve_case(stepped)
        print(json.dumps(row),flush=True)
        rows.append(row)
    rows.append(pad_case())
    floor,_ = fixture(False)
    messages = []
    _,info = guarded_level_pads(noise(floor,72),floor,{},messages.append,curved=True)
    assert not info and len(messages)==1 and "no_nearby_floor" in messages[0], messages
    print(messages[0],flush=True)
    Path(sys.argv[1]).write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
