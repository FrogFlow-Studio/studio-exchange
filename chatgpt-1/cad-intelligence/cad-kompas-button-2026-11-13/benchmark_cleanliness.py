"""Synthetic acceptance of exact islands and leveled floors."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cadquery as cq
import numpy as np
from shapely.geometry import Point

from button_fixture import scan_mesh
from cad_cleanliness import cad_cleanliness
from candidate_guard import compare_candidate
from kompas_build import islands, sections
from level_pads import level_pads


def round_trip(body):
    with TemporaryDirectory() as folder:
        path = str(Path(folder)/"candidate.step")
        cq.exporters.export(body,path)
        result = cq.importers.importStep(path).val()
    assert result.isValid() and len(result.Solids())==1
    return result


def cylinder_case():
    base = cq.Solid.makeBox(24,18,1,cq.Vector(-12,-9,0))
    rim = cq.Solid.makeBox(24,18,4,cq.Vector(-12,-9,.5)).cut(
        cq.Solid.makeBox(22,16,5,cq.Vector(-11,-8,.5)))
    shell = base.fuse(rim).clean()
    truth = shell
    for x,outer,inner in ((-5,2.,.8),(5,3.,1.2)):
        boss = cq.Solid.makeCylinder(outer,3.1,cq.Vector(x,0,.9)).cut(
            cq.Solid.makeCylinder(inner,3.2,cq.Vector(x,0,.9)))
        truth = truth.fuse(boss).clean()
    from nurbs import body_mesh

    scan = body_mesh(truth,.006)
    scan.merge_vertices(digits_vertex=6)
    scan.update_faces(scan.nondegenerate_faces() & scan.unique_faces())
    scan = scan.subdivide().subdivide()
    scan.vertices += np.random.default_rng(68).normal(0,.02,len(scan.vertices))[:,None]*scan.vertex_normals
    layers = [(z,p) for z,p in sections(scan,.2) if p and 1.15<z<3.95]
    stock = islands(layers,4.5)
    fitted = islands(layers,4.5,exact=True)
    expected = [4.,6.]
    report = sorted([p._cylindrical_island for p in fitted if hasattr(p,"_cylindrical_island")],
                    key=lambda row:row["diameter_mm"])
    assert len(report)==2 and all(abs(r["diameter_mm"]-v)<=.05 for r,v in zip(report,expected))
    assert [r["bore_diameter_mm"] for r in report]==[1.6,2.4]
    assert all(sum(f.geomType()=="CYLINDER" for f in p.Faces())==2 for p in fitted)
    from cylindrical_islands import cylindrical_island

    noncircle = Point(0,0).buffer(3)
    from shapely.affinity import scale

    assert cylindrical_island([(z,scale(noncircle,xfact=1.1)) for z in (1,2,3)],0,4) is None
    before,after = (shell.fuse(*parts).clean() for parts in (stock,fitted))
    after = round_trip(after)
    assert sum(f.geomType()=="CYLINDER" for f in after.Faces())==4
    accepted,audit = compare_candidate(scan,before,after,allowance=0.)
    assert accepted
    return dict(task=68,diameters=report,before=cad_cleanliness(before),
                after=cad_cleanliness(after),coverage=audit)


def pad_case():
    body = cq.Solid.makeBox(28,22,7,cq.Vector(-14,-11,0)).cut(
        cq.Solid.makeBox(24,18,6,cq.Vector(-12,-9,2)))
    rib = cq.Solid.makeBox(1,18,.8,cq.Vector(-.5,-9,1.99))
    real_step = cq.Solid.makeBox(3,4,.036,cq.Vector(6,2,2))
    truth = body.fuse(rib,real_step).clean()
    bump = cq.Solid.makeBox(3,4,.036,cq.Vector(-7,2,2))
    dip = cq.Solid.makeBox(3,4,.037,cq.Vector(-7,-6,1.965))
    before = truth.fuse(bump).cut(dip).clean()
    scan = scan_mesh(truth).subdivide().subdivide()
    scan.vertices += np.random.default_rng(69).normal(0,.02,len(scan.vertices))[:,None]*scan.vertex_normals
    after,count = level_pads(before,scan)
    assert count==2
    assert after.isInside(cq.Vector(7,3,2.025))
    assert not after.isInside(cq.Vector(-6,3,2.025))
    assert after.isInside(cq.Vector(-6,-5,1.995))
    after = round_trip(after)
    accepted,audit = compare_candidate(scan,before,after,allowance=0.)
    assert accepted
    return dict(task=69,adjusted=count,before=cad_cleanliness(before),
                after=cad_cleanliness(after),coverage=audit)


if __name__=="__main__":
    rows = [cylinder_case(),pad_case()]
    result = json.dumps(rows,indent=2,allow_nan=False)+"\n"
    Path(sys.argv[1]).write_text(result,encoding="utf8")
    print(result)
