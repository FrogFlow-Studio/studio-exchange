"""Build and reopen synthetic STEP controls for tasks 71 and 72."""
import json
from pathlib import Path
import sys

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface

from benchmark_cleanliness import round_trip
from cad_cleanliness import cad_cleanliness
from candidate_guard import compare_candidate
from kompas_build import islands, sections
from level_pads import level_pads
from nurbs import body_mesh


def noise(body,seed):
    mesh = body_mesh(body,.006)
    mesh.merge_vertices(digits_vertex=6)
    mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
    for _ in range(3):
        mesh = mesh.subdivide()
    mesh.vertices += np.random.default_rng(seed).normal(0,.02,len(mesh.vertices))[:,None]*mesh.vertex_normals
    return mesh


def boss_case():
    v = cq.Vector
    base = cq.Solid.makeBox(34,22,1,v(-17,-11,0))
    wall = cq.Solid.makeBox(34,22,9,v(-17,-11,.9)).cut(cq.Solid.makeBox(32,20,10,v(-16,-10,.9)))
    shell = base.fuse(wall).clean()
    first = cq.Solid.makeCylinder(3.75,7.1,v(-7,0,.9))
    first = first.fuse(cq.Solid.makeBox(12,1,3.1,v(-13,-.5,.9))).clean()
    second = cq.Workplane("XY").center(7,0).polygon(8,7.5).extrude(7.1).val().translate((0,0,.9))
    bosses = [b.cut(cq.Solid.makeCylinder(1.25,7.3,v(x,0,.89))).clean()
              for b,x in ((first,-7),(second,7))]
    truth = shell.fuse(*bosses).clean()
    scan = noise(truth,71)
    layers = [(z,p) for z,p in sections(scan,.2) if p and 1.15<z<7.85]
    old = islands(layers,9.9)
    new = islands(layers,9.9,exact="partial")
    before = shell.fuse(*old).clean()
    after = round_trip(shell.fuse(*new).clean())
    diameters = sorted(2*BRepAdaptor_Surface(f.wrapped).Cylinder().Radius()
                       for f in after.Faces() if f.geomType()=="CYLINDER")
    assert len(diameters)==3 and np.allclose(diameters,[2.5,2.5,7.5],atol=.05),diameters
    assert after.isInside(v(-12,0,2.5))
    assert after.isInside(v(-2,0,2.5))
    assert not after.isInside(v(-12,0,5))
    assert not after.isInside(v(-7,0,2.5)) and not after.isInside(v(7,0,2.5))
    accepted,audit = compare_candidate(scan,before,after,allowance=0.)
    assert accepted,audit
    return dict(task=71,diameters_mm=diameters,coverage=audit,faces=[len(before.Faces()),len(after.Faces())],
                islands=[p._cylindrical_island for p in new if hasattr(p,"_cylindrical_island")])


def pad_case():
    v = cq.Vector
    cutter = cq.Workplane().box(24,18,6).edges("|Z").fillet(2.).val().translate((0,0,5))
    base = cq.Solid.makeBox(28,22,7,v(-14,-11,0)).cut(cutter)
    ribs = [cq.Solid.makeBox(1,18,.8,v(x,-9,1.99)) for x in (-2.,2.)]
    step = cq.Solid.makeBox(3,4,.3,v(6,2,2))
    truth = base.fuse(*ribs,step).clean()
    bump = cq.Solid.makeCylinder(1.4,.032,v(-6,3,2))
    dip = cq.Workplane().box(3,4,.07).edges("|Z").fillet(.6).val().translate((-6,-3,2.005))
    t = np.linspace(0,2*np.pi,32,endpoint=False)
    loop = [v(6+1.2*np.cos(a),-3+.8*np.sin(a),2) for a in t]
    wire = cq.Wire.assembleEdges([cq.Edge.makeSpline(loop,periodic=True)])
    spline = cq.Solid.extrudeLinear(wire,[],v(0,0,.04))
    before = truth.fuse(bump,spline).cut(dip).clean()
    scan = noise(truth,72)
    after,count = level_pads(before,scan,curved=True)
    after = round_trip(after)
    assert count==3,count
    assert after.isInside(v(7,3,2.25))
    assert not after.isInside(v(-6,3,2.025))
    assert after.isInside(v(-6,-3,1.995))
    assert not after.isInside(v(6,-3,2.025))
    accepted,audit = compare_candidate(scan,before,after,allowance=0.)
    assert accepted,audit
    return dict(task=72,adjusted=count,before=cad_cleanliness(before),
                after=cad_cleanliness(after),coverage=audit)


if __name__=="__main__":
    output = []
    for case in (boss_case,pad_case):
        row = case()
        output.append(row)
        print(json.dumps(row),flush=True)
    Path(sys.argv[1]).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf8")
