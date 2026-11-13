"""Synthetic controls for the upper wall, pocket floor, and variable fillet."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

import kompas_build as stock
from benchmark_button import quality
from button_fixture import scan_mesh
from local_surfaces import trim_pocket, upper_wall
from tapered_fixture import end_edge, outline
from variable_fillet import _law, variable_fillet


def box(w,l,h,z):
    return cq.Workplane().box(w,l,h).translate((0,0,z)).val()


def wall_loft(levels):
    loft = BRepOffsetAPI_ThruSections(True,False,1e-6)
    loft.SetMaxDegree(2)
    loft.CheckCompatibility(False)
    for ring,z in levels:
        loft.AddWire(stock._wire(ring,z).wrapped)
    loft.Build()
    return cq.Solid(loft.Shape())


def rim_body(levels):
    outer = wall_loft(levels)
    outer = stock.cut_above(outer,np.array([11.65,0,.042,0,0,0]),(-12,-26,12,26))
    return outer.fillet(.3,[end_edge(outer,top=True)]).clean()


def wall_case(work):
    base = [(outline(z,count=64),z) for z in (0.,3.,6.,8.5,10.)]
    ring = base[-1][0]
    # The nose retreats above the last full contour, under an inclined rim.
    y = np.maximum(ring[:,1],0.)/24.5
    tip = ring-np.column_stack((np.zeros(len(ring)),.43*y**6))
    stock_levels = [*base,(ring,13.4)]
    reference = rim_body([*base,(tip,13.4)])
    observed = scan_mesh(reference,.035,seed=64)
    started = perf_counter()
    fitted = upper_wall(observed,stock_levels,ring,10.,13.4)
    if fitted is None:
        return dict(task=64,rejected=True)
    after = rim_body(fitted)
    elapsed = perf_counter()-started
    before = rim_body(stock_levels)
    for name,body in (("wall-truth",reference),("wall-stock",before),("wall-fit",after)):
        cq.exporters.export(body,str(work/f"{name}.step"))
    truth = scan_mesh(reference,refine=False)
    local = [float(np.max(np.abs(levels[-1][0]-tip))) for levels in (stock_levels,fitted)]
    def nose_error(body):
        from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
        from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

        probe = IntCurvesFace_ShapeIntersector()
        probe.Load(body.wrapped,1e-6)
        heights = np.linspace(10.15,12.15,21)
        values = []
        for z in heights:
            probe.Perform(gp_Lin(gp_Pnt(.2,0,z),gp_Dir(0,1,0)),22,26)
            values.append(max(probe.WParameter(i) for i in range(1,probe.NbPnt()+1)))
        return np.asarray(values)

    reference_nose = nose_error(reference)
    return dict(task=64,before=quality(truth,before,0),after=quality(truth,after,elapsed),
                end_profile_max_mm=local,nose_max_mm=[float(max(abs(nose_error(b)-reference_nose)))
                for b in (before,after)],top_fillet_mm=.3)


def floor_case(work, curved):
    shell = box(22,35,9,4.5)
    tool = box(16,27,12,7.5)
    coef = np.array([1.8,0.,.035,0.,0.,.004 if curved else 0.])
    exact = shell.cut(stock.cut_above(tool,coef,(-8,-13.5,8,13.5),below=True))
    observed = scan_mesh(exact,.045,seed=65)
    started = perf_counter()
    smooth = trim_pocket(observed,tool)
    reference = scan_mesh(exact,refine=False)
    if smooth is None:
        return dict(task=65,curved=curved,rejected=True)
    result = shell.cut(smooth)
    elapsed = perf_counter()-started
    cq.exporters.export(result,str(work/f"floor-{curved}.step"))
    # Compare against actual section-step cutters spaced 0.2 mm in Z.
    stepped = shell
    for i in range(27):
        y = -13.+i
        f = np.floor((1.8+.035*y+(.004*y*y if curved else 0.))/.2)*.2
        band = cq.Workplane().box(16,1.,12-f).translate((0,y,(12+f)/2)).val()
        stepped = stepped.cut(band)
    return dict(task=65,curved=curved,before=quality(reference,stepped,0),
                after=quality(reference,result,elapsed))


def variable_case():
    sharp = box(14,28,6,3)
    edge = next(e for e in sharp.Edges() if e.Length()>27 and e.Center().x>6 and e.Center().z>5)
    values = [(0.,.3),(.14,.34),(.38,1.02),(.62,1.02),(.86,.34),(1.,.3)]
    truth = _law(sharp,edge,values)
    if truth is None:
        return dict(task=67,rejected="reference law")
    scan = scan_mesh(truth,.035,seed=67)
    started = perf_counter()
    candidate = variable_fillet(scan,sharp,edge)
    elapsed = perf_counter()-started
    reference = scan_mesh(truth,refine=False)
    baseline = sharp.fillet(.3,[edge])
    row = dict(task=67,before=quality(reference,baseline,0))
    if candidate is not None:
        body,law = candidate
        row.update(after=quality(reference,body,elapsed),law=law.tolist())
    else:
        row["rejected"] = "estimated law"
    return row


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    rows = []
    for run in (lambda:wall_case(args.work),lambda:floor_case(args.work,False),
                lambda:floor_case(args.work,True),variable_case):
        row = run()
        rows.append(row)
        print(json.dumps(row),flush=True)
    (args.work/"results.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
    wall, slope, bowl, radius = rows
    assert wall["nose_max_mm"][1] < wall["nose_max_mm"][0]
    for row in rows:
        assert row["after"]["valid"]
        assert all(a>=b for a,b in zip(row["after"]["coverage"],row["before"]["coverage"]))
    assert all(row["after"]["faces"]<row["before"]["faces"] for row in (slope,bowl))
    assert max(a-b for a,b in zip(radius["after"]["coverage"],radius["before"]["coverage"]))>.0003
