"""Reproduce the button work for tasks 53–57 using only synthetic geometry."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
from shapely.geometry import Polygon

import kompas_build
from benchmark_button import quality
from button_fixture import make_button, scan_mesh
from defect_delta import compare_defects
from periodic_wall import curvatures, periodic_wall
from stack_check import stack_check


def bracket():
    """A fork with two unsupported arms and interrupted transverse sections."""
    return (cq.Workplane().box(48,18,5).translate((0,0,2.5))
            .union(cq.Workplane().box(6,18,32).translate((-20,0,18)))
            .union(cq.Workplane().box(6,18,32).translate((20,0,18)))
            .union(cq.Workplane().box(48,5,6).translate((0,-6.5,30)))
            .union(cq.Workplane().box(12,18,6).translate((0,19,2.5))).val())


def run(work):
    button = make_button()
    truth = scan_mesh(button,refine=False)
    cq.exporters.export(button,str(work/"truth.step"))
    scans = {s:scan_mesh(button,s) for s in (0.,.05,.1)}
    rows = []
    os.environ["CAD_KOMPAS_STABLE_POCKETS"] = "1"
    for sigma,scan in scans.items():
        for rounding in (False,True):
            os.environ["CAD_KOMPAS_STABLE_ROUND"] = str(int(rounding))
            started = perf_counter()
            body,info = kompas_build.assemble(scan)
            row = dict(task=53 if rounding else 54,sigma=sigma,
                       **quality(truth,body,perf_counter()-started),info=info)
            rows.append(row)
            print(json.dumps(row),flush=True)

    before = scan_mesh(make_button(defect="corner_cut"))
    after = scan_mesh(make_button(defect="bump"))
    delta = compare_defects(scans[.1],before,after)
    assert any(row["status"]=="removed" and row["before"]["center_mm"][1]<-17 for row in delta["patches"])
    assert any(row["status"]=="added" and row["after"]["center_mm"][0]>11.5 for row in delta["patches"])
    rows.append(dict(task=55,**delta))
    rows.extend(wall_benchmark(work))
    for name,scan in (("button",scans[.05]),("bracket",scan_mesh(bracket(),refine=False))):
        rows.append(dict(task=57,case=name,**stack_check(scan)))
    assert all(row["valid"] and row["info"]["stable_pockets"] and min(row["coverage"])>=.97
               for row in rows if row["task"]==54)
    assert all(row["valid"] and row["info"]["bottom_rounded"] and row["info"]["top_rounded"]
               for row in rows if row["task"]==53)
    assert rows[-2]["suitable"] and rows[-2]["axis"]==[0.,0.,1.] and not rows[-1]["suitable"]
    return rows


def wall_benchmark(work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.linspace(0,2*np.pi,160,endpoint=False)
    z = np.linspace(0,12,25)
    clean = np.array([np.column_stack((11*(1+.07*np.cos(theta))*np.cos(theta)+.05*h,
                                     24*(1+.05*np.sin(2*theta))*np.sin(theta))) for h in z])
    rings = lambda p: [np.vstack((r,r[:1])) for r in p]
    truth, _ = periodic_wall(rings(clean),z,modes=46,axial_penalty=.0001)
    reference = scan_mesh(truth,refine=False)
    figure, panels = plt.subplots(2,3,figsize=(12,5),layout="constrained")
    rows = []
    for j,sigma in enumerate((0.,.05,.1)):
        observed = clean+np.random.default_rng(56).normal(0,sigma,clean.shape)
        rng = np.random.default_rng(156)
        measured = rings([np.roll(r,rng.integers(len(r)),axis=0) for r in observed])
        stock_rings = np.array([r for r,_ in kompas_build.align(
            [(kompas_build.ring_pts(r,32,.05),h) for r,h in zip(measured,z)])])
        keep = np.arange(0,len(z),3)
        stock = cq.Solid.makeLoft([kompas_build._wire(stock_rings[i],z[i]) for i in keep])
        started = perf_counter()
        candidate = periodic_wall(measured,z)
        if candidate is None:
            rows.append(dict(task=56,sigma=sigma,rejected=True))
            continue
        body, fitted = candidate
        base = quality(reference,stock,0.)
        trial = quality(reference,body,perf_counter()-started)
        metrics = {}
        for i, (old,new) in enumerate(zip(curvatures(stock_rings,z),curvatures(fitted,z))):
            name = ("axial","transverse")[i]
            metrics[name] = [float(np.quantile(k,.99)) for k in (old,new)]
            ax = panels[i,j]
            ax.plot(z,np.quantile(old,.95,axis=1),label="32 points")
            ax.plot(z,np.quantile(new,.95,axis=1),label="Periodic fit")
            ax.set(title=f"{name}, σ={sigma:g} mm",xlabel="Height (mm)",ylabel="p95 curvature (1/mm)")
            ax.grid(alpha=.2)
        rows.append(dict(task=56,sigma=sigma,before=base,after=trial,curvature_p99=metrics))
    panels[0,0].legend()
    figure.savefig(work/"wall-56.png",dpi=150)
    plt.close(figure)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    result = run(args.work)
    (args.work/"followups.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
