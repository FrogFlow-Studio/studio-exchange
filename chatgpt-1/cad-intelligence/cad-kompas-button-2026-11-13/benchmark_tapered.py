"""Tasks 58–61 on a tapered, flared housing with a stepped cavity."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np

from benchmark_button import quality
from button_fixture import scan_mesh
from candidate_guard import compare_candidate
from fillet_radius import fillet_radius
from periodic_wall import curvatures, periodic_wall
from tapered_fixture import end_edge, loft, outline, sharp_outer, tapered_button


def run(work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    sharp = sharp_outer()
    for radius in (0.,.3,.48,.8):
        body = tapered_button(radius)
        scan = scan_mesh(body,.05)
        cq.exporters.export(body,str(work/f"truth-{radius:g}.step"))
        scan.export(work/f"scan-{radius:g}.stl")
        start = perf_counter()
        estimate = fillet_radius(scan,sharp,end_edge(sharp))
        rows.append(dict(task=61,radius=radius,estimate=estimate,seconds=perf_counter()-start))
        if radius==.48:
            for label,candidate in (("same",body),("shifted",body.translate((1.1,0,0)))):
                accepted,audit = compare_candidate(scan,body,candidate)
                rows.append(dict(task=59,case=label,accepted=accepted,**audit))

    z = np.linspace(0,13,25)
    clean = np.array([outline(h) for h in z])
    stock = loft([0.,3.,7.,10.,13.])
    truth = scan_mesh(stock,refine=False)
    fig,axes = plt.subplots(2,3,figsize=(11,5),layout="constrained")
    for i,sigma in enumerate((0.,.05,.1)):
        measured = clean+np.random.default_rng(60).normal(0,sigma,clean.shape)
        contours = [np.vstack((r,r[:1])) for r in measured]
        start = perf_counter()
        trial = periodic_wall(contours,z,degree=2,modes=30)
        if trial is None:
            rows.append(dict(task=60,sigma=sigma,rejected=True))
            continue
        body, rings = trial
        result = quality(truth,body,perf_counter()-start)
        measured_curvatures = curvatures(rings,z)
        result["curvature_p99"] = [float(np.quantile(k,.99)) for k in measured_curvatures]
        for j,(name,raw,fitted) in enumerate(zip(("Axial","Transverse"),curvatures(measured,z),measured_curvatures)):
            ax = axes[j,i]
            ax.plot(z,np.quantile(raw,.95,axis=1),label="Measured rings")
            ax.plot(z,np.quantile(fitted,.95,axis=1),label="Periodic fit")
            ax.set(title=f"{name}, σ={sigma:g} mm",xlabel="Height (mm)",ylabel="p95 curvature (1/mm)")
            ax.grid(alpha=.2)
        for name,radius,top in (("bottom",.48,False),("top",.3,True)):
            try:
                candidate = body.fillet(radius,[end_edge(body,top=top)])
                result[name+"_rounded"] = bool(candidate.isValid() and len(candidate.Solids())==1)
                if result[name+"_rounded"]:
                    body = candidate
            except Exception as error:
                result[name+"_rounded"] = False
                result[name+"_error"] = str(error)
        result["rounded_quality"] = quality(truth,body,perf_counter()-start)
        rows.append(dict(task=60,sigma=sigma,**result))
    axes[0,0].legend()
    fig.savefig(work/"tapered-curvature.png",dpi=150)
    plt.close(fig)
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",required=True,type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    rows = run(args.work)
    (args.work/"tapered.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
    assert all((row["estimate"] is None if row["radius"]==0 else
                row["estimate"] is not None and abs(row["estimate"]-row["radius"])<=.08)
               for row in rows if row["task"]==61)
    assert all(row["accepted"]==(row["case"]=="same") for row in rows if row["task"]==59)
