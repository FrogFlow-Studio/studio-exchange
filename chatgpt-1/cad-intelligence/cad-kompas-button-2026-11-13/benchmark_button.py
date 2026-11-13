"""Run the feature checks and optionally replay the full 0с build on synthetic scans."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np

import kompas_build as stock
from bottom_round import estimate_radius, round_bottom
from button_defects import defects
from button_fixture import make_button, rim_height, scan_mesh
from button_trial import assemble
from compare import compare
from ring_fit import correspond
from rim_fit import fit_rim
from shapely.geometry import Polygon


def wall_plot(work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, panels = plt.subplots(1,3,figsize=(10,3.2),layout="constrained",sharey=True)
    for sigma,panel in zip((0.,.05,.1),panels):
        path = work/f"wall-{sigma}.csv"
        if path.exists():
            z,before,after = np.loadtxt(path,delimiter=",",skiprows=1).T
            panel.plot(z,before,label="0c")
            panel.plot(z,after,label="Fixed rays")
        panel.set(title=f"σ = {sigma:g} mm",xlabel="Height (mm)",yscale="symlog")
        panel.grid(alpha=.2)
    panels[0].set_ylabel("Max profile acceleration (1/mm)")
    panels[0].legend()
    fig.savefig(work/"wall-curvature.png",dpi=150)
    plt.close(fig)


def quality(reference, body, elapsed):
    from tempfile import TemporaryDirectory
    import trimesh
    from stl_out import write_stl_checked

    with TemporaryDirectory() as temporary:
        path = Path(temporary)/"candidate.stl"
        ratio,_ = write_stl_checked(body.wrapped,path,.04,False,.4)
        if abs(ratio-1)>.05:
            return dict(coverage=None,faces=len(body.Faces()),valid=body.isValid(),seconds=elapsed,
                        mesh_volume_ratio=ratio)
        mesh = trimesh.load(path,force="mesh")
    numbers = compare(reference,mesh,.3,n=12000,seed=49)
    return dict(coverage=[numbers["within_tol_share"],round(1-numbers["added_surface_share"],4)],
                faces=len(body.Faces()),valid=bool(body.isValid() and len(body.Solids())==1),
                mesh_volume_ratio=ratio,seconds=elapsed)


def actual_rim_error(body, ring):
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

    tool = IntCurvesFace_ShapeIntersector()
    tool.Load(body.wrapped,1e-6)
    # Probe in the middle of the solid outer rim, away from its sharp edges.
    polygon = Polygon(ring).buffer(-.55)
    if polygon.geom_type!="Polygon" or polygon.is_empty:
        return None
    rim = polygon.exterior
    errors = []
    for fraction in np.linspace(0,1,240,endpoint=False):
        x,y = rim.interpolate(fraction,normalized=True).coords[0]
        tool.Perform(gp_Lin(gp_Pnt(x,y,0),gp_Dir(0,0,1)),10.5,14.5)
        if tool.NbPnt()==0:
            return None
        height = max(tool.WParameter(i) for i in range(1,tool.NbPnt()+1))
        errors.append(abs(height-rim_height(x,y)))
    return max(errors)


def asymmetric_wall(sigma):
    """Drifting rounded contour with a shoulder and uneven section origins."""
    t = np.linspace(0,2*np.pi,140,endpoint=False)
    levels = np.linspace(1,11,21)
    random = np.random.default_rng(51)
    exact,observed = [],[]
    for z in levels:
        radius = 1+.12*np.cos(t)+.08*np.sin(2*t)+.025*(z/11)*np.cos(3*t)
        clean = np.column_stack((10*radius*np.cos(t)+.025*z,22*radius*np.sin(t)))
        exact.append(clean)
        noisy = clean+random.normal(0,sigma,clean.shape)
        noisy = np.roll(noisy,random.integers(len(t)),axis=0)
        observed.append(np.vstack((noisy,noisy[:1])))
    keep = np.arange(0,len(levels),4)
    truth = cq.Solid.makeLoft([stock._wire(exact[i],levels[i]) for i in keep])
    reference = scan_mesh(truth)
    old = stock.align([(stock.ring_pts(observed[i],64,.05),levels[i]) for i in keep])
    baseline = cq.Solid.makeLoft([stock._wire(p,z) for p,z in old])
    candidate_rings = correspond(observed,levels)
    if candidate_rings is None:
        return dict(rejected=True)
    candidate = cq.Solid.makeLoft([stock._wire(candidate_rings[i],levels[i]) for i in keep])
    return dict(before=quality(reference,baseline,0.),after=quality(reference,candidate,0.))


def feature_checks(work):
    reference = make_button()
    cq.exporters.export(reference,str(work/"truth.step"))
    clean = scan_mesh(reference)
    sharp = make_button(rounded=False)
    sharp_quality = quality(clean,sharp,0.)
    altered = {key:scan_mesh(make_button(defect=key)) for key in ("corner_cut","bump","rim_shift")}
    rows = []
    for sigma in (0.,.05,.1):
        scan = scan_mesh(reference,sigma)
        scan.export(work/f"scan-{sigma}.stl")
        started = perf_counter()
        false = defects(scan,clean)
        cases = {key:defects(scan,mesh) for key,mesh in altered.items()}
        hits = {
            "corner_cut":any(r["source"]=="scan" and r["sign"]=="above" and r["center_mm"][0]>7
                             and r["center_mm"][1]<-17 for r in cases["corner_cut"]),
            "bump":any(r["source"]=="model" and r["sign"]=="above" and r["center_mm"][0]>11.5
                       and abs(r["center_mm"][1])<3 for r in cases["bump"]),
            "rim_shift":any(r["source"]=="scan" and r["sign"]=="above" and r["center_mm"][1]>17
                             and r["center_mm"][2]>11 for r in cases["rim_shift"]),
        }
        rows.append(dict(task=52,sigma=sigma,hits=hits,false=len(false),counts={k:len(v) for k,v in cases.items()},
                         seconds=perf_counter()-started))
        sections = stock.sections(scan,.10)
        levels,polygons = zip(*[(z,max(polys,key=lambda p:Polygon(p.exterior).area)) for z,polys in sections if polys])
        contours = [p.exterior.coords for p in polygons]
        ring = stock.ring_pts(contours[len(contours)//2],128,.03)
        started = perf_counter()
        height = fit_rim(scan,ring,12.8)
        y = np.linspace(-24.5,24.5,501)
        error = float(np.max(abs(height(0,y)-rim_height(0,y)))) if height is not None else None
        if height is None:
            rows.append(dict(task=50,sigma=sigma,rejected=True,seconds=perf_counter()-started))
        else:
            rim_body = make_button(height=height)
            rows.append(dict(task=50,sigma=sigma,max_rim_mm=error,
                             max_brep_rim_mm=actual_rim_error(rim_body,ring),
                             **quality(clean,rim_body,perf_counter()-started)))
        started = perf_counter()
        radius = estimate_radius(contours,levels,0.)
        rounded, accepted = round_bottom(sharp,radius,selection_band=.42)
        rows.append(dict(task=49,sigma=sigma,radius_mm=radius,rounded=accepted,
                         baseline_coverage=sharp_quality["coverage"],
                         **quality(clean,rounded,perf_counter()-started)))
        selected = [i for i,z in enumerate(levels) if 1<z<11.5]
        z = np.array(levels)[selected]
        outlines = [contours[i] for i in selected]
        old = np.array([r for r,_ in stock.align([(stock.ring_pts(p,64,.05),h) for p,h in zip(outlines,z)])])
        started = perf_counter()
        new = correspond(outlines,z)
        if new is not None:
            acceleration = lambda rings: np.linalg.norm(np.gradient(np.gradient(rings,z,axis=0),z,axis=0),axis=2)
            from scipy.interpolate import UnivariateSpline
            # Compare after the same longitudinal smoothing used by 0с.
            weights = 1+6*np.exp(-(z-z[0]))
            old = np.column_stack([UnivariateSpline(z,col,w=weights,s=len(z)*.08**2)(z)
                                   for col in old.reshape(len(z),-1).T]).reshape(old.shape)
            old_accel,new_accel = acceleration(old),acceleration(new)
            series = np.column_stack((z,old_accel.max(axis=1),new_accel.max(axis=1)))
            np.savetxt(work/f"wall-{sigma}.csv",series,delimiter=",",
                       header="z_mm,stock_max_per_mm,aligned_max_per_mm",comments="")
            rows.append(dict(task=51,sigma=sigma,p99_before=float(np.quantile(old_accel,.99)),
                             p99_after=float(np.quantile(new_accel,.99)),
                             max_before=float(old_accel.max()),max_after=float(new_accel.max()),
                             seconds=perf_counter()-started,asymmetric=asymmetric_wall(sigma)))
        else:
            rows.append(dict(task=51,sigma=sigma,rejected=True,seconds=perf_counter()-started))
        print(json.dumps(rows[-4:],ensure_ascii=False),flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--assembly", action="store_true")
    parser.add_argument("--sigma", type=float, choices=(0.,.05,.1))
    parser.add_argument("--flag", choices=("stock","round_edge","raised_rim","fixed_rings"))
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    if args.assembly:
        import trimesh
        clean = scan_mesh(cq.importers.importStep(str(args.work/"truth.step")).val(),refine=False)
        rows = []
        for sigma in (0.,.05,.1):
            if args.sigma is not None and sigma!=args.sigma:
                continue
            scan = trimesh.load(args.work/f"scan-{sigma}.stl",force="mesh")
            for flag in ("stock","round_edge","raised_rim","fixed_rings"):
                if args.flag is not None and flag!=args.flag:
                    continue
                start = perf_counter()
                try:
                    body,info = assemble(scan,**({} if flag=="stock" else {flag:True}))
                    row = {**quality(clean,body,perf_counter()-start),"info":info}
                except (RuntimeError,ValueError) as exc:
                    row = dict(error=str(exc),seconds=perf_counter()-start)
                rows.append(dict(sigma=sigma,flag=flag,**row))
                (args.work/"assembly.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
                print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    else:
        rows = feature_checks(args.work)
        (args.work/"features.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
        wall_plot(args.work)
        assert all(all(r["hits"].values()) and not r["false"] for r in rows if r["task"]==52)
        assert all(r["rounded"] and r["valid"] and
                   all(a>=b for a,b in zip(r["coverage"],r["baseline_coverage"]))
                   for r in rows if r["task"]==49)
        assert all(r["valid"] and r["max_brep_rim_mm"] <= .1 for r in rows if r["task"]==50)
        assert all(r["p99_after"]<r["p99_before"] and r["asymmetric"]["after"]["valid"]
                   and all(a>=b for a,b in zip(r["asymmetric"]["after"]["coverage"],
                                              r["asymmetric"]["before"]["coverage"]))
                   for r in rows if r["task"]==51)
