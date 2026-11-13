"""Try task 47 on an isolated, fully supported zone and compare with a baseline STEP."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cadquery as cq
import numpy as np
import trimesh

from compare import compare
from large_surfaces import build, rectangular_chart, step_audit
from nurbs import body_mesh


def coverage(scan, solid, tol):
    tess = body_mesh(solid, min(.04, tol/6))
    tess.merge_vertices()
    report = compare(scan, tess, tol, n=12000, seed=47)
    return [report["within_tol_share"], round(1-report["added_surface_share"],4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan",type=Path)
    parser.add_argument("spec",type=Path,help="JSON with zone_faces, support_normals, support_origins and bottom_point")
    parser.add_argument("baseline",type=Path)
    parser.add_argument("out",type=Path,help="candidate STEP, written only on acceptance")
    parser.add_argument("--tol",type=float,default=.3)
    args=parser.parse_args()
    if args.out.resolve() in {p.resolve() for p in (args.scan,args.spec,args.baseline)}:
        parser.error("output must be distinct from inputs")
    if args.out.exists():
        parser.error("output path already exists")
    started=perf_counter()
    try:
        scan=trimesh.load(args.scan,force="mesh",process=False)
        spec=json.loads(args.spec.read_text(encoding="utf-8"))
        zone=np.asarray(spec["zone_faces"],dtype=int)
        if zone.ndim!=1 or not len(zone) or min(zone)<0 or max(zone)>=len(scan.faces):
            raise ValueError("invalid zone face indices")
        # Preserve triangle indexing while joining coincident STL vertices.
        scan.merge_vertices()
        points,chart=rectangular_chart(scan,zone,np.asarray(spec["support_normals"]),
                                       np.asarray(spec["support_origins"]),np.asarray(spec["bottom_point"]),args.tol)
        solid,fit=build(points,chart,tol=args.tol)
        restored,audit=step_audit(solid)
        baseline=cq.importers.importStep(str(args.baseline)).val()
        before,after=coverage(scan,baseline,args.tol),coverage(scan,restored,args.tol)
        ok=(audit["valid"] and audit["mesh_cover_min"]>=.98 and audit["seam_angle_max_deg"]<=2.
            and 25*audit["spline_faces"]<=audit["spline_area_mm2"] and all(a>=b for a,b in zip(after,before)))
        report={**fit,**audit,"coverage_before":before,"coverage_after":after,"accepted":ok}
        if ok:
            cq.exporters.export(solid,str(args.out))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        report={"accepted":False,"reason":str(exc)}
    print(json.dumps({**report,"seconds":perf_counter()-started},indent=2,allow_nan=False))
    return 0 if report["accepted"] else 1


if __name__=="__main__":
    raise SystemExit(main())
