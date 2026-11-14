"""Reproduce the reviewed Infiniti STEP from an aligned scan and the stock STEP.

This fixture uses a measured region and dimensions specific to this button.
"""
import argparse
import os
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from shapely import union_all
from shapely.affinity import scale
from shapely.geometry import box

from candidate_guard import compare_candidate
from cad_cleanliness import cad_cleanliness
from kompas_build import _assemble_stock, prism_poly


def tab_from_scan(scan):
    region = box(.95,-17.24,2.22,-16.88)
    profiles = []
    for z in np.linspace(10.2,11.22,7):
        section = scan.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
        if section is None:
            raise ValueError(f"missing tab section at {z:.3f} mm")
        path,_ = section.to_2D(to_2D=np.eye(4))
        profiles.append(union_all(path.polygons_full).intersection(region))
    profile = union_all(profiles).buffer(.008)
    if profile.is_empty:
        raise ValueError("no tab material in the measured region")
    if profile.geom_type!="Polygon":
        profile = max(profile.geoms,key=lambda part:part.area)
    profile = scale(profile,yfact=.9,origin=(0,-17.24))
    return prism_poly(profile,10.17,11.20,tol=.018)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    scan = trimesh.load(args.scan,force="mesh")
    reference = cq.importers.importStep(str(args.reference)).val()
    os.environ.update(CAD_SECTIONS_CORRECT="1",CORR_ADD="0",THIN_RIB="0")
    raw,info = _assemble_stock(scan,exact_islands="hooked",fit_pocket_floor=True,void_n=48)
    if info.get("bosses")!=5 or not info.get("cylindrical_islands"):
        raise ValueError("expected all five bosses and the reconstructed sleeve")
    # Repeat the same STEP round trip used during fit and coverage verification.
    from benchmark_cleanliness import round_trip

    raw = round_trip(raw)
    tab = tab_from_scan(scan)
    candidate = round_trip(raw.fuse(tab).clean())
    delta = candidate.Volume(tol=1e-7)-raw.Volume(tol=1e-7)
    if not 0<delta<=tab.Volume(tol=1e-7)+.05:
        raise ValueError(f"tab changed model volume by {delta:.5f} mm³")
    passed,audit = compare_candidate(scan,reference,candidate,allowance=0.)
    print("Coverage",audit,flush=True)
    if not passed:
        raise ValueError("candidate failed scan coverage")
    print("CAD",cad_cleanliness(candidate),flush=True)
    cq.exporters.export(candidate,str(args.output))


if __name__=="__main__":
    main()
