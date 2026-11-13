"""Opt-in experiments for the 0с assembly; keep the shared pipeline untouched."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import cadquery as cq
import numpy as np
from shapely.geometry import Polygon
import trimesh

import kompas_build
from bottom_round import estimate_radius, round_bottom
from button_defects import defects
from ring_fit import correspond
from rim_fit import cut_rim, fit_rim
from stl_out import write_stl_checked


def assemble(scan, *, round_edge=False, raised_rim=False, fixed_rings=False, detect=False):
    spec = importlib.util.spec_from_file_location("sever_button_stock",Path(kompas_build.__file__))
    stock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stock)
    old_top, old_cut = stock.top_surface,stock.cut_above
    local_height = None
    used_rim = False
    used_rings = False

    def top_surface(mesh, ring, height, band=3., down=False):
        nonlocal local_height
        if raised_rim and not down:
            local_height = fit_rim(mesh,ring,max(height,float(mesh.bounds[1,2])))
        return old_top(mesh,ring,height,band,down)

    def cut_above(body, coef, bounds, margin=5., below=False):
        nonlocal used_rim
        if local_height is not None and not below:
            try:
                result = cut_rim(body,local_height,bounds)
                if result.isValid() and len(result.Solids())==1 and result.Volume()>0:
                    used_rim = True
                    return result
            except (ValueError,RuntimeError):
                pass
        return old_cut(body,coef,bounds,margin,below)

    old_align = stock.align
    first_alignment = True
    first_loft = True

    def align(rings):
        nonlocal first_alignment, used_rings
        outer = first_alignment
        first_alignment = False
        if outer and fixed_rings and len(rings)>=4:
            profiles,z = zip(*rings)
            aligned = correspond(profiles,z,count=len(profiles[0]))
            if aligned is not None:
                used_rings = True
                return list(zip(aligned,z))
        return old_align(rings)

    def loft(rings, ruled=False):
        nonlocal first_loft
        outer = first_loft
        first_loft = False
        # The upstream worker requests fork, unavailable on Windows.
        if sys.platform != "win32" and not (outer and used_rings):
            return kompas_build.loft(rings,ruled)
        if not (outer and used_rings):
            rings = old_align(rings)
        return cq.Solid.makeLoft([stock._wire(p,z) for p,z in rings],ruled)

    stock.top_surface,stock.cut_above,stock.loft = top_surface,cut_above,loft
    stock.align = align
    body, info = stock.assemble(scan)
    if raised_rim:
        info["raised_rim_used"] = used_rim
    if fixed_rings:
        info["fixed_rings_used"] = used_rings
    if round_edge:
        samples = scan.triangles_center
        outer = abs(samples[:,:2]-scan.bounds[:,:2].mean(axis=0)) > .38*scan.extents[:2]
        bottom = samples[(scan.face_normals[:,2]<-.85) & outer.any(axis=1),2]
        radius = None
        if len(bottom):
            floor = float(np.quantile(bottom,.2))
            sections = stock.sections(scan,.1)
            profiles = [(z,max(polys,key=lambda p:Polygon(p.exterior).area).exterior.coords)
                        for z,polys in sections if polys and floor<z<floor+2.5]
            if profiles:
                levels,rings = zip(*profiles)
                radius = estimate_radius(rings,levels,floor)
        body, success = round_bottom(body,radius)
        info.update(bottom_radius_mm=radius,bottom_rounded=success,faces=len(body.Faces()))
    if detect:
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"comparison.stl"
            ratio,_ = write_stl_checked(body.wrapped,path,.04,False,.4)
            if abs(ratio-1)<=.05:
                info["detected_patches"] = defects(scan,trimesh.load(path,force="mesh"))
            else:
                info["detection_skipped_mesh_volume_ratio"] = ratio
    return body,info
