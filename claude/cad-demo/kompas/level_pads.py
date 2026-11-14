"""Merge shallow steps into a neighboring, scan-supported pocket floor."""
import cadquery as cq
import numpy as np
import shapely
from OCP.Standard import Standard_Failure
from shapely.geometry import Polygon

from cad_cleanliness import horizontal_planes


def _footprint(face, *, curved=False):
    if len(face.Wires())!=1:
        return None
    if any(e.geomType()!="LINE" for e in face.Edges()):
        if not curved:
            return None
        wire = face.outerWire()
        # The polygon is only for evidence selection. The boolean uses the
        # original OCC wire and retains exact arcs and splines.
        points,_ = wire.sample(min(4096,max(96,int(wire.Length()/.06))))
        poly = Polygon([(p.x,p.y) for p in points])
        return poly if poly.is_valid and poly.area>0 else None
    # Preserve corners by using the actual wire vertices in traversal order.
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.BRep import BRep_Tool

    walk = BRepTools_WireExplorer(face.outerWire().wrapped,face.wrapped)
    xy = []
    while walk.More():
        v = BRep_Tool.Pnt_s(walk.CurrentVertex())
        xy.append((v.X(),v.Y()))
        walk.Next()
    poly = Polygon(xy) if len(xy)>=3 else None
    return poly if poly is not None and poly.is_valid and poly.area>0 else None


def level_pads(body, scan, *, curved=False):
    """Level supported shallow bumps and depressions; preserve an unsupported step."""
    import trimesh

    points,indices = trimesh.sample.sample_surface(scan,90000,seed=69)
    points = points[scan.face_normals[indices,2]>(.6 if curved else .92)]
    rows = [(f,z) for f,z in horizontal_planes(body,angle_deg=.05) if f.normalAt().z>.99]
    targets = sorted(rows,key=lambda row:-row[0].Area())
    footprints = {}
    accepted = 0
    for face,z in sorted(rows,key=lambda row:row[0].Area()):
        if not any(f.isSame(face) for f in body.Faces()):
            continue
        poly = _footprint(face,curved=curved)
        if poly is None:
            continue
        for parent,height in targets:
            offset = z-height
            if not .005<abs(offset)<.05 or parent.Area()<4*face.Area():
                continue
            if parent not in footprints:
                support = parent if len(parent.Wires())==1 else cq.Face.makeFromWires(parent.outerWire())
                footprints[parent] = _footprint(support,curved=curved)
            outer = footprints[parent]
            if outer is None or not outer.buffer(.002).covers(poly):
                continue
            mask = shapely.contains_xy(poly,points[:,0],points[:,1])
            samples = points[mask]
            samples = samples[abs(samples[:,2]-height)<.10]
            if len(samples)<12:
                continue
            delta = samples[:,2]-height
            median = np.median(delta)
            limit = max(.006,.3*abs(offset))
            if curved and median*offset<0:
                # Evidence on the opposite side of the floor cannot
                # support the existing step. Still require proximity.
                limit = max(limit,.025)
            if abs(median)>limit or np.quantile(abs(delta),.9)>.045:
                continue
            # A narrow overlap makes the filling prism fuse to the floor.
            low,high = sorted((z,height))
            try:
                wire = face.outerWire().translate((0,0,low-z-(.003 if offset<0 else 0)))
                tool = cq.Solid.extrudeLinear(wire,[],cq.Vector(0,0,high-low+.003))
                candidate = (body.cut(tool) if offset>0 else body.fuse(tool)).clean()
                if (candidate.isValid() and len(candidate.Solids())==1
                        and len(horizontal_planes(candidate))<len(horizontal_planes(body))
                        and abs(candidate.Volume()-body.Volume())<=1.2*poly.area*abs(offset)):
                    body = candidate
                    accepted += 1
                    break
            except (RuntimeError,ValueError,Standard_Failure):
                continue
    return body,accepted


def guarded_level_pads(scan, body, info, log, *, curved=False):
    from candidate_guard import compare_candidate

    candidate,count = level_pads(body,scan,curved=curved)
    if not count:
        return body,info
    accepted,audit = compare_candidate(scan,body,candidate,allowance=0.)
    log(f"    level pads guard: {audit}")
    info = {**info,"level_pads_guard":{**audit,"accepted":accepted}}
    if accepted:
        return candidate,{**info,"leveled_pads":count,"faces":len(candidate.Faces())}
    return body,info
