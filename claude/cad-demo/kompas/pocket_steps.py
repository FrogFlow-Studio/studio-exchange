"""Remove shallow ledges from finished pocket walls by rebuilding closed wall bands."""
from collections import Counter, defaultdict
from math import cos, radians
from statistics import median
import json
from subprocess import run, TimeoutExpired
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt2d
from OCP.Standard import Standard_Failure
from OCP.StdFail import StdFail_NotDone

from cad_cleanliness import horizontal_planes

LIMIT = .08
JOIN_OFFSET = .15
JOIN_HEIGHT = .05
WALL_Z = cos(radians(65))
CAD_ERRORS = (ValueError, RuntimeError, Standard_Failure,StdFail_NotDone)


def _samples(edge):
    return [edge.positionAt(t) for t in (.0,.25,.5,.75,1.)]


def _is_wall(face,edges):
    return all(abs(face.normalAt(point).z)<WALL_Z for edge in edges for point in _samples(edge)[1:-1])


def _side_at_edge(face,edge):
    # Classify a small step in either tangent-plane direction against the trimmed face.
    votes = []
    for t in (.25,.5,.75):
        point = edge.positionAt(t)
        direction = face.normalAt(point).cross(edge.tangentAt(t)).normalized()
        for distance in (.06,.025):
            local = []
            for sign in (-1,1):
                u,v = face.paramAt(point+direction*(sign*distance))
                inside = BRepClass_FaceClassifier(face.wrapped,gp_Pnt2d(u,v),1e-7)
                if inside.State()==TopAbs_IN:
                    dz = face.positionAt(u,v).z-point.z
                    if abs(dz)>1e-4:
                        local.append(1 if dz>0 else -1)
            if len(local)==1:
                votes.extend(local)
                break
    return votes[0] if len(votes)>=2 and len(set(votes))==1 else 0


def _ledges(body, rejected_offsets=None):
    adjacency = defaultdict(list)
    face_edges = {}
    for face in body.Faces():
        edges = face.Edges()
        face_edges[face] = edges
        for edge in edges:
            adjacency[edge].append(face)
    reasons = Counter()
    shelves = {}
    for face,z in horizontal_planes(body):
        perimeter = sum(w.Length() for w in face.Wires())
        if not perimeter or 2*face.Area()/perimeter>=LIMIT:
            continue
        above,below = [],[]
        neighbors = set()
        for edge in face_edges[face]:
            other = [f for f in adjacency[edge] if f!=face]
            if len(other)!=1 or not _is_wall(other[0],[edge]):
                continue
            wall = other[0]
            side = _side_at_edge(wall,edge)
            if not side:
                continue
            (above if side>0 else below).append(edge)
            neighbors.add(wall)
        if not above or not below:
            reasons["without_two_wall_sides"] += 1
            continue
        sides = [(edges,cq.Compound.makeCompound(edges)) for edges in (above,below)]
        offset = max(cq.Vertex.makeVertex(*point.toTuple()).distance(target)
                     for (edges,_),(_,target) in (sides,sides[::-1])
                     for edge in edges for point in _samples(edge))
        if offset>=JOIN_OFFSET:
            reasons["offset_above_limit"] += 1
            if rejected_offsets is not None:
                rejected_offsets.append(offset)
            continue
        shelves[face] = neighbors
    return shelves,face_edges,adjacency,reasons


def _bands(shelves,face_edges,adjacency):
    pending = set(shelves)
    order = list(face_edges)
    for first in shelves:
        if first not in pending:
            continue
        todo = [first]
        band = set()
        ledges = set()
        while todo:
            face = todo.pop()
            if face in band:
                continue
            band.add(face)
            if face in shelves:
                ledges.add(face)
                todo.extend(shelves[face])
                continue
            for edge in face_edges[face]:
                for neighbor in adjacency[edge]:
                    if neighbor in band:
                        continue
                    if neighbor in shelves:
                        if face in shelves[neighbor]:
                            todo.append(neighbor)
                    elif edge.BoundingBox().zlen>1e-4 and _is_wall(neighbor,[edge]):
                        todo.append(neighbor)
        pending.difference_update(ledges)
        yield [f for f in order if f in band],len(ledges)


def _bridge(band,face_edges,adjacency):
    region = set(band)
    boundary = list(dict.fromkeys(e for f in band for e in face_edges[f]
                                  if any(g not in region for g in adjacency[e])))
    wires = cq.Wire.combine(boundary,tol=1e-6)
    if len(wires)!=2 or any(not w.IsClosed() or w.BoundingBox().zlen>1e-4 for w in wires):
        return None,"open_band"
    wires.sort(key=lambda w:w.Center().z)
    loft = cq.Solid.makeLoft(wires,ruled=True)
    limits = [wire.Center().z for wire in wires]
    walls = [f for f in loft.Faces() if
             not any(abs(f.Center().z-z)<1e-5 and f.BoundingBox().zlen<1e-4 for z in limits)]
    if not walls or len(walls)>=len(band):
        return None,"no_face_reduction"
    original = cq.Compound.makeCompound(band)
    replacement = cq.Compound.makeCompound(walls)
    # Guard seam displacement along its full boundary and curvature across the wall.
    for faces,target in ((band,replacement),(walls,original)):
        points = [point for f in faces for edge in f.Edges() for point in _samples(edge)]
        for face in faces:
            if face.geomType()=="PLANE":
                continue
            u0,u1,v0,v1 = face.uvBounds()
            for x in (.2,.5,.8):
                for y in (.2,.5,.8):
                    u,v = u0+x*(u1-u0),v0+y*(v1-v0)
                    sample = BRepClass_FaceClassifier(face.wrapped,gp_Pnt2d(u,v),1e-6)
                    if sample.State()==TopAbs_IN:
                        points.append(face.positionAt(u,v))
        if any(cq.Vertex.makeVertex(*p.toTuple()).distance(target)>JOIN_OFFSET for p in points):
            return None,"wall_displacement"
    return walls,None


def _rebuild_bands(body):
    shelves,face_edges,adjacency,reasons = _ledges(body)
    removed = set()
    inserted = []
    count = 0
    for band,n in _bands(shelves,face_edges,adjacency):
        try:
            new,reason = _bridge(band,face_edges,adjacency)
        except CAD_ERRORS:
            new,reason = None,"wall_loft_failed"
        if new is None:
            reasons[reason] += n
            continue
        removed.update(band)
        inserted.extend(new)
        count += n
    audit = dict(found=len(shelves),removed=0,reasons=dict(reasons))
    if not count:
        return body,audit
    sew = BRepBuilderAPI_Sewing(1e-6)
    for face in [f for f in face_edges if f not in removed]+inserted:
        sew.Add(face.wrapped)
    sew.Perform()
    shells = cq.Shape.cast(sew.SewedShape()).Shells()
    if sew.NbFreeEdges() or len(shells)!=1:
        audit["reasons"]["sewing_open"] = count
        return body,audit
    shape = cq.Solid.makeSolid(shells[0]).clean()
    if not shape.isValid() or len(shape.Solids())!=1 or shape.Volume()<=0:
        audit["reasons"]["invalid_solid"] = count
        return body,audit
    audit.update(removed=count,faces=len(shape.Faces()))
    return shape,audit


def _join_groups(shelves):
    groups = []
    for z in sorted(face.Center().z for face in shelves):
        if not groups or z-groups[-1][0]>2*JOIN_HEIGHT:
            groups.append([])
        groups[-1].append(z)
    return groups


def _rebuild_local(body,shelves,reasons):
    # Work around each join inside a thin slab. Faces that span the join
    # retain their original surfaces beyond the slab.
    if not shelves:
        return body,dict(found=0,removed=0,reasons=dict(reasons))
    # Simple closed bands still get the prior fast, guarded repair.
    spans_join = False
    for shelf,walls in shelves.items():
        z = shelf.Center().z
        for wall in walls:
            box = wall.BoundingBox()
            if box.zmin<z-1e-4 and box.zmax>z+1e-4:
                spans_join = True
                break
        if spans_join:
            break
    if not spans_join:
        whole,full = _rebuild_bands(body)
        if full["removed"]==len(shelves):
            return whole,full
    groups = _join_groups(shelves)
    protected = []
    for face,z in horizontal_planes(body):
        if face in shelves:
            continue
        perimeter = sum(w.Length() for w in face.Wires())
        if perimeter and 2*face.Area()/perimeter>=LIMIT:
            protected.append((face.BoundingBox(),z))
    current = body
    removed = 0
    for i,group in enumerate(groups):
        count = len(group)
        low,high = group[0],group[-1]
        left = low-groups[i-1][-1] if i else 1.
        right = groups[i+1][0]-high if i+1<len(groups) else 1.
        footprints = [face.BoundingBox() for face in shelves if low<=face.Center().z<=high]
        nearby = (z for box,z in protected if any(
            box.xmin<=b.xmax+.15 and box.xmax>=b.xmin-.15
            and box.ymin<=b.ymax+.15 and box.ymax>=b.ymin-.15 for b in footprints))
        nearest = min((min(abs(low-z),abs(high-z)) for z in nearby),default=1.)
        span = min(.30,.40*left,.40*right,.40*nearest)
        if span<.065:
            reasons["nearby_horizontal_feature"] += count
            continue
        try:
            planes = [cq.Face.makePlane(basePnt=cq.Vector(0,0,z),dir=cq.Vector(0,0,1))
                      for z in (low-span,high+span)]
            parts = current.split(*planes).Solids()
            middle = [p for p in parts if p.BoundingBox().zmin>=low-span-1e-5
                      and p.BoundingBox().zmax<=high+span+1e-5]
            if not middle:
                reasons["missing_wall_slab"] += count
                continue
            replacements = {}
            slab_reasons = Counter()
            for part in middle:
                fixed,local = _rebuild_bands(part)
                if local["removed"]:
                    replacements[part] = fixed
                else:
                    slab_reasons.update(local["reasons"])
            if not replacements:
                reason = slab_reasons.most_common(1)[0][0] if slab_reasons else "no_removable_band"
                reasons["slab_"+reason] += count
                continue
            repaired = [replacements.get(part,part) for part in parts]
            joined = repaired[0].fuse(*repaired[1:]).clean()
            if not joined.isValid() or len(joined.Solids())!=1 or joined.Volume()<=0:
                reasons["invalid_wall_slab"] += count
                continue
            remaining,_,_,_ = _ledges(joined)
            removed_here = max(0,count-sum(low-1e-4<=f.Center().z<=high+1e-4 for f in remaining))
            if removed_here:
                current = joined
                removed += removed_here
            if removed_here<count:
                reasons["shelf_remains_after_join"] += count-removed_here
        except CAD_ERRORS:
            reasons["wall_slab_failed"] += count
    return current,dict(found=len(shelves),removed=removed,reasons=dict(reasons),
                        faces=len(current.Faces()))


def _rebuild(body):
    offsets = []
    shelves,_,_,reasons = _ledges(body,offsets)
    screened = len(shelves)+sum(reasons.values())
    result,audit = _rebuild_local(body,shelves,reasons)
    audit["screened"] = screened
    audit["join_groups"] = [dict(z_mm=[round(g[0],4),round(g[-1],4)],count=len(g))
                            for g in _join_groups(shelves)]
    audit["rejected_offset_mm"] = dict(
        count=len(offsets),min=round(min(offsets),4),median=round(median(offsets),4),
        max=round(max(offsets),4)) if offsets else dict(count=0)
    return result,audit


def _worker(folder):
    try:
        original = cq.Shape.importBrep(str(folder/"original.brep"))
        body,audit = _rebuild(original)
        if (folder/"scan.npz").exists() and audit["removed"]:
            import numpy as np
            import trimesh
            from candidate_guard import compare_candidate

            with np.load(folder/"scan.npz") as data:
                scan = trimesh.Trimesh(vertices=data["vertices"],faces=data["faces"],process=False)
            passed,coverage = compare_candidate(scan,original,body,allowance=0.)
            audit["coverage"] = coverage
            if not passed:
                audit["reasons"]["coverage_guard"] = audit["removed"]
                audit["removed"] = 0
        if audit["removed"] and not body.exportBrep(str(folder/"clean.brep")):
            audit["reasons"]["export_failed"] = audit["removed"]
            audit["removed"] = 0
    except CAD_ERRORS:
        audit = dict(found=0,removed=0,reasons={"cad_failure":1})
    (folder/"audit.json").write_text(json.dumps(audit),encoding="utf8")


def clean_pocket_steps(body, *, scan=None, timeout_s=60.):
    """Return the original solid if the repair, including scan validation, times out."""
    start = monotonic()
    timeout = dict(found=0,removed=0,reasons={"timeout":1})
    if timeout_s<=0:
        return body,timeout
    with TemporaryDirectory(prefix="pocket-steps-") as path:
        folder = Path(path)
        try:
            if not body.exportBrep(str(folder/"original.brep")):
                return body,dict(found=0,removed=0,reasons={"export_failed":1})
            if scan is not None:
                import numpy as np

                np.savez(folder/"scan.npz",vertices=scan.vertices,faces=scan.faces)
            remaining = timeout_s-(monotonic()-start)
            if remaining<=0:
                return body,timeout
            process = run([sys.executable,__file__,path],capture_output=True,timeout=remaining)
            audit_file = folder/"audit.json"
            if process.returncode or not audit_file.exists():
                return body,dict(found=0,removed=0,reasons={"worker_failed":1})
            audit = json.loads(audit_file.read_text(encoding="utf8"))
            shape = cq.Shape.importBrep(str(folder/"clean.brep")) if audit["removed"] else body
            return (shape,audit) if monotonic()-start<timeout_s else (body,timeout)
        except TimeoutExpired:
            return body,timeout
        except CAD_ERRORS+(OSError,):
            return body,dict(found=0,removed=0,reasons={"transfer_failed":1})


if __name__=="__main__":
    # The CLI can also run without the environment of kompas_build.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    _worker(Path(sys.argv[1]))
