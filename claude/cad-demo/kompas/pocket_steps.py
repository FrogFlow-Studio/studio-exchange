"""Remove shallow ledges from finished pocket walls by rebuilding closed wall bands."""
from collections import Counter, defaultdict
from math import cos, radians
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt2d
from OCP.Standard import Standard_Failure

from cad_cleanliness import horizontal_planes

LIMIT = .08
WALL_Z = cos(radians(65))
CAD_ERRORS = (ValueError, RuntimeError, Standard_Failure)


def _samples(edge):
    return [edge.positionAt(t) for t in (.0,.25,.5,.75,1.)]


def _is_wall(face,edges):
    return all(abs(face.normalAt(point).z)<WALL_Z for edge in edges for point in _samples(edge)[1:-1])


def _ledges(body):
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
            box = wall.BoundingBox()
            if box.zmin>=z-1e-4 and box.zmax>z+1e-4:
                above.append(edge)
            elif box.zmax<=z+1e-4 and box.zmin<z-1e-4:
                below.append(edge)
            else:
                continue
            neighbors.add(wall)
        if not above or not below:
            reasons["without_two_wall_sides"] += 1
            continue
        sides = [(edges,cq.Compound.makeCompound(edges)) for edges in (above,below)]
        if any(cq.Vertex.makeVertex(*point.toTuple()).distance(target)>=LIMIT
               for (edges,_),(_,target) in (sides,sides[::-1])
               for edge in edges for point in _samples(edge)):
            reasons["offset_above_limit"] += 1
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
        if any(cq.Vertex.makeVertex(*p.toTuple()).distance(target)>LIMIT for p in points):
            return None,"wall_displacement"
    return walls,None


def _rebuild(body):
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


def _worker(folder,scan,response):
    try:
        original = cq.Shape.importBrep(str(folder/"original.brep"))
        body,audit = _rebuild(original)
        if scan is not None and audit["removed"]:
            from candidate_guard import compare_candidate

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
    response.send(audit)
    response.close()


def clean_pocket_steps(body, *, scan=None, timeout_s=60.):
    """Return the original solid if the repair, including scan validation, times out."""
    start = monotonic()
    failed = dict(found=0,removed=0,reasons={"timeout":1})
    if timeout_s<=0:
        return body,failed
    with TemporaryDirectory(prefix="pocket-steps-") as path:
        folder = Path(path)
        try:
            exported = body.exportBrep(str(folder/"original.brep"))
        except CAD_ERRORS+(OSError,):
            exported = False
        if not exported:
            return body,dict(found=0,removed=0,reasons={"export_failed":1})
        ctx = get_context("spawn")
        receiver,sender = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_worker,args=(folder,scan,sender))
        try:
            process.start()
        except (OSError,RuntimeError):
            receiver.close()
            sender.close()
            return body,dict(found=0,removed=0,reasons={"worker_failed":1})
        sender.close()
        try:
            if not receiver.poll(max(0.,timeout_s-(monotonic()-start))):
                return body,failed
            try:
                audit = receiver.recv()
            except EOFError:
                return body,dict(found=0,removed=0,reasons={"worker_failed":1})
            try:
                shape = cq.Shape.importBrep(str(folder/"clean.brep")) if audit["removed"] else body
            except CAD_ERRORS+(OSError,):
                audit["reasons"]["import_failed"] = audit["removed"]
                audit["removed"] = 0
                shape = body
            return (shape,audit) if monotonic()-start<timeout_s else (body,failed)
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            receiver.close()
