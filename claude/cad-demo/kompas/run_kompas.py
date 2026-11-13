import sys, time, trimesh, cadquery as cq
sys.path.insert(0, __import__("os").path.dirname(__file__))
import kompas_build as stack
t = time.time()
m = trimesh.load(sys.argv[1], force="mesh")
pieces, cut, top = stack.build2(m)
print("outer faces", len(pieces[0].Faces()), pieces[0].isValid(), round(pieces[0].Volume(),1), flush=True)
cq.exporters.export(cq.Compound.makeCompound(pieces), sys.argv[2] + "_outer.stl", tolerance=0.05, angularTolerance=0.2)
print("outer exported", round(time.time()-t,1), flush=True)
cq.exporters.export(cq.Compound.makeCompound(cut), sys.argv[2] + "_voids.stl", tolerance=0.05, angularTolerance=0.2)
print("export", round(time.time() - t, 1), "top slices", len(top), flush=True)
body = pieces[0]
skipped = 0
for v in sorted(cut, key=lambda c: -c.Volume()):
    try:
        nb = body.cut(v)
        ok = nb.isValid() and nb.Volume() > 0.5 * body.Volume()
        print('cut', round(v.Volume(), 1), ok, round(time.time() - t, 1), flush=True)
        if ok:
            body = nb
        else:
            skipped += 1
    except Exception:
        skipped += 1
if stack.TOPINFO.get("coef") is not None:
    bb = pieces[0].BoundingBox()
    nb = stack.cut_above(body, stack.TOPINFO["coef"], (bb.xmin, bb.ymin, bb.xmax, bb.ymax))
    print("top cut", nb.isValid(), round(nb.Volume(), 1), round(time.time() - t, 1), flush=True)
    if nb.isValid(): body = nb
if stack.TOPINFO.get("bcoef") is not None:
    bb = pieces[0].BoundingBox()
    nb = stack.cut_above(body, stack.TOPINFO["bcoef"], (bb.xmin, bb.ymin, bb.xmax, bb.ymax), below=True)
    print("bottom cut", nb.isValid(), round(nb.Volume(), 1), flush=True)
    if nb.isValid(): body = nb
body = body.clean()
print("cuts skipped", skipped, "of", len(cut), flush=True)
print("cut", round(time.time() - t, 1), body.isValid(), round(body.Volume(), 1), flush=True)
cq.exporters.export(body, sys.argv[2] + "_low.step")
cq.exporters.export(body, sys.argv[2] + "_low.stl", tolerance=0.05, angularTolerance=0.2)
print("low saved", len(body.Faces()), flush=True)
tp = top
cq.exporters.export(cq.Compound.makeCompound(tp), sys.argv[2] + "_isl.stl", tolerance=0.05, angularTolerance=0.2)
for k, p in enumerate(tp):
    try:
        nb = body.fuse(p)
        ok = nb.isValid()
        print("fuse", k, ok, round(time.time() - t, 1), flush=True)
        if ok: body = nb
    except Exception as e:
        print("fuse fail", k, str(e)[:40], flush=True)
body = body.clean()
import tempfile, os as _os
if _os.environ.get("CORRECT", "1") == "1":
    tf = _os.path.join(tempfile.gettempdir(), "_kcorr.stl")
    cq.exporters.export(body, tf, tolerance=0.03, angularTolerance=0.15)
    mm = trimesh.load(tf, force="mesh")
    cuts = stack.correction_prisms(m, mm)
    nc = 0
    for v in sorted(cuts, key=lambda c: -c.Volume()):
        try:
            nb = body.cut(v)
            if nb.isValid() and nb.Volume() > 0.9 * body.Volume(): body = nb; nc += 1
        except Exception: pass
    print("correction cuts", nc, "of", len(cuts), round(time.time() - t, 1), flush=True)
    body = body.clean()
    cq.exporters.export(body, tf, tolerance=0.03, angularTolerance=0.15)
    mm = trimesh.load(tf, force="mesh")
    adds = stack.correction_prisms(m, mm, extra=True)
    na = 0
    for v in sorted(adds, key=lambda c: -c.Volume()):
        try:
            nb = body.fuse(v)
            if nb.isValid() and len(nb.Solids()) == 1: body = nb; na += 1
        except Exception: pass
    print("correction adds", na, "of", len(adds), round(time.time() - t, 1), flush=True)
    body = body.clean()
print("final", body.isValid(), len(body.Faces()), round(body.Volume(), 1), flush=True)
cq.exporters.export(body, sys.argv[2] + ".step")
cq.exporters.export(body, sys.argv[2] + ".stl", tolerance=0.05, angularTolerance=0.2)
