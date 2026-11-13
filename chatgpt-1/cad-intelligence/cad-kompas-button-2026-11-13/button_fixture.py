"""Analytic synthetic seat-button housing and deterministic noisy scans."""
import cadquery as cq
import numpy as np
import trimesh

from nurbs import body_mesh
from rim_fit import cut_rim


def rim_height(x, y):
    return 12.15 + .8*(y/49)**2 + .8*(y/49)**3 + .8*y/49


def _outline(inset=0.):
    width, length, radius = 23-2*inset, 49-2*inset, 4-inset
    wire = cq.Sketch().rect(width,length).vertices().fillet(radius)._faces.Faces()[0].outerWire()
    return cq.Face.makeFromWires(wire)


def make_button(*, defect=None, rounded=True, height=rim_height):
    shell = cq.Solid.extrudeLinear(_outline(),cq.Vector(0,0,14))
    bottom_edges = [e for e in shell.Edges() if abs(e.Center().z)<1e-6]
    if rounded:
        shell = shell.fillet(.48,bottom_edges)
    # Three pockets leave two 1.2 mm partitions and a 1.2 mm outside wall.
    void = cq.Solid.extrudeLinear(_outline(1.2),cq.Vector(0,0,16)).translate((0,0,1.7))
    for y in (-9.,8.):
        void = void.cut(cq.Workplane().box(30,1.2,20).translate((0,y,8)).val())
    body = shell.cut(void)
    for x,y in ((-4.,-15.),(4.,15.)):
        boss = (cq.Workplane("XY").center(x,y).circle(3.15).circle(1.55)
                .extrude(9.4).translate((0,0,1.55)).val())
        if x<0:
            notch = cq.Workplane().box(4,1.1,4.2).translate((x+2.5,y,9.7)).val()
            boss = boss.cut(notch)
            tab = cq.Workplane().box(1.8,1.05,.85).translate((x-.95,y,7.6)).val()
            boss = boss.fuse(tab)
        body = body.fuse(boss)
    # A small dome on the underside shares the floor material.
    dome = cq.Solid.makeSphere(24,cq.Vector(0,0,23.65),angleDegrees1=-90,angleDegrees2=0)
    slab = cq.Workplane().box(16,17,1.).translate((0,0,.15)).val()
    body = body.fuse(dome.intersect(slab))
    bbox = (-11.5,-24.5,11.5,24.5)
    if height is not None:
        body = cut_rim(body,height,bbox)
    if defect == "rim_shift":
        cap = cq.Workplane().box(24,12,2.2).translate((0,22,12.95)).val()
        body = body.cut(cap)
    if defect == "corner_cut":
        cutter = cq.Workplane().box(7,8,15).translate((11.8,-23.8,6)).val()
        body = body.cut(cutter)
    if defect == "bump":
        body = body.fuse(cq.Solid.makeSphere(1.75,cq.Vector(11.2,0,6)))
    return body.clean()


def scan_mesh(body, sigma=0., seed=49, *, refine=True):
    mesh = body_mesh(body,.045)
    mesh.merge_vertices(digits_vertex=6)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if refine:
        for _ in range(2):
            mesh = mesh.subdivide()
    if sigma:
        offsets = np.random.default_rng(seed).normal(0,sigma,len(mesh.vertices))
        mesh.vertices += offsets[:,None]*mesh.vertex_normals
    return mesh
