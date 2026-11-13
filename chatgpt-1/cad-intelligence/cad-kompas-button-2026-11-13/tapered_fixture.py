"""Independent CAD reference for a tapered seat-button shell."""
import cadquery as cq
import numpy as np
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections


def outline(z, inset=0., count=96):
    t = np.linspace(0,2*np.pi,count,endpoint=False)
    y = (24.5-inset)*np.sign(np.sin(t))*abs(np.sin(t))**.52
    longitudinal = (y+24.5)/49
    width = 11.5-4*longitudinal-inset
    flare = .75*(z/13)*longitudinal**2
    x = (width+flare)*np.sign(np.cos(t))*abs(np.cos(t))**.52 + .25*longitudinal**2
    return np.column_stack((x,y))


def loft(levels, inset=0.):
    tool = BRepOffsetAPI_ThruSections(True,False,1e-6)
    tool.SetMaxDegree(2)
    tool.CheckCompatibility(False)
    for z in levels:
        points = [cq.Vector(float(x),float(y),float(z)) for x,y in outline(z,inset)]
        tool.AddWire(cq.Wire.assembleEdges([cq.Edge.makeSpline(points,periodic=True)]).wrapped)
    tool.Build()
    return cq.Solid(tool.Shape())


def end_edge(body, *, top=False):
    edges = [e for e in body.Edges() if e.Length()>70]
    return (max if top else min)(edges,key=lambda edge:edge.Center().z)


def sharp_outer():
    from kompas_build import cut_above

    bottom = np.array([-.26,0.,0.,.00045,0.,.00034])
    # Extend the loft below the lowest point of the curved floor, so the
    # bottom cutter defines the whole face before rounding its outer edge.
    outer = loft([-.6,3.,7.,10.,13.])
    return cut_above(outer,bottom,(-12.,-25.,12.,25.),below=True)


def tapered_button(radius=.48, *, top_radius=.3):
    from kompas_build import cut_above

    outer = sharp_outer()
    if radius:
        outer = outer.fillet(radius,[end_edge(outer)])
    # The wall is slightly flared. A stepped inset above the floor alters
    # the pocket profile without changing the outer contour.
    cavity = loft([1.55,5.,8.,14.],1.3)
    lower = loft([1.55,2.7,3.15],1.95)
    upper = cavity.cut(cq.Workplane().box(60,70,10).translate((0,0,-1.9)).val())
    pocket = lower.fuse(upper)
    for y in (-8.,8.):
        pocket = pocket.cut(cq.Workplane().box(40,1.3,16).translate((0,y,7)).val())
    body = outer.cut(pocket)
    for x,y in ((-3.8,-15.),(3.,15.)):
        boss = (cq.Workplane("XY").center(x,y).circle(3.).circle(1.5)
                .extrude(8.6).translate((0,0,1.35)).val())
        if x<0:
            boss = boss.cut(cq.Workplane().box(4,1.,3.).translate((x+2.3,y,8.6)).val())
        body = body.fuse(boss)
    top = np.array([12.1,0.,.015,0.,0.,.0008])
    body = cut_above(body,top,(-12.,-25.,12.,25.))
    if top_radius:
        edge = end_edge(body,top=True)
        body = body.fillet(top_radius,[edge])
    body = body.clean()
    if not body.isValid() or len(body.Solids())!=1:
        raise ValueError("invalid tapered fixture")
    return body
