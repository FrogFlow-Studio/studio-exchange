"""Reconstruct only the scan-supported cylindrical parts of an island."""
import cadquery as cq
from OCP.Standard import Standard_Failure
from OCP.StdFail import StdFail_NotDone
from shapely.geometry import Polygon

from cylindrical_islands import _fit


def _runs(sections,dz):
    run = []
    for row in sections:
        if run and row[0]-run[-1][0]>1.5*dz:
            yield run
            run = []
        run.append(row)
    if run:
        yield run


def _circle(rings):
    return _fit(rings,diameter_step=.05,sample_segments=True)


def partial_island(sections, low, high, dz, make_loft, make_prism, ring_points, n):
    """Return a mixed solid and its measured diameters, or defer to stock."""
    if len(sections)<3 or len({z for z,_ in sections})!=len(sections):
        return None
    try:
        return _build(sections,low,high,dz,make_loft,make_prism,ring_points,n)
    except (RuntimeError,ValueError,Standard_Failure,StdFail_NotDone):
        return None


def _build(sections,low,high,dz,make_loft,make_prism,ring_points,n):
    def cylinder(fit,a,b):
        center,diameter = fit
        return cq.Solid.makeCylinder(diameter/2,b-a,cq.Vector(*center,a))

    def outer_piece(rows,a,b):
        reference = rows[len(rows)//2][1]
        outline = Polygon(reference.exterior)
        if all(outline.hausdorff_distance(Polygon(p.exterior))<.11 for _,p in rows):
            return make_prism(ring_points(reference.exterior.coords,n,.03),a,b)
        rings = [(ring_points(p.exterior.coords,n,.03),z) for z,p in rows]
        if len(rings)<=2:
            return make_prism(rings[len(rings)//2][0],a,b)
        return make_loft([(rings[0][0],a),*rings[1:-1],(rings[-1][0],b)],ruled=True)

    spans = []
    circular = [(z,p) for z,p in sections if _circle([p.exterior]) is not None]
    for rows in _runs(circular,dz):
        first,last = rows[0][0],rows[-1][0]
        if last-first<1.-1e-8:
            continue
        fit = _circle([p.exterior for _,p in rows])
        if fit is not None:
            # Keep material represented by the adjacent noncircular slice.
            a = low if first==sections[0][0] else first-dz/2
            b = high if last==sections[-1][0] else last+dz/2
            spans.append((first,last,a,b,fit))

    bores = []
    hole_rows = [(z,p) for z,p in sections if len(p.interiors)==1]
    if (hole_rows and all(len(p.interiors)<=1 for _,p in sections)
            and len(list(_runs(hole_rows,dz)))==1):
        fit = _circle([p.interiors[0] for _,p in hole_rows])
        if fit is not None and len(hole_rows)>=3:
            first,last = hole_rows[0][0],hole_rows[-1][0]
            # A hole open in the topmost slice goes through the island cap.
            # Its bottom does not inherit the extra embed depth.
            bores.append((first-dz/2,high+.02 if last==sections[-1][0] else last+dz/2,fit))
    if not spans and not bores:
        return None
    if not spans:
        body = outer_piece(sections,low,high)
    else:
        pieces = []
        previous = low
        remaining = sections
        for first,last,a,b,fit in spans:
            lower = [(z,p) for z,p in remaining if z<first]
            if lower:
                pieces.append(outer_piece(lower,previous,a+.015))
            pieces.append(cylinder(fit,a,b))
            previous = b-.015
            remaining = [(z,p) for z,p in remaining if z>last]
        if remaining:
            pieces.append(outer_piece(remaining,previous,high))
        body = pieces[0].fuse(*pieces[1:]).clean() if len(pieces)>1 else pieces[0]
    if bores:
        for a,b,fit in bores:
            drilled = body.cut(cylinder(fit,a,b)).clean()
            if drilled.Volume()>=body.Volume():
                return None
            body = drilled
    elif hole_rows or any(p.interiors for _,p in sections):
        # Defer to the regular path rather than fill an unrecognized hole.
        return None
    if not body.isValid() or len(body.Solids())!=1:
        return None
    return body,dict(outer_spans=[dict(z_mm=[a,b],diameter_mm=fit[1]) for _,_,a,b,fit in spans],
                     bores=[dict(z_mm=[a,b],diameter_mm=fit[1]) for a,b,fit in bores])
