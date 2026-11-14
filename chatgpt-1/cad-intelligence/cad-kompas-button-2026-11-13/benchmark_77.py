"""Regression checks for a sparse window flank and a pocket wall spanning a join."""
import json
from pathlib import Path
import sys
from time import monotonic

import cadquery as cq
import numpy as np
from shapely.geometry import Polygon

from benchmark_71_72 import noise
from benchmark_cleanliness import round_trip
from benchmark_74 import run as rounded_fixture
from cad_cleanliness import cad_cleanliness, horizontal_planes
from pocket_steps import _ledges, clean_pocket_steps
from windowed_islands import _window_polygons
from hooked_islands import _sampled_flanks


def sparse_flanks():
    center = np.array([1.46,-12.78])
    inner,outer = 2.225,3.75
    petals = []
    for start,end in ((27,87),(117,140),(207,267),(297,320)):
        angles = np.deg2rad(np.linspace(start,end,24))
        unit = np.column_stack((np.cos(angles),np.sin(angles)))
        # A sparse mesh can give no vertex along a flank between the two arcs.
        petals.append(Polygon(np.vstack((center+outer*unit,center+inner*unit[::-1]))))
    reference = [(float(z),petals) for z in np.arange(8.94,10.04,.2)]
    assert _window_polygons(reference,center,inner,outer) is None
    recovered = _sampled_flanks(reference,center,inner,outer)
    assert recovered is not None and len(recovered)==4
    return dict(levels=len(reference),windows=len(recovered),fit="angular_crossings")


def spanning_wall():
    vector = cq.Vector
    outer = cq.Solid.makeBox(12,12,8,vector(-6,-6,0))
    def profile(z,r):
        return cq.Wire.makePolygon([vector(-r,-3,z),vector(3,-3,z),vector(3,3,z),vector(-r,3,z)],close=True)

    void = cq.Solid.makeLoft([profile(z,r) for z,r in ((1,2.95),(4,3.1),(8.5,3.15))])
    lower = cq.Solid.makeBox(.04,2,7.5,vector(3,1,1))
    upper = cq.Solid.makeBox(.04,2,5,vector(3,-1,4))
    stock = outer.cut(void,lower,upper).clean()
    shelves,edges,_,_ = _ledges(stock)
    assert len(shelves)==1
    shelf,walls = next(iter(shelves.items()))
    box = [face.BoundingBox() for face in walls]
    spanning = sum(b.zmin<=1.01 and b.zmax>=7.99 for b in box)
    assert spanning>=2 and len(edges[shelf])>=4
    scan = noise(stock,77)
    start = monotonic()
    out,audit = clean_pocket_steps(stock,scan=scan)
    elapsed = monotonic()-start
    assert audit["found"]==audit["removed"]==1 and elapsed<60,audit
    result = round_trip(out)
    assert not any(abs(z-4)<1e-4 for _,z in horizontal_planes(result))
    assert any(f.geomType()=="BSPLINE" and f.BoundingBox().zmax>=7.99 for f in result.Faces())
    before,after = (cad_cleanliness(body) for body in (stock,result))
    assert after["narrow_faces"]<=before["narrow_faces"],(before,after)
    return dict(audit=audit,seconds=round(elapsed,3),spanning_neighbors=spanning,
                shelf_edges=len(edges[shelf]),faces=[before["faces"],after["faces"]],
                narrow_faces=[before["narrow_faces"],after["narrow_faces"]])


if __name__=="__main__":
    report = dict(sparse_window=sparse_flanks(),wall=spanning_wall(),rounded_window=rounded_fixture(rounded=True))
    rendered = json.dumps(report,indent=2)
    print(rendered,flush=True)
    Path(sys.argv[1]).write_text(rendered+"\n",encoding="utf8")
