"""Noisy three-loft pocket with two shallow joins and a deliberate ledge."""
import json
from pathlib import Path
import sys
from time import monotonic

import cadquery as cq
import numpy as np

from benchmark_71_72 import noise
from benchmark_cleanliness import round_trip
from cad_cleanliness import cad_cleanliness
from kompas_build import _wire
from pocket_steps import clean_pocket_steps


def pocket_wire(radius,z):
    angle = np.arange(32)*2*np.pi/32
    ellipse = np.column_stack((radius*np.cos(angle),.85*radius*np.sin(angle)))
    return _wire(ellipse,z)


def fixture():
    outer = cq.Solid.makeBox(12,12,8,cq.Vector(-6,-6,0))
    profiles = [(3.,3.,1.,2.6),(3.04,3.04,2.6,4.1),(3.015,3.015,4.1,6.)]
    # Last cut gives the 0.3 mm feature independently of the three wall lofts.
    ledge = cq.Solid.makeLoft([pocket_wire(3.315,6.),pocket_wire(3.315,9.)],ruled=True)
    cuts = [cq.Solid.makeLoft([pocket_wire(a,low),pocket_wire(b,high)],ruled=True)
            for a,b,low,high in profiles]
    stock = outer.cut(*cuts,ledge).clean()
    nominal = cq.Solid.makeLoft([pocket_wire(3.,1.),pocket_wire(3.015,6.)],ruled=True)
    truth = outer.cut(nominal,ledge).clean()
    return stock,truth


def run():
    stock,truth = fixture()
    scan = noise(truth,76)
    started = monotonic()
    candidate,audit = clean_pocket_steps(stock,scan=scan)
    elapsed = monotonic()-started
    assert audit["found"]==2 and audit["removed"]==2,audit
    assert elapsed<60
    result = round_trip(candidate)
    for body in (stock,result):
        solid = body.Solids()[0]
        assert solid.isInside(cq.Vector(3.15,0,5.95))
        assert not solid.isInside(cq.Vector(3.15,0,6.05))
    old,new = (cad_cleanliness(body) for body in (stock,result))
    assert old["narrow_faces"]==2 and new["narrow_faces"]==0,(old,new)
    untouched,timeout = clean_pocket_steps(stock,scan=scan,timeout_s=.01)
    assert untouched is stock and timeout["reasons"]=={"timeout":1},timeout
    return dict(audit=audit,seconds=round(elapsed,3),
                faces=[old["faces"],new["faces"]],narrow_faces=[old["narrow_faces"],new["narrow_faces"]],
                preserved_step_mm=.30,timeout=timeout)


if __name__=="__main__":
    report = json.dumps(run(),indent=2)
    print(report,flush=True)
    Path(sys.argv[1]).write_text(report+"\n",encoding="utf8")
