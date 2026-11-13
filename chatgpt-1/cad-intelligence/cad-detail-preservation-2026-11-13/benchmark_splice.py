"""Compare local face substitution on the six task 40 STEP results."""
import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
from compare import compare
from nurbs import body_mesh
from local_faces import splice


def run(work):
    rows=[]
    for part in ("angle","block"):
        for noise in (0.,.1):
            stock=cq.importers.importStep(str(work/f"{part}-{noise}-stock.step")).val()
            candidate=cq.importers.importStep(str(work/f"{part}-{noise}-40.step")).val()
            cylinders=[face for face in candidate.Faces() if face.geomType()=="CYLINDER"]
            samples=[p.toTuple() for face in cylinders for p in face.tessellate(.1)[0]]
            samples.extend(edge.positionAt(.5).toTuple() for face in cylinders for edge in face.Edges())
            result,audit=splice(stock,candidate,np.asarray(samples),1.2)
            if not audit["accepted"] and samples:
                result, expanded=splice(stock,candidate,np.asarray(samples),1.2,include_incident=True)
                audit={**expanded,"narrow_scope":audit}
            out=work/f"{part}-{noise}-45.step"
            cq.exporters.export(result,str(out))
            restored=cq.importers.importStep(str(out)).val()
            reference=body_mesh(stock,.04)
            actual=body_mesh(restored,.04)
            reference.merge_vertices()
            actual.merge_vertices()
            # On rollback the entire stock body is the outside control.
            cov=compare(reference,actual,.3,n=1500,seed=45)
            rows.append(dict(case=part,noise=noise,valid=restored.isValid(),result_cylinders=sum(
                f.geomType()=="CYLINDER" for f in restored.Faces()),
                coverage_to_stock=[cov["within_tol_share"],1-cov["added_surface_share"]],**audit))
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows=run(a.work)
    a.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(rows)
