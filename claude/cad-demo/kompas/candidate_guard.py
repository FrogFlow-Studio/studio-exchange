"""Compare a candidate and the stock 0с body against the same aligned scan."""
from pathlib import Path
from tempfile import TemporaryDirectory
from math import isfinite

import trimesh


def compare_candidate(scan, stock, candidate, tol=.3, allowance=.005):
    from compare import ScanComparison, compare
    from stl_out import write_stl_checked
    from OCP.Standard import Standard_Failure
    from OCP.StdFail import StdFail_NotDone

    try:
        if not candidate.isValid() or len(candidate.Solids())!=1:
            return False, {"reason":"invalid_candidate"}
        prepared = ScanComparison(scan)
        audit = {}
        with TemporaryDirectory() as temporary:
            for name,body in (("stock",stock),("candidate",candidate)):
                path = Path(temporary)/f"{name}.stl"
                ratio,_ = write_stl_checked(body.wrapped,path,.04,False,.4)
                if not isfinite(ratio) or abs(ratio-1)>.05:
                    return False, {"reason":f"{name}_mesh_volume", "ratio":ratio}
                mesh = trimesh.load(path,force="mesh")
                report = compare(scan,mesh,tol,seed=59,prepared=prepared)
                audit[name] = [report["within_tol_share"],round(1-report["added_surface_share"],4)]
        accept = all(a >= b-allowance-1e-9 for a,b in zip(audit["candidate"],audit["stock"]))
        return accept, {**audit,"reason":"coverage_ok" if accept else "coverage_regression"}
    except (RuntimeError,ValueError,Standard_Failure,StdFail_NotDone,OSError) as error:
        return False, {"reason":"comparison_failed","error":str(error)}
