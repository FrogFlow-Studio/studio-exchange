import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import trimesh
sys.path.insert(0,str(Path('claude/cad-demo').resolve()))
import geo_worker
import nurbs
import surface

def main():
    raw = trimesh.creation.icosphere(subdivisions=5)
    raw.apply_scale([32, 24, 16])
    p = Path('speed-ellipsoid.stl')
    raw.export(p)
    scan = trimesh.load(p, force='mesh')
    mesh_results = []
    for i, cached in enumerate((False, True, True)):
        os.environ['CAD_SURFACE_CACHE'] = str(int(cached))
        start = time.perf_counter()
        m = geo_worker._clean_surface(scan, p, Path('surface-result.json'))
        mesh_results.append((round(time.perf_counter()-start,3),m))
    assert all(np.array_equal(m.vertices,mesh_results[0][1].vertices) and np.array_equal(m.faces,mesh_results[0][1].faces) for _,m in mesh_results)
    print(json.dumps({'cleanup_seconds_disabled_write_hit':[v[0] for v in mesh_results],'mesh_identical':True}),flush=True)
    mesh=surface.flatten_planes(mesh_results[-1][1])
    baseline=None
    for jobs in (1,4):
        os.environ['CAD_JOBS'] = str(jobs)
        fitted_times=[]
        fn=nurbs._fit_patches
        def time_fit(grids, basis):
            t=time.perf_counter()
            v=fn(grids,basis)
            fitted_times.append(round(time.perf_counter()-t,3))
            return v
        nurbs._fit_patches=time_fit
        start=time.perf_counter()
        body,info=nurbs.smooth_body(mesh,scan)
        elapsed=time.perf_counter()-start
        nurbs._fit_patches=fn
        assert body is not None and body.isValid() and len(body.Solids())==1
        cover=nurbs.cover_share(body,nurbs._Projector(mesh),0.3)
        result={'jobs':jobs,'elapsed_s':round(elapsed,3),'fit_s':fitted_times,'faces':info['faces'],'near':info['near'],'score':info['score'],'cover':cover,'volume':body.Volume()}
        print(json.dumps(result),flush=True)
        if baseline is None:
            baseline=result
        else:
            for k in ('faces','near','score','cover'):
                assert baseline[k]==result[k], k
            assert abs(baseline['volume']-result['volume'])<1e-6

if __name__=='__main__':
    main()
