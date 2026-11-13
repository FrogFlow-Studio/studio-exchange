"""Сборка «как в Компасе» из выровненного скана: python run_kompas.py scan_aligned.stl out_prefix"""
import os
import sys

import cadquery as cq
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kompas_build  # noqa: E402

if __name__ == "__main__":
    scan = trimesh.load(sys.argv[1], force="mesh")
    body, info = kompas_build.assemble(scan)
    print("годное" if body.isValid() else "негодное", info, flush=True)
    cq.exporters.export(body, sys.argv[2] + ".step")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stl_out import write_stl_checked

    print("STL: объём сетки / тела, попытка", write_stl_checked(body.wrapped, sys.argv[2] + ".stl", tol=0.1))
