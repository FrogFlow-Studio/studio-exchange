"""Reuse identical smooth patches between candidate B-rep builds."""
from collections import OrderedDict
from hashlib import sha256
import importlib.util
from pathlib import Path

import numpy as np
import planar_brep


class CachedBuilder:
    """Load an isolated builder so the shared CAD-demo module stays untouched."""

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        spec = importlib.util.spec_from_file_location("isolated_planar_brep", Path(planar_brep.__file__))
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original = self.module._patches
        self.module._patches = self._patches

    def _patches(self, vertices, faces, parts, tol, edge_tol, deadline):
        keys = {}
        missing = []
        for part in parts:
            ids, local = np.unique(faces[part], return_inverse=True)
            data = np.ascontiguousarray(vertices[ids]).tobytes() + local.tobytes()
            key = (tol, edge_tol, len(ids), len(part), sha256(data).digest())
            keys[int(part.min())] = key
            if key in self.cache:
                self.hits += 1
            else:
                self.misses += 1
                missing.append(part)
        fresh = {int(part.min()):(face,reason) for part,face,reason in
                 self.original(vertices,faces,missing,tol,edge_tol,deadline)}
        for part in parts:
            index = int(part.min())
            key = keys[index]
            if key in self.cache:
                self.cache.move_to_end(key)
                face, reason = self.cache[key]
            else:
                face, reason = fresh.get(index,(None,None))
                if face is not None or reason is not None:
                    self.cache[key] = (face,reason)
                    if len(self.cache) > self.capacity:
                        self.cache.popitem(last=False)
            # Sewing may change edge tolerances on the supplied topology.
            yield part, _copy_face(face), reason

    def __call__(self, *args, **kwargs):
        body, audit = self.module.build(*args, **kwargs)
        return body, {**audit,"patch_cache_hits":self.hits,"patch_cache_misses":self.misses}


def _copy_face(face):
    if face is None:
        return None
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.TopoDS import TopoDS

    return TopoDS.Face_s(BRepBuilderAPI_Copy(face).Shape())
