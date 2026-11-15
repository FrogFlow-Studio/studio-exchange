"""Compare old and fast contours on aligned STL files in millimetres."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import shapely

from scan_analysis import measurement_sheet
from section_contours import fast_polygons, section_polygons


def measure(path):
    _, mesh = measurement_sheet(str(path), units="mm")
    low, high = mesh.bounds[:, 2]
    heights = np.arange(low + .1, high, .2)
    start = perf_counter()
    old = [section_polygons(mesh, height, fast=False) for height in heights]
    legacy_s = perf_counter() - start
    start = perf_counter()
    candidates = [fast_polygons(mesh, height) for height in heights]
    faults = sum(candidate is None for candidate in candidates)
    new = [section_polygons(mesh, height, fast=False) if candidate is None else candidate
           for height, candidate in zip(heights, candidates)]
    fast_s = perf_counter() - start
    max_delta = 0.
    for height, reference, result in zip(heights, old, new):
        assert sorted(len(p.interiors) for p in reference) == sorted(len(p.interiors) for p in result), height
        delta = shapely.union_all(reference).symmetric_difference(shapely.union_all(result)).area
        assert delta < .01, (height, delta)
        max_delta = max(max_delta, delta)
    return dict(levels=len(heights), fallback_levels=faults, legacy_s=legacy_s, fast_s=fast_s,
                speedup=legacy_s / fast_s, max_difference_mm2=max_delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.scans:
        stats = measure(path)
        print(json.dumps({path.stem: stats}, ensure_ascii=False), flush=True)
