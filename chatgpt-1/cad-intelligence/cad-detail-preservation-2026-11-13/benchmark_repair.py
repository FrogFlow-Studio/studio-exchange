"""Run task 44 against the task 39 fixture set."""
import argparse
import json
import os
from pathlib import Path

import benchmark_followup
import repair_geodesic
from patch_cache import CachedBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    os.environ["CAD_PATCH_WORKERS"] = str(args.workers)
    args.work.mkdir(parents=True, exist_ok=True)
    benchmark_followup.repair_adaptive = repair_geodesic
    builder = CachedBuilder() if args.cache else None
    if builder:
        benchmark_followup.stock = builder
    rows = benchmark_followup.folds(args.work)
    if builder:
        rows.append({"cache_hits":builder.hits,"cache_misses":builder.misses})
    (args.work/"task44.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                                        encoding="utf8")


if __name__ == "__main__":
    main()
