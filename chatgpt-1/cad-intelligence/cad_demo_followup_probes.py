"""Independent follow-up probes for snapshot 8b791e6; no LLM/GPU calls.

python cad_demo_followup_probes.py --source COPIED_CAD_DEMO --out EVIDENCE_DIR
Run with the dependencies of the reviewed demo. Use a disposable source copy.
These are observations on gates and metrics, not a reliability benchmark.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(args.source.resolve()))
    import cadquery as cq
    import compare
    import gates
    import holes
    import llm
    import numpy as np
    import scan_analysis as sa
    import trimesh

    source = (args.source / "demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    good = next(n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == "good")
    env = {"a": SimpleNamespace(target=.95)}
    exec(compile(ast.Module(body=[good], type_ignores=[]), "reviewed_good", "exec"), env)

    def check(name, scan_shape, candidate):
        src = args.out / f"{name}_source.stl"
        dst = args.out / f"{name}_candidate.stl"
        cq.exporters.export(scan_shape, str(src), tolerance=.02)
        cq.exporters.export(candidate, str(dst), tolerance=.02)
        sheet, scan = sa.measurement_sheet(str(src))
        model = trimesh.load(dst, force="mesh")
        model.apply_transform(np.array(sheet["transform_from_scan_4x4"]))
        run = {"ok": True, "valid": bool(candidate.val().isValid()), "solids": len(candidate.val().Solids())}
        failed = gates.check(run, model, sheet, scan)
        cmp = compare.compare(scan, model, tol=.3)
        hole_rows = holes.check(scan, model)
        return {"probe": name, "gates": failed,
                "good": env["good"]({"cmp": cmp, "gates": failed, "run": run}),
                "within_tol_share": cmp["within_tol_share"], "outside_share": cmp["outside_share"],
                "holes_report": hole_rows}

    box = cq.Workplane("XY").box(100, 100, 10, centered=(True, True, False))
    vertical = box.faces(">Z").workplane().hole(2)
    side_cut = cq.Solid.makeCylinder(1, 102, cq.Vector(-51, 0, 5), cq.Vector(1, 0, 0))
    horizontal = box.cut(side_cut)
    short_boss = box.union(cq.Workplane("XY").workplane(offset=10).rect(2, 2).extrude(1.5))
    tall_boss = box.union(cq.Workplane("XY").workplane(offset=10).rect(2, 2).extrude(100))
    results = [check("vertical_hole_filled", vertical, box),
               check("horizontal_hole_filled", horizontal, box),
               check("unwanted_short_boss", box, short_boss),
               check("unwanted_tall_boss", box, tall_boss)]
    tiny_path = args.out / "tiny_mm.stl"
    trimesh.creation.box(extents=[1, .8, .5]).export(tiny_path)
    sheet, mesh = sa.measurement_sheet(str(tiny_path))
    results.append({"probe": "tiny_mm_preserved", "extents_mm": mesh.extents.tolist(),
                    "diagnostics": sheet["diagnostics"]})

    class FakeClient:
        def __init__(self):
            self.messages = self
            self.requests = []

        def create(self, **kw):
            self.requests.append(kw)
            return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(
                input_tokens=1000, output_tokens=32000,
                cache_creation_input_tokens=0, cache_read_input_tokens=0),
                content=[SimpleNamespace(type="text", text="offline test")])

    fake = FakeClient()
    provider = llm.Anthropic("sonnet-review", client=fake)
    provider.budget_usd = .3
    provider.chat([{"role": "user", "content": "test"}])
    results.append({"probe": "soft_budget", "fake_requests": len(fake.requests),
                    "budget": provider.budget_usd, "estimate_using_code_prices": provider.cost_usd(),
                    "requested_max_tokens": fake.requests[0]["max_tokens"],
                    "note": "Synthetic usage, no provider called or charged."})
    payload = {"snapshot": "8b791e6", "platform": platform.platform(), "python": sys.version,
               "llm_calls": 0, "results": results}
    (args.out / "followup-results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
