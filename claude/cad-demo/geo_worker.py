"""Попытка 0 в отдельном процессе, чтобы на сложной детали её можно было остановить по времени.

python geo_worker.py скан_выровненный.stl sheet.json 1|0 ответ.json
"""
import json
import sys
from pathlib import Path

import trimesh

import geometric

scan = trimesh.load(sys.argv[1], force="mesh")
sheet = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
code, meta = geometric.build(scan, sheet, restore=sys.argv[3] == "1")
Path(sys.argv[4]).write_text(json.dumps({"code": code, "meta": meta}, ensure_ascii=False), encoding="utf-8")
