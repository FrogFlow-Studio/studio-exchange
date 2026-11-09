"""Попытка 0 в отдельном процессе, чтобы на сложной детали её можно было остановить по времени.

python geo_worker.py скан_выровненный.stl sheet.json 1|0 ответ.json [walls]
"""
import json
import sys
from pathlib import Path

import trimesh

import geometric

scan = trimesh.load(sys.argv[1], force="mesh")
sheet = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if len(sys.argv) > 5 and sys.argv[5] == "walls":
    import walls

    import pickle

    ws = walls.walls(scan)
    code = walls.draft_code(ws)
    # те же стенки отдаст инструмент walls для ИИ, чтобы не считать их второй раз
    Path(sys.argv[4]).with_suffix(".pkl").write_bytes(pickle.dumps(ws))
    meta = {"plan": f"черновик из стенок: найдено {len(ws)} стенок, каждая выдавлена по контуру",
            "restored": [], "assumptions": ["деталь тонкостенная: собрана из пластин"]}
else:
    code, meta = geometric.build(scan, sheet, restore=sys.argv[3] == "1")
Path(sys.argv[4]).write_text(json.dumps({"code": code, "meta": meta}, ensure_ascii=False), encoding="utf-8")
