"""Кандидат CADENA: ИИ-модель (Qwen2-VL-2B) по снимкам скана пишет программу CAD по шагам
(выдавливание, вращение, отверстие…), после каждого шага сверяя результат со сканом.

Код CADENA (MIT) клонируют рядом: github.com/kulibinai/cadena, путь — в CADENA_DIR. Веса (MIT)
скачиваются с Hugging Face (kulibinai/cadena, папка rl, ~4 ГБ) при первом запуске или берутся из
CADENA_WEIGHTS. Оригинал раздаёт модель через vLLM; без него поднимаем cadena_shim/server.py —
та же генерация через transformers на видеокарте NVIDIA или на процессоре. Готовый сервер
можно указать в CADENA_URL.

python cadena_worker.py скан.stl папка ответ.json [секунд]

В ответе: STEP в мм (рядом с ответом) и редактируемая программа построения (*.cadena.py).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
HF_REPO = "kulibinai/cadena"


def cadena_dir() -> Path | None:
    for p in (os.environ.get("CADENA_DIR"), HERE / "ext" / "cadena", HERE.parent / "cadena"):
        if p and (Path(p) / "inference" / "main.py").exists():
            return Path(p)
    return None


def weights_dir() -> Path:
    p = os.environ.get("CADENA_WEIGHTS")
    if p and (Path(p) / "config.json").exists():
        return Path(p)
    local = HERE / "ext" / "cadena_ckpt"
    if not (local / "rl" / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(HF_REPO, allow_patterns=["rl/*"], local_dir=str(local))
    return local / "rl"


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    t = path.read_text(encoding="utf-8")
    for a, b in pairs:
        if a not in t:
            print(f"CADENA: правка не применилась в {path.name}: {a[:60]}", flush=True)
        t = t.replace(a, b)
    path.write_text(t, encoding="utf-8")


def windows_copy(src: Path, work: Path) -> Path:
    """Копия CADENA с правками для Windows: там нет fork и /tmp (на Linux можно включить CADENA_SPAWN=1)."""
    run = work / "cadena_run"
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(src, run, ignore=shutil.ignore_patterns(".git", "__pycache__", "work_dirs"))
    patch(run / "inference" / "main.py", [("mp.set_start_method('fork', force=True)",
                                            "mp.set_start_method('spawn', force=True)")])
    # клиент OpenAI держит блокировку и не переносится в процесс spawn: создаём его уже в процессе
    patch(run / "inference" / "cad_worker.py", [
        ('self.openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")',
         'self.base_url = base_url\n        self.openai_client = None'),
        ("        self.plotter = Plotter()\n",
         '        self.plotter = Plotter()\n        self.openai_client = OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY")\n'),
    ])
    # снимки рисуются без окна через OSMesa, а на Windows его нет: там обычный OpenGL с off_screen
    patch(run / "visualization.py", [("os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'",
                                      "if os.name != 'nt':\n    os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'")])
    patch(run / "rl" / "cad_pool.py", [('get_context("forkserver")', 'get_context("spawn")'),
                                       ('ctx = get_context("fork")', 'ctx = get_context("spawn")')])
    patch(run / "dataset" / "create_hf_dataset.py", [('_TMP_DIR = f"/tmp/{getpass.getuser()}"',
                                                      '_TMP_DIR = os.path.join(__import__("tempfile").gettempdir(), getpass.getuser())')])
    return run


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_server(url: str, proc: subprocess.Popen | None, limit: float) -> None:
    end = time.time() + limit
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            raise SystemExit("сервер модели CADENA не запустился")
        try:
            urllib.request.urlopen(url + "/v1/models", timeout=5).read()
            return
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise SystemExit("сервер модели CADENA не ответил вовремя")


def candidates(res: Path) -> list[tuple[float, str]]:
    """(совпадение по объёму, программа) для каждого шага, у которого построилась деталь."""
    out = []
    for f in glob.glob(str(res / "**" / "*.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or "prefixes" not in d:
            continue
        ious = d.get("step_iou") or []
        for i in range(min(len(d.get("mesh_paths") or []), len(ious), len(d["prefixes"]) - 1)):
            if ious[i] is not None:
                out.append((float(ious[i]), d["prefixes"][i + 1]))
    return sorted(out, key=lambda t: -t[0])


def main() -> None:
    scan, work, answer = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 900
    t_start = time.time()
    src = cadena_dir()
    if src is None:
        raise SystemExit("CADENA не найдена: склонируйте github.com/kulibinai/cadena и задайте CADENA_DIR")
    import torch
    import trimesh

    gpu = torch.cuda.is_available()
    work.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" or os.environ.get("CADENA_SPAWN") == "1":
        src = windows_copy(src, work)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]),
               PYTHONIOENCODING="utf-8")
    server, url = None, os.environ.get("CADENA_URL")
    try:
        if not url:
            port = free_port()
            url = f"http://127.0.0.1:{port}"
            log = open(work / "server.log", "w", encoding="utf-8")
            server = subprocess.Popen([sys.executable, str(HERE / "cadena_shim" / "server.py"), str(weights_dir()), str(port)],
                                      env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_server(url, server, 600)
        # снимки скана с нескольких сторон — то, что видит модель
        meshes = work / "meshes"
        if meshes.exists():
            shutil.rmtree(meshes)
        meshes.mkdir()
        shutil.copy(scan, meshes / "part.stl")
        subprocess.run([sys.executable, "dataset/create_hf_dataset.py"], cwd=src, check=True, timeout=600,
                       env=dict(env, NUM_PROC="1", MESH_ROOT=str(meshes), OUT_DIR=str(work / "ds")))
        res = work / "res"
        if res.exists():
            shutil.rmtree(res)
        # на видеокарте — поиск по нескольким вариантам на каждом шаге, на процессоре — один вариант
        mode = os.environ.get("CADENA_MODE") or ("beam_topk" if gpu else "greedy")
        width, exp = ("2", "4") if mode != "greedy" else ("1", "12")
        left = budget - (time.time() - t_start)
        if left < 60:
            raise SystemExit("CADENA: не осталось времени на построение")
        cmd = [sys.executable, "inference/main.py", "--batch_size", "100", "--base_urls", url,
               "--inference_folder", str(res / "x"), "--num_workers_per_gpu", "1", "--model_name", "cadena",
               "--max_rollout_steps", "20", "--max_tokens", "500", "--dataset_path", str(work / "ds" / "meshes"),
               "--temperature", "1.0", "--beam_temperature", "1.0", "--num_expansions", exp,
               "--beam_width", width, "--mode", mode, "--get_metrics", "gms"]
        try:
            subprocess.run(cmd, cwd=src, env=env, timeout=left, check=False)
        except subprocess.TimeoutExpired:
            print("CADENA: время вышло, беру лучшее из готового", flush=True)
    finally:
        if server is not None:
            server.kill()
    cands = candidates(work / "res")
    if not cands:
        raise SystemExit("CADENA не построила ни одной детали по этому скану")
    iou, prog = cands[0]
    # программа CADENA — в координатах, где деталь вписана в куб 200; возвращаем в мм скана
    m = trimesh.load(scan, force="mesh")
    centre = [float(x) for x in (m.bounds[0] + m.bounds[1]) / 2]
    k = float(max(m.extents)) / 200.0
    sys.path.insert(0, str(src))
    ns: dict = {}
    cwd = os.getcwd()
    os.chdir(src)
    try:
        exec(prog, ns)  # noqa: S102 — программа той же модели, что CADENA сама только что исполняла
    finally:
        os.chdir(cwd)
    import cadquery as cq

    r = ns["r"]
    shape = r.val() if hasattr(r, "val") else r
    if hasattr(r, "vals") and len(r.vals()) > 1:
        shape = cq.Compound.makeCompound(r.vals())
    shape = shape.scale(k).translate(cq.Vector(*centre))
    step = answer.with_suffix(".step")
    cq.exporters.export(cq.Workplane().add(shape), str(step))
    editable = answer.with_suffix(".cadena.py")
    editable.write_text(
        "# Программа построения CADENA (нужна библиотека cadgen из github.com/kulibinai/cadena).\n"
        "# Размеры — в кубе 200; в конце деталь переводится в мм скана.\n"
        + prog + f"\nresult = r.val().scale({k!r}).translate(cq.Vector(*{centre!r}))\n", encoding="utf-8")
    steps = prog.count("\nr=")
    meta = {"plan": f"CADENA ({'видеокарта' if gpu else 'процессор'}, {mode}): программа из {steps} операций, "
                    f"совпадение по объёму у самой модели {iou:.0%}",
            "restored": [], "assumptions": ["ИИ-модель CADENA, лицензия кода и весов MIT",
                                            f"редактируемая программа: {editable.name}"]}
    code = f"import cadquery as cq\nresult = cq.importers.importStep({str(step)!r}).val()\n"
    answer.write_text(json.dumps({"code": code, "meta": meta}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
