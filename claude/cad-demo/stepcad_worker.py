"""Кандидат StepCAD: ИИ-модель (Qwen2.5-Coder 1.5B + энкодер облака точек) пишет программу CadQuery по скану.

Код StepCAD в репозиторий не кладём (у репозитория StepCAD нет лицензии на код): его клонируют
рядом, путь — в переменной STEPCAD_DIR. Веса (Apache-2.0) скачиваются с Hugging Face при первом
запуске. Оригинал требует vLLM и видеокарту; если vLLM нет (Windows, процессор), подставляется
stepcad_shim — та же генерация через transformers на видеокарте NVIDIA или на процессоре.

python stepcad_worker.py скан.stl папка ответ.json [секунд]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def stepcad_dir() -> Path | None:
    for p in (os.environ.get("STEPCAD_DIR"), HERE / "ext" / "StepCAD", HERE.parent / "StepCAD"):
        if p and (Path(p) / "infer.py").exists():
            return Path(p)
    return None


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    t = path.read_text(encoding="utf-8")
    for a, b in pairs:
        if a not in t:
            print(f"StepCAD: правка не применилась в {path.name}: {a[:60]}", flush=True)
        t = t.replace(a, b)
    path.write_text(t, encoding="utf-8")


def main() -> None:
    scan, work, answer = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 900
    src = stepcad_dir()
    if src is None:
        raise SystemExit("StepCAD не найден: склонируйте github.com/ghadinehme/StepCAD и задайте STEPCAD_DIR")
    import torch

    gpu = torch.cuda.is_available()
    try:
        import vllm  # noqa: F401

        have_vllm = True
    except ImportError:
        have_vllm = False
    work.mkdir(parents=True, exist_ok=True)
    # запускаем копию кода с правками для процессора и Windows; на Linux с видеокартой они ничего не меняют
    run = work / "stepcad_run"
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(src, run, ignore=shutil.ignore_patterns("weights", ".git"))
    patch(run / "src" / "stepcad" / "fast_infer.py", [
        # оригинал жёстко ставит устройство 'cuda'
        ("dev = 'cuda'", "dev = 'cuda' if torch.cuda.is_available() else 'cpu'"),
        ("torch.autocast('cuda'", "torch.autocast(dev"),
        # в Windows нет coreutils timeout: срок и так держит цикл ожидания с kill()
        ("['timeout', '-s', 'KILL', '%d' % max(2, int(left - 2)), sys.executable, '-c',",
         "(['timeout', '-s', 'KILL', '%d' % max(2, int(left - 2))] if os.name != 'nt' else []) + [sys.executable, '-c',"),
    ])
    # в Windows открытый NamedTemporaryFile нельзя открыть второй раз по имени
    patch(run / "src" / "stepcad" / "refine.py", [
        ("tempfile.NamedTemporaryFile(suffix='.stl') as f:", "tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:"),
    ])
    weights = src / "weights" if (src / "weights" / "encoders.pt").exists() else None
    # на видеокарте с vLLM — как задумано авторами (128 попыток), иначе меньше попыток в отведённое время
    k = "128" if (gpu and have_vllm) else ("16" if gpu else "4")
    inf = (run / "infer.py").read_text(encoding="utf-8")
    inf = inf.replace("'--rollouts', '128', '--n0', '128', '--n', '16', '--t_max', '12'",
                      f"'--rollouts', '{k}', '--n0', '{k}', '--n', '2', '--t_max', '3'")
    inf = inf.replace("'1min': dict(budget=60, policy_frac=0.6, R=12)",
                      f"'1min': dict(budget=60, policy_frac=0.6, R=12),\n    'ours': dict(budget={budget}, policy_frac=0.8, R=4)")
    (work / "stepcad_infer.py").write_text(inf.replace("HERE = os.path.dirname(os.path.abspath(__file__))",
                                                       f"HERE = {str(run)!r}"), encoding="utf-8")
    env = dict(os.environ)
    extra = ([str(HERE / "stepcad_shim")] if not have_vllm else []) + ([str(HERE / "stepcad_shim_win")] if os.name == "nt" else [])
    if extra:
        env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")])
    inp = work / "in"
    inp.mkdir(exist_ok=True)
    shutil.copy(scan, inp / "part.stl")
    cmd = [sys.executable, str(work / "stepcad_infer.py"), "--input", str(inp / "part.stl"), "--out", str(work / "res"),
           "--mode", "ours", "--cpus", str(os.cpu_count() or 4)]
    if weights:
        cmd += ["--weights", str(weights)]
    subprocess.run(cmd, env=env, check=True)
    if not (work / "res" / "part.py").exists():
        raise SystemExit("StepCAD не написал ни одной рабочей программы для этой детали")
    prog = (work / "res" / "part.py").read_text(encoding="utf-8")
    frame = json.loads((work / "res" / "part.json").read_text(encoding="utf-8"))
    # программа StepCAD — в нормированных координатах; возвращаем в мм скана
    code = (prog + "\n_s = final_result if 'final_result' in dir() else result\n"
            "_s = _s.val() if hasattr(_s, 'val') else _s\n"
            f"result = _s.scale({frame['scale']!r}).translate(cq.Vector(*{frame['centre']!r}))\n")
    if "import cadquery as cq" not in code:
        code = "import cadquery as cq\n" + code
    meta = {"plan": f"StepCAD ({'видеокарта' if gpu else 'процессор'}, попыток: {k}): программа CadQuery по облаку точек скана",
            "restored": [], "assumptions": ["ИИ-модель StepCAD, лицензия весов Apache-2.0"]}
    answer.write_text(json.dumps({"code": code, "meta": meta}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
