"""Подготовка к показу: прогнать несколько сканов подряд и сложить всё нужное в одну папку.

Для каждой детали в out/demo_pack/<имя>/ кладутся: скан, model.step, report.html и журнал
запуска; в конце — сводка summary.txt (время, итог, путь к отчёту). Так перед встречей есть
готовые отчёты, а живьём запускается только быстрая деталь.

python prepare_demo.py скан1.stl [скан2.stl ...] [--mode geometric]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    args = sys.argv[1:]
    extra = []
    if "--mode" in args:
        i = args.index("--mode")
        extra = ["--mode", args[i + 1]]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    pack = HERE / "out" / "demo_pack"
    pack.mkdir(parents=True, exist_ok=True)
    summary = []
    for scan in map(Path, args):
        name = scan.stem
        dst = pack / name
        dst.mkdir(exist_ok=True)
        runs = dst / "runs"
        print(f"=== {name}: запуск…", flush=True)
        t = time.time()
        with open(dst / "log.txt", "w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, str(HERE / "demo.py"), str(scan), "--out", str(runs), "--no-open",
                                    *extra], stdout=log, stderr=subprocess.STDOUT)
        sec = time.time() - t
        last = sorted(runs.glob("*/report.html"))
        if code == 0 and last:
            run = last[-1].parent
            for f in ("report.html", "model.step"):
                if (run / f).exists():
                    shutil.copy2(run / f, dst / f)
            shutil.copy2(scan, dst / scan.name)
            done = [ln for ln in (dst / "log.txt").read_text(encoding="utf-8").splitlines() if "в допуске" in ln]
            line = f"{name}: готово за {sec / 60:.1f} мин; {done[-1].strip() if done else ''}; отчёт {dst / 'report.html'}"
        else:
            line = f"{name}: НЕ ПОЛУЧИЛОСЬ (код {code}) за {sec / 60:.1f} мин, журнал {dst / 'log.txt'}"
        print(line, flush=True)
        summary.append(line)
    (pack / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nСводка: {pack / 'summary.txt'}")


if __name__ == "__main__":
    main()
