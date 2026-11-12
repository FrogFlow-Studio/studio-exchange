"""Запуск рабочего окна из ярлыка установленной программы (без чёрного окна консоли).

Если программа уже запущена, просто открывает её страницу в браузере. Иначе поднимает
локальный сервер на первом свободном порту 8770–8779 и открывает страницу. Ошибки пишутся
в launcher.log в папке данных (%LOCALAPPDATA%\\FrogFlow\\CAD Workspace).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback
import urllib.request
import webbrowser

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "FrogFlow" / "CAD Workspace"
PORTS = range(8770, 8780)


def running(port: int) -> bool:
    """Отвечает ли на порту именно наше рабочее окно (а не чужая программа)."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/config", headers={"Host": f"127.0.0.1:{port}"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return "ai_budget_usd" in json.load(resp)
    except Exception:  # noqa: BLE001 — любой отказ значит «не наше»
        return False


def free(port: int) -> bool:
    import socket

    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_running():
    return next((p for p in PORTS if running(p)), None)


def main() -> None:
    import threading
    import time

    DATA.mkdir(parents=True, exist_ok=True)
    lock = DATA / "starting.lock"
    # двойной щелчок по ярлыку: первая копия ещё грузит библиотеки (до полуминуты), вторая ждёт её
    try:
        waiting = time.time() - lock.stat().st_mtime < 90
    except OSError:
        waiting = False
    deadline = time.time() + (90 if waiting else 0)
    while True:
        port = find_running()
        if port:
            webbrowser.open(f"http://127.0.0.1:{port}/")
            return
        if time.time() >= deadline:
            break
        time.sleep(1)
    port = next((p for p in PORTS if free(p)), None)
    if port is None:
        raise RuntimeError("порты 8770–8779 заняты другими программами")
    lock.write_text(str(os.getpid()), encoding="utf-8")

    def unlock():
        for _ in range(180):
            if running(port):
                break
            time.sleep(0.5)
        lock.unlink(missing_ok=True)

    threading.Thread(target=unlock, daemon=True).start()
    sys.path.insert(0, str(HERE))
    try:
        import server
    except BaseException:
        lock.unlink(missing_ok=True)
        raise
    sys.argv = [str(HERE / "server.py"), "--port", str(port), "--data", str(DATA)]
    server.main()


if __name__ == "__main__":
    # у pythonw нет консоли: всё, что напечатали бы, уходит в журнал
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        log = open(DATA / "launcher.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    except OSError:
        log = None
    try:
        main()
    except Exception:  # noqa: BLE001
        if log:
            log.write(traceback.format_exc())
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, "CAD Intelligence не запустился:\n\n" + traceback.format_exc()[-800:]
                                             + f"\n\nЖурнал: {DATA / 'launcher.log'}", "CAD Intelligence", 0x10)
        except Exception:  # noqa: BLE001
            raise
