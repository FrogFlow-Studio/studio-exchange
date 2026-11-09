"""Локальная страница демо: http://127.0.0.1:8765

Только стандартная библиотека Python. Скан загружается с этого же компьютера и
целиком не отправляется (ИИ, если включён, получает замеры и ответы на свои запросы к сетке: сечения, точки); построение идёт здесь же (demo.py в отдельном процессе).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).parent
OUT = HERE / "out"
UPLOADS = OUT / "_uploads"
PORT = 8765

JOB: dict = {"running": False, "lines": [], "report": None, "started": None}
LOCK = threading.Lock()

PAGE = """<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Скан → CAD</title><style>
:root{--bg:#fff;--fg:#1d2330;--mut:#5b6474;--line:#e3e6eb;--acc:#2f6fd6}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1100px;padding:20px 16px}
h1{font-size:24px;margin:.2em 0 .6em}.row{display:grid;grid-template-columns:340px 1fr;gap:20px}
@media(max-width:800px){.row{grid-template-columns:1fr}}
.drop{border:2px dashed #b9c3d3;border-radius:10px;padding:26px 14px;text-align:center;cursor:pointer;color:var(--mut)}
.drop.on{border-color:var(--acc);background:#eef4ff}textarea{width:100%;box-sizing:border-box;min-height:70px;font:inherit;margin:10px 0;padding:8px;border:1px solid var(--line);border-radius:6px}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:10px 16px;font:inherit;cursor:pointer}button:disabled{opacity:.5}
.small{color:var(--mut);font-size:13px}pre#log{background:#0f1420;color:#d6e2f5;padding:12px;border-radius:8px;height:330px;overflow:auto;font-size:13px;white-space:pre-wrap}
iframe{width:100%;height:1200px;border:1px solid var(--line);border-radius:8px;margin-top:16px}
.step{color:#7fd1ff}.ok{color:#8ce99a}.bad{color:#ff8a80}</style></head><body>
<h1>Скан повреждённой детали → CAD-модель</h1>
<div class=row><div>
<div id=drop class=drop>Перетащите скан сюда<br><span class=small>STL, OBJ или PLY</span></div>
<input id=file type=file accept=".stl,.obj,.ply" hidden>
<p id=fname class=small>или <a href="#" id=sample>взять тестовую деталь</a> (сломана нами, не деталь студии)</p>
<label>Единицы файла: <select id=units><option value=mm selected>мм</option><option value=cm>см</option><option value=m>м</option><option value=in>дюймы</option></select></label>
<textarea id=notes placeholder="Что сломано, что важно (необязательно)">угол плиты отколот, отверстие в нём потеряно</textarea>
<button id=go disabled>Построить модель</button>
<p class=small id=mode></p></div>
<div><pre id=log>Журнал появится здесь.</pre></div></div>
<iframe id=rep hidden></iframe>
<script>
let file=null,useSample=false;const $=id=>document.getElementById(id);
const pick=f=>{file=f;useSample=false;$('fname').textContent='Файл: '+f.name;$('go').disabled=false};
$('drop').onclick=()=>$('file').click();$('file').onchange=e=>e.target.files[0]&&pick(e.target.files[0]);
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('on')};$('drop').ondragleave=()=>$('drop').classList.remove('on');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('on');e.dataTransfer.files[0]&&pick(e.dataTransfer.files[0])};
$('sample').onclick=e=>{e.preventDefault();useSample=true;file=null;$('fname').textContent='Тестовая деталь: bracket_scan.stl';$('go').disabled=false};
fetch('/api/mode').then(r=>r.json()).then(j=>$('mode').textContent='Режим: '+j.label);
$('go').onclick=async()=>{$('go').disabled=true;$('rep').hidden=true;$('log').textContent='Запуск…';
 const h={'X-Notes':encodeURIComponent($('notes').value),'X-Units':$('units').value,'X-Name':encodeURIComponent(file?file.name:'')};
 const r=await fetch(useSample?'/api/run-sample':'/api/run',{method:'POST',headers:h,body:useSample?'':file});
 if(!r.ok){$('log').textContent=await r.text();$('go').disabled=false;return}poll()};
const color=l=>l.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/^(.*?(\\d\\/4).*)$/gm,'<span class=step>$1</span>')
 .replace(/^(.*(сошлось|готово).*)$/gm,'<span class=ok>$1</span>').replace(/^(.*(ошибка|не ответил|Traceback).*)$/gm,'<span class=bad>$1</span>');
async function poll(){const j=await (await fetch('/api/status')).json();const L=$('log');L.innerHTML=color(j.lines.join('\\n'));L.scrollTop=L.scrollHeight;
 if(j.running){setTimeout(poll,700);return}$('go').disabled=false;if(j.report){$('rep').src=j.report;$('rep').hidden=false}}
</script></body></html>"""


def start_job(scan: Path, notes: str, units: str = "mm") -> None:
    def work():
        cmd = [sys.executable, "-u", str(HERE / "demo.py"), str(scan), "--notes", notes, "--no-open", "--units", units]
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            line = line.rstrip()
            with LOCK:
                JOB["lines"].append(line)
            m = re.search(r"готово: (.*report\.html)", line)
            if m:
                rel = Path(m.group(1)).resolve().relative_to(OUT.resolve())
                with LOCK:
                    JOB["report"] = "/out/" + rel.as_posix()
        p.wait()
        with LOCK:
            JOB["running"] = False
            if p.returncode and not JOB["report"]:
                JOB["lines"].append(f"ошибка: процесс завершился с кодом {p.returncode}")

    with LOCK:
        JOB.update(running=True, lines=[], report=None, started=time.time())
    threading.Thread(target=work, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/status":
            with LOCK:
                return self.send(200, json.dumps(JOB, ensure_ascii=False).encode(), "application/json")
        if path == "/api/mode":
            import llm

            labels = {"anthropic": "живой ИИ (Anthropic)", "openai": "живой ИИ (OpenAI)",
                      "geometric": "БЕЗ ИИ: ключа нет, модель строит геометрический алгоритм"}
            return self.send(200, json.dumps({"label": labels[llm.auto_mode()]}, ensure_ascii=False).encode(),
                             "application/json")
        if path.startswith("/out/"):
            f = (OUT / path[5:]).resolve()
            if OUT.resolve() in f.parents and f.is_file():
                ctype = {".html": "text/html; charset=utf-8", ".png": "image/png"}.get(f.suffix, "application/octet-stream")
                return self.send(200, f.read_bytes(), ctype)
        self.send(404, "не найдено".encode())

    def do_POST(self):
        with LOCK:
            if JOB["running"]:
                return self.send(409, "Уже идёт построение, дождитесь конца.".encode())
        notes = unquote(self.headers.get("X-Notes", ""))
        units = self.headers.get("X-Units", "mm")
        units = units if units in ("mm", "cm", "m", "in") else "mm"
        if self.path == "/api/run-sample":
            start_job(HERE / "samples" / "bracket_scan.stl", notes, units)
            return self.send(200, b"ok")
        if self.path == "/api/run":
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > 300 * 1024 * 1024:
                return self.send(400, "Пустой или слишком большой файл.".encode())
            name = Path(unquote(self.headers.get("X-Name", "scan.stl"))).name
            if Path(name).suffix.lower() not in (".stl", ".obj", ".ply"):
                return self.send(400, "Нужен STL, OBJ или PLY.".encode())
            UPLOADS.mkdir(parents=True, exist_ok=True)
            dst = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
            dst.write_bytes(self.rfile.read(n))
            start_job(dst, notes, units)
            return self.send(200, b"ok")
        self.send(404, b"")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Демо открыто: {url}  (закрыть: Ctrl+C)")
    if "--no-open" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
