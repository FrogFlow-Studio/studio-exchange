"""Минимальный OpenAI-совместимый сервер для CADENA без vLLM (Windows, процессор).

Отвечает на /v1/models и /v1/chat/completions ровно в том объёме, в каком их вызывает
inference/cad_worker.py CADENA: одно сообщение пользователя с картинкой (base64) и текстом,
n вариантов, temperature, top_p, max_tokens, top_k. Генерация через transformers на
видеокарте NVIDIA, если есть, иначе на процессоре.

python server.py путь_к_весам [порт]
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = int(os.environ.get("SHIM_BATCH", "4"))
torch.set_num_threads(int(os.environ.get("SHIM_THREADS", str(os.cpu_count() or 4))))
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
LOCK = threading.Lock()  # одна модель — запросы по очереди
print(f"cadena shim ready on {DEV}, port {PORT}", flush=True)


def generate(body: dict) -> list[str]:
    msg = body["messages"][0]
    content, images = [], []
    for b in msg["content"]:
        if b.get("type") == "image_url":
            data = b["image_url"]["url"].split(",", 1)[1]
            images.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
            content.append({"type": "image"})
        else:
            content.append({"type": "text", "text": b.get("text", "")})
    text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    inp = proc(text=[text], images=images or None, return_tensors="pt").to(DEV)
    n, t = int(body.get("n", 1)), float(body.get("temperature", 0.0))
    kw = dict(max_new_tokens=int(body.get("max_tokens", 500)))
    if t > 0:
        kw.update(do_sample=True, temperature=t, top_p=float(body.get("top_p", 1.0)), top_k=int(body.get("top_k", 50)))
    else:
        kw.update(do_sample=False)
    outs: list[str] = []
    with LOCK, torch.no_grad():
        left = n if t > 0 else 1
        while left > 0:
            k = min(left, BATCH)
            g = model.generate(**inp, num_return_sequences=k, **kw)
            outs += proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            left -= k
    if t == 0:
        outs = outs * n  # жадный выбор даёт один и тот же ответ
    return outs


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # тише
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        t0 = time.time()
        try:
            texts = generate(body)
        except Exception as e:  # noqa: BLE001
            print("shim error:", repr(e)[:300], flush=True)
            return self._send({"error": {"message": str(e)[:300]}}, 500)
        print(f"shim: {len(texts)} x {max(len(x) for x in texts)} chars in {time.time() - t0:.0f} s", flush=True)
        self._send({"id": "x", "object": "chat.completion", "created": int(t0), "model": MODEL,
                    "choices": [{"index": i, "finish_reason": "stop", "message": {"role": "assistant", "content": x}}
                                for i, x in enumerate(texts)]})


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
