"""Шаг 2: модель ИИ. Пишет код CadQuery по листу замеров и правит его по отчёту.

Режимы:
  anthropic — живой прогон, нужен ANTHROPIC_API_KEY;
  openai    — живой прогон, нужен OPENAI_API_KEY;
  replay    — повтор ответов ИИ из прошлого живого прогона (llm_log.json);
  scripted  — без ИИ: заготовленные ответы для проверки конвейера.
В отчёте всегда написано, какой режим был, чтобы запись не выдать за живой прогон.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path

SYSTEM = """Ты инженер-конструктор. Восстанавливаешь CAD-модель повреждённой детали по скану.
Тебе дают «лист замеров» скана (уже выровненного: опорная плоскость на Z=0, ось X вдоль
главной стороны, центр в плане в 0,0; всё в мм), картинку скана с четырёх сторон (видно
форму и где скол), иногда фото и заметки мастера. После каждой попытки — картинку, где твоя
модель наложена на скан: серый — совпало, красный — модели не хватает материала (ошибка),
синий — скан внутри модели (скол, который ты достроил, или модель велика), зелёный — твоя
достройка там, где скана нет. Смотри на картинки, но размеры бери из чисел.

Задача: написать Python-код на CadQuery, который строит ЦЕЛУЮ деталь, какой она была до
поломки (замысел конструктора), в той же системе координат, что и скан.
- Сломанные и отсутствующие места восстанавливай по замыслу: симметрия, повтор
  отверстий, ровные стенки. Не копируй скол.
- Размеры бери из замеров, округляй до разумных конструкторских значений
  (например 6.52 → 6.5, если это похоже на стандарт), но не подгоняй ради красоты.
- Строй через понятные операции (эскиз, вытягивание, отверстия, скругления), чтобы
  инженер мог повторить дерево построения в SolidWorks.
- Итог положи в переменную `result` (cq.Workplane или cq.Shape), это одно тело.
- Скругления делай осторожно: если fillet падает, лучше без него, чем ошибка.
- Не читай и не пиши файлы, не импортируй ничего, кроме cadquery и math.

Ответ строго в двух блоках:
```python
# код
```
```json
{"plan": "коротко, как строишь", "assumptions": ["что предположил"],
 "restored": ["что достроил, чего нет в скане"], "check_by_engineer": ["что проверить штангенциркулем"]}
```
Пиши по-русски."""

TOOLS_NOTE = """

У тебя есть инструменты, которыми ты сам смотришь сетку скана: сечения по X, Y, Z на любой
высоте, лучи, точки с нормалями в любой области, вписывание плоскости и окружности, ближайшая
точка поверхности. Картинок нет: форму и скол ищи по самой сетке. Лист замеров — только
подсказка, проверяй по скану то, в чём сомневаешься (особенно место скола: где поверхности
нет, что там было по симметрии и соседним граням). Вызовов на одну попытку не больше {limit};
не повторяй уже сделанные запросы. Когда всё ясно, дай ответ в двух блоках, как сказано выше."""

# Цены за 1 млн токенов (вход, выход), $; кэш: запись 1.25×, чтение 0.1× входа.
PRICES = {"sonnet": (2.0, 10.0), "opus": (5.0, 25.0), "haiku": (1.0, 5.0)}


def first_prompt(sheet: dict, notes: str) -> str:
    rest = {k: v for k, v in sheet.items() if k not in ("summary", "transform_from_scan_4x4", "file")}
    s = json.dumps(rest, ensure_ascii=False)
    summ = "\n".join("- " + line for line in sheet.get("summary", []))
    return (f"Сводка скана (посчитана геометрией, не догадка):\n{summ}\n\nПолный лист замеров:\n{s}\n\n"
            f"Заметки мастера: {notes or 'нет'}\n\n"
            "Построй модель целой детали.")


def feedback_prompt(run: dict, cmp: dict | None, target: float) -> str:
    if not run.get("ok"):
        return ("Код не выполнился.\n" + run.get("error", "") + "\n" + run.get("traceback", "")[-1500:] +
                "\nИсправь код. Ответ в тех же двух блоках.")
    lines = [f"Код выполнился. Тело валидно: {run.get('valid')}, тел: {run.get('solids')}, "
             f"объём {run.get('volume_mm3')} мм³, габарит {run.get('bbox_mm')}.",
             f"Сравнение со сканом (допуск {cmp['tolerance_mm']} мм):",
             f"- точек скана в допуске: {cmp['within_tol_share']*100:.1f}% (цель {target*100:.0f}%)",
             f"- средн. отклонение {cmp['mean_abs_dev_mm']} мм, 95% точек ближе {cmp['p95_abs_dev_mm']} мм",
             f"- скан СНАРУЖИ модели (модели не хватает материала, это ошибка): {cmp['outside_share']*100:.2f}%",
             f"- скан ВНУТРИ модели (скол, который ты восстановил, или модель велика): {cmp['inside_share']*100:.2f}%",
             f"- поверхность модели, которой нет в скане (достроено): {cmp['added_surface_share']*100:.1f}%"]
    for title, key in (("Где модели не хватает материала", "clusters_outside"),
                       ("Где модель больше скана", "clusters_inside"),
                       ("Где модель достроена (в скане пусто)", "clusters_added")):
        cl = cmp[key][:6]
        if cl:
            lines.append(title + ":")
            lines += [f"  · около {c['center_mm']}, область {c['bbox_mm']}, точек {c['points']}, "
                      f"отклонение средн. {c['mean_dev_mm']} / макс. {c['max_dev_mm']} мм" for c in cl]
    lines.append("Если «модель больше скана» и «достроено» совпадают с местом скола, который ты "
                 "восстановил сознательно, это нормально, оставь и скажи об этом в restored. "
                 "Всё остальное исправь. Ответ в тех же двух блоках.")
    return "\n".join(lines)


def parse(text: str) -> tuple[str, dict]:
    code = re.search(r"```python\s*\n(.*?)```", text, re.S)
    meta = re.search(r"```json\s*\n(.*?)```", text, re.S)
    info: dict = {}
    if meta:
        try:
            info = json.loads(meta.group(1))
        except json.JSONDecodeError:
            info = {"plan": meta.group(1)[:500]}
    return (code.group(1) if code else ""), info


def _images(m: dict) -> list[Path]:
    """Картинки сообщения: фото детали, виды скана, карта отклонений попытки."""
    out = list(m.get("images") or [])
    if m.get("image"):
        out.insert(0, m["image"])
    return [Path(p) for p in out if p and Path(p).exists()]


def _mt(p: Path) -> str:
    return "image/png" if p.suffix.lower() == ".png" else "image/jpeg"


class Provider:
    label = ""
    supports_tools = False

    def chat(self, messages: list[dict], tools=None) -> str:  # messages: [{"role","content", "images"?: [путь, ...]}]
        raise NotImplementedError


class Anthropic(Provider):
    supports_tools = True

    def __init__(self, model: str, client=None, max_tool_calls: int = 30):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=240.0, max_retries=0)
        self.client = client
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.label = f"живой прогон, Anthropic {model}"
        self.usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "requests": 0}
        self.trace: list[dict] = []  # вызовы инструментов последней попытки

    def cost_usd(self) -> float:
        pin, pout = next((v for k, v in PRICES.items() if k in self.model), PRICES["sonnet"])
        u = self.usage
        return round((u["input"] * pin + u["cache_write"] * pin * 1.25 + u["cache_read"] * pin * 0.1
                      + u["output"] * pout) / 1e6, 4)

    def _count(self, r) -> None:
        u = getattr(r, "usage", None)
        self.usage["requests"] += 1
        if u is None:
            return
        self.usage["input"] += getattr(u, "input_tokens", 0) or 0
        self.usage["output"] += getattr(u, "output_tokens", 0) or 0
        self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0

    def chat(self, messages, tools=None):
        conv = []
        for m in messages:
            content: list = []
            for p in _images(m):
                content.append({"type": "image", "source": {"type": "base64", "media_type": _mt(p),
                                                            "data": base64.b64encode(p.read_bytes()).decode()}})
            content.append({"type": "text", "text": m["content"]})
            conv.append({"role": m["role"], "content": content})
        system = SYSTEM + (TOOLS_NOTE.format(limit=self.max_tool_calls) if tools else "")
        kw: dict = {"model": self.model, "max_tokens": 8000,
                    "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}
        if tools:
            from scan_tools import TOOLS

            kw["tools"] = TOOLS[:-1] + [dict(TOOLS[-1], cache_control={"type": "ephemeral"})]
        self.trace = []
        while True:
            # кэш на последний блок: каждый следующий запрос цикла читает прошлое из кэша
            req = [dict(m) for m in conv]
            last = dict(req[-1]["content"][-1], cache_control={"type": "ephemeral"})
            req[-1] = {"role": req[-1]["role"], "content": req[-1]["content"][:-1] + [last]}
            if tools and len(self.trace) >= self.max_tool_calls:
                kw["tool_choice"] = {"type": "none"}
            r = self.client.messages.create(messages=req, **kw)
            self._count(r)
            if getattr(r, "stop_reason", "") != "tool_use" or not tools:
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            blocks = [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b) for b in r.content]
            conv.append({"role": "assistant", "content": blocks})
            results = []
            for b in blocks:
                if b.get("type") != "tool_use":
                    continue
                if len(self.trace) >= self.max_tool_calls:
                    out = json.dumps({"error": "лимит вызовов на попытку исчерпан, дай ответ"}, ensure_ascii=False)
                else:
                    out = tools.call(b["name"], b.get("input") or {})
                self.trace.append({"tool": b["name"], "input": b.get("input"), "result": out[:2000]})
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out})
            conv.append({"role": "user", "content": results})


class OpenAI(Provider):
    def __init__(self, model: str):
        self.key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.label = f"живой прогон, OpenAI {model}"

    def chat(self, messages, tools=None):
        if tools:
            raise RuntimeError("инструменты сетки пока только для Anthropic")
        conv = [{"role": "system", "content": SYSTEM}]
        for m in messages:
            imgs = _images(m)
            if imgs:
                parts = [{"type": "text", "text": m["content"]}]
                for p in imgs:
                    url = f"data:{_mt(p)};base64," + base64.b64encode(p.read_bytes()).decode()
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                conv.append({"role": m["role"], "content": parts})
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": self.model, "messages": conv}).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]


class Replay(Provider):
    """Повтор ответов из прошлого прогона: тот же порядок ответов."""

    def __init__(self, log_path: str, label: str):
        self.answers = [e["answer"] for e in json.loads(Path(log_path).read_text(encoding="utf-8"))]
        self.i = 0
        self.label = label

    def chat(self, messages, tools=None):
        if self.i >= len(self.answers):
            raise RuntimeError("в записи больше нет ответов")
        a = self.answers[self.i]
        self.i += 1
        return a


def make(mode: str, model: str | None, replay_log: str | None) -> Provider:
    if mode == "anthropic":
        return Anthropic(model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    if mode == "openai":
        return OpenAI(model or os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    if mode == "replay":
        src = json.loads(Path(replay_log).read_text(encoding="utf-8"))
        lab = src[0].get("provider", "?") if src else "?"
        return Replay(replay_log, f"ПОВТОР записанного прогона ({lab}), ИИ сейчас не вызывается")
    if mode == "scripted":
        return Replay(replay_log, "БЕЗ ИИ: заготовленные ответы для проверки конвейера")
    raise ValueError(f"неизвестный режим {mode}")


def auto_mode() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "geometric"  # без ключа: честный расчёт без ИИ, а не заготовки
