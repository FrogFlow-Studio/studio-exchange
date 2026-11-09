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
главной стороны, центр в плане в 0,0; всё в мм), иногда фото или эскиз и заметки мастера.

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


class Provider:
    label = ""

    def chat(self, messages: list[dict]) -> str:  # messages: [{"role","content", "image"?}]
        raise NotImplementedError


class Anthropic(Provider):
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic(timeout=240.0, max_retries=0)
        self.model = model
        self.label = f"живой прогон, Anthropic {model}"

    def chat(self, messages):
        conv = []
        for m in messages:
            content: list = []
            if m.get("image"):
                p = Path(m["image"])
                mt = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                content.append({"type": "image", "source": {"type": "base64", "media_type": mt,
                                                            "data": base64.b64encode(p.read_bytes()).decode()}})
            content.append({"type": "text", "text": m["content"]})
            conv.append({"role": m["role"], "content": content})
        r = self.client.messages.create(model=self.model, max_tokens=8000, system=SYSTEM, messages=conv)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAI(Provider):
    def __init__(self, model: str):
        self.key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.label = f"живой прогон, OpenAI {model}"

    def chat(self, messages):
        conv = [{"role": "system", "content": SYSTEM}]
        for m in messages:
            if m.get("image"):
                p = Path(m["image"])
                mt = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                url = f"data:{mt};base64," + base64.b64encode(p.read_bytes()).decode()
                conv.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]},
                                                            {"type": "image_url", "image_url": {"url": url}}]})
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

    def chat(self, messages):
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
