"""Картинки для установщика: значок (icon.ico) и полосы мастера (welcome.bmp, header.bmp).

Рисуются кодом в цветах рабочего окна, чтобы сборка не зависела от внешних файлов.
Запуск: python make_art.py <папка_вывода>
"""
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

BG = (13, 21, 32)
PANEL = (21, 32, 46)
TEAL = (78, 205, 196)
AMBER = (245, 180, 84)
TEXT = (232, 238, 245)
MUTED = (138, 151, 168)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def hexagon(draw, cx, cy, r, width, color, fill=None):
    """Кольцо-шестигранник: внешний залитый цветом, внутренний — фоном (без зазора на стыке линий)."""
    def pts(rr):
        return [(cx + rr * math.cos(math.radians(60 * k - 90)), cy + rr * math.sin(math.radians(60 * k - 90)))
                for k in range(6)]
    draw.polygon(pts(r), fill=color)
    draw.polygon(pts(r - width / math.cos(math.radians(30))), fill=fill or (0, 0, 0, 0))


def mark(size):
    """Знак: деталь-кубик в изометрии на тёмной плашке; три грани в оттенках бирюзы, рёбра янтарём
    (рёбра — то, что программа восстанавливает по скану)."""
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([s * 0.02, s * 0.02, s * 0.98, s * 0.98], radius=s * 0.2, fill=BG + (255,),
                        outline=(36, 52, 70, 255), width=max(2, s // 64))
    cx, cy, r = s / 2, s * 0.53, s * 0.36
    def p(k, rr=r):
        return (cx + rr * math.cos(math.radians(60 * k - 90)), cy + rr * math.sin(math.radians(60 * k - 90)))
    c = (cx, cy)
    top = [p(0), p(1), c, p(5)]
    right = [p(1), p(2), p(3), c]
    left = [p(5), c, p(3), p(4)]
    d.polygon(top, fill=(132, 230, 222, 255))
    d.polygon(right, fill=(50, 160, 154, 255))
    d.polygon(left, fill=(78, 205, 196, 255))
    # скол на верхнем углу: срезанный уголок другого цвета — «достроено»
    w = max(3, s // 40)
    for a_, b_ in ((p(1), c), (p(5), c), (p(3), c)):
        d.line([a_, b_], fill=BG + (255,), width=w)
    d.polygon([p(k) for k in range(6)], outline=AMBER + (255,), width=w)
    return img.resize((size, size), Image.LANCZOS)


def icon(out: Path):
    sizes = [16, 24, 32, 48, 64, 128, 256]
    big = mark(256)
    big.save(out / "icon.ico", sizes=[(n, n) for n in sizes])


def welcome(out: Path):
    # боковая полоса страниц «Добро пожаловать» и «Готово»: 164×314
    img = Image.new("RGB", (164, 314), BG)
    d = ImageDraw.Draw(img)
    for y in range(0, 314, 16):  # тонкая сетка, как фон вьювера
        d.line([(0, y), (164, y)], fill=(19, 29, 42))
    for x in range(0, 164, 16):
        d.line([(x, 0), (x, 314)], fill=(19, 29, 42))
    img.paste(mark(96), (34, 50), mark(96))
    d.text((82, 170), "CAD", font=ImageFont.truetype(BOLD, 22), fill=TEXT, anchor="mm")
    d.text((82, 196), "Intelligence", font=ImageFont.truetype(BOLD, 18), fill=TEXT, anchor="mm")
    small = ImageFont.truetype(FONT, 11)
    for i, line in enumerate(["скан → CAD-модель", "STEP · без интернета"]):
        d.text((82, 236 + 16 * i), line, font=small, fill=MUTED, anchor="mm")
    d.text((82, 296), "FrogFlow Studio", font=ImageFont.truetype(FONT, 10), fill=TEAL, anchor="mm")
    img.save(out / "welcome.bmp")


def header(out: Path):
    # картинка в шапке остальных страниц: 150×57, справа
    img = Image.new("RGB", (150, 57), (255, 255, 255))
    d = ImageDraw.Draw(img)
    img.paste(mark(44), (100, 6), mark(44))
    img.save(out / "header.bmp")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    target.mkdir(parents=True, exist_ok=True)
    icon(target)
    welcome(target)
    header(target)
