"""Synthetic training images for the cell classifiers.

Every sample is a single board cell drawn with a random font, size, offset,
colour scheme, grid-line fragments, blur, noise and JPEG compression. Fonts
and chess piece sets used here are the *training* styles; the evaluation
uses styles that never appear in this file's output (see bench_vision.py).
"""
import io
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 32          # classifier input size
CANVAS = 96        # drawing size before downscaling
PIECES = "KQRBNP"

# Glyph sets: name -> class labels
SETS = {
    "digits": ["blank"] + [str(d) for d in range(1, 10)],
    "zipnum": ["blank"] + [str(d) for d in range(1, 31)],
    "tango": ["empty", "sun", "moon"],
    "tango_sign": ["none", "=", "x"],
    "letters": ["wall"] + [chr(c) for c in range(65, 91)],
    "patches_num": ["none"] + [str(d) for d in range(1, 37)],
    "patches_shape": ["none", "square", "wide", "tall", "any"],
    "chess": ["empty"] + [c + p for c in "wb" for p in PIECES],
    "checkers": ["empty", "man", "king"],
}


@lru_cache(maxsize=1)
def fonts(list_file="/home/user/data/train_fonts.txt"):
    files = [f for f in Path(list_file).read_text().split() if f]
    return files


@lru_cache(maxsize=1)
def symbol_fonts():
    from fontTools.ttLib import TTFont
    out = []
    for f in fonts():
        try:
            if 0x265E in TTFont(f, lazy=True).getBestCmap():
                out.append(f)
        except Exception:
            pass
    return out


@lru_cache(maxsize=512)
def _font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=1)
def piece_images(root="/home/user/data/pieces"):
    sets = {}
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            sets[d.name] = {f.stem: Image.open(f).convert("RGBA") for f in d.glob("*.png")}
    return sets


def _rgb(rng, light):
    if light:
        return tuple(rng.randint(200, 255) for _ in range(3))
    return tuple(rng.randint(0, 70) for _ in range(3))


def _scheme(rng):
    """Background and ink colours; mostly light boards, sometimes dark mode or tinted cells."""
    dark_mode = rng.random() < 0.45
    bg = _rgb(rng, not dark_mode)
    if rng.random() < 0.3:  # tinted cell (region colours, highlights)
        bg = tuple(rng.randint(120, 255) if not dark_mode else rng.randint(20, 90) for _ in range(3))
    ink = _rgb(rng, dark_mode)
    if rng.random() < 0.3:
        ink = tuple(rng.randint(0, 255) for _ in range(3))
        if abs(sum(ink) - sum(bg)) < 200:
            ink = _rgb(rng, dark_mode)
    return bg, ink


def _text(draw, rng, text, ink, scale=(0.4, 0.75), font_list=None, center=(0.5, 0.5)):
    path = rng.choice(font_list or fonts())
    size = int(CANVAS * rng.uniform(*scale))
    font = _font(path, max(8, size))
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    jx, jy = rng.uniform(-0.08, 0.08) * CANVAS, rng.uniform(-0.08, 0.08) * CANVAS
    x = center[0] * CANVAS - w / 2 - bbox[0] + jx
    y = center[1] * CANVAS - h / 2 - bbox[1] + jy
    draw.text((x, y), text, font=font, fill=ink)


def _grid_lines(draw, rng, ink):
    if rng.random() < 0.6:
        w = rng.randint(1, 5)
        col = ink if rng.random() < 0.5 else tuple(rng.randint(80, 200) for _ in range(3))
        for side in rng.sample(["l", "r", "t", "b"], rng.randint(1, 4)):
            off = rng.randint(0, 4)
            if side == "l":
                draw.rectangle([0, 0, off + w, CANVAS], fill=col)
            elif side == "r":
                draw.rectangle([CANVAS - off - w, 0, CANVAS, CANVAS], fill=col)
            elif side == "t":
                draw.rectangle([0, 0, CANVAS, off + w], fill=col)
            else:
                draw.rectangle([0, CANVAS - off - w, CANVAS, CANVAS], fill=col)


def _finish(img, rng):
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.6)))
    # random crop/zoom: cell crops at read time are never perfectly aligned
    m = rng.uniform(0, 0.1) * CANVAS
    dx, dy = rng.uniform(-0.06, 0.06) * CANVAS, rng.uniform(-0.06, 0.06) * CANVAS
    img = img.crop((m + dx, m + dy, CANVAS - m + dx, CANVAS - m + dy))
    small = rng.choice([SIZE, 14, 16, 20, 24, 28, 40, 48])  # tiny cells: heavily downscaled screenshots
    img = img.resize((small, small), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, np.float32)
    if rng.random() < 0.5:
        arr = arr + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 12), arr.shape)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(20, 95))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def _sun(draw, rng, ink):
    c, r = CANVAS / 2 + rng.uniform(-6, 6), CANVAS * rng.uniform(0.15, 0.25)
    col = rng.choice([(232, 150, 46), (245, 190, 40), (240, 120, 30), (250, 210, 60), (255, 176, 32), ink])
    outline = tuple(int(v * 0.7) for v in col) if rng.random() < 0.4 else None  # LinkedIn-style rim
    draw.ellipse([c - r, c - r, c + r, c + r], fill=col, outline=outline, width=rng.randint(1, 3))
    if rng.random() < 0.6:
        for k in range(rng.choice([8, 12])):
            a = 2 * math.pi * k / (8 if k < 8 else 12)
            draw.line([c + 1.3 * r * math.cos(a), c + 1.3 * r * math.sin(a),
                       c + 1.8 * r * math.cos(a), c + 1.8 * r * math.sin(a)], fill=col, width=rng.randint(2, 5))


def _moon(draw, rng, bg, ink):
    c, r = CANVAS / 2 + rng.uniform(-6, 6), CANVAS * rng.uniform(0.2, 0.3)
    col = rng.choice([(61, 111, 216), (70, 90, 160), (120, 140, 200), (40, 60, 120), (74, 134, 232), ink])
    # the bite can come from any side: crescents point left, right, up-left...
    o, a = r * rng.uniform(0.35, 0.6), rng.uniform(0, 2 * math.pi)
    bx, by = c + o * math.cos(a), c + o * math.sin(a)
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    m = ImageDraw.Draw(mask)
    m.ellipse([c - r, c - r, c + r, c + r], fill=255)
    m.ellipse([bx - r, by - r, bx + r, by + r], fill=0)
    if rng.random() < 0.4:  # LinkedIn-style darker rim
        draw.bitmap((0, 0), mask, fill=tuple(int(v * 0.6) for v in col))
        mask = mask.filter(ImageFilter.MinFilter(2 * rng.randint(1, 2) + 1))
    draw.bitmap((0, 0), mask, fill=col)


def render(kind, label, rng):
    """One RGB SIZExSIZE sample of class `label` (an index into SETS[kind])."""
    name = SETS[kind][label]
    bg, ink = _scheme(rng)
    img = Image.new("RGB", (CANVAS, CANVAS), bg)
    d = ImageDraw.Draw(img)
    if kind == "digits":
        _grid_lines(d, rng, ink)
        if name != "blank":
            _text(d, rng, name, ink)
    elif kind == "zipnum":
        if name != "blank":
            style = rng.random()
            if style < 0.7:  # dark disc with light number
                disc = _rgb(rng, False) if rng.random() < 0.7 else tuple(rng.randint(0, 255) for _ in range(3))
                r = CANVAS * rng.uniform(0.28, 0.4)
                c = CANVAS / 2 + rng.uniform(-4, 4)
                d.ellipse([c - r, c - r, c + r, c + r], fill=disc)
                _text(d, rng, name, _rgb(rng, True), scale=(0.25, 0.4), center=(c / CANVAS, c / CANVAS))
            else:
                _text(d, rng, name, ink, scale=(0.3, 0.55))
        if rng.random() < 0.3:  # partial path stroke passing by
            col = tuple(rng.randint(0, 255) for _ in range(3))
            y = rng.choice([0, CANVAS])
            d.line([CANVAS / 2, CANVAS / 2, CANVAS / 2, y], fill=col, width=rng.randint(6, 20))
        _grid_lines(d, rng, ink)
    elif kind == "tango":
        _grid_lines(d, rng, ink)
        if name == "sun":
            if rng.random() < 0.4:
                _text(d, rng, rng.choice(["☀", "☀", "☼"]), rng.choice([(232, 150, 46), (240, 180, 40), ink]),
                      font_list=_sym_fonts(0x2600))
            else:
                _sun(d, rng, ink)
        elif name == "moon":
            if rng.random() < 0.4:
                _text(d, rng, rng.choice(["☾", "☽"]), rng.choice([(61, 111, 216), (90, 110, 190), ink]),
                      font_list=_sym_fonts(0x263E))
            else:
                _moon(d, rng, bg, ink)
    elif kind == "tango_sign":
        # crop centred on the border between two cells; the border line passes through the middle
        col = tuple(rng.randint(60, 200) for _ in range(3))
        w = rng.randint(1, 4)
        if rng.random() < 0.5:
            d.rectangle([CANVAS / 2 - w, 0, CANVAS / 2 + w, CANVAS], fill=col)
        else:
            d.rectangle([0, CANVAS / 2 - w, CANVAS, CANVAS / 2 + w], fill=col)
        if name == "=":
            _text(d, rng, "=", ink, scale=(0.3, 0.5))
        elif name == "x":
            _text(d, rng, rng.choice(["×", "x", "X", "✕"]), ink, scale=(0.3, 0.5),
                  font_list=rng.choice([None, _sym_fonts(0x2715)]))
    elif kind == "letters":
        if name == "wall":
            img = Image.new("RGB", (CANVAS, CANVAS), tuple(rng.randint(20, 110) for _ in range(3)) if rng.random() < 0.8
                            else tuple(rng.randint(140, 190) for _ in range(3)))
            d = ImageDraw.Draw(img)
        else:
            _text(d, rng, name if rng.random() < 0.7 else name.lower(), ink)
        _grid_lines(d, rng, ink)
    elif kind in ("patches_num", "patches_shape"):
        _grid_lines(d, rng, ink)
        num = name if kind == "patches_num" else rng.choice(["none"] + [str(k) for k in range(1, 13)])
        shape = name if kind == "patches_shape" else rng.choice(["none", "square", "wide", "tall", "any"])
        if kind == "patches_shape" and shape == "none":
            num = "none"  # no clue at all
        _patch_clue(d, rng, num if num != "none" else None, None if shape == "none" else shape, ink)
    elif kind == "chess":
        img = Image.new("RGB", (CANVAS, CANVAS), _square_colour(rng))
        d = ImageDraw.Draw(img)
        if rng.random() < 0.15:  # board coordinates printed in a corner
            _text(d, rng, rng.choice("abcdefgh12345678"), tuple(rng.randint(0, 255) for _ in range(3)),
                  scale=(0.12, 0.2), center=(rng.choice([0.1, 0.9]), rng.choice([0.1, 0.9])))
        if name != "empty":
            _piece(img, d, rng, name)
        if rng.random() < 0.1:  # last-move highlight
            over = Image.new("RGB", img.size, tuple(rng.randint(150, 255) for _ in range(3)))
            img = Image.blend(img, over, 0.3)
    elif kind == "checkers":
        img = Image.new("RGB", (CANVAS, CANVAS), _square_colour(rng))
        d = ImageDraw.Draw(img)
        if name != "empty":
            dark = rng.random() < 0.5
            col = (tuple(rng.randint(0, 80) for _ in range(3)) if dark else
                   rng.choice([(200, 49, 43), (230, 230, 230), (210, 60, 60), (240, 220, 180), (180, 30, 30)]))
            r = CANVAS * rng.uniform(0.3, 0.42)
            c = CANVAS / 2 + rng.uniform(-3, 3)
            d.ellipse([c - r, c - r, c + r, c + r], fill=col, outline=tuple(rng.randint(150, 255) for _ in range(3)),
                      width=rng.randint(0, 4))
            if name == "king":
                kc = rng.choice([(245, 215, 110), (255, 255, 255), (20, 20, 20), (240, 200, 60)])
                if rng.random() < 0.5:
                    _text(d, rng, rng.choice(["♛", "♔", "♚"]), kc, scale=(0.3, 0.45),
                          font_list=symbol_fonts())
                else:
                    pts = [(c - r * .5, c + r * .25), (c - r * .5, c - r * .3), (c - r * .25, c), (c, c - r * .45),
                           (c + r * .25, c), (c + r * .5, c - r * .3), (c + r * .5, c + r * .25)]
                    d.polygon(pts, fill=kc)
    img = _finish(img, rng)
    return np.asarray(img, np.uint8)


@lru_cache(maxsize=8)
def _sym_fonts(code):
    from fontTools.ttLib import TTFont
    out = []
    for f in fonts():
        try:
            if code in TTFont(f, lazy=True).getBestCmap():
                out.append(f)
        except Exception:
            pass
    return out or list(fonts())


def _square_colour(rng):
    return rng.choice([(240, 217, 181), (181, 136, 99), (238, 238, 210), (118, 150, 86), (222, 227, 230),
                       (140, 162, 173), (236, 230, 211), (109, 138, 90), (255, 255, 255), (200, 200, 200)]) \
        if rng.random() < 0.7 else tuple(rng.randint(60, 255) for _ in range(3))


def _piece(img, d, rng, name):
    color, kind = name[0], name[1]
    sets = piece_images()
    if sets and rng.random() < 0.75:
        style = sets[rng.choice(sorted(sets))]
        sprite = style[color + kind]
        s = int(CANVAS * rng.uniform(0.75, 0.98))
        sprite = sprite.resize((s, s), Image.LANCZOS)
        off = ((CANVAS - s) // 2 + rng.randint(-3, 3), (CANVAS - s) // 2 + rng.randint(-3, 3))
        img.paste(sprite, off, sprite)
    else:
        glyph = chr((0x2654 if color == "w" else 0x265A) + "KQRBNP".index(kind))
        style = rng.random()
        if style < 0.35:  # web style: white = outline glyph painted white with a dark halo; black = solid dark glyph
            path = rng.choice(symbol_fonts())
            font = _font(path, int(CANVAS * rng.uniform(0.6, 0.85)))
            bbox = d.textbbox((0, 0), glyph, font=font)
            x = (CANVAS - (bbox[2] - bbox[0])) / 2 - bbox[0]
            y = (CANVAS - (bbox[3] - bbox[1])) / 2 - bbox[1]
            if color == "w":
                d.text((x, y), glyph, font=font, fill=(255, 255, 255), stroke_width=rng.randint(1, 2),
                       stroke_fill=tuple(rng.randint(0, 60) for _ in range(3)))
            else:
                d.text((x, y), chr(0x265A + "KQRBNP".index(kind)), font=font, fill=tuple(rng.randint(0, 40) for _ in range(3)))
        elif style < 0.65:  # solid glyph filled with the piece colour, outlined
            glyph = chr(0x265A + "KQRBNP".index(kind))
            fill = (255, 255, 255) if color == "w" else (0, 0, 0)
            path = rng.choice(symbol_fonts())
            font = _font(path, int(CANVAS * rng.uniform(0.6, 0.8)))
            bbox = d.textbbox((0, 0), glyph, font=font)
            x = (CANVAS - (bbox[2] - bbox[0])) / 2 - bbox[0]
            y = (CANVAS - (bbox[3] - bbox[1])) / 2 - bbox[1]
            d.text((x, y), glyph, font=font, fill=fill, stroke_width=rng.randint(1, 3),
                   stroke_fill=(0, 0, 0) if color == "w" else (255, 255, 255) if rng.random() < 0.3 else (0, 0, 0))
        else:
            _text(d, rng, glyph, (0, 0, 0), scale=(0.6, 0.8), font_list=symbol_fonts())


def _patch_clue(d, rng, num, shape, ink):
    layout = rng.random()
    if shape is not None and layout < 0.45:
        # number inside an outline whose proportions show the shape
        w, h = {"square": (0.6, 0.6), "wide": (0.75, 0.45), "tall": (0.45, 0.75), "any": (0.6, 0.6)}[shape]
        w *= rng.uniform(0.85, 1.1) * CANVAS
        h *= rng.uniform(0.85, 1.1) * CANVAS
        c = CANVAS / 2
        if shape == "any":
            if rng.random() < 0.5:
                d.ellipse([c - w / 2, c - h / 2, c + w / 2, c + h / 2], outline=ink, width=rng.randint(2, 4))
            else:
                d.rectangle([c - w / 2, c - h / 2, c + w / 2, c + h / 2], outline=ink, width=rng.randint(2, 4))
                _text(d, rng, "*", ink, scale=(0.15, 0.22), center=(0.8, 0.22))
        else:
            d.rectangle([c - w / 2, c - h / 2, c + w / 2, c + h / 2], outline=ink, width=rng.randint(2, 4))
        if num:
            _text(d, rng, num, ink, scale=(0.22, 0.34))
        return
    # number with a small icon beside it
    icon = {"square": "■", "wide": "▬", "tall": "▮", "any": rng.choice(["✱", "*", "∗"])}.get(shape)
    if num and icon:
        side = rng.random() < 0.5
        _text(d, rng, num, ink, scale=(0.3, 0.45), center=(0.35, 0.5) if side else (0.5, 0.38))
        _text(d, rng, icon, ink, scale=(0.18, 0.3), center=(0.78, 0.5) if side else (0.5, 0.78),
              font_list=_sym_fonts(ord(icon)) if ord(icon) > 127 else None)
    elif num:
        _text(d, rng, num, ink, scale=(0.3, 0.5))
    elif icon:
        _text(d, rng, icon, ink, scale=(0.3, 0.5), font_list=_sym_fonts(ord(icon)) if ord(icon) > 127 else None)


def dataset(kind, count, seed=0):
    rng = random.Random(seed)
    n = len(SETS[kind])
    xs = np.zeros((count, SIZE, SIZE, 3), np.uint8)
    ys = np.zeros(count, np.int64)
    for i in range(count):
        y = i % n
        xs[i] = render(kind, y, rng)
        ys[i] = y
    return xs, ys
