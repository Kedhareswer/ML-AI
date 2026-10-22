"""Synthetic document pages with known content, for end-to-end OCR benchmarks.

A page has a title, section headings, wrapped paragraphs, a bullet list, a ruled table,
a borderless key/value table and sometimes a two-column section. "scan" pages are
rotated, blurred, noised and JPEG-compressed after drawing.
"""
import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .ocr.render import TEST_FONTS, TRAIN_FONTS, _font, _number, _prose, available, sample_text

PAGE = (1240, 1754)  # A4 at 150 dpi


def _family(fonts, r):
    reg = [f for f in fonts if not any(k in f for k in ("Bold", "Italic", "Oblique", "Light", "Medium", "Semi"))]
    base = r.choice(reg)
    bold = next((f for f in fonts if f != base and f.startswith(base.split("-")[0].rsplit(".", 1)[0]) and "Bold" in f
                 and "Italic" not in f), base)
    return base, bold


def _wrap(text, font, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def make_page(seed, split="test", scan=False):
    r = random.Random(seed)
    fonts = available(TEST_FONTS if split == "test" else TRAIN_FONTS)
    regular, bold = _family(fonts, r)
    body = r.randint(20, 26)
    f_body, f_h1, f_h2 = _font(regular, body), _font(bold, int(body * 1.9)), _font(bold, int(body * 1.4))
    img = Image.new("L", PAGE, 255)
    d = ImageDraw.Draw(img)
    M = 100
    y = M
    W = PAGE[0] - 2 * M
    truth = []

    def text_block(kind, text, font, x, width, level=None):
        nonlocal y
        lines = _wrap(text, font, width)
        lh = int(font.size * 1.35)
        for ln in lines:
            d.text((x, y), ln, font=font, fill=0)
            y += lh
        b = {"type": kind, "text": " ".join(lines)}
        if level:
            b["level"] = level
        truth.append(b)

    def para(n_lo=120, n_hi=320):
        return _prose(r, n_lo, n_hi, split)

    text_block("heading", " ".join(w.capitalize() for w in _prose(r, 15, 40, split).split()[:6]), f_h1, M, W, 1)
    y += body
    for section in range(r.randint(2, 3)):
        if y > PAGE[1] - 400:
            break
        text_block("heading", f"{section + 1}. " + " ".join(_prose(r, 10, 30, split).split()[:4]).capitalize(), f_h2, M, W, 2)
        y += body // 2
        kind = r.choice(["para", "list", "table", "kv", "columns"])
        if kind == "para" or section == 0:
            text_block("paragraph", para(), f_body, M, W)
            y += body
        if kind == "list":
            for _ in range(r.randint(2, 4)):
                text_block("list_item", "• " + _prose(r, 20, 70, split), f_body, M + 20, W - 20)
            y += body
        elif kind == "table":
            cols = r.randint(3, 4)
            head = ["Item", "Region", "Units", "Amount", "Status", "Date"]
            r.shuffle(head)
            rows = [head[:cols]] + [[(sample_text(r, split).split() or ["x"])[0][:12] if c == 0 else _number(r)
                                     for c in range(cols)] for _ in range(r.randint(3, 5))]
            cw = [max(f_body.getlength(str(row[c])) for row in rows) + 30 for c in range(cols)]
            rh = int(body * 1.9)
            x0 = M
            if y + rh * len(rows) > PAGE[1] - M:
                continue
            for i, row in enumerate(rows):
                xx = x0
                for c, v in enumerate(row):
                    d.text((xx + 15, y + (rh - body) // 2), str(v), font=_font(bold if i == 0 else regular, body), fill=0)
                    xx += cw[c]
                d.line([x0, y, xx, y], fill=0, width=2)
                y += rh
            d.line([x0, y, x0 + sum(cw), y], fill=0, width=2)
            xx = x0
            for c in range(cols + 1):
                d.line([xx, y - rh * len(rows), xx, y], fill=0, width=2)
                xx += cw[c] if c < cols else 0
            truth.append({"type": "table", "rows": [[str(v) for v in row] for row in rows], "ruled": True})
            y += body * 2
        elif kind == "kv":
            keys = r.sample(["Invoice No.", "Date", "Customer", "Total", "Due date", "Reference", "Tax"], 3)
            rows = [[k, sample_text(r, split)[:24]] for k in keys]
            lh = int(body * 1.6)
            for k, v in rows:
                d.text((M, y), k, font=_font(bold, body), fill=0)
                d.text((M + 320, y), v, font=f_body, fill=0)
                y += lh
            truth.append({"type": "table", "rows": rows, "ruled": False})
            y += body
        elif kind == "columns":
            gap = 60
            cw = (W - gap) // 2
            top = y
            text_block("paragraph", para(250, 400), f_body, M, cw)
            left_bottom, y = y, top
            text_block("paragraph", para(250, 400), f_body, M + cw + gap, cw)
            y = max(y, left_bottom) + body
    if scan:
        img = _scan(img, r)
    return img, truth


def _scan(img, r):
    img = img.rotate(r.uniform(-2.0, 2.0), resample=Image.BICUBIC, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.4, 1.0)))
    a = np.asarray(img, np.float32)
    a = a * r.uniform(0.85, 1.0) + r.uniform(0, 20)
    a = a + np.random.default_rng(r.randint(0, 2 ** 31)).normal(0, r.uniform(4, 10), a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=r.randint(45, 80))
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def truth_text(truth):
    out = []
    for b in truth:
        if b["type"] == "table":
            out += [" ".join(row) for row in b["rows"]]
        else:
            out.append(b["text"])
    return " ".join(out)
