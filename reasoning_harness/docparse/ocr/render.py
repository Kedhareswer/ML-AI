"""Synthetic text-line images for training and testing the OCR network.

Every line is drawn with a real font, then degraded the way scans, photos and
screenshots degrade text. Fonts and text are split: the test side uses fonts and
sentences that never appear in training (Carlito/Caladea are metric clones of
Calibri/Cambria, the default Office fonts).
"""
import io
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .charset import CHARS, normalize

HEIGHT = 32
MAX_WIDTH = 1200
DATA = Path(__file__).resolve().parents[1] / "data"
FONT_DIR = Path("/usr/share/fonts")

TRAIN_FONTS = [
    "truetype/dejavu/DejaVuSans.ttf", "truetype/dejavu/DejaVuSans-Bold.ttf", "truetype/dejavu/DejaVuSerif.ttf",
    "truetype/dejavu/DejaVuSerif-Bold.ttf", "truetype/dejavu/DejaVuSansMono.ttf",
    "truetype/liberation/LiberationSans-Regular.ttf", "truetype/liberation/LiberationSans-Bold.ttf",
    "truetype/liberation/LiberationSans-Italic.ttf", "truetype/liberation/LiberationSerif-Regular.ttf",
    "truetype/liberation/LiberationSerif-Bold.ttf", "truetype/liberation/LiberationSerif-Italic.ttf",
    "truetype/liberation/LiberationMono-Regular.ttf", "truetype/freefont/FreeSans.ttf",
    "truetype/freefont/FreeSansBold.ttf", "truetype/freefont/FreeSansOblique.ttf", "truetype/freefont/FreeSerif.ttf",
    "truetype/freefont/FreeSerifBold.ttf", "truetype/freefont/FreeSerifItalic.ttf", "truetype/freefont/FreeMono.ttf",
    "opentype/inter/Inter-Regular.otf", "opentype/inter/Inter-Bold.otf", "opentype/inter/Inter-Light.otf",
    "opentype/inter/Inter-Medium.otf", "opentype/inter/Inter-Italic.otf", "opentype/inter/Inter-SemiBold.otf",
    "truetype/jetbrains-mono/JetBrainsMono-Regular.ttf", "truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
]
TEST_FONTS = [
    "truetype/crosextra/Carlito-Regular.ttf", "truetype/crosextra/Carlito-Bold.ttf",
    "truetype/crosextra/Carlito-Italic.ttf", "truetype/crosextra/Caladea-Regular.ttf",
    "truetype/crosextra/Caladea-Bold.ttf", "truetype/crosextra/Caladea-Italic.ttf",
]


def available(fonts):
    return [str(FONT_DIR / f) for f in fonts if (FONT_DIR / f).exists()]


@lru_cache(maxsize=4096)
def _font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=2)
def corpus(split):
    lines = (DATA / f"corpus_{split}.txt").read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.strip()]


# ---------- text sampling ----------

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WORDS = ["Invoice", "Total", "Subtotal", "Tax", "Amount", "Date", "Name", "Address", "Phone", "Email", "Qty",
         "Price", "Description", "Item", "Balance", "Due", "Account", "Reference", "Page", "Table", "Figure",
         "Summary", "Introduction", "Results", "Method", "Revenue", "Q1", "Q2", "Q3", "Q4", "Report", "ID",
         "Status", "Approved", "Pending", "Customer", "Order", "Units", "Region", "North", "South", "Notes"]


def _number(r):
    k = r.random()
    if k < 0.3:
        return f"{r.randint(0, 999999):,}"
    if k < 0.55:
        return f"{r.uniform(-9999, 99999):,.2f}"
    if k < 0.7:
        return f"{r.uniform(0, 100):.1f}%"
    if k < 0.85:
        return r.choice("$€£") + f"{r.uniform(0, 50000):,.2f}"
    return str(r.randint(0, 10 ** r.randint(1, 9)))


def _structured(r):
    k = r.randint(0, 11)
    if k == 0:
        return _number(r)
    if k == 1:
        y, m, d = r.randint(1990, 2030), r.randint(1, 12), r.randint(1, 28)
        return r.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{MONTHS[m - 1]} {d}, {y}",
                         f"{d} {MONTHS[m - 1][:3]} {y}"])
    if k == 2:
        user = "".join(r.choice("abcdefghijklmnopqrstuvwxyz._0123456789") for _ in range(r.randint(3, 12)))
        return f"{user}@{r.choice(['gmail.com', 'example.org', 'mail.co.uk', 'corp.io'])}"
    if k == 3:
        path = "/".join("".join(r.choice("abcdefghijklmnopqrstuvwxyz-") for _ in range(r.randint(2, 8)))
                        for _ in range(r.randint(1, 3)))
        return f"https://www.{r.choice(['example', 'docs', 'data'])}.com/{path}"
    if k == 4:
        return f"{r.choice(['INV', 'PO', 'ID', 'REF', 'SKU', 'No.'])}-{r.randint(0, 99999):05d}"
    if k == 5:
        return f"+{r.randint(1, 99)} {r.randint(100, 999)} {r.randint(100, 999)} {r.randint(1000, 9999)}"
    if k == 6:
        return f"{r.choice(WORDS)}: {_number(r)}"
    if k == 7:
        return " ".join(_number(r) for _ in range(r.randint(2, 5)))
    if k == 8:
        return r.choice(["• ", "- ", "* ", f"{r.randint(1, 20)}. ", f"({r.choice('abcdef')}) "]) + _prose(r, 4, 40)
    if k == 9:
        return _prose(r, 3, 40).upper()
    if k == 10:
        return " ".join(w.capitalize() for w in _prose(r, 6, 45).split())
    return "".join(r.choice(CHARS[:95]) for _ in range(r.randint(3, 25))).strip() or "x"


def _prose(r, lo, hi, split="train"):
    para = r.choice(corpus(split))
    n = r.randint(lo, hi)
    if len(para) <= n:
        return para
    i = r.randint(0, len(para) - n)
    j = para.rfind(" ", 0, i) + 1  # start on a word boundary
    end = para.find(" ", j + n)
    return para[j:end if end > 0 else None].strip() or para[:n]


def sample_text(r, split="train"):
    k = r.random()
    if k < 0.62:
        t = _prose(r, 3, 70, split)
    elif k < 0.95:
        t = _structured(r)
    else:
        t = r.choice(WORDS)
    return normalize(t).strip() or "a"


# ---------- rendering ----------

def render_line(text, font_path, size, r, degrade=True):
    """Draw one line of text and return a uint8 grayscale array of height HEIGHT."""
    font = _font(font_path, size)
    ink = r.randint(0, 70)
    bg = r.randint(200, 255)
    invert = degrade and r.random() < 0.04
    if invert:
        ink, bg = r.randint(225, 255), r.randint(20, 90)
    left, top, right, bottom = font.getbbox(text)
    asc, desc = font.getmetrics()
    pad = int(size * 0.6) + 8
    w, h = right - left + 2 * pad, asc + desc + 2 * pad
    img = Image.new("L", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.text((pad - left, pad), text, font=font, fill=ink)
    if degrade and r.random() < 0.06:  # underline
        y = pad + asc + max(1, desc // 3)
        d.line([pad, y, w - pad, y], fill=ink, width=max(1, size // 16))
    if degrade and r.random() < 0.5:
        img = img.rotate(r.uniform(-1.2, 1.2), resample=Image.BILINEAR, fillcolor=bg, expand=False)

    # Crop: either the tight ink box or the font's full line box, plus random padding.
    if r.random() < 0.5:
        y0, y1 = pad + top, pad + bottom
    else:
        y0, y1 = pad, pad + asc + desc
    x0, x1 = pad, pad + right - left
    ph = (y1 - y0)
    py0, py1 = (r.uniform(0.02, 0.25) * ph, r.uniform(0.02, 0.25) * ph) if degrade else (0.1 * ph, 0.1 * ph)
    px0, px1 = (r.uniform(0.0, 0.6) * ph, r.uniform(0.0, 0.6) * ph) if degrade else (0.2 * ph, 0.2 * ph)
    img = img.crop((int(x0 - px0), int(y0 - py0), int(x1 + px1) + 1, int(y1 + py1) + 1))

    if degrade:
        dr = ImageDraw.Draw(img)
        if r.random() < 0.08:  # remnants of table rules at the crop edge
            side = r.randint(0, 3)
            W, H = img.size
            lw = r.randint(1, 2)
            [lambda: dr.line([0, 0, W, 0], fill=ink, width=lw), lambda: dr.line([0, H - 1, W, H - 1], fill=ink, width=lw),
             lambda: dr.line([0, 0, 0, H], fill=ink, width=lw), lambda: dr.line([W - 1, 0, W - 1, H], fill=ink, width=lw)][side]()
        if r.random() < 0.35:
            scale = r.uniform(0.45, 0.85)  # resolution loss: low-dpi scans, small screenshots
            small = img.resize((max(4, int(img.width * scale)), max(4, int(img.height * scale))), Image.BILINEAR)
            img = small.resize(img.size, Image.BILINEAR)
        elif r.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.9)))
    out_w = max(8, min(MAX_WIDTH, round(img.width * HEIGHT / img.height)))
    img = img.resize((out_w, HEIGHT), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    if degrade:
        if r.random() < 0.4:
            a = a + np.random.default_rng(r.randint(0, 2 ** 31)).normal(0, r.uniform(2, 14), a.shape)
        if r.random() < 0.15:  # uneven lighting
            a = a * np.linspace(r.uniform(0.75, 1.0), r.uniform(0.75, 1.0), a.shape[1])[None, :]
    a = np.clip(a, 0, 255).astype(np.uint8)
    if degrade and r.random() < 0.25:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=r.randint(25, 85))
        a = np.asarray(Image.open(buf).convert("L"))
    return a


def sample(seed, split="train", degrade=True):
    """One (image, text) pair. split chooses both the font list and the corpus."""
    r = random.Random(seed)
    fonts = available(TRAIN_FONTS if split == "train" else TEST_FONTS)
    text = sample_text(r, split)
    size = r.randint(14, 44)
    return render_line(text, r.choice(fonts), size, r, degrade), text


def dataset(count, seed, split="train", degrade=True):
    images, texts = [], []
    for i in range(count):
        a, t = sample(seed * 1_000_003 + i, split, degrade)
        images.append(a)
        texts.append(t)
    return images, texts
