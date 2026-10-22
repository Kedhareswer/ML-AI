"""One entry point for every format: extract(path) -> document dict.

Document shape (JSON-serialisable):
    {"source", "format", "metadata", "pages": int, "blocks": [...], "sheets": [...], "warnings": [...]}
Blocks, in reading order, are one of
    heading   {text, level}
    paragraph {text}
    list_item {text, level?}
    table     {rows: [[cell, ...], ...]}
    figure    {}                       (pictures; their text is read only with ocr_images=True)
    header / footer {text}             (Word page headers and footers)
Every block can carry page, bbox [x0, y0, x1, y1] (PDF points or image pixels), sheet, source
("text-layer", "ocr", "structure") and, for OCR output, confidence (mean per-character probability).
"""
import csv
import datetime as dt
import html
import io
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "docparse" / "ocr.pt"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
CONVERT = {".doc": "docx", ".odt": "docx", ".rtf": "docx", ".xls": "xlsx", ".ods": "xlsx",
           ".ppt": "pdf", ".pptx": "pdf", ".odp": "pdf"}


@lru_cache(maxsize=2)
def recognizer(path=None):
    from .ocr.model import Recognizer
    path = Path(path or DEFAULT_MODEL)
    if not path.exists():
        raise FileNotFoundError(f"OCR weights not found at {path}. Train them with `docparse train-ocr`.")
    return Recognizer(str(path))


def extract(path, ocr="auto", ocr_images=False, model=None, dpi=200, password=None):
    """ocr: "auto" (only pages without a text layer), "always", or "never"."""
    path = Path(path)
    ext = path.suffix.lower()
    doc = {"source": path.name, "format": ext.lstrip("."), "metadata": {}, "pages": 0, "blocks": [],
           "sheets": [], "warnings": []}
    opts = {"ocr": ocr, "ocr_images": ocr_images, "model": model, "dpi": dpi, "password": password}
    if ext in CONVERT:
        with tempfile.TemporaryDirectory() as tmp:
            converted = _libreoffice(path, CONVERT[ext], tmp)
            doc["warnings"].append(f"converted {ext} to .{CONVERT[ext]} with LibreOffice before parsing")
            _dispatch(converted, converted.suffix.lower(), doc, opts)
    else:
        _dispatch(path, ext, doc, opts)
    return doc


def _dispatch(path, ext, doc, opts):
    if ext == ".pdf":
        pdf(path, doc, **opts)
    elif ext in IMAGE_EXT:
        image(path, doc, opts["model"])
    elif ext in (".docx", ".docm", ".dotx"):
        docx(path, doc, opts["ocr_images"], opts["model"])
    elif ext in (".xlsx", ".xlsm", ".xltx"):
        xlsx(path, doc, opts["ocr_images"], opts["model"])
    elif ext in (".csv", ".tsv"):
        delimited(path, doc)
    elif ext in (".txt", ".md"):
        text_file(path, doc)
    else:
        raise ValueError(f"unsupported file type: {ext}")


def _libreoffice(path, target, out_dir):
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        raise RuntimeError(f"{path.suffix} files need LibreOffice (soffice) for conversion")
    subprocess.run([exe, "--headless", "--convert-to", target, "--outdir", out_dir, str(path)],
                   check=True, capture_output=True, timeout=180)
    out = Path(out_dir) / (path.stem + "." + target)
    if not out.exists():
        raise RuntimeError(f"LibreOffice could not convert {path.name}")
    return out


# ---------- images ----------

def image(path, doc, model=None):
    from PIL import Image, ImageSequence
    from .ocr.page import analyze
    rec = recognizer(model)
    img = Image.open(path)
    frames = [f.copy() for f in ImageSequence.Iterator(img)] if getattr(img, "n_frames", 1) > 1 else [img]
    doc["metadata"].update({"width": img.width, "height": img.height, "mode": img.mode})
    for n, frame in enumerate(frames, 1):
        doc.setdefault("page_sizes", []).append([frame.width, frame.height])
        blocks, meta = analyze(frame, rec)
        mark_page_furniture(blocks, frame.height)
        for b in blocks:
            b.update(page=n, source="ocr")
        doc["blocks"] += blocks
        doc["metadata"].setdefault("ocr", []).append({"page": n, **meta})
    doc["pages"] = len(frames)


# ---------- PDF ----------

def pdf(path, doc, ocr="auto", ocr_images=False, model=None, dpi=200, password=None):
    import pdfplumber
    with pdfplumber.open(path, password=password) as pdf_:
        doc["metadata"].update({k: str(v) for k, v in (pdf_.metadata or {}).items() if isinstance(v, (str, int, float))})
        doc["pages"] = len(pdf_.pages)
        for n, page in enumerate(pdf_.pages, 1):
            doc.setdefault("page_sizes", []).append([float(page.width), float(page.height)])
            chars = page.chars
            scanned = len("".join(c["text"] for c in chars).strip()) < 20
            if ocr == "always" or (ocr == "auto" and scanned):
                if ocr == "never":
                    continue
                blocks = _ocr_pdf_page(path, n - 1, page, model, dpi, password)
            else:
                blocks = _pdf_text_page(page)
                if ocr_images:
                    _ocr_pdf_figures(path, n - 1, blocks, model, dpi, password)
                if scanned:
                    doc["warnings"].append(f"page {n} has no text layer; run with ocr='auto' to read it")
            mark_page_furniture(blocks, float(page.height))
            for b in blocks:
                b["page"] = n
            doc["blocks"] += blocks
            try:
                links = [h.get("uri") for h in page.hyperlinks if h.get("uri")]
            except (UnicodeDecodeError, ValueError, KeyError, TypeError):  # malformed annotations: skip links
                links = []
                doc["warnings"].append(f"page {n}: hyperlink annotations could not be decoded")
            if links:
                doc["metadata"].setdefault("links", []).extend({"page": n, "uri": u} for u in links)


def _pdf_text_page(page):
    from .ocr.page import group, text_blocks
    S = 2.0  # mask pixels per PDF point
    W, H = int(page.width * S) + 2, int(page.height * S) + 2
    tables, candidates = [], []
    for t in page.find_tables():
        candidates.append(t.bbox)
        rows = [[_clean_cell(c) for c in row] for row in _split_stacked_rows(t.extract())]
        cells = [c for row in rows for c in row]
        # Boxes drawn around code or text frames also look like grids; real tables are mostly filled.
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2 and sum(1 for c in cells if c) >= 0.5 * len(cells):
            tables.append((t.bbox, rows))
    tboxes = [b for b, _ in tables]

    def in_table(w):
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        return any(x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1 for x0, y0, x1, y1 in tboxes)
    all_words = [w for w in page.extract_words(extra_attrs=["size", "fontname", "upright"], use_text_flow=False,
                                               keep_blank_chars=False) if not in_table(w) and w["text"].strip()]
    # Rotated text (margin stamps, vertical axis labels) would line up with every body line and look
    # like a table column; it is kept out of the layout pass and reported as its own block.
    figures = _vector_figures(page, candidates, all_words)
    figures = [f for f in figures if not any(g != f and _inside(f, g) for g in figures)]
    if any(_inside(b, f) for b in tboxes for f in figures):  # a "table" drawn inside a chart is part of it
        tables = [(b, r) for b, r in tables if not any(_inside(b, f) for f in figures)]
        tboxes = [b for b, _ in tables]
        all_words = [w for w in page.extract_words(extra_attrs=["size", "fontname", "upright"], use_text_flow=False,
                                                   keep_blank_chars=False) if not in_table(w) and w["text"].strip()]

    def in_figure(w):
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        return next((k for k, (x0, y0, x1, y1) in enumerate(figures) if x0 <= cx <= x1 and y0 <= cy <= y1), None)
    fig_words = [[] for _ in figures]
    for w in all_words:
        k = in_figure(w)
        if k is not None:
            fig_words[k].append(w)
    all_words = [w for w in all_words if in_figure(w) is None]
    words = [w for w in all_words if w["upright"]]
    rotated = [w for w in all_words if not w["upright"]]
    mask = np.zeros((H, W), bool)
    boxes = []
    for w in words:
        h = w["bottom"] - w["top"]
        y0, y1 = int((w["top"] + 0.18 * h) * S), int((w["bottom"] - 0.18 * h) * S) + 1
        x0, x1 = int(w["x0"] * S), int(w["x1"] * S) + 1
        y0, y1 = max(0, min(y0, H - 1)), max(1, min(y1, H))
        x0, x1 = max(0, min(x0, W - 1)), max(1, min(x1, W))
        mask[y0:y1, x0:x1] = True
        boxes.append((y0, y1, x0, x1, w))
    sizes = [w["size"] for w in words]
    body = float(np.median(sizes)) if sizes else 10.0

    def read(segs):
        out = []
        for y0, y1, x0, x1 in segs:
            ws = [b[4] for b in boxes if y0 <= (b[0] + b[1]) / 2 <= y1 and x0 <= (b[2] + b[3]) / 2 <= x1]
            out.append((_styled_text(ws, body), None))
        return out
    th = body * 0.64 * S  # height of a shrunken word box at body size
    blocks, th = text_blocks(mask, th, read, figures=False)
    for (x0, y0, x1, y1), rows in tables:
        blocks.append({"kind": "table", "box": (int(y0 * S), int(y1 * S), int(x0 * S), int(x1 * S)), "rows": rows,
                       "confidence": None})
    for im in page.images:
        x0, y0, x1, y1 = im["x0"], im["top"], im["x1"], im["bottom"]
        if (x1 - x0 >= 10 and y1 - y0 >= 10 and (x1 - x0) * (y1 - y0) > 400
                and not any(_inside((x0, y0, x1, y1), f) for f in figures)):
            blocks.append({"kind": "figure", "box": (int(y0 * S), int(y1 * S), int(x0 * S), int(x1 * S))})
    for (x0, y0, x1, y1), ws in zip(figures, fig_words):
        blocks.append({"kind": "figure", "box": (int(y0 * S), int(y1 * S), int(x0 * S), int(x1 * S)),
                       "text": " ".join(w["text"] for w in sorted(ws, key=lambda w: (round(w["top"]), w["x0"])))})
    out = group(blocks, th)
    if rotated:
        x0, y0 = min(w["x0"] for w in rotated), min(w["top"] for w in rotated)
        x1, y1 = max(w["x1"] for w in rotated), max(w["bottom"] for w in rotated)
        text = " ".join(w["text"] for w in sorted(rotated, key=lambda w: (w["x0"], -w["bottom"])))
        out.append({"type": "paragraph", "text": text, "rotated": True,
                    "box": (int(y0 * S), int(y1 * S), int(x0 * S), int(x1 * S))})
    for b in out:
        y0, y1, x0, x1 = b.pop("box")
        b["bbox"] = [round(x0 / S, 1), round(y0 / S, 1), round(x1 / S, 1), round(y1 / S, 1)]
        b["source"] = "text-layer"
        b.pop("confidence", None)
        _pdf_heading_by_font(b, words, body)
    return out


def _inside(a, b, tol=2.0):
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _vector_figures(page, tables, words):
    """Charts and diagrams drawn with vector graphics (not embedded images).

    Drawing primitives are rasterised onto a coarse grid, dilated and grouped into connected
    regions. A region is a figure when it is large, holds enough primitives and little text
    (shaded tables and boxed text are dense with words; charts and diagrams are not)."""
    from .ocr.page import components
    W, H = float(page.width), float(page.height)
    S = 0.5  # grid cells per point
    prims = []
    for o in page.curves + page.lines + page.rects:
        x0, y0, x1, y1 = o["x0"], o["top"], o["x1"], o["bottom"]
        if (x1 - x0) * (y1 - y0) > 0.5 * W * H or any(_inside((x0, y0, x1, y1), t, 6) for t in tables):
            continue  # page backgrounds and table rules
        prims.append((x0, y0, x1, y1, o["object_type"] == "curve"))
    if len(prims) < 4:
        return []
    grid = np.zeros((int(H * S) + 2, int(W * S) + 2), bool)
    pad = 6
    for x0, y0, x1, y1, _ in prims:
        grid[max(0, int((y0 - pad) * S)):int((y1 + pad) * S) + 1, max(0, int((x0 - pad) * S)):int((x1 + pad) * S) + 1] = True
    rows, starts, ends, groups = components(grid)
    out = []
    for g in groups:
        x0, x1 = starts[g].min() / S + pad, ends[g].max() / S - pad
        y0, y1 = rows[g].min() / S + pad, (rows[g].max() + 1) / S - pad
        if x1 - x0 < 50 or y1 - y0 < 40 or (x1 - x0) * (y1 - y0) < 0.01 * W * H:
            continue
        inside = [p for p in prims if _inside(p[:4], (x0, y0, x1, y1), 4)]
        curves = sum(p[4] for p in inside)
        if len(inside) < 6 and curves < 2:
            continue
        text_area = sum((w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) for w in words
                        if _inside((w["x0"], w["top"], w["x1"], w["bottom"]), (x0, y0, x1, y1)))
        if text_area > 0.2 * (x1 - x0) * (y1 - y0):
            continue
        area = (x1 - x0) * (y1 - y0)
        if any(_overlap((x0, y0, x1, y1), t) > 0.3 * area for t in tables):
            continue  # the rules of a table (pdfplumber's box can be a little smaller than the grid)
        out.append((x0, y0, x1, y1))
    return out


def _font_style(w):
    name = w["fontname"].lower().split("+")[-1]
    bold = any(k in name for k in ("bold", "black", "heavy", "semibold", ",b")) or name.endswith("-b")
    italic = any(k in name for k in ("italic", "oblique", ",i")) or name.endswith("-i")
    return bold, italic


def _styled_text(ws, body):
    """Words of one text segment -> text with Markdown emphasis and <sup>/<sub> from the PDF's fonts.

    Characters in a different size come back as separate words ("x" "2"); they are joined without
    a space when they touch, and marked as superscript or subscript from their baseline."""
    if not ws:
        return ""
    lines = {}
    for w in sorted(ws, key=lambda w: w["top"]):
        key = next((k for k in lines if abs(k - w["bottom"]) < 0.6 * max(w["size"], body)
                    or (w["top"] >= k - 0.9 * body and w["bottom"] <= k + 0.4 * body)), None)
        lines.setdefault(w["bottom"] if key is None else key, []).append(w)
    pieces = []  # (text, bold, italic, joined_to_previous)
    for key in sorted(lines):
        line = sorted(lines[key], key=lambda w: w["x0"])
        big = max(w["size"] for w in line)
        base = float(np.median([w["bottom"] for w in line if w["size"] >= 0.9 * big]))
        prev = None
        for w in line:
            text = w["text"]
            small = w["size"] < 0.85 * big
            if small and w["bottom"] < base - 0.2 * w["size"]:
                text = f"<sup>{text}</sup>"
            elif small and w["bottom"] > base + 0.1 * w["size"]:
                text = f"<sub>{text}</sub>"
            touching = prev is not None and w["x0"] - prev["x1"] < 0.12 * big
            pieces.append((text, *_font_style(w), touching))
            prev = w
    if all(p[1] for p in pieces):  # a fully bold segment is a heading or label: no inline marks
        pieces = [(t, False, it, j) for t, _, it, j in pieces]
    if all(p[2] for p in pieces):
        pieces = [(t, b, False, j) for t, b, _, j in pieces]
    out, i = [], 0
    while i < len(pieces):
        bold, italic = pieces[i][1], pieces[i][2]
        j = i
        run = pieces[i][0]
        while j + 1 < len(pieces) and pieces[j + 1][1:3] == (bold, italic):
            j += 1
            run += ("" if pieces[j][3] else " ") + pieces[j][0]
        mark = "**" if bold else ""
        mark += "*" if italic else ""
        out.append(("" if pieces[i][3] and out else " ") + f"{mark}{run}{mark[::-1]}")
        i = j + 1
    return "".join(out).strip()


_INLINE_MARKS = re.compile(r"\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|</?su[pb]>")


def plain(text):
    """Text without the inline Markdown marks added from PDF fonts."""
    return _INLINE_MARKS.sub(lambda m: next((g for g in m.groups() if g is not None), ""), text)


def _pdf_heading_by_font(b, words, body):
    """Use real font sizes (not box heights) to decide headings in born-digital PDFs."""
    if b["type"] not in ("paragraph", "heading"):
        return
    x0, y0, x1, y1 = b["bbox"]
    ws = [w for w in words if x0 - 1 <= w["x0"] and w["x1"] <= x1 + 1 and y0 - 2 <= w["top"] and w["bottom"] <= y1 + 2]
    if not ws:
        return
    size = float(np.median([w["size"] for w in ws]))
    all_bold = all("bold" in w["fontname"].lower() for w in ws)
    short = len(b["text"]) < 120
    if len(b["text"]) < 200 and size >= body * 1.15:  # long titles are still titles when set large
        b["type"], b["level"] = "heading", 1 if size >= body * 1.7 else 2 if size >= body * 1.35 else 3
        b["text"] = plain(b["text"])
    elif short and all_bold and len(ws) <= 14:
        b["type"], b["level"] = "heading", 3
        b["text"] = plain(b["text"])
    elif b["type"] == "heading":
        b["type"] = "paragraph"
        b.pop("level", None)


PAGE_NUMBER = re.compile(r"^(page|p\.?|pg\.?)?\s*[-–—(]?\s*([0-9]{1,4}|[ivxlcdm]{1,6})\s*[-–—)]?(\s*(of|/)\s*[0-9]{1,4})?$", re.I)


def mark_page_furniture(blocks, page_height, band=0.08):
    """Running headers and footers: short text blocks inside the top or bottom band of the page.
    Headings stay in the body (a document title is often near the top of page one)."""
    for b in blocks:
        if b["type"] not in ("paragraph", "list_item") or not b.get("bbox") or len(b.get("text", "")) > 200:
            continue
        x0, y0, x1, y1 = b["bbox"]
        if y1 <= band * page_height:
            b["type"] = "header"
        elif y0 >= (1 - band) * page_height:
            b["type"] = "footer"
        else:
            continue
        if PAGE_NUMBER.match(plain(b["text"]).strip()):
            b["page_number"] = True


def _split_stacked_rows(rows):
    """Tables without horizontal rules come back from pdfplumber with several records stacked in one
    row, one line each per cell ("8833\n8837\n8838"). When every filled cell of a row holds the same
    number of lines, those lines are separate records: split them into rows."""
    out = []
    for row in rows:
        parts = [str(c).split("\n") if c else [] for c in row]
        counts = {len(p) for p in parts if p}
        if len(counts) == 1 and (k := counts.pop()) > 1 and sum(1 for p in parts if p) >= 2:
            out += [[p[i] if p else "" for p in parts] for i in range(k)]
        else:
            out.append(row)
    return out


def _clean_cell(c):
    return "" if c is None else " ".join(str(c).split())


def _render_pdf_page(path, index, dpi, password):
    import pypdfium2 as pdfium
    pdf_ = pdfium.PdfDocument(str(path), password=password)
    try:
        return pdf_[index].render(scale=dpi / 72).to_pil()
    finally:
        pdf_.close()


def _ocr_pdf_page(path, index, page, model, dpi, password):
    from .ocr.page import analyze
    img = _render_pdf_page(path, index, dpi, password)
    blocks, _ = analyze(img, recognizer(model))
    k = 72 / dpi  # pixels -> PDF points
    for b in blocks:
        b["bbox"] = [round(v * k, 1) for v in b["bbox"]]
        b["source"] = "ocr"
    return blocks


def _ocr_pdf_figures(path, index, blocks, model, dpi, password):
    """Read text inside pictures on a born-digital page; it is stored on the figure block."""
    from .ocr.page import analyze
    figs = [b for b in blocks if b["type"] == "figure"]
    if not figs:
        return
    img = _render_pdf_page(path, index, dpi, password)
    k = dpi / 72
    for f in figs:
        x0, y0, x1, y1 = f["bbox"]
        sub, _ = analyze(img.crop((int(x0 * k), int(y0 * k), int(x1 * k), int(y1 * k))), recognizer(model))
        text = "\n".join(_block_text(b) for b in sub if _block_text(b))
        if text:
            f["text"] = text
            f["text_source"] = "ocr"


# ---------- Word ----------

def docx(path, doc, ocr_images=False, model=None):
    import docx as docx_
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx_.Document(str(path))
    cp = d.core_properties
    for k in ("title", "author", "subject", "keywords", "created", "modified", "last_modified_by"):
        v = getattr(cp, k, None)
        if v:
            doc["metadata"][k] = v.isoformat() if isinstance(v, dt.datetime) else str(v)
    seen = set()
    for s in d.sections:
        for kind, part in (("header", s.header), ("footer", s.footer)):
            if part.is_linked_to_previous:
                continue
            text = "\n".join(p.text for p in part.paragraphs if p.text.strip())
            if text and (kind, text) not in seen:
                seen.add((kind, text))
                doc["blocks"].append({"type": kind, "text": text, "source": "structure"})
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            b = _docx_paragraph(Paragraph(el, d))
            if b:
                doc["blocks"].append(b)
            for box in el.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}txbxContent"):
                text = "\n".join(Paragraph(q, d).text for q in box.iterchildren() if q.tag.endswith("}p")).strip()
                if text:  # text boxes and shapes are not part of paragraph.text
                    doc["blocks"].append({"type": "paragraph", "text": text, "source": "structure", "text_box": True})
        elif tag == "tbl":
            doc["blocks"].append({"type": "table", "rows": _docx_table(Table(el, d)), "source": "structure"})
    images = [r.target_part for r in d.part.rels.values() if "image" in r.reltype]
    doc["metadata"]["images"] = len(images)
    if ocr_images:
        _ocr_blobs([im.blob for im in images], doc, model)
    doc["pages"] = 1


def _docx_paragraph(p):
    text = p.text.strip()
    if not text:
        return None
    style = (p.style.name if p.style is not None else "") or ""
    b = {"type": "paragraph", "text": text, "source": "structure"}
    low = style.lower()
    if low == "title":
        b.update(type="heading", level=1)
    elif low.startswith("heading"):
        digits = "".join(ch for ch in style if ch.isdigit())
        b.update(type="heading", level=int(digits) if digits else 1)
    else:
        ppr = p._p.pPr
        num = ppr.numPr if ppr is not None else None
        if num is not None or "list" in low:
            b["type"] = "list_item"
            if num is not None and num.ilvl is not None:
                b["level"] = int(num.ilvl.val)
            elif style[-1:].isdigit():  # "List Bullet 2" -> second level
                b["level"] = int(style[-1]) - 1
            if "number" in low:
                b["ordered"] = True
    if b["type"] == "paragraph" and p.runs and all(r.bold for r in p.runs if r.text.strip()) and len(text) < 100:
        b.update(type="heading", level=3)
    return b


def _docx_table(table):
    rows = []
    for row in table.rows:
        out, last = [], None
        for cell in row.cells:
            if cell._tc is last:  # horizontally merged cells repeat the same element
                out.append("")
                continue
            last = cell._tc
            parts = [p.text for p in cell.paragraphs if p.text.strip()]
            for inner in cell.tables:
                parts += [" | ".join(r) for r in _docx_table(inner)]
            out.append(" ".join(" ".join(parts).split()))
        rows.append(out)
    return rows


def _ocr_blobs(blobs, doc, model):
    from PIL import Image
    from .ocr.page import analyze
    for k, blob in enumerate(blobs, 1):
        try:
            img = Image.open(io.BytesIO(blob))
            img.load()
        except Exception:
            doc["warnings"].append(f"embedded image {k} could not be decoded")
            continue
        blocks, _ = analyze(img, recognizer(model))
        for b in blocks:
            b.update(source="ocr", image=k)
        doc["blocks"] += blocks


# ---------- Excel / CSV ----------

def _cell_value(v):
    if isinstance(v, dt.datetime) and v.time() == dt.time(0):
        return v.date().isoformat()
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _regions(grid):
    """Split a sheet into blocks of cells separated by fully empty rows, then empty columns."""
    def empty(v):
        return v is None or (isinstance(v, str) and not v.strip())
    if not grid:
        return []
    out = []
    rows = [any(not empty(v) for v in r) for r in grid]
    for r0, r1 in _true_runs(rows):
        band = grid[r0:r1]
        cols = [any(not empty(r[c]) for r in band) for c in range(len(band[0]))]
        for c0, c1 in _true_runs(cols):
            block = [r[c0:c1] for r in band]
            keep = [any(not empty(v) for v in r) for r in block]
            first, last = keep.index(True), len(keep) - keep[::-1].index(True)
            out.append((r0 + first, c0, block[first:last]))
    return out


def _true_runs(flags):
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    return runs


def xlsx(path, doc, ocr_images=False, model=None):
    import openpyxl
    from openpyxl.utils import get_column_letter
    values = openpyxl.load_workbook(path, data_only=True)
    formulas = openpyxl.load_workbook(path, data_only=False)
    props = values.properties
    for k in ("title", "creator", "created", "modified", "lastModifiedBy"):
        v = getattr(props, k, None)
        if v:
            doc["metadata"][k] = v.isoformat() if isinstance(v, dt.datetime) else str(v)
    for ws, wf in zip(values.worksheets, formulas.worksheets):
        info = {"name": ws.title, "state": ws.sheet_state, "dimensions": ws.dimensions,
                "merged": [str(m) for m in ws.merged_cells.ranges], "formulas": {}, "uncalculated": 0}
        grid = []
        for row_v, row_f in zip(ws.iter_rows(), wf.iter_rows()):
            out = []
            for cv, cf in zip(row_v, row_f):
                v = _cell_value(cv.value)
                if isinstance(cf.value, str) and cf.value.startswith("="):
                    info["formulas"][cf.coordinate] = cf.value
                    if v is None:  # saved without cached results (e.g. written by a script)
                        info["uncalculated"] += 1
                        v = cf.value
                out.append(v)
            grid.append(out)
        # Values of merged ranges live in the top-left cell; the rest are empty, which is correct for tables.
        info["images"] = len(getattr(ws, "_images", []))
        info["charts"] = len(getattr(ws, "_charts", []))
        doc["sheets"].append(info)
        doc["blocks"].append({"type": "heading", "level": 2, "text": ws.title, "sheet": ws.title, "source": "structure"})
        for r0, c0, rows in _regions(grid):
            a = f"{get_column_letter(c0 + 1)}{r0 + 1}"
            b = f"{get_column_letter(c0 + len(rows[0]))}{r0 + len(rows)}"
            doc["blocks"].append({"type": "table", "sheet": ws.title, "range": f"{a}:{b}", "source": "structure",
                                  "rows": [["" if v is None else v for v in r] for r in rows]})
        if ocr_images:
            blobs = []
            for im in getattr(ws, "_images", []):
                try:
                    blobs.append(im._data())
                except Exception:
                    pass
            _ocr_blobs(blobs, doc, model)
    doc["pages"] = len(values.worksheets)


def delimited(path, doc):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if Path(path).suffix.lower() == ".tsv" else csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect)]
    doc["blocks"].append({"type": "table", "rows": rows, "source": "structure"})
    doc["metadata"].update({"encoding": enc, "delimiter": dialect.delimiter, "rows": len(rows)})
    doc["pages"] = 1


def text_file(path, doc):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if para.startswith("#"):
            level = len(para) - len(para.lstrip("#"))
            doc["blocks"].append({"type": "heading", "level": level, "text": para.lstrip("# ").strip(), "source": "structure"})
        else:
            doc["blocks"].append({"type": "paragraph", "text": para, "source": "structure"})
    doc["pages"] = 1


# ---------- output ----------

def _block_text(b):
    if b["type"] == "table":
        return "\n".join("\t".join(str(c) for c in row) for row in b["rows"])
    return b.get("text", "")


def to_text(doc):
    return "\n\n".join(plain(t) for t in (_block_text(b) for b in doc["blocks"]) if t)


def _md_cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def _html_table(rows):
    esc = lambda v: html.escape(str(v)).replace("\n", "<br>")  # noqa: E731
    head, *body = rows
    out = ["<table>", "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in head) + "</tr>"]
    out += ["<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in body]
    return "\n".join(out + ["</table>"])


def to_markdown(doc, tables="markdown", page_marks=True):
    """tables="html" writes <table> elements, which keep cells containing pipes or line breaks intact."""
    out, page = [], None
    for b in doc["blocks"]:
        if (page_marks and b.get("page") and doc["pages"] > 1 and b["page"] != page
                and doc["format"] not in ("xlsx", "xlsm", "xltx")):
            page = b["page"]
            out.append(f"<!-- page {page} -->")
        t = b["type"]
        if t == "heading":
            out.append("#" * min(6, max(1, b.get("level", 1))) + " " + b["text"])
        elif t == "list_item":
            text = re.sub(r"^(•|\*|-)\s+", "", b["text"])
            prefix = "" if re.match(r"^\d{1,3}[.)]", text) else "1. " if b.get("ordered") else "- "
            out.append("  " * b.get("level", 0) + prefix + text)
        elif t == "table":
            rows = b["rows"]
            if not rows:
                continue
            n = max(len(r) for r in rows)
            rows = [list(r) + [""] * (n - len(r)) for r in rows]
            if tables == "html":
                out.append(_html_table(rows))
                continue
            lines = ["| " + " | ".join(_md_cell(c) for c in rows[0]) + " |", "|" + " --- |" * n]
            lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows[1:]]
            out.append("\n".join(lines))
        elif t == "figure":
            out.append("[figure]" + (f"\n\n> {b['text']}" if b.get("text") else ""))
        elif t in ("header", "footer"):
            out.append(f"<!-- {t}: {b['text']} -->")
        else:
            out.append(b["text"])
    return "\n\n".join(out) + "\n"
