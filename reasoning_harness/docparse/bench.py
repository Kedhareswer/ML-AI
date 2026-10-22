"""Benchmarks: line OCR, synthetic pages, and real PDF pages read from pixels.

1. Lines: held-out fonts (Carlito/Caladea) + held-out text, clean and degraded; the
   training fonts on unseen text are reported separately, as is an untrained network.
2. Pages: synthetic A4 pages (held-out fonts/text), clean renders and simulated scans.
   Scores: page CER in reading order, order-free word F1, heading recall, table recovery.
3. Real pages: born-digital PDFs from this repository rendered to images and OCR'd; the
   PDF's own text layer is the reference. Reading order of complex pages differs, so
   the order-free word F1 is the fairer score there.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .ocr.metrics import cer, summary
from .ocr.model import LineOCR, Recognizer
from .ocr.render import TRAIN_FONTS, available, dataset, render_line, sample_text
from .synth_pages import make_page, truth_text

ROOT = Path(__file__).resolve().parents[1]
REAL_PDFS = ["../Credit Card/Credit Card Fraud Detection.pdf", "../ML/PPT.pdf",
             "../Customer Analysis/vertopal.com_Customer_Behavior_Analysis.pdf"]


def _norm(t):
    return " ".join(str(t).split())


def word_f1(pred, ref):
    p, r = Counter(pred.split()), Counter(ref.split())
    hit = sum((p & r).values())
    if not hit:
        return 0.0
    prec, rec = hit / sum(p.values()), hit / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def lines(rec, count=2000):
    out = {}
    for name, split, degrade, seed in [("heldout_fonts_clean", "test", False, 7_000_001),
                                       ("heldout_fonts_degraded", "test", True, 7_000_002),
                                       ("train_fonts_unseen_text_degraded", "train", True, 7_000_003)]:
        if split == "train":  # training fonts with held-out text, so no sentence was seen in training
            ims, txt = [], []
            fonts = available(TRAIN_FONTS)
            for i in range(count):
                r = random.Random(seed * 1_000_003 + i)
                txt.append(sample_text(r, "test"))
                ims.append(render_line(txt[-1], r.choice(fonts), r.randint(14, 44), r, degrade))
        else:
            ims, txt = dataset(count, seed, split, degrade)
        t0 = time.time()
        preds = [p for p, _ in rec.read(ims, batch=64)]
        res = summary(preds, txt)
        res["ms_per_line"] = round(1000 * (time.time() - t0) / count, 2)
        res["examples"] = [{"truth": t, "pred": p} for t, p in list(zip(txt, preds))[:8]]
        out[name] = res
    # Untrained control: same architecture, random weights.
    torch.manual_seed(0)
    blank = Recognizer.__new__(Recognizer)
    blank.net, blank.device = LineOCR().eval(), "cpu"
    ims, txt = dataset(500, 7_000_002, "test", True)
    out["untrained_network_heldout_degraded"] = summary([p for p, _ in blank.read(ims)], txt)
    return out


def _flat(blocks):
    out = []
    for b in blocks:
        if b["type"] == "table":
            out += [" ".join(str(c) for c in row if str(c)) for row in b["rows"]]
        elif b.get("text"):
            out.append(b["text"])
    return _norm(" ".join(out))


def pages(rec, count=40):
    from .ocr.page import analyze
    res = {}
    for scan in (False, True):
        cers, f1s, heads, head_hits, tabs, tab_hits, cells, cell_hits, secs = [], [], 0, 0, 0, 0, 0, 0, []
        for i in range(count):
            img, truth = make_page(8_000_000 + i + (100_000 if scan else 0), "test", scan)
            t0 = time.time()
            blocks, _ = analyze(img, rec)
            secs.append(time.time() - t0)
            ref, pred = _norm(truth_text(truth)), _flat(blocks)
            cers.append(cer(pred, ref))
            f1s.append(word_f1(pred, ref))
            pheads = [b["text"] for b in blocks if b["type"] == "heading"]
            for b in truth:
                if b["type"] == "heading":
                    heads += 1
                    head_hits += any(cer(p, b["text"]) < 0.2 for p in pheads)
                if b["type"] == "table":
                    tabs += 1
                    shape = (len(b["rows"]), len(b["rows"][0]))
                    match = [p for p in blocks if p["type"] == "table" and (len(p["rows"]), len(p["rows"][0])) == shape]
                    if match:
                        tab_hits += 1
                        best = max(match, key=lambda p: sum(_norm(x) == _norm(y) for rp, rt in zip(p["rows"], b["rows"])
                                                            for x, y in zip(rp, rt)))
                        cell_hits += sum(_norm(x) == _norm(y) for rp, rt in zip(best["rows"], b["rows"]) for x, y in zip(rp, rt))
                    cells += shape[0] * shape[1]
        res["scanned" if scan else "clean"] = {
            "pages": count, "page_cer_reading_order": round(float(np.mean(cers)), 4),
            "page_cer_median": round(float(np.median(cers)), 4), "word_f1": round(float(np.mean(f1s)), 4),
            "heading_recall": round(head_hits / max(1, heads), 4), "headings": heads,
            "tables_found_with_correct_shape": round(tab_hits / max(1, tabs), 4), "tables": tabs,
            "table_cells_exact": round(cell_hits / max(1, cells), 4), "cells": cells,
            "seconds_per_page": round(float(np.mean(secs)), 2)}
    return res


def real_pdfs(rec, dpi=150, max_pages=40):
    """Render born-digital pages to pixels, OCR them, and compare with the PDF's text layer."""
    import pdfplumber
    import pypdfium2 as pdfium
    from .ocr.page import analyze
    rows = []
    for rel in REAL_PDFS:
        path = (ROOT / rel).resolve()
        if not path.exists():
            continue
        pdf_ = pdfium.PdfDocument(str(path))
        with pdfplumber.open(path) as pl:
            for n, page in enumerate(pl.pages):
                ref = _norm(page.extract_text() or "")
                if len(ref) < 50 or len(rows) >= max_pages:
                    continue
                img = pdf_[n].render(scale=dpi / 72).to_pil()
                t0 = time.time()
                blocks, _ = analyze(img, rec)
                pred = _flat(blocks)
                rows.append({"file": path.name, "page": n + 1, "ref_chars": len(ref),
                             "cer_reading_order": round(cer(pred, ref), 4), "word_f1": round(word_f1(pred, ref), 4),
                             "seconds": round(time.time() - t0, 2)})
        pdf_.close()
    agg = {"pages": len(rows), "dpi": dpi,
           "mean_word_f1": round(float(np.mean([r["word_f1"] for r in rows])), 4) if rows else None,
           "median_word_f1": round(float(np.median([r["word_f1"] for r in rows])), 4) if rows else None,
           "mean_cer_reading_order": round(float(np.mean([r["cer_reading_order"] for r in rows])), 4) if rows else None}
    return {"summary": agg, "pages": rows}


def run(model, output, line_count=2000, page_count=40, real_pages=40):
    torch.set_num_threads(4)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recognizer(str(model))
    report = {"model": str(model), "parameters": sum(p.numel() for p in rec.net.parameters()),
              "weights_mb": round(Path(model).stat().st_size / 2 ** 20, 2)}
    report["lines"] = lines(rec, line_count)
    print(json.dumps({k: {m: v for m, v in r.items() if m != "examples"} for k, r in report["lines"].items()}, indent=1))
    report["pages"] = pages(rec, page_count)
    print(json.dumps(report["pages"], indent=1))
    report["real_pdf_pages"] = real_pdfs(rec, max_pages=real_pages)
    print(json.dumps(report["real_pdf_pages"]["summary"], indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
