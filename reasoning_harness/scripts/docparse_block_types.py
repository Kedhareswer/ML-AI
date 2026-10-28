"""Block-type check: synthetic PDF pages with known block types, scored against docparse.

Covers the cases the Space's Pages view showed going wrong (notebook code cells read as
tables, printed dataframes read as paragraphs, "Term: description" lists read as paragraphs)
and guard cases that must not change (typewriter prose, a two-line form, ruled and
borderless tables, normal prose). Not tuned on ParseBench.

usage: python scripts/docparse_block_types.py [--output benchmarks/docparse/block_types.json]
"""
import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docparse.extract import extract, plain  # noqa: E402

PROSE = ["Card fraud costs issuers billions every year, and most of it happens online.",
         "This notebook loads the transactions, explores them and trains a classifier.",
         "Every feature except time and amount is the output of a PCA transform."]
CODE = ["# Load, explore and plot data", "import numpy as np", "import pandas as pd",
        "credit = pd.read_csv('creditcard.csv')", "credit.head()"]
FRAME = ["   Time        V1        V2   Amount  Class",
         "0   0.0 -1.359807 -0.072781   149.62      0",
         "1   0.0  1.191857  0.266151     2.69      0",
         "2   1.0 -1.358354 -1.340163   378.66      0",
         "3   1.0 -0.966272 -0.185226   123.50      0"]
WARN = ["UserWarning: Unable to import Axes3D. This may be due to multiple",
        "versions of Matplotlib being installed (e.g. as a system package and as a",
        "pip package). As a result, the 3D projection is not available."]
DEFS = ["ID: Customer's unique identifier", "Year_Birth: Customer's birth year",
        "Education: Customer's education level", "Income: Customer's yearly household income"]
LETTER = ["Dear Mr. Patel, thank you for your letter of the fourth of March.",
          "We have reviewed the claim and attached the signed forms for you.",
          "Please call our office if any detail in them is not correct."]
FORM = ["Name: John Smith", "Date: 12 March 2024"]
RULED = [["Region", "Sales", "Growth"], ["North", "1200", "12%"], ["South", "950", "8%"]]
LOOSE = [["Quarter", "Revenue", "Margin"], ["Q1 2024", "4,210", "18.2%"], ["Q2 2024", "4,630", "19.0%"],
         ["Q3 2024", "5,020", "19.6%"]]

# (anchor text, expected type, expected (rows, cols) for tables)
TRUTH = ([("Fraud Detection Notebook", "heading", None), (PROSE[0], "paragraph", None)]
         + [(CODE[1], "code", None), (CODE[3], "code", None)]
         + [("-1.359807", "table", (5, 6)), ("378.66", "table", (5, 6)), ("3D projection", "code", None)]
         + [("Column Information", "heading", None)] + [(d.split(":")[0] + ":", "list_item", None) for d in DEFS]
         + [(LETTER[0][:30], "paragraph", None), (FORM[0], "paragraph", None)]
         + [("Growth", "table", (3, 3)), ("Q2 2024", "table", (4, 3)), ("Closing remarks", "heading", None)])


def build(path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4

    def text(x, y, s, font="Helvetica", size=11):
        c.setFont(font, size)
        c.drawString(x, y, s)

    def cell(y_top, lines, token_bg):
        """A notebook cell: grey background, monospace lines, optional per-token backgrounds."""
        h = 14 * len(lines) + 10
        c.setFillGray(0.96); c.rect(60, y_top - h, W - 120, h, stroke=0, fill=1)
        c.setStrokeGray(0.89); c.setLineWidth(0.25); c.rect(60, y_top - h, W - 120, h, stroke=1, fill=0)  # thin cell frame
        for k, s in enumerate(lines):
            y = y_top - 16 - 14 * k
            if token_bg:  # syntax highlighters paint touching boxes behind each token (with its trailing space)
                x = 66
                for tok in s.split(" "):
                    w = c.stringWidth(tok + " ", "Courier", 9)
                    c.setFillGray(0.96); c.rect(x, y - 4, w, 14, stroke=0, fill=1)
                    x += w
            c.setFillGray(0); text(66, y, s, "Courier", 9)
        c.setStrokeGray(0)
        return y_top - h - 18

    # page 1: notebook
    text(60, H - 70, "Fraud Detection Notebook", "Helvetica-Bold", 20)
    y = H - 105
    for s in PROSE:
        text(60, y, s); y -= 15
    y = cell(y - 10, CODE, token_bg=True)
    y = cell(y, FRAME, token_bg=False)
    y = cell(y, WARN, token_bg=False)  # console prose in a cell: code, never a table
    c.showPage()
    # page 2: definitions, typewriter letter, short form
    text(60, H - 70, "Column Information", "Helvetica-Bold", 16)
    text(60, H - 100, "People")
    y = H - 122
    for s in DEFS:
        text(60, y, s); y -= 22
    y -= 20
    for s in LETTER:
        text(60, y, s, "Courier", 10); y -= 13
    y -= 30
    for s in FORM:
        text(60, y, s); y -= 16
    c.showPage()
    # page 3: ruled and borderless tables
    y = H - 80
    for r, row in enumerate(RULED):
        for k, v in enumerate(row):
            c.rect(60 + 120 * k, y - 22 * r - 6, 120, 22, stroke=1, fill=0)
            text(66 + 120 * k, y - 22 * r, v)
    y -= 22 * len(RULED) + 50
    for r, row in enumerate(LOOSE):
        for k, v in enumerate(row):
            text(60 + 150 * k, y - 18 * r, v, "Helvetica-Bold" if r == 0 else "Helvetica")
    y -= 18 * len(LOOSE) + 40
    text(60, y, "Closing remarks", "Helvetica-Bold", 16)
    text(60, y - 25, "The model flags fewer than one in a thousand transactions for review.")
    c.save()


def block_text(b):
    if b["type"] == "table":
        return " ".join(str(v) for r in b["rows"] for v in r if v)
    return plain(b.get("text") or "")


def score(doc):
    rows, per = [], defaultdict(lambda: [0, 0])
    for anchor, want, shape in TRUTH:
        hit = next((b for b in doc["blocks"] if anchor in block_text(b)), None)
        got = hit["type"] if hit else "missing"
        ok = got == want
        if ok and shape:
            n_rows, n_cols = len(hit["rows"]), max(len(r) for r in hit["rows"])
            ok = (n_rows, n_cols) == shape
            got += f" {n_rows}x{n_cols}"
        per[want][0] += ok
        per[want][1] += 1
        rows.append({"anchor": anchor[:40], "expected": want, "got": got, "ok": ok})
    return {"correct": sum(r["ok"] for r in rows), "total": len(rows),
            "by_type": {k: f"{a}/{n}" for k, (a, n) in sorted(per.items())},
            "tables_found": sum(1 for b in doc["blocks"] if b["type"] == "table"), "tables_expected": 3,  # the dataframe has two checks
            "checks": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "block_types.pdf"
        build(path)
        res = score(extract(path, ocr="never"))
    for r in res["checks"]:
        print(f"{'ok ' if r['ok'] else 'BAD'}  {r['expected']:<10} got {r['got']:<16} {r['anchor']}")
    print(f"\n{res['correct']}/{res['total']} correct; by type {res['by_type']}; "
          f"tables found {res['tables_found']} (expected {res['tables_expected']})")
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
