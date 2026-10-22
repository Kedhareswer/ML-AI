# docparse: a very small document parser

`docparse` extracts **everything** from a document (text, headings, paragraphs,
lists, tables, sheets, formulas, headers/footers, metadata and figure positions)
and returns it as JSON, Markdown or plain text, with reading order preserved.

| Input | How it is read | Exact? |
|---|---|---|
| PDF with a text layer | Words and fonts from the PDF; layout analysis groups them into headings, paragraphs, lists, columns and tables | Text is exact |
| Scanned PDF pages (no text layer) | Rendered at 200 dpi, then the OCR pipeline below | OCR |
| Images: PNG, JPG, TIFF (multi-page), BMP, WEBP, GIF | OCR pipeline | OCR |
| Word `.docx` | Document structure: styles → headings/lists, tables (merged cells), text boxes, headers/footers, properties | Exact |
| Excel `.xlsx` | Every sheet: values, formulas, merged ranges, dates, separate tables split on empty rows/columns | Exact |
| CSV / TSV | Delimiter and encoding sniffed | Exact |
| `.doc`, `.rtf`, `.odt`, `.xls`, `.ods`, `.ppt`, `.pptx`, `.odp` | Converted with LibreOffice (if installed), then parsed as above | Exact |

The design choice: **a model is only used where one is needed.** Born-digital
files already contain their text; reading their structure is exact and much
faster than any network. The neural network reads pixels: images, scans and
(optionally, `--ocr-images`) pictures embedded inside documents.

## The model

A CRNN line recognizer trained **from scratch** (no pretrained weights) on a
4-core CPU:

- **1,484,357 parameters, 6.0 MB** (`artifacts/docparse/ocr.pt`): 6 convolution
  layers, a 2-layer bidirectional LSTM and CTC decoding.
- Input: a grayscale text line, 32 px high, any width. Output: the text plus a
  confidence (mean probability of the emitted characters).
- Characters: printable ASCII plus `• € £ ° ©`. Typographic quotes, dashes and
  accented letters are folded to ASCII (`é` → `e`).
- Training: 480,000 synthetic lines (4 rounds of 120,000 freshly rendered
  lines), 27 fonts (DejaVu, Liberation, FreeFont, Inter, JetBrains Mono),
  text from app reviews and medical abstracts in this repository, plus
  generated numbers, prices, dates, emails, URLs, IDs and list items. Lines are
  degraded like scans and screenshots: blur, low resolution, noise, JPEG,
  rotation, uneven light, inverted colours and table-rule remnants. Training
  took 80 minutes.

Everything around the network is classical image processing (numpy only):
adaptive binarization (dark-on-light and light-on-dark), speck removal,
deskew, ruled-table detection from long horizontal/vertical strokes, frame
and outline removal, recursive XY-cut for columns and lines, then grouping
lines into headings (by x-height), paragraphs (by line spacing), list items,
and borderless tables (rows whose gaps line up).

## Results

All numbers are from `docparse bench` (`benchmarks/docparse/report.json`).
CER = character error rate (edit distance / reference length).

**1. Single lines.** Test fonts **Carlito and Caladea were never used in
training** (they are metric clones of Calibri and Cambria, the Office
defaults), and the test sentences come from a separate corpus.

| Test set | Lines | CER | Word error | Lines exactly right |
|---|---:|---:|---:|---:|
| Held-out fonts, clean | 2,000 | **0.27%** | 2.3% | 93.3% |
| Held-out fonts, degraded | 2,000 | **0.93%** | 6.0% | 84.8% |
| Training fonts, unseen text, degraded | 2,000 | 0.24% | 1.2% | 95.6% |
| Untrained network (same architecture), degraded | 500 | 99.7% | 99.8% | 0% |

About 3 ms per line on CPU.

**2. Synthetic A4 pages** (held-out fonts and text; titles, section headings,
paragraphs, bullet lists, ruled tables, key/value tables, two-column sections).
"Scanned" pages are rotated up to ±2°, blurred, noised and JPEG-compressed.

| | Clean (40 pages) | Scanned (40 pages) |
|---|---:|---:|
| Page CER, in reading order (mean / median) | 0.32% / 0.00% | 1.65% / 0.19% |
| Word F1 (order-free) | 98.9% | 97.7% |
| Headings found | 86.7% | 85.7% |
| Tables found with the right rows × columns | 100% | 100% |
| Table cells exactly right | 98.5% | 90.2% |
| Seconds per page (CPU) | 0.9 | 1.3 |

A few hard pages pull the scanned mean up; the median page is almost perfect.
Missed headings are mostly short numbered section titles ("1. Results") read
as list items or paragraphs: headings are detected by letter size, not by
boldness.

**3. Real pages from this repository.** 40 pages of the three PDFs in this
repo (two Jupyter notebook exports and a slide deck) were rendered to
150 dpi images and read **from pixels only**. Each PDF's own text layer is the
reference.

| PDF | Pages | Median CER (reading order) | Median word F1 |
|---|---:|---:|---:|
| Customer Behavior Analysis (notebook) | 15 | 0.7% | 92.5% |
| Credit Card Fraud Detection (notebook) | 11 | 2.3% | 94.8% |
| PPT.pdf (slides with graphics) | 14 | 32.3% | 82.5% |
| **All** | **40** | — | **91.9%** (mean 86.3%) |

Most slide-deck error is reading order: slides scatter text boxes, and the
PDF's own order differs from top-to-bottom reading. Text inside charts (axis
labels, tick numbers) is also read as text. The weakest notebook pages are
mostly plots with a few short labels.

## ParseBench (LlamaIndex)

[ParseBench](https://huggingface.co/datasets/llamaindex/ParseBench)
([paper](https://arxiv.org/abs/2604.08538), [code](https://github.com/run-llama/ParseBench))
is LlamaIndex's public benchmark for document parsers: **2,078 real enterprise
pages** (insurance filings, financial reports, government documents, papers),
1,211 documents and 169,011 human-verified test rules in five dimensions.
Scoring is deterministic rule matching, with no LLM judge.

docparse was run on the **full** dataset with the official harness
(parse-bench 1.0.4) on a 4-core CPU: all 2,078 pages, 0 failures, median
0.5 s per page. The comparison rows are the "Open Source - Local" entries of
ParseBench's published leaderboard.

| Parser | Overall | Tables | Charts | Content faithfulness | Semantic formatting | Visual grounding |
|---|---:|---:|---:|---:|---:|---:|
| PyMuPDF4LLM | 53.5 | 72.0 | 2.2 | 79.3 | 52.0 | 61.9 |
| Warp Ingest | 40.2 | 57.5 | 7.0 | 70.8 | 45.8 | 19.8 |
| LiteParse (no OCR) | 36.9 | 42.4 | 1.8 | 70.0 | 38.5 | 31.8 |
| **docparse** | **36.2** | **42.1** | **1.0** | **72.5** | **35.5** | **30.0** |
| OpenDataLoader | 29.4 | 35.2 | 0.9 | 66.1 | 34.1 | 10.8 |
| pdf-inspector | 26.6 | 26.6 | 5.3 | 56.1 | 35.1 | 9.9 |
| MarkItDown | 18.6 | 15.8 | 2.0 | 64.5 | 0.9 | 9.9 |
| PyMuPDF (text) | 16.0 | 0.0 | 0.0 | 68.3 | 1.0 | 10.9 |
| pypdf | 14.9 | 0.0 | 0.0 | 62.5 | 0.9 | 10.9 |

Overall is the mean of the five dimensions, as on the official leaderboard.
For scale: the top of the full leaderboard (cloud services and multi-billion
parameter vision-language models) reaches 75–90.

**What this says honestly**

- docparse is mid-table among local open-source parsers: ahead of pypdf,
  PyMuPDF, MarkItDown, pdf-inspector and OpenDataLoader; just behind LiteParse
  (36.9); behind Warp Ingest and clearly behind PyMuPDF4LLM.
- Its **content faithfulness (72.5)** is second among local parsers. It is
  strongest on simple (89.0) and multi-column (76.5) pages, and weakest on
  handwriting (27.4), scans (56.1) and non-English text (58.8). That matches the
  OCR model's limits: printed Latin text only.
- **Charts (1.0):** ParseBench wants each chart's data points as a table.
  docparse finds charts and keeps their labels but does not convert them to
  data. No local parser scores above 7.
- **Tables (42.1)** are limited by pdfplumber's ruled-grid detection on
  borderless financial tables; 32.3 on "hard" tables.
- **Formatting (35.5):** bold 29%, italic 23%, superscript 59%, subscript 57%
  of rules pass. Underline, strikeout, highlight, LaTeX and code blocks are not
  produced (0%).
- **Visual grounding (30.0):** paragraph boxes and reading order are decent;
  figure regions and box granularity often differ from the annotation.

**What the benchmark changed in docparse.** The first full run scored 33.66; the final run scored 36.2.
Its failures led to fixes that are now part of docparse itself:

| Change | Effect |
|---|---|
| Bold/italic from PDF font names, `<sup>`/`<sub>` from glyph size and baseline | Semantic formatting 24.0 → 35.5 |
| Running header/footer and printed page-number detection | Header rules 0 → 9.9%, footer rules 0 → 12.2%; content faithfulness 71.8 → 72.5 |
| Rotated margin text (e.g. arXiv stamps) kept out of the reading flow | It no longer turns body text into a fake table |
| Vector charts and diagrams detected as figures, their labels attached | Visual grounding 28.9 → 30.0 |
| Rows stacked in one cell of rule-less tables split into records | Correct rows in insurance rate tables |
| A malformed PDF link annotation no longer fails the whole page | 9 failing pages → 0 |

Reproduce (parse-bench needs Python 3.12+):

```bash
pip install "parse-bench[fast]"
python scripts/parsebench_docparse.py download          # ~600 MB
python scripts/parsebench_docparse.py run docparse --max_concurrent 3
python scripts/parsebench_summary.py output/docparse leaderboard.csv
```

`docparse/parsebench.py` registers the pipeline through ParseBench's public
extension API. Each page is sent as Markdown with HTML tables, page
header/footer fields, and layout boxes in ParseBench's canonical classes.
Aggregate results are in `benchmarks/parsebench/report.json`.

**Structure parsers** (PDF text layer, DOCX, XLSX, CSV) are tested for exact
output in `tests/test_docparse.py`.

## Use it

```bash
pip install -e '.[docs]'          # torch, numpy, pillow, pdfplumber, pypdfium2, python-docx, openpyxl

docparse extract report.pdf                      # Markdown to stdout
docparse extract scan.png --format json -o scan.json
docparse extract invoice.pdf --ocr always        # OCR every page, even with a text layer
docparse extract deck.docx --ocr-images          # also read text inside embedded pictures
docparse extract *.xlsx --format md -o out/      # many files -> a directory
docparse ui                                      # browser UI on http://127.0.0.1:7862
```

```python
from docparse import extract
from docparse.extract import to_markdown

doc = extract("statement.pdf")          # dict: source, format, metadata, pages, blocks, sheets, warnings
for b in doc["blocks"]:
    print(b["type"], b.get("page"), b.get("text") or b.get("rows"))
print(to_markdown(doc))
```

Each block has a `type` (`heading`, `paragraph`, `list_item`, `table`,
`figure`, `header`, `footer`), a `source` (`text-layer`, `structure` or
`ocr`), and where meaningful a `page`, `bbox`, `level`, `sheet`, `range` and
an OCR `confidence`. Excel sheets also list formulas, merged ranges, charts and
images under `sheets`.

## Train or fine-tune your own

```bash
python scripts/build_doc_corpus.py                        # rebuild the text corpora
docparse train-ocr --run runs/ocr --rounds 4              # ~80 min on 4 CPU cores
docparse train-ocr --run runs/ocr --rounds 6 --resume     # continue / extend training
docparse bench --model runs/ocr/ocr.pt --output runs/ocr-bench
```

To specialise it (your fonts, your vocabulary, e.g. invoices in one
typeface), add font files to `TRAIN_FONTS` and text to `docparse/data/corpus_train.txt`
in `docparse/ocr/render.py`, then train or resume. A checkpoint is written
after every round.

## Limits

- **Printed text only.** No handwriting. Latin script only (ASCII plus a few
  symbols); accents are dropped and other scripts are not read.
- Tables: ruled grids and borderless tables with aligned columns are found.
  Cells that span several rows or columns are kept in the first cell. Very
  irregular tables may come out as text.
- Reading order is top-to-bottom, columns left to right. Magazine layouts,
  slides with scattered boxes, sidebars and text running around pictures can
  be read out of order.
- Pictures are reported as `figure` blocks with a position. Charts are not
  converted to data.
- Photos of documents taken at an angle (perspective), curved pages and
  very low resolution (text under about 10 px high) are not handled well.
- Born-digital PDFs are only as good as their text layer: some PDFs embed
  fonts without a Unicode mapping. Use `--ocr always` for those.
- Besides ParseBench, the benchmarks are synthetic pages and three real PDFs from
  this repository. No direct comparison with Tesseract was run here.