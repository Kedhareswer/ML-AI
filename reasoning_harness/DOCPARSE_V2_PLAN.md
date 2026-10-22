# docparse v2 plan: tables, layout and reading order

Status: plan (nothing here is built yet). Every target below is measured with the same
ParseBench harness as v1 (`benchmarks/parsebench/report.json`), so progress is comparable.

## Where v1 stands (facts from the full ParseBench run)

| Dimension | v1 | Best local parser (PyMuPDF4LLM) | What the sub-metrics say |
|---|---:|---:|---|
| Tables | 42.1 | 72.0 | Cell content is half right (GriTS 55.5), but whole records rarely are (TableRecordMatch 26.9; 17% of tables perfect). 85% of predicted tables match nothing (false positives); 11% of real tables are missed. |
| Visual grounding | 30.0 | 61.9 | Per page, 19 ground-truth regions go unmatched vs 11 predicted ones: v1 **merges** regions. Localization passes 49%, classification 38%, reading order 64%. Class F1: Text 0.32, Section 0.27, Table 0.26, Picture 0.38. |
| Content: reading order | 60.2% of order rules | — | Content is mostly found (missing words pass 75%), but its order is often wrong on multi-column, form and slide-like pages. |
| Overall | 36.2 | 53.5 | Charts (1.0) are excluded from this plan. |

Why v1 is weak here: tables come from pdfplumber's ruled-grid finder plus one heuristic for
borderless tables; layout is a hand-tuned XY-cut; reading order is whatever the XY-cut yields.
None of these parts learn from data. The OCR model is the only learned part.

## Principles for v2

1. **Stay small and CPU-only at inference.** Each new model is ≤ 10 M parameters and runs
   in under 1 s per page on 4 CPU cores. Total package under ~40 MB.
2. **Learn from data, but use the PDF's own structure as features.** For born-digital PDFs
   (about 90% of ParseBench's non-chart pages; 150 of ~1,510 are scans or images), the text layer gives exact words, fonts and positions. A model
   over those features is far cheaper and more accurate than one reading pixels. Pixels are
   used only for scans and pictures.
3. **No tuning on ParseBench.** v1's fixes were found by looking at ParseBench failures, which
   is acceptable once but would become overfitting. v2 develops on public training/validation
   sets and runs ParseBench only at milestones, reporting every run.
4. **Permissive data only.** The sets below are CDLA-Permissive unless marked "verify".

## Workstream A: tables (target 42 → 60+)

| Step | What | Data | Expected effect |
|---|---|---|---|
| A1 | **Table detector**: classify candidate regions (pdfplumber grids, aligned word blocks, vector rules) as table / not table with a small gradient-boosted or MLP model on geometry features (column alignment, numeric density, rule coverage, gaps) | PubTables-1M and FinTabNet page boxes; DocLayNet "Table" regions | Remove most of the 85% false-positive tables and recover missed borderless ones |
| A2 | **Table structure model**: predict row and column separators from projection profiles of words and rules (SPLERGE-style "split"), ~1–3 M params; then a "merge" step for spanning cells | PubTabNet, FinTabNet, PubTables-1M structure labels | Fixes merged/stacked rows and misaligned columns, which is what limits TableRecordMatch |
| A3 | **Header understanding**: multi-row headers, header vs body rows, colspan/rowspan written as real HTML | Same | TableRecordMatch keys records by header, so correct headers lift it directly |
| A4 | Scanned tables: run A2 on OCR word boxes instead of text-layer words | Same, rendered with scan noise | Tables in scans stop being plain text |

## Workstream B: layout (target 30 → 50+)

| Step | What | Data | Expected effect |
|---|---|---|---|
| B1 | **Line-level role classifier** (born-digital): each text line gets a label (Text, Section header, Title, List item, Caption, Footnote, Page header/footer, Table, Formula) from font size/weight relative to the page, position, indentation, gaps and neighbours; a small transformer or GBDT over line features | DocLayNet (80k pages with PDF-derived boxes, 11 classes) | Classification and Section/Title F1; replaces the size-only heading rule |
| B2 | **Region grouping**: merge lines into regions with a learned "same region?" decision between neighbouring lines, instead of fixed gap thresholds | DocLayNet regions | Fixes the over-merging (19 unmatched GT regions per page) |
| B3 | **Tiny visual detector** for Picture and Table regions and for scanned pages (YOLO-nano-class, ~3–6 M params, 640 px input). Trained on a GPU once (Colab / HF ZeroGPU, as the repo's Space already uses), CPU inference | DocLayNet, PubLayNet | Picture F1 (0.38 now) and layout on scans and images |
| B4 | Furniture: page header/footer from B1 plus repetition across pages in multi-page files | DocLayNet furniture labels | Header/footer rules (10–12% now) |

## Workstream C: reading order (target 60% → 75%+)

| Step | What | Data | Expected effect |
|---|---|---|---|
| C1 | **Learned pairwise order**: a small model scores "region i comes before region j" from geometry and roles; a greedy or topological pass builds the order (LayoutReader idea, far smaller) | ReadingBank (verify license); reading order from tagged PDFs' structure trees (self-labelled, free) | Multi-column, sidebars and forms |
| C2 | Use B1 roles in ordering: captions next to their figure/table, footnotes last, furniture outside the body | — | Order rules and layout reading-order pass rate |

## Milestones and gates

| Milestone | Contents | Gate (full ParseBench, same harness) |
|---|---|---|
| v2.0 | A1 + B1 + B2 (feature models only; no new vision model) | Overall ≥ 41 (above Warp Ingest's 40.2); no dimension below its v1 score |
| v2.1 | A2 + A3 + C1 + C2 | Tables ≥ 55, layout ≥ 45, order ≥ 70% |
| v2.2 | B3 + A4 + B4 | Overall ≥ 47; layout and tables also improve on scanned pages |
| Stretch | — | Overall ≥ 53.5 (PyMuPDF4LLM) |

Every milestone also keeps: v1's unit tests green, the internal OCR benchmarks unchanged or better,
a latency budget (median ≤ 1 s/page CPU), and a written report with all numbers, including
regressions.

## Risks

- **Domain gap.** DocLayNet and PubTables are mostly papers, reports and manuals; ParseBench
  pages are insurance and financial filings. Mitigation: synthetic table pages rendered from
  real layouts, and a small hand-labelled dev set from public filings (not from ParseBench).
- **CPU training budget.** Feature models (A1, B1, B2, C1) train on CPU in hours. The visual
  detector (B3) needs a GPU for a day; if none is available, B3 moves out of scope and pictures
  keep the v1 heuristics.
- **Targets are estimates.** They come from v1's sub-metrics (what fixing each failure mode can
  recover), not from experiments. They will be revised after v2.0.
