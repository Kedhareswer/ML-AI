# ML-AI

Small models trained from scratch, with their harness, weights, benchmarks and demos.
Everything lives in [`reasoning_harness/`](reasoning_harness/README.md):

| Project | What it is | Card | Try it |
|---|---|---|---|
| **docparse** | Document parser with a 6 MB OCR model (PDF, images, Word, Excel, PowerPoint); ParseBench 36.2 | [DOCPARSE.md](reasoning_harness/DOCPARSE.md) | [Space](https://huggingface.co/spaces/Kedhareswer/docparse) · [model](https://huggingface.co/Kedhareswer/docparse) |
| **Puzzle Arena** | Networks for Sudoku, Queens, Tango, Zip, chess and more | [ARENA.md](reasoning_harness/ARENA.md) | [Space](https://huggingface.co/spaces/Kedhareswer/puzzle-arena) |
| **Tiny System 2** | 32 MB maze-planning policy with verified search | [MODEL_CARD.md](reasoning_harness/MODEL_CARD.md) | — |

```bash
cd reasoning_harness
pip install -e '.[docs]'
docparse ui            # http://127.0.0.1:7862
```

[`archive/`](archive/) keeps older coursework notebooks and datasets. docparse's corpus builder
(`scripts/build_doc_corpus.py`) and real-PDF benchmark read a few files from there.
