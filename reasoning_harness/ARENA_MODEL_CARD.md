# Puzzle Arena: model card

Small neural networks **trained from random initialisation on a 4-core CPU**
for Sudoku, Mini Sudoku, Tango, Queens, Zip, Patches, Wend, chess puzzles and
checkers, plus a Pinpoint word-guessing experiment and nine screenshot cell
readers. No pretrained weights, no pretrained language model and no external
solver are used. Every answer the app calls "solved" is checked by an
independent rule verifier.

- **Try it:** [huggingface.co/spaces/Kedhareswer/puzzle-arena](https://huggingface.co/spaces/Kedhareswer/puzzle-arena)
- **Weights:** [huggingface.co/Kedhareswer/puzzle-arena](https://huggingface.co/Kedhareswer/puzzle-arena) (every model, plus the `arena` code that loads them)
- **Source:** [github.com/Kedhareswer/ML-AI](https://github.com/Kedhareswer/ML-AI/tree/main/reasoning_harness) (`reasoning_harness/`)
- **Full results and method notes:** [ARENA.md](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/ARENA.md)

![Held-out results](https://raw.githubusercontent.com/Kedhareswer/ML-AI/main/reasoning_harness/evidence/arena/benchmark-summary.png)

## At a glance

| Model | What it does | Architecture | Parameters | Trained on | Held-out result |
|---|---|---|---:|---|---|
| Sudoku | fills 9×9 cells | GridNet, width 128, 12 blocks | 3.37M | 383,200 real Sudoku-Extreme puzzles | 20.7% network only · 95.0% with search |
| Mini Sudoku | fills 6×6 cells | GridNet, width 64, 6 blocks | 0.43M | 10,000 generated puzzles | 99.3% · 100% |
| Tango | places suns and moons | GridNet, width 64, 6 blocks | 0.43M | 2,858 generated puzzles | 84.0% · 100% |
| Queens | places one queen per row | GridNet, width 64, 6 blocks | 0.43M | 2,000 generated puzzles | 90.3% · 100% |
| Zip | draws the path | GridNet, width 64, 6 blocks | 0.43M | 2,000 generated puzzles | 70.3% · 100% |
| Patches | cuts rectangles | GridNet, width 64, 6 blocks | 0.43M | 15,000 generated puzzles | 72.3% · 100% |
| Wend | splits letters into words | GridNet, width 64, 6 blocks | 0.44M | 15,000 generated puzzles | 22.0% · 100% |
| Chess | picks the next move | ResNet policy, width 96, 6 blocks | 1.02M | 1.18M Lichess puzzle positions | 44.0% · 58.7% of held-out Lichess puzzles |
| Checkers | scores positions | conv value net, width 64 | 0.64M | 5,000 self-play games | 60% score vs equal-depth engine |
| Pinpoint | guesses the linking word | corpus link counts + skip-gram word vectors (200-dim) | 8.0M vectors + 2.2M word-pair counts | 83M words of WikiText-103 | 30–35% top guess · 57.5% top 5 on handwritten puzzles; ~4% on abstract WordNet categories (**still weak**) |
| Screenshot readers | read one cell each | small CNN, 9 classifiers | ~1.1M each | 15k–74k synthetic cells each | 94% of 280 held-out screenshots read exactly |

"Network only" means the network makes every choice; the rules only block
illegal moves, and there is no backtracking. "With search" means depth-first
backtracking (alpha-beta for chess) whose choices are ordered by the network.
Untrained copies of every network solve 0–2.3%, so the gains come from
training. 300 held-out puzzles per game unless stated.

## Implementation

### Grid puzzles: GridNet ([arena/nets.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/nets.py))

One architecture serves all seven grid games. Input planes describe the board
(givens, clues, walls, region one-hots), plus a mask of real cells so one
network handles several board sizes.

- **Stem:** 3×3 convolution over input planes + mask.
- **MixBlock (×6 or ×12):** a 3×3 convolution for local context, added to a
  1×1 convolution over `[x, row mean, column mean, region mean]`. Every cell
  sees the constraints of its whole row, column and box in every block.
  GroupNorm, GELU, a 2× feed-forward, residual connections.
- **Head:** 1×1 convolution to per-cell logits: the digit (Sudoku), sun or
  moon (Tango), queen or not (Queens), path order (Zip), rectangle edges
  (Patches), word boundaries (Wend).
- **Training** ([arena/train.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/train.py)): cross-entropy on the unique solution,
  AdamW at lr 0.001, batch 64, 2,000 steps (2,500 for Sudoku), bf16 autocast
  on CPU, seed 0, resumable checkpoints.
- **Solving** ([arena/search.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/search.py)): *network only* fills the most
  confident legal cell, re-runs the network and repeats. *Network + search* is
  depth-first backtracking that tries the network's most likely values first.
  The *classical solver* is the same search with no network.

### Chess ([arena/games/chess_net.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/games/chess_net.py), [chess_engine.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/games/chess_engine.py))

- 18 input planes (12 piece types, side to move, castling, en passant),
  stem + 6 residual blocks of width 96, and a policy head giving a logit for each
  (from-square, to-square) pair (4,096 outputs). Moves are masked to the legal
  ones.
- Trained on 1.18M positions from the Lichess puzzle database (CC0). Batch
  256, 4,000 steps. Training and test puzzles come from **different source
  games**.
- *Network + search* is alpha-beta with the network ordering moves and a
  material evaluation (node budget 6,000, depth 3 for puzzles).

### Checkers ([arena/games/checkers.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/games/checkers.py))

- Value network: 4 piece planes, four 3×3 convolutions (width 64), a
  128-unit dense layer and a tanh output: expected result for the side to move.
- Learned only from **self-play**: 20 generations × 250 games, TD(λ=0.7)
  targets, 1-ply search with softmax sampling (temperature 0.05) and 3% random
  moves. It never saw an expert game or the engines it is tested against.
- A first attempt (Monte Carlo targets, 15% random moves) lost to a random
  mover and was replaced. That run is kept on record.

### Pinpoint ([arena/games/pinpoint.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/games/pinpoint.py))

Real Pinpoint answers are often not a "category" in the dictionary sense. "Poker,
baby, clock" → **face** (poker face, baby face, clock face); "butter, dragon,
fire" → **fly** (butterfly, dragonfly, firefly). The guesser
([`LinkGuesser`](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/games/pinpoint.py)) therefore looks for a word that **links to
every clue**, using counts taken from the same Wikipedia text
([scripts/build_pinpoint_links.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/scripts/build_pinpoint_links.py), about 2 minutes on CPU):

| Link | Example | Counted from |
|---|---|---|
| two-word phrase | "poker **face**", "**face** value" | 2.2M adjacent word pairs (seen ≥3 times), scored by PMI |
| compound word | "butter" + "**fly**" = butterfly | 161,515 corpus words seen ≥5 times |
| category pattern | "**planets** such as Mars", "Mars and other **planets**", "Paris is a **city** in…" | 46,392 category/member pairs from five text patterns |
| similar meaning (fallback) | guitar ~ piano | skip-gram word vectors, 200 dims, 3 epochs, 40,000 words |

A linked answer must connect to every clue (3 of 4 when there are 3–5 clues);
otherwise the guess falls back on word-vector similarity. The app shows the
reason for each clue, e.g. `poker: "poker face"`. No pretrained model and no list
of answers is used. A small WordNet "category head" from an earlier version is
still in the repo but no longer used by the app.

### Screenshot reading ([arena/vision](https://github.com/Kedhareswer/ML-AI/tree/main/reasoning_harness/arena/vision))

1. Find the square board from edge contours, snapped to its border lines.
2. Find the grid size from *line coverage* (how far each edge runs across the
   board). Queens also checks that each cell is one solid colour.
3. Read every cell with a small CNN (5 conv layers, BatchNorm, 2 dense
   layers) trained **only on synthetic cells** drawn with random fonts,
   colours, blur, crops, noise and JPEG compression. Queens regions are found by
   clustering cell colours, with no learning.
4. Check the reading against the rules. If the puzzle is not valid and
   solvable, try the runner-up readings of the least confident cells, then the
   next grid size, and report every correction.

## Results

### Held-out puzzles

| Game | Network only | Untrained network | Network + search | Classical solver |
|---|---:|---:|---:|---:|
| Sudoku (real, Sudoku-Extreme) | 20.7% | 0.0% | 95.0%\* | 100% |
| Mini Sudoku | 99.3% | 0.7% | 100% | 100% |
| Tango | 84.0% | 0.0% | 100% | 100% |
| Queens | 90.3% | 0.3% | 100% | 100% |
| Zip | 70.3% | 0.0% | 100% | 100% |
| Patches | 72.3% | 2.3% | 100% | 100% |
| Wend | 22.0% | 2.3% | 100% | 100% |
| Chess, held-out Lichess puzzles | 44.0% | 0.3% | 58.7% | 52.3% (search, no network) |
| Chess, 297 Lichess daily puzzles | 18.2% | 0.0% | 30.0% | 22.6% (search, no network) |

\* 2,000-state budget; 15 of the hardest puzzles ran out of budget.

### Games

| Match | Win | Draw | Loss |
|---|---:|---:|---:|
| Chess: network only vs random mover (20 games) | 15 | 5 | 0 |
| Chess: network only vs 1-ply material engine (20) | 0 | 10 | 10 |
| Checkers: network (depth 2) vs material engine (depth 2) (40) | 15 | 18 | 7 |
| Checkers: network (depth 2) vs material engine (depth 4) (40) | 4 | 20 | 16 |
| Checkers: **untrained** network (depth 2) vs material engine (depth 2) (40) | 0 | 0 | 40 |

### Screenshots

280 held-out screenshots of grid puzzles drawn in a browser with fonts and
sizes the readers never trained on: **263 read exactly right (94%)**, **265
solved (95%)**. Chess boards from Lichess's image service with three unseen
piece sets: 58/60 read exactly. Per-game numbers are in [ARENA.md](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/ARENA.md#solving-from-a-screenshot).

**Tango reader update (October 2026).** A LinkedIn-style Tango screenshot
(filled crescent moons with a dark rim, pointing up-left) was misread: every
moon came out as a sun. The training renderer drew crescents facing one way
only, with no rim. It now draws crescents facing any direction with an
optional rim (suns too), and the Tango cell classifier was retrained (30,000
synthetic cells, 6 epochs, 99.9% on fresh synthetic cells). It now reads all 8
given cells of that screenshot correctly, and gives the same reading as before
on the three Tango screenshots kept in the repository. The 280-image browser
benchmark above was **not** re-run after this change.

**Patches and Wend readers (October 2026).** Patches readings that put a
square shape on an area no square can have (3, 5, 6…), or a wide/tall shape on
area 1, made the puzzle unsolvable. Those shapes are now ruled out before
decoding. Wend screenshots carry no word lengths, and the app crashed drawing
them; any split into dictionary words now counts, as the benchmark always
assumed. When a reading has no solution at all, the app now says the reading
is wrong instead of only "not solved".

### Pinpoint

Answer is the top guess / in the top 5, after 1–5 clues
([report](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/benchmarks/arena/results/pinpoint/report.json)):

| Test set | Link guesser, top 1 (after 4 / 5 clues) | Link guesser, top 5 (5 clues) | Earlier word-vector guesser, top 1 / top 5 | Random |
|---|---:|---:|---:|---:|
| 40 handwritten Pinpoint-style puzzles | 35.0% / 30.0% | 57.5% | 5.0% / 37.5% | 0% |
| 300 WordNet categories | 4.7% / 3.7% | 15.0% | 5.0% / 14.3% | 0% |
| 18 dev puzzles (used for tuning, not a test) | 77.8% / 66.7% | 77.8% | 11.1% / 27.8% | 0% |

**How this was tuned, honestly.** 18 dev puzzles were written for tuning, with
no answer shared with either test set. A first test run hurt the WordNet set
(top 1 fell to 2%), and the "link to every clue" rule was then added, so that change was
made **after seeing test results**. Links help a lot on puzzles built from
phrases and compounds. On WordNet's abstract categories (stripe, bar, band,
plaque, streak → "marking") they make no real difference.

## What this does well, and what it does not

- **Training clearly works.** Untrained networks solve 0–2% alone; trained
  ones solve 70–99% of five LinkedIn-style games with no backtracking.
- **Network-guided search reaches 100%** on every LinkedIn-style game and needs
  fewer states than search with random ordering (about 5× fewer on Queens).
- **Classical solvers are faster** on every grid puzzle here (milliseconds). The
  networks are a learning result, not a speed improvement.
- **Real Sudoku-Extreme is hard for a small network**: 2% of the hardest puzzles
  without search, and the network does not make search cheaper.
- **The chess network is a puzzle solver, not a strong player.** In normal
  games against a material engine it draws or loses.
- **Pinpoint is still weak.** Corpus links solve phrase and compound puzzles
  (face, fly, board, light) and many plain categories (planet, country,
  disease). They miss abstract or rare categories and anything Wikipedia text
  rarely says. Full Pinpoint skill needs the world knowledge of a large
  pretrained language model, which this project deliberately does not use.
- **LinkedIn puzzles are imitated, not copied.** LinkedIn's daily puzzles are
  not public, so the six LinkedIn-style games are generated from the official
  rules, each with exactly one solution. Real LinkedIn screenshots can still
  use drawing styles the readers have not seen.

## Data

| Source | Used for | Licence |
|---|---|---|
| [Sudoku-Extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) | Sudoku training (train split) and test (test split) | see dataset card |
| [Lichess puzzle database](https://database.lichess.org/#puzzles) | chess training and test, split by source game | CC0 |
| Generated by this code | Mini Sudoku, Tango, Queens, Zip, Patches, Wend | — |
| WikiText-103 | Pinpoint word vectors and link counts | CC BY-SA 3.0 |
| WordNet | Pinpoint category head and test set | WordNet licence |
| ENABLE + FrequencyWords | Wend dictionary (12,000 common words) | public domain / MIT |

## Training compute

CPU only, sharing four cores between jobs, PyTorch 2.6.0+cpu. Wall-clock times
are approximate: Sudoku 105 min, chess 68 min (resumed part only), checkers
75 min, Queens 28 min, Tango 14 min, the other grid games 5–7 min each, cell
readers about 1 h together.

## The demo app

The Space runs [arena/ui.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/arena/ui.py), a standard-library HTTP server, with
the same weights. Each tab shows where the puzzle came from, which solver
answered (the highlighted button), and whether the answer passed the
independent verifier. Chess and checkers can also play full games: network vs
classical search or a random mover in chess, and network vs a material engine
or random mover in checkers.

It runs on Hugging Face's free ZeroGPU hardware, but all inference is on CPU.
[app.py](https://github.com/Kedhareswer/ML-AI/blob/main/reasoning_harness/deploy/huggingface/app.py) registers an unused `@spaces.GPU` function because
ZeroGPU will not start without one. Most answers take under 2 seconds; the
hardest Sudoku-Extreme puzzles take 10–30 seconds, and a full chess game up to
about a minute.

## Use the weights

The model repo holds `artifacts/arena/<game>/weights.pt` (plus a `manifest.json`
and training log per model), the Pinpoint vectors and link counts, the `arena`
package that loads them, and a root `config.json` listing every model and its
files. The Hub counts a download each time `config.json` is fetched. It needs
PyTorch, NumPy, `chess` and (for screenshots) OpenCV and Pillow.

```bash
hf download Kedhareswer/puzzle-arena --local-dir puzzle-arena
cd puzzle-arena
pip install torch numpy chess opencv-python-headless pillow
```

```python
from arena.nets import load_net
from arena.registry import from_json
from arena.search import solve

net, _ = load_net("artifacts/arena/sudoku/weights.pt")
grid = [int(c) for c in
        "000000034530090600700000000000005080907001000650900100009300000000600002160009300"]
puzzle = from_json({"kind": "sudoku", "grid": grid})
result = solve(puzzle, net, "search", budget=4000)  # or "neural" (network only), "exact" (no network)
print(result["solved"], result["answer"])

# chess: best move for a position
import chess
from arena.games.chess_engine import load, Searcher
best = Searcher(load("artifacts/arena/chess/weights.pt"), 8000, 3).choose(chess.Board())
print(best.uci())
```

Run the whole app locally with
`python -m arena.cli ui --artifacts artifacts/arena` (the held-out example
puzzles are only in the source repository).

## Reproduce

```bash
python -m pip install -e '.[games]'
python -m unittest discover -s tests -v
arena ui --artifacts artifacts/arena --port 7861
python scripts/build_datasets.py --out data
arena train --data data/tango/train.jsonl --val data/tango/test.jsonl --run runs/tango --steps 2000 --width 64 --blocks 6
python scripts/train_vision.py artifacts/arena/vision
python scripts/bench_vision.py
```

Raw per-puzzle records (every answer, state count and time) are in
`benchmarks/arena/results/*/records.jsonl`.
