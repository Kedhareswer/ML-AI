# Tiny System 2

> **New: [Puzzle Arena](ARENA.md)**: networks trained from scratch for Sudoku (real Sudoku-Extreme), Mini Sudoku, Tango, Queens, Zip, Patches, Wend, chess (real Lichess puzzles), checkers and a Pinpoint experiment, with held-out benchmarks, screenshots and a screen recording.

A **32.10 MB model trained from scratch**, with Python, CLI, a browser UI, a Colab notebook, resumable training and reproducible benchmarks. The current application solves small maze-planning problems. It is not a general chatbot or a multimodal training platform.

Two modes are deliberately separate:

- **Neural policy only:** predicts moves and follows a bounded self-avoiding rollout. Four-attempt mode resamples; this policy does not learn corrections from textual feedback.
- **Neural policy + backtracking:** the same network orders a depth-first exploration of legal states. Search can recover from a wrong branch and a verifier checks the returned route. At most 64 distinct states are expanded. Completeness on these grids comes from search, not from a claim of perfect neural reasoning.

Read the [current results and model card](MODEL_CARD.md), [benchmark evidence](benchmarks/POLICY_RESULTS.md), and [actual browser captures](evidence/README.md). Earlier failed autoregressive experiments are retained under `benchmarks/results/` and documented in the [historical report](benchmarks/README.md).

## Run the packaged model

Python 3.10+ and PyTorch 2.6+ are required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
# Optional on CPU-only machines: install the official CPU wheel first.
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

sys2 solve --run artifacts/policy-v1 --maze 'S.../XXX./..../GXXX' --mode model --temperature 0 --attempts 1
sys2 solve --run artifacts/policy-v1 --maze 'S.../XXX./..../GXXX' --mode guided
sys2 ui --run artifacts/policy-v1 --temperature 0 --port 7860
```

The UI binds to the local machine on port 7860. It has explicit model-only and backtracking choices, a maze display, actual verifier results, route playback and model metadata. Route playback happens after inference; it is not a recording of hidden reasoning.

Inference requires `weights.pt` and `config.json`. The included `manifest.json`, training logs and checksums document the released artifact. No pretrained weights, API key or external dataset is required. The earlier 38 MB decoder is still supported by `sys2 train`, but `train-policy` is the recommended working model.

## Python API

```python
from sys2.engine import load, solve
from sys2.tasks import Maze

model = load('artifacts/policy-v1', device='cpu')
maze = Maze(('S...', 'XXX.', '....', 'GXXX'))
print(solve(model, maze, attempts=1, temperature=0))  # neural decisions only
print(model.search(maze, budget=64))                 # explicit backtracking
```

## Train and resume

```bash
sys2 train-policy --run runs/policy --steps 1500 --mazes 3000 --batch-size 64 --seed 314
sys2 train-policy --run runs/policy --resume --steps 2000
```

This recipe uses random 3x3/4x4 mazes and expert labels for **every reachable current position**. BFS generates the offline training labels; it is not called to choose moves during neural inference or guided search. Tied optimal actions share the target probability mass. The verifier uses BFS to score optimality after a candidate is produced.

The model is a four-hidden-layer MLP of width 1600: 8,024,004 parameters, 207 input features, four move logits. Inputs include padded walls/current/goal planes, coordinates, local legality and local goal-distance features. Illegal moves and revisits are masked by the rollout wrapper. These features and constraints are supplied to the untrained controls too. The benchmark measures the added value of learned priorities, rather than crediting the network for the wrapper's guarantees.

Training uses FP32, AdamW, gradient clipping, periodic checkpoints and bounded CPU threading. GPU is selected if available, but CUDA execution remains unvalidated in this CPU-only environment. `--steps` is the total target. Resume restores the original data settings, architecture, optimizer and RNG; CPU resumed-versus-uninterrupted equality is tested. A checkpoint is saved every 250 steps and on ordinary interruption. An abrupt runtime termination can only recover the last periodic checkpoint.

Fresh run directories are required to prevent silent overwrites. Optimizer checkpoints are much larger than inference weights. The packaged release includes inference weights; the original full resumable checkpoint is outside the checkout at `/workspace/ml-ai-environment/policy-full-v1/last.pt` in this development instance. Future training runs create their own `last.pt`.

## Benchmarks and tests

```bash
python -m unittest discover -s tests -v
sys2 benchmark --run artifacts/policy-v1 --output runs/fresh-benchmark \
  --count 100 --sizes 4 5 6 8 --seeds 37000000 38000000 39000000 \
  --attempts 4 --temperature 0.8 --require-local-solver
```

The suite records all cases/paths, verifier checks, greedy/model resampling, untrained network, random paths, guided search, untrained-guided search, exact BFS, uncertainty intervals, latency, memory, held-out decision loss and fixed stress cases. For the policy, **the entire training wall-layout/goal combination is excluded, regardless of start position**, because all reachable starts were supervised. Test tasks are also deduplicated by that identity.

`--require-local-solver` enforces the separately reported **local hybrid solver** gates and saves evidence before a failure exit. `--require-ready` retains the earlier, stricter neural-feedback/GPU research gates; it will fail for this policy because the policy has no learned textual-feedback mechanism and GPU checks have not run. A successful evaluation process alone does not certify a model.

BFS is an exact, fast baseline for this task and often the better practical solver. The neural policy is a learning experiment; the hybrid's completeness must not be mistaken for evidence that it outperforms search. Performance and accuracy are limited to the synthetic maze distributions tested.

## Colab and evidence

[notebooks/colab_start.ipynb](notebooks/colab_start.ipynb) trains the same policy package and runs its benchmarks. Commit/push these files before cloning from GitHub. Optional Drive storage retains checkpoints across runtime deletion. The notebook supports CPU execution and GPU selection; a real Colab GPU/disconnect recovery has not been validated here.

Actual UI recording can be repeated after installing Playwright and its video support:

```bash
python -m pip install playwright
# Install a Chromium browser and Playwright's ffmpeg support, or use trusted system binaries.
python scripts/capture_demo.py --output evidence/new-recording --chromium /usr/bin/chromium
```

Start the UI with `--temperature 0` first. Screenshots and recordings come from browser interactions with the real model API. They are selected demonstrations, not substitutes for aggregate benchmarks.
