"""Generate unique-solution train/test sets; test seeds are disjoint and deduplicated."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.registry import build, identity, read_jsonl  # noqa: E402

SIZES = {"mini_sudoku": 6000, "tango": 2800, "queens": 2000, "zip": 2000, "patches": 15000, "wend": 15000}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--kinds", nargs="*", default=list(SIZES))
ap.add_argument("--test", type=int, default=500)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--reuse-train", action="store_true")
args = ap.parse_args()

for kind in args.kinds:
    out = Path(args.out) / kind
    if args.reuse_train and (out / "train.jsonl").exists():
        train = read_jsonl(out / "train.jsonl")  # keep an interrupted run's puzzles
    else:
        train = build(kind, int(SIZES[kind] * args.scale), 1, out / "train.jsonl", args.workers, append=True)
    build(kind, args.test, 50_000_000, out / "test.jsonl", args.workers, exclude={identity(r) for r in train})
    print(kind, "done", flush=True)
