"""Command line: arena train | bench | solve | ui | play."""
import argparse
import json
import sys


def _rows(kind, path, limit=None, skip=0, step=1):
    from .registry import read_jsonl, sudoku_rows
    if path.endswith(".csv"):
        return sudoku_rows(path, limit, skip, step)
    rows = read_jsonl(path)
    return rows[:limit] if limit else rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arena")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a puzzle network from scratch")
    t.add_argument("--data", required=True, help="train.jsonl, or Sudoku-Extreme train.csv")
    t.add_argument("--val", help="held-out jsonl/csv for validation loss")
    t.add_argument("--run", required=True)
    t.add_argument("--limit", type=int)
    t.add_argument("--csv-step", type=int, default=1, help="take every k-th CSV row")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--width", type=int, default=96)
    t.add_argument("--blocks", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--resume", action="store_true")

    b = sub.add_parser("bench", help="benchmark a trained puzzle network on held-out puzzles")
    b.add_argument("--run", required=True)
    b.add_argument("--test", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--limit", type=int, default=300)
    b.add_argument("--csv-step", type=int, default=1)
    b.add_argument("--budget", type=int, default=2000)
    b.add_argument("--threads", type=int, default=2)

    s = sub.add_parser("solve", help="solve one puzzle given as JSON")
    s.add_argument("--run", required=True)
    s.add_argument("--puzzle", required=True, help="JSON object, e.g. from a test.jsonl line")
    s.add_argument("--mode", default="search", choices=["oneshot", "neural", "search", "exact"])

    u = sub.add_parser("ui", help="local browser UI for every game")
    u.add_argument("--artifacts", default="artifacts/arena")
    u.add_argument("--port", type=int, default=7861)
    u.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept outside connections (e.g. in a container)")
    u.add_argument("--tests", default="benchmarks/arena", help="held-out puzzles used for the example buttons")

    args = ap.parse_args(argv)
    if args.cmd == "train":
        from .train import train
        rows = _rows(None, args.data, args.limit, 0, args.csv_step)
        val = _rows(None, args.val, 256) if args.val else None
        _, info = train(rows, args.run, args.steps, args.batch_size, args.width, args.blocks, args.lr,
                        args.seed, args.resume, args.threads, val)
        print(json.dumps(info, indent=2))
    elif args.cmd == "bench":
        from .bench import run_bench
        rows = _rows(None, args.test, args.limit, 0, args.csv_step)
        report = run_bench(args.run, rows, args.output, args.budget, args.threads)
        print(json.dumps(report["summary"], indent=2))
    elif args.cmd == "solve":
        from .nets import load_net
        from .registry import from_json
        from .search import solve
        net, _ = load_net(f"{args.run}/weights.pt")
        result = solve(from_json(json.loads(args.puzzle)), net, args.mode)
        print(json.dumps(result))
        return 0 if result["solved"] else 1
    elif args.cmd == "ui":
        from .ui import serve
        serve(args.artifacts, args.port, args.tests, args.host)


if __name__ == "__main__":
    sys.exit(main())
