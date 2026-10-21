"""Puzzle kinds, (de)serialisation and dataset files."""
import json
from pathlib import Path

from .puzzles import sudoku, tango, queens, zip as zip_, patches, wend

KINDS = ("sudoku", "mini_sudoku", "tango", "queens", "zip", "patches", "wend")


def from_json(d):
    kind = d["kind"]
    if kind == "sudoku":
        return sudoku.Sudoku(d["grid"])
    if kind == "mini_sudoku":
        return sudoku.MiniSudoku(d["grid"])
    if kind == "tango":
        return tango.Tango.from_json(d)
    if kind == "queens":
        return queens.Queens(d["regions"])
    if kind == "zip":
        return zip_.Zip.from_json(d)
    if kind == "patches":
        return patches.Patches.from_json(d)
    if kind == "wend":
        return wend.Wend.from_json(d)
    raise ValueError(f"unknown puzzle kind {kind}")


GENERATORS = {
    "mini_sudoku": sudoku.generate_mini,
    "tango": tango.generate,
    "queens": queens.generate,
    "zip": zip_.generate,
    "patches": patches.generate,
    "wend": wend.generate,
}


def identity(d):
    """Canonical clue identity, used to keep test puzzles out of training data."""
    return json.dumps({k: v for k, v in d.items() if k not in ("solution", "seed")}, sort_keys=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sudoku_rows(csv_path, limit=None, skip=0, step=1):
    """Real Sudoku-Extreme rows: question, answer, rating, source."""
    import csv
    out = []
    with open(csv_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < skip or (i - skip) % step:
                continue
            out.append({"kind": "sudoku", "grid": [0 if ch == "." else int(ch) for ch in row["question"]],
                        "solution": [int(ch) for ch in row["answer"]], "rating": int(row["rating"]),
                        "source": row["source"]})
            if limit and len(out) >= limit:
                break
    return out


def _make(args):
    kind, seed = args
    puzzle, solution = GENERATORS[kind](seed)
    return {**puzzle.to_json(), "solution": solution, "seed": seed}


def build(kind, count, seed0, out_path, workers=3, exclude=(), append=False):
    from multiprocessing import Pool
    exclude = set(exclude)
    seen, rows = set(), []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed = seed0
    if append and out_path.exists():  # continue an interrupted build
        rows = read_jsonl(out_path)
        seen = {identity(r) for r in rows}
        seed = max(r["seed"] for r in rows) + 1 if rows else seed0
    with Pool(workers) as pool, open(out_path, "a" if append else "w") as f:
        while len(rows) < count:
            batch = [(kind, s) for s in range(seed, seed + 4 * workers * 8)]
            seed += len(batch)
            for row in pool.imap(_make, batch):
                ident = identity(row)
                if ident in seen or ident in exclude:
                    continue
                seen.add(ident)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                if len(rows) >= count:
                    break
    return rows
