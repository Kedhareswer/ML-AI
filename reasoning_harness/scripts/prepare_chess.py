"""Lichess puzzles -> training arrays + held-out test sets (split by source game)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.games.chess_net import build_dataset  # noqa: E402

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
test_rows, daily_rows = build_dataset(src, out / "train.npz")
rng = random.Random(0)
rng.shuffle(test_rows)
keep = ("PuzzleId", "FEN", "Moves", "Rating", "Themes", "GameUrl", "DailyDate")
with open(out / "test.jsonl", "w") as f:
    for row in test_rows[:3000]:
        f.write(json.dumps({k: row[k] for k in keep}) + "\n")
with open(out / "daily.jsonl", "w") as f:
    for row in daily_rows:
        f.write(json.dumps({k: row[k] for k in keep}) + "\n")
print("test", len(test_rows), "daily", len(daily_rows))
