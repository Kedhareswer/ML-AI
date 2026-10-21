"""Benchmarks for chess (real Lichess puzzles + games), checkers (games) and Pinpoint."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from arena.bench import wilson  # noqa: E402


def rate(k, n):
    return {"solved": k, "total": n, "rate": round(100 * k / max(n, 1), 2), "wilson95": wilson(k, n)}


def chess_puzzles(args):
    from arena.games.chess_engine import load, make_player, solve_puzzle
    from arena.games.chess_net import ChessNet
    net = load(Path(args.artifacts) / "chess" / "weights.pt")
    torch.manual_seed(1234)
    untrained = ChessNet(**net.config).eval()
    sets = {"test": [json.loads(l) for l in open(Path(args.tests) / "chess" / "test.jsonl")][:args.limit],
            "daily": [json.loads(l) for l in open(Path(args.tests) / "chess" / "daily.jsonl")]}
    rng = random.Random(0)
    players = {
        "neural": ("Trained network only (top move, no search)", make_player("neural", net)),
        "neural_untrained": ("Random-initialised network (control)", make_player("neural", untrained)),
        "search": ("Trained network + alpha-beta search", make_player("search", net, args.budget, 3)),
        "classic": ("Alpha-beta search, no network (baseline)", make_player("classic", None, args.budget, 3)),
        "random": ("Random legal move (control)", make_player("random", rng=rng)),
    }
    out = Path(args.output) / "chess"
    out.mkdir(parents=True, exist_ok=True)
    report = {"budget_nodes_per_move": args.budget, "sets": {}}
    for name, rows in sets.items():
        recs = []
        for i, row in enumerate(rows):
            rec = {"PuzzleId": row["PuzzleId"], "Rating": int(row["Rating"]), "Themes": row["Themes"], "results": {}}
            for key, (_, player) in players.items():
                t = time.perf_counter()
                ok, played = solve_puzzle(row, player)
                rec["results"][key] = {"solved": ok, "played": played, "ms": round(1000 * (time.perf_counter() - t), 1)}
            recs.append(rec)
            if (i + 1) % 25 == 0:
                print(name, i + 1, flush=True)
        with open(out / f"{name}-records.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        summ = {}
        for key, (desc, _) in players.items():
            k = sum(r["results"][key]["solved"] for r in recs)
            bands = {}
            for lo, hi in ((0, 1200), (1200, 1600), (1600, 2000), (2000, 4000)):
                sub = [r for r in recs if lo <= r["Rating"] < hi]
                if sub:
                    bands[f"{lo}-{hi}"] = rate(sum(r["results"][key]["solved"] for r in sub), len(sub))
            summ[key] = {"description": desc, **rate(k, len(recs)), "by_rating": bands,
                         "p50_ms": float(np.median([r["results"][key]["ms"] for r in recs]))}
        report["sets"][name] = summ
    (out / "puzzles-report.json").write_text(json.dumps(report, indent=2))
    return report


def chess_games(args):
    from arena.games.chess_engine import load, make_player, play_game
    net = load(Path(args.artifacts) / "chess" / "weights.pt")
    rng = random.Random(7)
    matches = [
        ("Trained network only vs random mover", make_player("neural", net), make_player("random", rng=rng), args.games),
        ("Trained network only vs 1-ply material engine", make_player("neural", net), make_player("classic", None, 400, 1), args.games),
        ("Network + search vs same search without network", make_player("search", net, 1500, 2), make_player("classic", None, 1500, 2), args.games // 2),
    ]
    out = {}
    for desc, a, b, n in matches:
        score = {"win": 0, "draw": 0, "loss": 0}
        games = []
        for g in range(n):
            white_is_net = g % 2 == 0
            res, board, term = play_game(a if white_is_net else b, b if white_is_net else a)
            net_res = {"1-0": 1, "0-1": -1}.get(res, 0) * (1 if white_is_net else -1)
            score[{1: "win", 0: "draw", -1: "loss"}[net_res]] += 1
            games.append({"net_white": white_is_net, "result": res, "termination": term, "plies": board.ply()})
        out[desc] = {**score, "games": games}
        print(desc, score, flush=True)
    p = Path(args.output) / "chess"
    p.mkdir(parents=True, exist_ok=True)
    (p / "games-report.json").write_text(json.dumps(out, indent=2))
    return out


def checkers_games(args):
    from arena.games import checkers as ck
    net = ck.load(Path(args.artifacts) / "checkers" / "weights.pt")
    torch.manual_seed(1234)
    untrained = ck.ValueNet(**net.config).eval()
    rng = random.Random(11)
    trained = {"kind": "net", "net": net, "depth": 2}
    matches = [
        ("Trained network (depth 1) vs random mover", {"kind": "net", "net": net, "depth": 1}, {"kind": "random"}),
        ("Trained network (depth 1) vs material engine (depth 1)", {"kind": "net", "net": net, "depth": 1},
         {"kind": "material", "depth": 1}),
        ("Trained network (depth 2) vs random mover", trained, {"kind": "random"}),
        ("Trained network (depth 2) vs material engine (depth 2)", trained, {"kind": "material", "depth": 2}),
        ("Trained network (depth 2) vs material engine (depth 4)", trained, {"kind": "material", "depth": 4}),
        ("Untrained network (depth 2) vs material engine (depth 2) [control]", {"kind": "net", "net": untrained, "depth": 2},
         {"kind": "material", "depth": 2}),
    ]
    out = {}
    for desc, a, b in matches:
        score = {"win": 0, "draw": 0, "loss": 0}
        for g in range(args.games):
            net_black = g % 2 == 0
            res, _, _ = ck.play(a if net_black else b, b if net_black else a, rng, opening_random=2)
            r = res if net_black else -res
            score[{1: "win", 0: "draw", -1: "loss"}[r]] += 1
        n = args.games
        out[desc] = {**score, "games": n, "score_pct": round(100 * (score["win"] + 0.5 * score["draw"]) / n, 1),
                     "win_rate_wilson95": wilson(score["win"], n)}
        print(desc, out[desc], flush=True)
    p = Path(args.output) / "checkers"
    p.mkdir(parents=True, exist_ok=True)
    (p / "games-report.json").write_text(json.dumps(out, indent=2))
    return out


def pinpoint(args):
    from arena.games.pinpoint import Guesser, evaluate, random_guesser, wordnet_puzzles
    root = Path(args.artifacts) / "pinpoint"
    cands = (root / "candidates.txt").read_text().split()
    g = Guesser(root, cands)
    hand = [json.loads(l) for l in open(Path(args.tests) / "pinpoint" / "handwritten.jsonl")]
    wn = [json.loads(l) for l in open(Path(args.tests) / "pinpoint" / "wordnet.jsonl")]
    rng = random.Random(3)
    out = {}
    for name, puzzles in (("handwritten", hand), ("wordnet", wn)):
        res, recs = evaluate(g.rank, puzzles)
        plain, _ = evaluate(lambda c, k=5: g.rank(c, k, use_head=False), puzzles)
        ctrl, _ = evaluate(random_guesser(cands, rng), puzzles)
        out[name] = {"vectors_plus_category_head": res, "vectors_only": plain, "random_guess_control": ctrl}
        p = Path(args.output) / "pinpoint"
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{name}-records.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(name, json.dumps(out[name]), flush=True)
    (Path(args.output) / "pinpoint" / "report.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["chess-puzzles", "chess-games", "checkers", "pinpoint"])
    ap.add_argument("--artifacts", default="artifacts/arena")
    ap.add_argument("--tests", default="benchmarks/arena")
    ap.add_argument("--output", default="benchmarks/arena/results")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"chess-puzzles": chess_puzzles, "chess-games": chess_games, "checkers": checkers_games, "pinpoint": pinpoint}[a.what](a)
