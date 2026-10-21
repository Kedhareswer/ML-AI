"""End-to-end image benchmark: screenshot -> reader -> solver -> verified against the true puzzle.

Test images use styles the cell classifiers never saw in training:
  * browser: held-out puzzles drawn by the arena UI in Chromium (Inter font, which
    is excluded from training), light and dark colour schemes, several window
    sizes, then JPEG-compressed and rescaled like a phone screenshot;
  * lichess: held-out chess puzzles rendered by Lichess's own board-image service,
    including piece sets (merida, california, alpha) excluded from training.
"""
import argparse
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from arena.bench import wilson  # noqa: E402
from arena.registry import from_json  # noqa: E402
from arena.vision.read import read  # noqa: E402

GRIDS = ("sudoku", "mini_sudoku", "tango", "queens", "zip", "patches", "wend")


def canonical(kind, d):
    """Clue content only, in a comparable form."""
    if kind in ("sudoku", "mini_sudoku"):
        return tuple(d["grid"])
    if kind == "tango":
        return (frozenset(map(tuple, d["givens"])), frozenset(map(tuple, d["edges"])))
    if kind == "queens":
        reg = np.array(d["regions"])
        remap, out = {}, []
        for v in reg.ravel():
            remap.setdefault(int(v), len(remap))
            out.append(remap[int(v)])
        return (reg.shape, tuple(out))
    if kind == "zip":
        return (d["n"], frozenset(map(tuple, d["numbers"])), frozenset(tuple(sorted([tuple(w[:2]), tuple(w[2:])])) for w in d.get("walls", [])))
    if kind == "patches":
        return (d["n"], frozenset(tuple(c) for c in d["clues"]))
    if kind == "wend":
        return tuple(d["rows"])
    raise ValueError(kind)


def _check_against_truth(kind, row, got, answer):
    """Is the answer to the puzzle read from the image a valid answer to the true puzzle?"""
    truth = from_json(row)
    if kind == "patches":  # answers are listed in clue order; reorder to the true puzzle's clues
        rect_at = {(c[0], c[1]): a for c, a in zip(got["clues"], answer)}
        try:
            answer = [rect_at[(c[0], c[1])] for c in row["clues"]]
        except KeyError:
            return False
    if kind == "wend":  # the image does not show word lengths: any dictionary split of the true grid counts
        from arena.puzzles.wend import Wend
        truth = Wend(row["rows"], None)
    return truth.verify(answer)


def phone_like(png, rng):
    """JPEG + rescale, as a screenshot shared from a phone or chat app would be."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    scale = rng.uniform(0.6, 1.3)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(55, 92))
    return buf.getvalue()


def browser_images(url, items, rng, chromium):
    """items: list of (kind, js_expression_that_draws_it). Yields (index, scheme, jpeg bytes)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, args=["--no-sandbox"])
        for scheme in ("light", "dark"):
            for i, (kind, js) in enumerate(items):
                if (i % 2 == 0) != (scheme == "light"):
                    continue  # alternate schemes: half the cases light, half dark
                vw, vh = rng.choice([(1180, 760), (900, 1000), (1400, 900), (760, 1100)])
                page = browser.new_page(viewport={"width": vw, "height": vh}, color_scheme=scheme)
                page.goto(f"{url}/?kind={kind}")
                page.wait_for_selector("nav button.on")
                page.wait_for_timeout(400)
                page.evaluate(js)
                page.wait_for_timeout(150)
                png = page.screenshot()
                page.close()
                yield i, scheme, phone_like(png, rng)


def lichess_image(fen, piece, theme, orientation):
    q = urllib.parse.urlencode({"fen": fen, "color": orientation, "piece": piece, "theme": theme})
    with urllib.request.urlopen(f"https://lichess1.org/export/fen.gif?{q}", timeout=30) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    ap.add_argument("--artifacts", default="artifacts/arena")
    ap.add_argument("--tests", default="benchmarks/arena")
    ap.add_argument("--output", default="benchmarks/arena/results/vision")
    ap.add_argument("--per-kind", type=int, default=60)
    ap.add_argument("--lichess", type=int, default=120)
    ap.add_argument("--chromium", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    ap.add_argument("--kinds", nargs="*", default=list(GRIDS) + ["chess", "checkers"])
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    from arena.nets import load_net
    from arena.search import solve
    rng = random.Random(2026)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    vision = Path(args.artifacts) / "vision"
    report, records = {}, []

    for kind in [k for k in args.kinds if k in GRIDS]:
        rows = [json.loads(l) for l in open(Path(args.tests) / kind / "test.jsonl")]
        rows = rows[-args.per_kind:]  # the end of the test file: not the puzzles shown in the demo
        net = load_net(Path(args.artifacts) / kind / "weights.pt")[0]
        items = []
        for row in rows:
            show = {k: v for k, v in row.items() if k not in ("solution", "seed", "rating", "source")}
            items.append((kind, f"row={json.dumps(show)};draw(row,null,0)"))
        stats = {"cases": 0, "read_exact": 0, "solved": 0, "low_conf_flagged": 0, "by_scheme": {}}
        for i, scheme, img in browser_images(args.url, items, rng, args.chromium):
            row = rows[i]
            t = time.perf_counter()
            try:
                res = read(kind, img, root=vision)
                got = res["puzzle"]
                exact = canonical(kind, got) == canonical(kind, row)
                puzzle = from_json(got)
                sol = solve(puzzle, net, "search", budget=4000)
                solved = bool(sol["solved"] and _check_against_truth(kind, row, got, sol["answer"]))
                err = None
            except Exception as e:  # unreadable or invalid grid counts as a failure
                exact, solved, err, res = False, False, str(e)[:200], {"confidence": 0.0, "n": None}
            rec = {"kind": kind, "index": i, "scheme": scheme, "read_exact": exact, "solved_true_puzzle": solved,
                   "n_read": res.get("n"), "n_true": from_json(row).n, "confidence": round(res.get("confidence", 0), 4),
                   "error": err, "ms": round(1000 * (time.perf_counter() - t), 1)}
            records.append(rec)
            stats["cases"] += 1
            stats["read_exact"] += exact
            stats["solved"] += solved
            stats["low_conf_flagged"] += res.get("confidence", 0) < 0.6
            sc = stats["by_scheme"].setdefault(scheme, {"cases": 0, "read_exact": 0, "solved": 0})
            sc["cases"] += 1
            sc["read_exact"] += exact
            sc["solved"] += solved
        stats["read_exact_wilson95"] = wilson(stats["read_exact"], stats["cases"])
        stats["solved_wilson95"] = wilson(stats["solved"], stats["cases"])
        report[kind] = stats
        print(kind, json.dumps(stats), flush=True)

    if "chess" in args.kinds:
        import chess
        from arena.games.chess_engine import Searcher, load
        chess_net = load(Path(args.artifacts) / "chess" / "weights.pt")
        rows = [json.loads(l) for l in open(Path(args.tests) / "chess" / "test.jsonl")][-(args.per_kind + args.lichess):]
        cases = []
        for row in rows:
            b = chess.Board(row["FEN"])
            b.push_uci(row["Moves"].split()[0])
            cases.append((b, row["Moves"].split()[1]))
        sets = {}

        def score(source, b, best, img):
            to_move = "w" if b.turn else "b"
            try:
                res = read("chess", img, root=vision, to_move=to_move)
                got = chess.Board(res["fen"])
                exact = got.board_fen() == b.board_fen()
                got.turn = b.turn
                move = Searcher(chess_net, 3000, 3).choose(got) if got.is_valid() else None
                ok = move is not None and move in b.legal_moves and (move.uci() == best or _mates(b, move))
                err = None
            except Exception as e:
                exact, ok, err = False, False, str(e)[:200]
            s = sets.setdefault(source, {"cases": 0, "read_exact": 0, "first_move_correct": 0})
            s["cases"] += 1
            s["read_exact"] += exact
            s["first_move_correct"] += ok
            records.append({"kind": "chess", "source": source, "read_exact": exact, "first_move_correct": ok, "error": err})

        items = [("chess", f"row={{FEN:{json.dumps(b.fen())},_start:{json.dumps(b.fen())}}};drawChess(fenBoard({json.dumps(b.fen())}))")
                 for b, _ in cases[:args.per_kind]]
        for i, scheme, img in browser_images(args.url, items, rng, args.chromium):
            score(f"browser-{scheme}", cases[i][0], cases[i][1], img)
        pieces = ["merida", "california", "alpha", "cburnett"]
        for k, (b, best) in enumerate(cases[args.per_kind:]):
            piece = pieces[k % 4]
            img = lichess_image(b.fen(), piece, rng.choice(["brown", "blue", "green", "purple", "ic"]),
                                "white" if b.turn else "black")
            score(f"lichess-{piece}{' (training set)' if piece == 'cburnett' else ' (unseen)'}", b, best, img)
        for s in sets.values():
            s["read_exact_wilson95"] = wilson(s["read_exact"], s["cases"])
        report["chess"] = sets
        print("chess", json.dumps(sets), flush=True)

    if "checkers" in args.kinds:
        from arena.games import checkers as ck
        positions = []
        prng = random.Random(77)
        while len(positions) < args.per_kind:
            _, hist, _ = ck.play({"kind": "random"}, {"kind": "material", "depth": 1}, prng, record=True)
            b, side = hist[prng.randrange(len(hist))]
            positions.append(ck.absolute(b, side))
        items = [("checkers", f"drawCheckers({json.dumps(p.tolist())})") for p in positions]
        st = {"cases": 0, "read_exact": 0}
        for i, scheme, img in browser_images(args.url, items, rng, args.chromium):
            try:
                res = read("checkers", img, root=vision)
                exact = np.array_equal(np.array(res["board"]), positions[i])
            except Exception:
                exact = False
            st["cases"] += 1
            st["read_exact"] += exact
            records.append({"kind": "checkers", "scheme": scheme, "read_exact": exact})
        st["read_exact_wilson95"] = wilson(st["read_exact"], st["cases"])
        report["checkers"] = st
        print("checkers", json.dumps(st), flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _mates(board, move):
    b = board.copy()
    b.push(move)
    return b.is_checkmate()


if __name__ == "__main__":
    main()
