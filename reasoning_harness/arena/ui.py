"""Local browser UI for every trained model (Python standard library HTTP server)."""
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

PAGE = Path(__file__).with_name("ui.html")
PUZZLES = ("sudoku", "mini_sudoku", "tango", "queens", "zip", "patches", "wend")


class Models:
    def __init__(self, root, tests):
        self.root, self.tests, self.cache, self.lock = Path(root), Path(tests), {}, Lock()

    def available(self):
        out = {}
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            m = d / "manifest.json"
            if m.exists():
                out[d.name] = json.loads(m.read_text())
        return out

    def get(self, kind):
        with self.lock:
            if kind not in self.cache:
                if kind in PUZZLES:
                    from .nets import load_net
                    self.cache[kind] = load_net(self.root / kind / "weights.pt")[0]
                elif kind == "chess":
                    from .games.chess_engine import load
                    self.cache[kind] = load(self.root / "chess" / "weights.pt")
                elif kind == "checkers":
                    from .games.checkers import load
                    self.cache[kind] = load(self.root / "checkers" / "weights.pt")
                elif kind == "pinpoint":
                    from .games.pinpoint import Guesser, LinkGuesser
                    cands = (self.root / "pinpoint" / "candidates.txt").read_text().split()
                    self.cache[kind] = LinkGuesser(self.root / "pinpoint", Guesser(self.root / "pinpoint", cands))
            return self.cache[kind]

    def example(self, kind, index=None):
        name = {"chess": "chess/test.jsonl", "chess_daily": "chess/daily.jsonl",
                "pinpoint": "pinpoint/handwritten.jsonl"}.get(kind, f"{kind}/test.jsonl")
        lines = (self.tests / name).read_text().splitlines()
        i = random.randrange(len(lines)) if index is None else int(index) % len(lines)
        return {"index": i, "row": json.loads(lines[i])}


def handle(models, path, payload):
    from .registry import from_json
    from .search import solve
    if path == "/api/solve":
        row = payload["puzzle"]
        if row.get("kind") not in PUZZLES:
            raise ValueError("unknown puzzle kind")
        mode = payload.get("mode", "search")
        if mode not in ("neural", "search", "oneshot", "exact"):
            raise ValueError("unknown mode")
        puzzle = from_json(row)
        net = None if mode == "exact" else models.get(row["kind"])
        result = solve(puzzle, net, mode, budget=4000)
        if row["kind"] == "wend" and not result["solved"] and not row.get("extended"):
            # no split in the 12,000 common words: retry once with the larger dictionary ("compile")
            puzzle = from_json({**row, "extended": True})
            result = {**solve(puzzle, net, mode, budget=4000), "extended_dictionary": True}
        if row["kind"] == "wend" and result["answer"]:
            result["words"] = [puzzle.word([tuple(c) for c in p]) for p in result["answer"]]
        return result
    if path == "/api/chess/solve":
        import chess
        from .games.chess_engine import make_player, solve_puzzle
        mode = payload.get("mode", "neural")
        if mode not in ("neural", "search", "classic"):
            raise ValueError("unknown mode")
        row = payload["puzzle"]
        player = make_player(mode, models.get("chess") if mode != "classic" else None, budget=6000, depth=3)
        ok, played = solve_puzzle(row, player)
        board = chess.Board(row["FEN"])
        line = row["Moves"].split()
        board.push_uci(line[0])
        return {"solved": ok, "played": played, "solution": line[1:], "start_fen": board.fen(), "mode": mode}
    if path == "/api/chess/game":
        import chess
        from .games.chess_engine import make_player, play_game
        rng = random.Random(payload.get("seed", 0))
        opp = payload.get("opponent", "classic")
        if opp not in ("classic", "random"):
            raise ValueError("unknown opponent")
        board, opening = chess.Board(), []
        for _ in range(2):  # two random opening plies so games differ
            m = rng.choice(list(board.legal_moves))
            opening.append(m.uci())
            board.push(m)
        # small budgets keep a full game near a minute on two CPUs
        white = make_player("search", models.get("chess"), budget=600, depth=2)
        black = make_player(opp, None, budget=600, depth=2, rng=rng)
        result, end, how = play_game(white, black, max_plies=120, opening=opening)
        replay, fens, sans = chess.Board(), [chess.STARTING_FEN], []
        for m in end.move_stack:
            sans.append(replay.san(m))
            replay.push(m)
            fens.append(replay.fen())
        return {"result": result, "termination": how, "fens": fens, "sans": sans,
                "moves": [m.uci() for m in end.move_stack],
                "white": "trained network + search", "black": "classical search (no network)" if opp == "classic" else "random mover"}
    if path == "/api/checkers/game":
        from .games import checkers as ck
        rng = random.Random(payload.get("seed", 0))
        net = models.get("checkers")
        engine = {"kind": "net", "net": net, "depth": 2}
        opponent = {"random": {"kind": "random"}, "material": {"kind": "material", "depth": 2}}[payload.get("opponent", "material")]
        result, hist, plies = ck.play(engine, opponent, rng, opening_random=2, record=True)
        frames = [ck.absolute(b, side).tolist() for b, side in hist]
        return {"result": result, "plies": plies, "frames": frames,
                "black": "trained network (depth 2)", "white": f"{payload.get('opponent', 'material')} engine"}
    if path == "/api/read":
        import base64
        from .vision.read import read
        kind = payload["kind"]
        data = payload["image"]
        if "," in data[:100]:
            data = data.split(",", 1)[1]
        out = read(kind, base64.b64decode(data), root=str(models.root / "vision"),
                   n=payload.get("n") or None, to_move=payload.get("to_move", "w"))
        if "puzzle" in out:
            from_json(out["puzzle"])  # raises ValueError if the read grid is not a valid puzzle
        out.pop("quad", None)
        return out
    if path == "/api/chess/best":
        import chess
        from .games.chess_engine import Searcher
        board = chess.Board(payload["fen"])
        if not board.is_valid():
            raise ValueError("the board read from the image is not a legal chess position")
        move = Searcher(models.get("chess"), 8000, 3).choose(board)
        return {"uci": move.uci(), "san": board.san(move), "fen": board.fen()}
    if path == "/api/checkers/best":
        import numpy as np
        from .games import checkers as ck
        grid = np.array(payload["board"], np.int8)
        dark_to_move = payload.get("to_move", "dark") == "dark"
        internal = grid if dark_to_move else ck.flip(grid)
        move = ck.choose(internal, {"kind": "net", "net": models.get("checkers"), "depth": 4}, random.Random(0))
        if move is None:
            return {"move": None, "message": "no legal move: the side to move has lost"}
        path_ = [list(p) if dark_to_move else [7 - p[0], 7 - p[1]] for p in move[0]]
        result, _ = ck.apply(internal, move)  # comes back in the opponent's view
        after = ck.flip(result) if dark_to_move else result  # back to the absolute (dark-up) view
        return {"move": path_, "captures": len(move[1]), "after": after.tolist()}
    if path == "/api/pinpoint":
        clues = [c for c in payload["clues"] if isinstance(c, str) and c.strip()][:5]
        guesser = models.get("pinpoint")
        after = [guesser.rank(clues[:k], 5) for k in range(1, len(clues) + 1)]
        best = after[-1][0][0] if after and after[-1] else None
        return {"clues": clues, "after": after, "best": best, "why": guesser.explain(clues, best) if best else [],
                "unknown": [c for c in clues if not guesser.g.known(c)]}
    raise KeyError(path)


def serve(root="artifacts/arena", port=7861, tests="benchmarks/arena", host="127.0.0.1"):
    models = Models(root, tests)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, value, ctype="application/json"):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/models":
                    return self.send(200, models.available())
                if u.path == "/api/example":
                    return self.send(200, models.example(q["kind"][0], q.get("index", [None])[0]))
                self.send(404, {"error": "not found"})
            except (KeyError, ValueError, FileNotFoundError) as e:
                self.send(400, {"error": str(e)})

        def do_POST(self):
            origin = self.headers.get("Origin")
            # https when served behind a TLS proxy (Hugging Face Spaces)
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host"):
                return self.send(403, {"error": "cross-origin requests are disabled"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= (12_000_000 if self.path == "/api/read" else 65536):
                    raise ValueError("expected a small JSON body")
                self.send(200, handle(models, self.path, json.loads(self.rfile.read(n))))
            except KeyError:
                self.send(404, {"error": "not found"})
            except (ValueError, TypeError) as e:
                self.send(400, {"error": str(e)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"arena UI on http://{host}:{port}", flush=True)
    server.serve_forever()
