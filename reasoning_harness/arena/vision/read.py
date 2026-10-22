"""Image -> puzzle readers for every game.

Each reader finds the board, estimates the grid size, and classifies every
cell with a classifier trained from scratch on synthetic cells. The most
likely reading is checked against the game's rules (a valid, solvable puzzle;
a legal chess position). If it fails, the reader tries the runner-up labels of
its least confident cells, most probable first, and reports any correction.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import board as B
from .cells import Classifier

SIZES = {"sudoku": [9], "mini_sudoku": [6], "tango": [6], "queens": list(range(5, 11)), "zip": list(range(4, 9)),
         "patches": list(range(4, 8)), "wend": list(range(4, 8)), "chess": [8], "checkers": [8]}


@lru_cache(maxsize=16)
def clf(root, name):
    return Classifier(Path(root) / f"{name}.pt")


def _solvable(d, budget=30_000):
    from ..registry import from_json
    from ..search import search
    try:
        puzzle = from_json(d)
    except (ValueError, KeyError, IndexError):
        return False
    found, _, exhausted = search(puzzle, None, budget, count_limit=1)
    return bool(found)


def _decode(slots, build, check, max_tries=80):
    """slots: list of (probabilities, labels, name). Returns (result, corrections or None if unrepaired)."""
    choice = [int(p.argmax()) for p, _, _ in slots]
    make = lambda ch: build([labels[c] for (_, labels, _), c in zip(slots, ch)])  # noqa: E731
    first = make(choice)
    if check is None or check(first):
        return first, []
    conf = np.array([p[c] for (p, _, _), c in zip(slots, choice)])
    weak = list(np.argsort(conf)[:10])
    alts = {i: [int(j) for j in np.argsort(-slots[i][0])[1:3]] for i in weak}
    tries = [((i, j),) for i in weak for j in alts[i]]
    tries += [((a, ja), (b, jb)) for x, a in enumerate(weak[:6]) for b in weak[x + 1:6] for ja in alts[a] for jb in alts[b]]
    logit = lambda i, j: np.log(slots[i][0][j] + 1e-9) - np.log(conf[i] + 1e-9)  # noqa: E731
    tries.sort(key=lambda t: -sum(logit(i, j) for i, j in t))
    for t in tries[:max_tries]:
        ch = list(choice)
        for i, j in t:
            ch[i] = j
        cand = make(ch)
        if check(cand):
            return cand, [{"cell": slots[i][2], "read": slots[i][1][choice[i]], "corrected_to": slots[i][1][j],
                           "confidence": round(float(conf[i]), 3)} for i, j in t]
    return first, None


def read(kind, image, root="artifacts/arena/vision", n=None, to_move="w", repair=True):
    rgb = B.load(image)
    sizes = [n] if n else SIZES[kind]
    board, quad, n0 = B.find_board(rgb, sizes)
    # Grid sizes in order of how well they explain the board's lines. If the best one
    # does not give a valid, solvable puzzle, try the next plausible size.
    scores = {m: B.gridness(board, m) for m in sizes}
    ranked = [n0] + [m for m in sorted(scores, key=scores.get, reverse=True)
                     if m != n0 and scores[m] >= scores[n0] - 0.35][:2]
    if kind == "queens" and len(sizes) > 1:
        # at the true size every cell is one solid region colour; at a wrong size many cells straddle two
        plausible = [m for m in sizes if scores[m] >= max(scores.values()) - 0.5]
        ranked = sorted(plausible, key=lambda m: (_mixed_cells(board, m), -scores[m]))
    first = None
    for m in ranked:
        out = _read_at(kind, board, m, root, to_move, repair)
        out["quad"] = np.asarray(quad).round(1).tolist()
        if first is None:
            first = out
        if out["consistent"] or not repair:
            return out
    return first


def _read_at(kind, board, n, root, to_move, repair):
    out = {"kind": kind, "n": n}
    C = lambda name: clf(str(root), name)  # noqa: E731
    cell_names = [f"r{i // n + 1}c{i % n + 1}" for i in range(n * n)]

    def slots_for(name, crops, names):
        model = C(name)
        return [(p, model.labels, nm) for p, nm in zip(model.probs(crops), names)]

    check = _solvable if repair else None
    if kind in ("sudoku", "mini_sudoku"):
        slots = slots_for("digits", B.cells(board, n), cell_names)
        extra = {"n": 9, "box": [3, 3]} if kind == "sudoku" else {"n": 6, "box": [2, 3]}
        build = lambda ls: {"kind": kind, "grid": [0 if l == "blank" else int(l) for l in ls], **extra}  # noqa: E731
        puzzle, fixes = _decode(slots, build, check)
    elif kind == "tango":
        crops, where = [], []
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    crops.append(B.edge_crop(board, n, r, c, False)), where.append((r, c, r, c + 1))
                if r + 1 < n:
                    crops.append(B.edge_crop(board, n, r, c, True)), where.append((r, c, r + 1, c))
        slots = slots_for("tango", B.cells(board, n), cell_names) + \
            slots_for("tango_sign", crops, [f"edge r{a + 1}c{b + 1}-r{c + 1}c{d + 1}" for a, b, c, d in where])

        def build(ls):
            cells, signs = ls[:n * n], ls[n * n:]
            return {"kind": "tango", "n": n, "givens": [[i // n, i % n, 0 if l == "sun" else 1] for i, l in enumerate(cells) if l != "empty"],
                    "edges": [[*w, s] for w, s in zip(where, signs) if s != "none"]}
        puzzle, fixes = _decode(slots, build, check)
    elif kind == "queens":
        cols = np.array([B.cell_colour(board, n, r, c) for r in range(n) for c in range(n)], np.float32)
        lab = cv2.cvtColor(cols.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, ids, centers = cv2.kmeans(lab, n, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        ids = ids.ravel()
        remap = {}
        for g in ids:
            remap.setdefault(int(g), len(remap))
        puzzle = {"kind": "queens", "regions": [[remap[int(ids[r * n + c])] for c in range(n)] for r in range(n)]}
        d = np.linalg.norm(lab - centers[ids], axis=1)
        spread = float(np.linalg.norm(centers[:, None] - centers[None], axis=2)[np.triu_indices(n, 1)].min())
        out["confidence"] = float(max(0.0, 1 - d.max() / max(spread, 1e-3)))
        fixes = [] if (not repair or _solvable(puzzle)) else None
    elif kind == "zip":
        slots = slots_for("zipnum", B.cells(board, n, inset=0.1), cell_names)
        strengths, where = [], []
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    strengths.append(B.edge_strength(board, n, r, c, False)), where.append((r, c, r, c + 1))
                if r + 1 < n:
                    strengths.append(B.edge_strength(board, n, r, c, True)), where.append((r, c, r + 1, c))
        v = np.array(strengths)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-3
        walls = [list(w) for w, s in zip(where, v) if s > med + max(25.0, 8 * mad)]
        build = lambda ls: {"kind": "zip", "n": n, "walls": walls,  # noqa: E731
                            "numbers": [[i // n, i % n, int(l)] for i, l in enumerate(ls) if l != "blank"]}
        puzzle, fixes = _decode(slots, build, check)
    elif kind == "patches":
        crops = B.cells(board, n, inset=0.04)
        nums = slots_for("patches_num", crops, [f"{c} number" for c in cell_names])
        shapes = slots_for("patches_shape", crops, [f"{c} shape" for c in cell_names])
        for (pn, ln, _), (ps, ls, _) in zip(nums, shapes):
            # rule out shapes no rectangle of the read area can have: a square needs a square area,
            # a wide or tall rectangle needs area >= 2
            num = ln[int(pn.argmax())]
            if num != "none":
                area = int(num)
                for j, s in enumerate(ls):
                    if (s == "square" and int(area ** 0.5) ** 2 != area) or (s in ("wide", "tall") and area < 2):
                        ps[j] = 0.0
                ps /= max(ps.sum(), 1e-9)
        slots = nums + shapes

        def build(ls):
            nums, shapes = ls[:n * n], ls[n * n:]
            return {"kind": "patches", "n": n, "clues": [
                [i // n, i % n, None if num == "none" else int(num), "any" if shape == "none" else shape]
                for i, (num, shape) in enumerate(zip(nums, shapes)) if not (num == "none" and shape == "none")]}
        puzzle, fixes = _decode(slots, build, check)
    elif kind == "wend":
        slots = slots_for("letters", B.cells(board, n), cell_names)
        build = lambda ls: {"kind": "wend", "lengths": None,  # noqa: E731
                            "rows": ["".join("#" if l == "wall" else l.lower() for l in ls[r * n:(r + 1) * n]) for r in range(n)]}
        puzzle, fixes = _decode(slots, build, check)
    elif kind == "chess":
        slots = slots_for("chess", B.cells(board, n, inset=0.02), [f"square {i}" for i in range(64)])

        def build(ls):
            grid = [ls[r * 8:(r + 1) * 8] for r in range(8)]
            if to_move == "b":  # board shown from black's side: rotate to the standard orientation
                grid = [row[::-1] for row in grid[::-1]]
            return _fen(grid, to_move)

        def legal(fen):
            import chess
            return chess.Board(fen).is_valid()
        fen, fixes = _decode(slots, build, legal if repair else None)
        for f in fixes or []:  # square names in chess notation
            k = int(f["cell"].split()[1])
            r, c = divmod(k, 8)
            if to_move == "b":
                r, c = 7 - r, 7 - c
            f["cell"] = "abcdefgh"[c] + str(8 - r)
        out["fen"] = fen
        out["confidence"] = float(min(s[0].max() for s in slots))
        puzzle = None
    elif kind == "checkers":
        crops = B.cells(board, n, inset=0.02)
        labels, conf = C("checkers").predict(crops)
        occupied = [i for i, l in enumerate(labels) if l != "empty"]
        lum = {i: _piece_luma(board, n, i // n, i % n) for i in occupied}
        split = 128
        if occupied:
            vals = np.array(list(lum.values()), np.float32)
            if vals.max() - vals.min() > 25:
                split = (vals.max() + vals.min()) / 2
        grid = [[0] * 8 for _ in range(8)]
        for i in occupied:
            val = 2 if labels[i] == "king" else 1
            grid[i // 8][i % 8] = val if lum[i] < split else -val  # +: dark pieces (moving up), -: light pieces
        out.update(board=grid, confidence=float(conf.min()))
        puzzle, fixes = None, []
    else:
        raise ValueError(f"no image reader for {kind}")
    if puzzle is not None:
        out["puzzle"] = puzzle
    if "confidence" not in out:
        out["confidence"] = float(min(s[0].max() for s in slots))
    out["corrections"] = fixes or []
    out["consistent"] = fixes is not None  # False: no small correction made the reading valid
    out["low_confidence"] = out["confidence"] < 0.6 or fixes is None
    return out


def _mixed_cells(board, n):
    """Fraction of cells whose centre area mixes clearly different colours."""
    lab = cv2.cvtColor(board, cv2.COLOR_RGB2LAB).astype(np.float32)
    s = board.shape[0] / n
    mixed = 0
    for r in range(n):
        for c in range(n):
            patch = lab[int((r + .2) * s):int((r + .8) * s), int((c + .2) * s):int((c + .8) * s)].reshape(-1, 3)
            lo, hi = np.percentile(patch, 10, axis=0), np.percentile(patch, 90, axis=0)
            mixed += np.linalg.norm(hi - lo) > 25
    return mixed / (n * n)


def _piece_luma(board, n, r, c):
    """Median brightness of a ring inside the piece's rim: avoids crown marks in the centre."""
    s = board.shape[0] / n
    g = cv2.cvtColor(board[int(r * s):int((r + 1) * s), int(c * s):int((c + 1) * s)], cv2.COLOR_RGB2GRAY)
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    d = np.hypot(yy - g.shape[0] / 2, xx - g.shape[1] / 2) / s
    ring = g[(d > 0.2) & (d < 0.28)]
    return float(np.median(ring)) if ring.size else float(g.mean())


def _fen(grid, to_move):
    rows = []
    for row in grid:
        s, empty = "", 0
        for l in row:
            if l == "empty":
                empty += 1
                continue
            if empty:
                s, empty = s + str(empty), 0
            s += l[1].upper() if l[0] == "w" else l[1].lower()
        rows.append(s + (str(empty) if empty else ""))
    placement = "/".join(rows)
    castle = ""
    flat = [l for row in grid for l in row]
    if flat[60] == "wK":
        castle += ("K" if flat[63] == "wR" else "") + ("Q" if flat[56] == "wR" else "")
    if flat[4] == "bK":
        castle += ("k" if flat[7] == "bR" else "") + ("q" if flat[0] == "bR" else "")
    return f"{placement} {to_move} {castle or '-'} - 0 1"
