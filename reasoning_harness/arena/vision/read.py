"""Image -> puzzle readers for every game.

Each reader finds the board, estimates the grid size, classifies every cell
with a classifier trained from scratch on synthetic cells, and returns the
puzzle in the solver's JSON format plus the lowest per-cell confidence.
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


def read(kind, image, root="artifacts/arena/vision", n=None, to_move="w"):
    rgb = B.load(image)
    sizes = [n] if n else SIZES[kind]
    board, quad, n = B.find_board(rgb, sizes)
    out = {"kind": kind, "n": n, "quad": np.asarray(quad).round(1).tolist()}
    C = lambda name: clf(str(root), name)  # noqa: E731
    if kind in ("sudoku", "mini_sudoku"):
        labels, conf = C("digits").predict(B.cells(board, n))
        grid = [0 if l == "blank" else int(l) for l in labels]
        out.update(puzzle={"kind": kind, "grid": grid, **({"n": 9, "box": [3, 3]} if kind == "sudoku" else {"n": 6, "box": [2, 3]})},
                   confidence=float(conf.min()))
    elif kind == "tango":
        labels, conf = C("tango").predict(B.cells(board, n))
        givens = [[i // n, i % n, 0 if l == "sun" else 1] for i, l in enumerate(labels) if l != "empty"]
        crops, where = [], []
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    crops.append(B.edge_crop(board, n, r, c, False)), where.append((r, c, r, c + 1))
                if r + 1 < n:
                    crops.append(B.edge_crop(board, n, r, c, True)), where.append((r, c, r + 1, c))
        signs, sconf = C("tango_sign").predict(crops)
        edges = [[*w, s] for w, s in zip(where, signs) if s != "none"]
        out.update(puzzle={"kind": "tango", "n": n, "givens": givens, "edges": edges},
                   confidence=float(min(conf.min(), sconf.min())))
    elif kind == "queens":
        cols = np.array([B.cell_colour(board, n, r, c) for r in range(n) for c in range(n)], np.float32)
        lab = cv2.cvtColor(cols.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, ids, centers = cv2.kmeans(lab, n, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        ids = ids.ravel()
        # relabel regions in reading order so the JSON is canonical
        order, remap = [], {}
        for g in ids:
            if g not in remap:
                remap[g] = len(remap)
        regions = [[remap[ids[r * n + c]] for c in range(n)] for r in range(n)]
        d = np.linalg.norm(lab - centers[ids], axis=1)
        spread = float(np.linalg.norm(centers[:, None] - centers[None], axis=2)[np.triu_indices(n, 1)].min())
        out.update(puzzle={"kind": "queens", "regions": regions},
                   confidence=float(max(0.0, 1 - d.max() / max(spread, 1e-3))))
    elif kind == "zip":
        labels, conf = C("zipnum").predict(B.cells(board, n, inset=0.1))
        numbers = [[i // n, i % n, int(l)] for i, l in enumerate(labels) if l != "blank"]
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
        out.update(puzzle={"kind": "zip", "n": n, "numbers": numbers, "walls": walls}, confidence=float(conf.min()))
    elif kind == "patches":
        crops = B.cells(board, n, inset=0.04)
        nums, nconf = C("patches_num").predict(crops)
        shapes, sconf = C("patches_shape").predict(crops)
        clues = []
        for i, (num, shape) in enumerate(zip(nums, shapes)):
            if num == "none" and shape == "none":
                continue
            clues.append([i // n, i % n, None if num == "none" else int(num), "any" if shape == "none" else shape])
        out.update(puzzle={"kind": "patches", "n": n, "clues": clues}, confidence=float(min(nconf.min(), sconf.min())))
    elif kind == "wend":
        labels, conf = C("letters").predict(B.cells(board, n))
        rows = ["".join("#" if l == "wall" else l.lower() for l in labels[r * n:(r + 1) * n]) for r in range(n)]
        out.update(puzzle={"kind": "wend", "rows": rows, "lengths": None}, confidence=float(conf.min()))
    elif kind == "chess":
        labels, conf = C("chess").predict(B.cells(board, n, inset=0.02))
        grid = [labels[r * 8:(r + 1) * 8] for r in range(8)]
        if to_move == "b":  # board shown from black's side: rotate to the standard orientation
            grid = [row[::-1] for row in grid[::-1]]
        out.update(fen=_fen(grid, to_move), confidence=float(conf.min()))
    elif kind == "checkers":
        crops = B.cells(board, n, inset=0.02)
        labels, conf = C("checkers").predict(crops)
        occupied = [i for i, l in enumerate(labels) if l != "empty"]
        lum = {i: _piece_luma(board, n, i // n, i % n) for i in occupied}
        if occupied:
            vals = np.array(list(lum.values()), np.float32)
            split = (vals.max() + vals.min()) / 2 if vals.max() - vals.min() > 25 else 128
        grid = [[0] * 8 for _ in range(8)]
        for i in occupied:
            dark = lum[i] < split
            val = 2 if labels[i] == "king" else 1
            grid[i // 8][i % 8] = val if dark else -val  # +: dark pieces (moving up), -: light pieces
        out.update(board=grid, confidence=float(conf.min()))
    else:
        raise ValueError(f"no image reader for {kind}")
    out["low_confidence"] = out["confidence"] < 0.6
    return out


def _piece_luma(board, n, r, c):
    s = board.shape[0] / n
    patch = board[int((r + .35) * s):int((r + .65) * s), int((c + .2) * s):int((c + .45) * s)]
    return float(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).mean())


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
