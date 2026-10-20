"""LinkedIn Patches (Shikaku family): cut the grid into rectangles.

Each rectangle holds exactly one clue. The clue's number (when shown) is the
area; its shape icon is square, wide (wider than tall), tall, or any.
"""
import random
import numpy as np

from .base import GridPuzzle, log_softmax

SHAPES = ("square", "wide", "tall", "any")


def shape_ok(shape, h, w):
    return {"square": h == w, "wide": w > h, "tall": h > w, "any": True}[shape]


class Patches(GridPuzzle):
    name = "patches"
    pad = 7
    max_regions = 1
    loss = "cell"
    in_ch, out_ch = 12, 4

    def __init__(self, n, clues):
        # clues: list of (r, c, area or None, shape)
        self.n = n
        self.clues = [(int(r), int(c), None if a is None else int(a), s) for r, c, a, s in clues]
        if any(s not in SHAPES for *_, s in self.clues):
            raise ValueError("unknown shape")
        self.where = {(r, c): i for i, (r, c, _, _) in enumerate(self.clues)}
        self.cands = []
        for i, (r, c, area, shape) in enumerate(self.clues):
            opts = []
            for h in range(1, n + 1):
                for w in range(1, n + 1):
                    if (area is not None and h * w != area) or not shape_ok(shape, h, w):
                        continue
                    for r0 in range(r - h + 1, r + 1):
                        for c0 in range(c - w + 1, c + 1):
                            if r0 < 0 or c0 < 0 or r0 + h > n or c0 + w > n:
                                continue
                            inside = [k for k in self.where if r0 <= k[0] < r0 + h and c0 <= k[1] < c0 + w]
                            if inside == [(r, c)]:
                                opts.append((r0, c0, h, w))
            self.cands.append(opts)

    def to_json(self):
        return {"kind": self.name, "n": self.n, "clues": [list(c) for c in self.clues]}

    @classmethod
    def from_json(cls, d):
        return cls(d["n"], d["clues"])

    # state: tuple of rect-or-None per clue
    def initial(self):
        return (None,) * len(self.clues)

    def assign(self, state, var, value):
        s = list(state)
        s[var] = value
        return tuple(s)

    def complete(self, state):
        return all(x is not None for x in state) and self._cover(state).all()

    def answer(self, state):
        return [list(x) for x in state]

    def _cover(self, state):
        g = np.zeros((self.n, self.n), bool)
        for rect in state:
            if rect is not None:
                r0, c0, h, w = rect
                g[r0:r0 + h, c0:c0 + w] = True
        return g

    def domains(self, state, prune=True):
        cover = self._cover(state)
        out = {}
        for i, rect in enumerate(state):
            if rect is None:
                out[i] = [q for q in self.cands[i] if not cover[q[0]:q[0] + q[2], q[1]:q[1] + q[3]].any()]
        if prune and out:
            reach = cover.copy()
            for opts in out.values():
                for r0, c0, h, w in opts:
                    reach[r0:r0 + h, c0:c0 + w] = True
            if not reach.all():
                return None
        return out

    def verify(self, answer):
        n = self.n
        if len(answer) != len(self.clues):
            return False
        g = np.zeros((n, n), np.int64)
        for (r, c, area, shape), rect in zip(self.clues, answer):
            r0, c0, h, w = rect
            if min(r0, c0) < 0 or r0 + h > n or c0 + w > n or h < 1 or w < 1:
                return False
            if not (r0 <= r < r0 + h and c0 <= c < c0 + w):
                return False
            if (area is not None and h * w != area) or not shape_ok(shape, h, w):
                return False
            g[r0:r0 + h, c0:c0 + w] += 1
        return bool((g == 1).all())

    def _owner(self, state):
        own = -np.ones((self.n, self.n), np.int64)
        for i, rect in enumerate(state):
            if rect is not None:
                r0, c0, h, w = rect
                own[r0:r0 + h, c0:c0 + w] = i
        return own

    def encode(self, state):
        n = self.n
        x = np.zeros((12, n, n), np.float32)
        for r, c, area, shape in self.clues:
            x[0, r, c] = 1
            x[1, r, c] = (area or 0) / (n * n)
            x[2 + SHAPES.index(shape), r, c] = 1
            x[6, r, c] = area is None
        own = self._owner(state)
        x[7] = own >= 0
        for r in range(n):
            for c in range(n):
                if own[r, c] < 0:
                    continue
                if c + 1 < n and own[r, c + 1] >= 0:
                    x[8 if own[r, c + 1] != own[r, c] else 10, r, c] = 1
                if r + 1 < n and own[r + 1, c] >= 0:
                    x[9 if own[r + 1, c] != own[r, c] else 11, r, c] = 1
        return x

    def labels(self, solution, state):
        own = self._owner(tuple(tuple(x) for x in solution))
        n = self.n
        t = np.zeros((n, n), np.int64)
        for r in range(n):
            for c in range(n):
                br = c + 1 >= n or own[r, c + 1] != own[r, c]
                bd = r + 1 >= n or own[r + 1, c] != own[r, c]
                t[r, c] = 2 * br + bd
        return t, np.ones((n, n), np.float32)

    def normalise(self, out):
        lp = log_softmax(out[:, :self.n, :self.n], 0)
        # marginals: [P(no right boundary), P(right boundary), P(no down), P(down)] in log space
        br1 = np.logaddexp(lp[2], lp[3])
        br0 = np.logaddexp(lp[0], lp[1])
        bd1 = np.logaddexp(lp[1], lp[3])
        bd0 = np.logaddexp(lp[0], lp[2])
        return np.stack([br0, br1, bd0, bd1])

    def rect_score(self, logp, rect):
        r0, c0, h, w = rect
        n, total, k = self.n, 0.0, 0
        for r in range(r0, r0 + h):
            for c in range(c0 - 1, c0 + w):
                if 0 <= c < n - 1:
                    total += logp[1 if c in (c0 - 1, c0 + w - 1) else 0, r, c]
                    k += 1
        for c in range(c0, c0 + w):
            for r in range(r0 - 1, r0 + h):
                if 0 <= r < n - 1:
                    total += logp[3 if r in (r0 - 1, r0 + h - 1) else 2, r, c]
                    k += 1
        return total / max(k, 1)

    def scores(self, logp, state, var, values):
        return [float(self.rect_score(logp, q)) for q in values]

    def random_partial(self, solution, rng):
        keep = rng.random() * 0.8
        return tuple(tuple(x) if rng.random() < keep else None for x in solution)


def _partition(n, rng):
    own = -np.ones((n, n), np.int64)
    rects = []
    while (own < 0).any():
        r, c = map(int, np.argwhere(own < 0)[0])
        maxw = 0
        while c + maxw < n and own[r, c + maxw] < 0:
            maxw += 1
        options = []
        for h in range(1, n - r + 1):
            for w in range(1, maxw + 1):
                if (own[r:r + h, c:c + w] < 0).all() and 2 <= h * w <= 8:
                    options.append((h, w))
        if not options:
            options = [(1, 1)]
        h, w = rng.choice(options)
        own[r:r + h, c:c + w] = len(rects)
        rects.append((r, c, h, w))
    return rects


def generate(seed, n=None):
    from ..search import search
    rng = random.Random(seed)
    n = n or rng.choice([5, 6])
    while True:
        rects = _partition(n, rng)
        clues = []
        for r0, c0, h, w in rects:
            r, c = r0 + rng.randrange(h), c0 + rng.randrange(w)
            exact = "square" if h == w else ("wide" if w > h else "tall")
            shape = "any" if rng.random() < 0.3 else exact
            area = None if rng.random() < 0.25 else h * w
            clues.append([r, c, area, shape])
        for attempt in range(3):
            puzzle = Patches(n, clues)
            found, _, exhausted = search(puzzle, None, 100_000, count_limit=2)
            if len(found) == 1 and not exhausted:
                return puzzle, found[0]
            for cl, (r0, c0, h, w) in zip(clues, rects):
                if cl[2] is None or cl[3] == "any":
                    cl[2] = h * w
                    cl[3] = "square" if h == w else ("wide" if w > h else "tall")
                    break
