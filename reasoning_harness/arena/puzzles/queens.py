"""LinkedIn Queens: one queen per row, column and colour region; queens never touch."""
import random
from collections import deque
import numpy as np

from .base import GridPuzzle


class Queens(GridPuzzle):
    name = "queens"
    pad = 10
    max_regions = 10
    loss = "cell"
    in_ch, out_ch = 7, 2

    def __init__(self, regions):
        self.regions = np.array(regions, np.int64)
        self.n = len(self.regions)
        if self.regions.shape != (self.n, self.n) or not 4 <= self.n <= 10:
            raise ValueError("regions must be an n x n grid, 4 <= n <= 10")
        if sorted(set(self.regions.reshape(-1).tolist())) != list(range(self.n)):
            raise ValueError("need exactly n regions numbered 0..n-1")

    def to_json(self):
        return {"kind": self.name, "regions": self.regions.tolist()}

    def region_map(self):
        return self.regions

    def initial(self):
        return -np.ones(self.n, np.int64)  # column of the queen in each row

    def assign(self, state, var, value):
        s = state.copy()
        s[var] = value
        return s

    def complete(self, state):
        return bool((state >= 0).all())

    def answer(self, state):
        return state.tolist()

    def _free(self, state, r, c):
        if c in state:
            return False
        reg = self.regions[r, c]
        for rr, cc in enumerate(state):
            if cc < 0:
                continue
            if self.regions[rr, cc] == reg or (abs(rr - r) <= 1 and abs(cc - c) <= 1):
                return False
        return True

    def domains(self, state, prune=True):
        out = {r: [c for c in range(self.n) if self._free(state, r, c)] for r in range(self.n) if state[r] < 0}
        if prune and out:
            used = {self.regions[r, c] for r, c in enumerate(state) if c >= 0}
            reachable = {self.regions[r, c] for r, cols in out.items() for c in cols}
            if any(g not in used and g not in reachable for g in range(self.n)):
                return None
        return out

    def verify(self, answer):
        n = self.n
        if len(answer) != n or sorted(answer) != list(range(n)):
            return False
        if len({int(self.regions[r, c]) for r, c in enumerate(answer)}) != n:
            return False
        return all(abs(answer[r] - answer[r + 1]) > 1 for r in range(n - 1))

    def encode(self, state):
        n = self.n
        x = np.zeros((7, n, n), np.float32)
        used_regions = {self.regions[r, c] for r, c in enumerate(state) if c >= 0}
        for r in range(n):
            for c in range(n):
                if state[r] == c:
                    x[0, r, c] = 1
                elif not self._free(state, r, c) or state[r] >= 0:
                    x[1, r, c] = 1
                x[2, r, c] = self.regions[r, c] in used_regions
                x[3, r, c] = state[r] >= 0
                x[4, r, c] = c in state
                x[5, r, c] = c + 1 < n and self.regions[r, c] != self.regions[r, c + 1]
                x[6, r, c] = r + 1 < n and self.regions[r, c] != self.regions[r + 1, c]
        return x

    def labels(self, solution, state):
        n = self.n
        t = np.zeros((n, n), np.int64)
        t[np.arange(n), solution] = 1
        m = np.repeat((state < 0)[:, None], n, 1).astype(np.float32)
        return t, m

    def scores(self, logp, state, var, values):
        return [float(logp[1, var, c] - logp[0, var, c]) for c in values]

    def oneshot(self, logp):
        return logp[1].argmax(1).tolist()

    def random_partial(self, solution, rng):
        keep = rng.random(self.n) < rng.random() * 0.8
        return np.where(keep, np.array(solution), -1)


def _placement(n, rng):
    while True:
        cols, ok = [], True
        for r in range(n):
            options = [c for c in range(n) if c not in cols and (not cols or abs(c - cols[-1]) > 1)]
            if not options:
                ok = False
                break
            cols.append(rng.choice(options))
        if ok:
            return cols


def _grow(n, cols, rng):
    reg = -np.ones((n, n), np.int64)
    frontier = []
    for r, c in enumerate(cols):
        reg[r, c] = r
        frontier.append((r, c))
    while (reg < 0).any():
        r, c = frontier[rng.randrange(len(frontier))]
        nbrs = [(r + dr, c + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if 0 <= r + dr < n and 0 <= c + dc < n and reg[r + dr, c + dc] < 0]
        if not nbrs:
            frontier.remove((r, c))
            continue
        rr, cc = rng.choice(nbrs)
        reg[rr, cc] = reg[r, c]
        frontier.append((rr, cc))
    return reg


def _connected(reg, g):
    cells = list(zip(*np.nonzero(reg == g)))
    if not cells:
        return False
    seen, q = {cells[0]}, deque([cells[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nxt = (r + dr, c + dc)
            if nxt not in seen and 0 <= nxt[0] < len(reg) and 0 <= nxt[1] < len(reg) and reg[nxt] == g:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def generate(seed, n=None):
    """Plant a valid queen layout, grow colour regions around it, then repair the
    regions until the planted layout is the only solution."""
    from ..search import search
    rng = random.Random(seed)
    n = n or rng.choice([7, 8, 9])
    while True:
        cols = _placement(n, rng)
        regions = _grow(n, cols, rng)
        planted = {(r, c) for r, c in enumerate(cols)}
        for _ in range(60):
            found, _, exhausted = search(Queens(regions), None, 50_000, count_limit=2)
            if exhausted:
                break
            if len(found) == 1:
                return Queens(regions), found[0]
            alt = next(f for f in found if f != cols)
            moved = False
            rows = [r for r in range(n) if alt[r] != cols[r]]
            rng.shuffle(rows)
            for r in rows:
                cell = (r, alt[r])
                if cell in planted:
                    continue
                g = regions[cell]
                nbrs = [regions[r + dr, cell[1] + dc] for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                        if 0 <= r + dr < n and 0 <= cell[1] + dc < n]
                choices = [h for h in set(nbrs) if h != g]
                rng.shuffle(choices)
                for h in choices:
                    regions[cell] = h
                    if _connected(regions, g):
                        moved = True
                        break
                    regions[cell] = g
                if moved:
                    break
            if not moved:
                break
