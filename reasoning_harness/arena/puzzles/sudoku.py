"""Sudoku (9x9, 3x3 boxes) and LinkedIn-style Mini Sudoku (6x6, 2x3 boxes)."""
import random
import numpy as np

from .base import GridPuzzle


class Sudoku(GridPuzzle):
    name = "sudoku"
    pad = 9
    max_regions = 9
    loss = "cell"

    def __init__(self, grid, n=9, box=(3, 3)):
        self.n, self.box = n, box
        self.clues = np.array(grid, dtype=np.int8).reshape(n, n)
        self.in_ch, self.out_ch = n + 2, n
        br, bc = box
        self._rmap = np.array([[(r // br) * (n // bc) + c // bc for c in range(n)] for r in range(n)])

    # ---- serialisation -------------------------------------------------
    @classmethod
    def parse(cls, text, n=9, box=(3, 3)):
        text = "".join(ch for ch in text if not ch.isspace())
        if len(text) != n * n:
            raise ValueError(f"expected {n*n} cells")
        cells = [0 if ch in ".0" else int(ch) for ch in text]
        if any(not 0 <= v <= n for v in cells):
            raise ValueError("digit out of range")
        return cls(cells, n, box)

    def to_json(self):
        return {"kind": self.name, "n": self.n, "box": list(self.box), "grid": self.clues.reshape(-1).tolist()}

    def region_map(self):
        return self._rmap

    # ---- rules ------------------------------------------------------------
    def initial(self):
        return self.clues.copy()

    def assign(self, state, var, value):
        s = state.copy()
        s[var] = value
        return s

    def complete(self, state):
        return bool((state > 0).all())

    def answer(self, state):
        return state.reshape(-1).tolist()

    def _used(self, state):
        n = self.n
        bits = np.where(state > 0, 1 << state.astype(np.int64), 0)
        rows = np.bitwise_or.reduce(bits, axis=1)
        cols = np.bitwise_or.reduce(bits, axis=0)
        boxes = np.zeros(n, np.int64)
        for b in range(n):
            boxes[b] = np.bitwise_or.reduce(bits[self._rmap == b])
        return rows, cols, boxes

    def domains(self, state, prune=True):
        n = self.n
        rows, cols, boxes = self._used(state)
        out = {}
        for r, c in zip(*np.nonzero(state == 0)):
            used = rows[r] | cols[c] | boxes[self._rmap[r, c]]
            out[(int(r), int(c))] = [d for d in range(1, n + 1) if not used >> d & 1]
        if prune and out:
            # Hidden singles make the solver far stronger; a digit with one home is forced.
            for unit in self._units():
                empties = [cell for cell in unit if cell in out]
                for d in range(1, n + 1):
                    if any(state[cell] == d for cell in unit):
                        continue
                    homes = [cell for cell in empties if d in out[cell]]
                    if not homes:
                        return None
                    if len(homes) == 1 and len(out[homes[0]]) > 1:
                        out[homes[0]] = [d]
        return out

    def _units(self):
        if not hasattr(self, "_unit_cache"):
            n = self.n
            units = [[(r, c) for c in range(n)] for r in range(n)]
            units += [[(r, c) for r in range(n)] for c in range(n)]
            units += [[(r, c) for r in range(n) for c in range(n) if self._rmap[r, c] == b] for b in range(n)]
            self._unit_cache = units
        return self._unit_cache

    def verify(self, answer):
        n = self.n
        g = np.array(answer).reshape(n, n)
        if g.shape != (n, n):
            return False
        full = set(range(1, n + 1))
        clue = self.clues > 0
        if not (g[clue] == self.clues[clue]).all():
            return False
        for i in range(n):
            if set(g[i].tolist()) != full or set(g[:, i].tolist()) != full:
                return False
        return all(set(g[self._rmap == b].tolist()) == full for b in range(n))

    # ---- learning interface ------------------------------------------------
    def encode(self, state):
        n = self.n
        x = np.zeros((n + 2, n, n), np.float32)
        x[state, np.arange(n)[:, None], np.arange(n)[None, :]] = 1
        x[n + 1] = self.clues > 0
        return x

    def labels(self, solution, state):
        sol = np.array(solution).reshape(self.n, self.n)
        return (sol - 1).astype(np.int64), (state == 0).astype(np.float32)

    def scores(self, logp, state, var, values):
        r, c = var
        return [float(logp[v - 1, r, c]) for v in values]

    def oneshot(self, logp):
        g = self.clues.copy()
        pred = logp.argmax(0) + 1
        g[g == 0] = pred[g == 0]
        return g.reshape(-1).tolist()

    def random_partial(self, solution, rng):
        sol = np.array(solution, dtype=np.int8).reshape(self.n, self.n)
        keep = rng.random((self.n, self.n)) < rng.random() * 0.9
        return np.where((self.clues > 0) | keep, sol, 0).astype(np.int8)


class MiniSudoku(Sudoku):
    name = "mini_sudoku"
    pad = 6
    max_regions = 6

    def __init__(self, grid, n=6, box=(2, 3)):
        super().__init__(grid, 6, (2, 3))

    @classmethod
    def parse(cls, text, n=6, box=(2, 3)):
        return super().parse.__func__(cls, text, 6, (2, 3))


def random_full(n, box, rng):
    puzzle = Sudoku([0] * n * n, n, box)
    state = puzzle.initial()

    def rec(s):
        doms = puzzle.domains(s, prune=False)
        if not doms:
            return s
        var = min(doms, key=lambda v: len(doms[v]))
        vals = doms[var][:]
        rng.shuffle(vals)
        for v in vals:
            out = rec(puzzle.assign(s, var, v))
            if out is not None:
                return out
        return None

    return rec(state)


def generate_mini(seed):
    """Unique-solution 6x6 puzzle: random full grid, then remove clues while unique."""
    from ..search import count_solutions
    rng = random.Random(seed)
    sol = random_full(6, (2, 3), rng)
    grid = sol.reshape(-1).copy()
    order = list(range(36))
    rng.shuffle(order)
    target = rng.randint(10, 16)  # LinkedIn minis are gentle; leave a handful of givens
    for i in order:
        if (grid > 0).sum() <= target:
            break
        keep = grid[i]
        grid[i] = 0
        if count_solutions(MiniSudoku(grid))[0] != 1:
            grid[i] = keep
    return MiniSudoku(grid), sol.reshape(-1).tolist()
