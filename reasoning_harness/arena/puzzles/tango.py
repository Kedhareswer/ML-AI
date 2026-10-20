"""LinkedIn Tango (binary puzzle): fill a 6x6 grid with suns (0) and moons (1).

Rules: each row and column has 3 of each; never three identical symbols in a
line; '=' joins cells that must match, 'x' joins cells that must differ.
"""
import random
import numpy as np

from .base import GridPuzzle

SYMBOLS = "SM"  # S = sun (0), M = moon (1)


class Tango(GridPuzzle):
    name = "tango"
    pad = 6
    max_regions = 1
    loss = "cell"
    in_ch, out_ch = 8, 2

    def __init__(self, givens, edges, n=6):
        # givens: {(r,c): 0/1}; edges: {((r,c),(r2,c2)): '=' or 'x'} with (r2,c2) right of / below (r,c)
        self.n = n
        self.givens = {tuple(k): int(v) for k, v in givens.items()}
        self.edges = {(tuple(a), tuple(b)): s for (a, b), s in edges.items()}
        self._nbr = {}
        for (a, b), s in self.edges.items():
            self._nbr.setdefault(a, []).append((b, s))
            self._nbr.setdefault(b, []).append((a, s))

    def to_json(self):
        return {"kind": self.name, "n": self.n,
                "givens": [[r, c, v] for (r, c), v in self.givens.items()],
                "edges": [[a[0], a[1], b[0], b[1], s] for (a, b), s in self.edges.items()]}

    @classmethod
    def from_json(cls, d):
        return cls({(r, c): v for r, c, v in d["givens"]},
                   {((r, c), (r2, c2)): s for r, c, r2, c2, s in d["edges"]}, d.get("n", 6))

    def initial(self):
        s = -np.ones((self.n, self.n), np.int8)
        for (r, c), v in self.givens.items():
            s[r, c] = v
        return s

    def assign(self, state, var, value):
        s = state.copy()
        s[var] = value
        return s

    def complete(self, state):
        return bool((state >= 0).all())

    def answer(self, state):
        return state.tolist()

    def _ok(self, s, r, c, v):
        n, half = self.n, self.n // 2
        if (s[r] == v).sum() >= half or (s[:, c] == v).sum() >= half:
            return False
        for line, i in ((s[r], c), (s[:, c], r)):
            for start in range(i - 2, i + 1):
                if 0 <= start and start + 2 < n:
                    seg = [v if k == i else line[k] for k in range(start, start + 3)]
                    if seg[0] == seg[1] == seg[2]:
                        return False
        for (rr, cc), sign in self._nbr.get((r, c), []):
            other = s[rr, cc]
            if other >= 0 and ((sign == "=") != (other == v)):
                return False
        return True

    def domains(self, state, prune=True):
        out = {}
        for r, c in zip(*np.nonzero(state < 0)):
            r, c = int(r), int(c)
            out[(r, c)] = [v for v in (0, 1) if self._ok(state, r, c, v)]
        return out

    def verify(self, answer):
        g = np.array(answer)
        n = self.n
        if g.shape != (n, n) or not np.isin(g, (0, 1)).all():
            return False
        if any(g[r, c] != v for (r, c), v in self.givens.items()):
            return False
        for line in list(g) + list(g.T):
            if line.sum() != n // 2:
                return False
            if any(line[i] == line[i + 1] == line[i + 2] for i in range(n - 2)):
                return False
        return all((g[a] == g[b]) == (s == "=") for (a, b), s in self.edges.items())

    def encode(self, state):
        n = self.n
        x = np.zeros((8, n, n), np.float32)
        x[0] = state == 0
        x[1] = state == 1
        x[2] = state < 0
        for (r, c) in self.givens:
            x[3, r, c] = 1
        for (a, b), s in self.edges.items():
            down = b[0] == a[0] + 1
            x[4 + 2 * down + (s == "x"), a[0], a[1]] = 1
        return x

    def labels(self, solution, state):
        return np.array(solution, np.int64), (state < 0).astype(np.float32)

    def scores(self, logp, state, var, values):
        return [float(logp[v, var[0], var[1]]) for v in values]

    def oneshot(self, logp):
        g = logp.argmax(0)
        for (r, c), v in self.givens.items():
            g[r, c] = v
        return g.tolist()

    def random_partial(self, solution, rng):
        sol = np.array(solution, np.int8)
        keep = rng.random(sol.shape) < rng.random() * 0.9
        s = np.where(keep, sol, -1).astype(np.int8)
        for (r, c), v in self.givens.items():
            s[r, c] = v
        return s


def random_solution(n, rng):
    p = Tango({}, {}, n)

    def rec(s):
        doms = p.domains(s)
        if not doms:
            return s
        var = min(doms, key=lambda k: len(doms[k]))
        vals = doms[var][:]
        rng.shuffle(vals)
        for v in vals:
            out = rec(p.assign(s, var, v))
            if out is not None:
                return out
        return None

    return rec(p.initial())


def generate(seed, n=6):
    """Random valid grid, then strip clues (cells and signs) while the answer stays unique."""
    from ..search import count_solutions
    rng = random.Random(seed)
    sol = random_solution(n, rng)
    givens = {(r, c): int(sol[r, c]) for r in range(n) for c in range(n)}
    edges = {}
    for r in range(n):
        for c in range(n):
            for b in ((r, c + 1), (r + 1, c)):
                if b[0] < n and b[1] < n and rng.random() < 0.35:
                    edges[((r, c), b)] = "=" if sol[r, c] == sol[b] else "x"
    clues = [("g", k) for k in givens] + [("e", k) for k in edges]
    rng.shuffle(clues)
    for kind, key in clues:
        store = givens if kind == "g" else edges
        val = store.pop(key)
        if count_solutions(Tango(givens, edges, n))[0] != 1:
            store[key] = val
    return Tango(givens, edges, n), sol.tolist()
