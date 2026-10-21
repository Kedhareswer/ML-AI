"""LinkedIn Zip: draw one path through every cell, visiting the numbers 1..K in order.

The path starts on 1, ends on K, moves orthogonally, never crosses a wall and
visits each cell exactly once.
"""
import random
from collections import deque
import numpy as np

from .base import GridPuzzle

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class Zip(GridPuzzle):
    name = "zip"
    pad = 8
    max_regions = 1
    loss = "spatial"
    in_ch, out_ch = 10, 1

    def __init__(self, n, numbers, walls=()):
        # numbers: {(r,c): k} with k = 1..K; walls: {((r,c),(r2,c2))} blocked adjacent pairs
        self.n = n
        self.numbers = {tuple(k): int(v) for k, v in numbers.items()}
        self.K = max(self.numbers.values())
        if sorted(self.numbers.values()) != list(range(1, self.K + 1)) or self.K < 2:
            raise ValueError("numbers must be 1..K")
        self.walls = {frozenset((tuple(a), tuple(b))) for a, b in walls}
        self.start = next(k for k, v in self.numbers.items() if v == 1)
        self.end = next(k for k, v in self.numbers.items() if v == self.K)

    def to_json(self):
        return {"kind": self.name, "n": self.n,
                "numbers": [[r, c, k] for (r, c), k in self.numbers.items()],
                "walls": [[*sorted(w)[0], *sorted(w)[1]] for w in self.walls]}

    @classmethod
    def from_json(cls, d):
        return cls(d["n"], {(r, c): k for r, c, k in d["numbers"]},
                   [((a, b), (c, e)) for a, b, c, e in d.get("walls", [])])

    def moves(self, cell):
        r, c = cell
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < self.n and 0 <= nxt[1] < self.n and frozenset((cell, nxt)) not in self.walls:
                yield nxt

    # state = tuple(path)
    def initial(self):
        return (self.start,)

    def assign(self, state, var, value):
        return state + (value,)

    def complete(self, state):
        return len(state) == self.n * self.n and state[-1] == self.end

    def answer(self, state):
        return [list(c) for c in state]

    def _next_number(self, path):
        return 1 + max(self.numbers.get(c, 0) for c in path)

    def _basic(self, path):
        seen, head = set(path), path[-1]
        target = self._next_number(path)
        total = self.n * self.n
        out = []
        for nxt in self.moves(head):
            if nxt in seen:
                continue
            k = self.numbers.get(nxt)
            if k is not None and k != target:
                continue
            if nxt == self.end and len(path) + 1 != total:
                continue
            out.append(nxt)
        return out

    def _alive(self, path):
        """Cheap dead-end test after a move: unvisited cells stay connected and degree-feasible."""
        seen, head = set(path), path[-1]
        rest = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in seen]
        if not rest:
            return head == self.end
        rest_set = set(rest)
        q, reach = deque([head]), {head}
        while q:
            cur = q.popleft()
            for nxt in self.moves(cur):
                if nxt in rest_set and nxt not in reach:
                    reach.add(nxt)
                    q.append(nxt)
        if len(reach) - 1 != len(rest):
            return False
        dead = 0
        for cell in rest:
            deg = sum(1 for nxt in self.moves(cell) if nxt in rest_set or nxt == head)
            if deg <= 1 and cell != self.end:
                dead += 1
        return dead == 0

    def domains(self, state, prune=True):
        if self.complete(state):
            return {}
        options = self._basic(state)
        if prune:
            options = [o for o in options if self._alive(state + (o,))]
        return {"next": options}

    def verify(self, answer):
        path = [tuple(c) for c in answer]
        n = self.n
        if len(path) != n * n or len(set(path)) != n * n:
            return False
        if any(not (0 <= r < n and 0 <= c < n) for r, c in path):
            return False
        if path[0] != self.start or path[-1] != self.end:
            return False
        for a, b in zip(path, path[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 or frozenset((a, b)) in self.walls:
                return False
        order = [self.numbers[c] for c in path if c in self.numbers]
        return order == list(range(1, self.K + 1))

    def encode(self, state):
        n = self.n
        x = np.zeros((10, n, n), np.float32)
        target = self._next_number(state)
        for c in state:
            x[0][c] = 1
        x[1][state[-1]] = 1
        x[2] = 1 - x[0]
        for (r, c), k in self.numbers.items():
            x[3, r, c] = 1
            x[4, r, c] = k / self.K
            x[5, r, c] = k == target
            x[6, r, c] = k == self.K
        for w in self.walls:
            a, b = sorted(w)
            x[7 if a[0] == b[0] else 8][a] = 1
        x[9] = len(state) / (n * n)
        return x

    def labels(self, solution, state):
        nxt = tuple(solution[len(state)])
        t = np.zeros((self.n, self.n), np.int64)
        t[nxt] = 1
        return t, np.ones((self.n, self.n), np.float32)

    def scores(self, logp, state, var, values):
        return [float(logp[0][v]) for v in values]

    def random_partial(self, solution, rng):
        k = int(rng.integers(1, len(solution))) if hasattr(rng, "integers") else rng.randrange(1, len(solution))
        return tuple(tuple(c) for c in solution[:k])


def _hamiltonian(n, rng, steps=None):
    """Random Hamiltonian path via backbite moves from a serpentine start."""
    path = [(r, c if r % 2 == 0 else n - 1 - c) for r in range(n) for c in range(n)]
    for _ in range(steps or 20 * n * n * n):
        if rng.random() < 0.5:
            path.reverse()
        head = path[0]
        nbrs = [(head[0] + dr, head[1] + dc) for dr, dc in DIRS]
        nbrs = [p for p in nbrs if 0 <= p[0] < n and 0 <= p[1] < n and p != path[1]]
        if not nbrs:
            continue
        pick = rng.choice(nbrs)
        i = path.index(pick)
        path = path[:i][::-1] + path[i:]
    return path


def generate(seed, n=None):
    """Random Hamiltonian path; add number clues until the path is the only solution."""
    from ..search import search
    rng = random.Random(seed)
    n = n or rng.choice([6, 7])
    path = _hamiltonian(n, rng)
    index = {cell: i for i, cell in enumerate(path)}
    marks = {0, n * n - 1}
    marks.update(rng.sample(range(1, n * n - 1), rng.randint(n - 3, n)))
    edges_on_path = {frozenset((a, b)) for a, b in zip(path, path[1:])}
    walls = []
    for _ in range(rng.choice([0, 0, 2, 3, 4])):
        r, c = rng.randrange(n), rng.randrange(n)
        b = (r, c + 1) if rng.random() < .5 else (r + 1, c)
        if b[0] < n and b[1] < n and frozenset(((r, c), b)) not in edges_on_path:
            walls.append(((r, c), b))
    while True:
        ordered = sorted(marks)
        puzzle = Zip(n, {path[i]: k + 1 for k, i in enumerate(ordered)}, walls)
        found, _, exhausted = search(puzzle, None, 30_000, count_limit=2)
        if len(found) == 1 and not exhausted:
            return puzzle, [list(c) for c in path]
        if exhausted or not found:
            marks.update(rng.sample(range(1, n * n - 1), 2))
            continue
        alt = next(f for f in found if [tuple(c) for c in f] != path) if len(found) > 1 else found[0]
        alt = [tuple(c) for c in alt]
        split = next(i for i, (a, b) in enumerate(zip(alt, path)) if a != b)
        options = [i for i in (split, index[alt[split]]) if i not in marks]
        free = [i for i in range(1, n * n - 1) if i not in marks]
        marks.add(options[0] if options else rng.choice(free))
