"""LinkedIn Wend: split a letter grid into hidden words.

Words run through orthogonally adjacent tiles and may bend; every open tile is
used exactly once; '#' tiles are walls. The word lengths are given. The
dictionary is a common-English list (ENABLE words ranked by OpenSubtitles
frequency); generated puzzles have exactly one solution in that dictionary.
"""
import random
from functools import lru_cache
from pathlib import Path
import numpy as np

from .base import GridPuzzle

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # R D L U; class 4 = end of word
WORDS_FILE = Path(__file__).with_name("wend_words.txt")
# fallback for boards with rarer words ("compile"): ENABLE words seen >= 50 times in WikiText-103
EXTENDED_FILE = Path(__file__).with_name("wend_words_extended.txt")


@lru_cache(maxsize=2)
def lexicon(extended=False):
    words = [w.strip() for w in WORDS_FILE.read_text().split() if w.strip()]
    if extended:
        words += [w.strip() for w in EXTENDED_FILE.read_text().split() if w.strip()]
    trie = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True
    return words, trie


class Wend(GridPuzzle):
    name = "wend"
    pad = 7
    max_regions = 1
    loss = "cell"
    in_ch, out_ch = 30, 5

    def __init__(self, rows, lengths, extended=False):
        self.extended = extended
        self.rows = [r.lower() for r in rows]
        self.n = len(rows)
        if any(len(r) != self.n for r in self.rows) or not 3 <= self.n <= 7:
            raise ValueError("grid must be square, 3-7 tiles per side")
        if any(not (ch == "#" or "a" <= ch <= "z") for r in self.rows for ch in r):
            raise ValueError("tiles are letters a-z or '#' walls")
        # lengths=None: word lengths unknown (e.g. read from a screenshot); any 3-8 letter words may be used
        self.lengths = sorted(int(x) for x in lengths) if lengths else None
        self.open = [(r, c) for r in range(self.n) for c in range(self.n) if self.rows[r][c] != "#"]
        if self.lengths and sum(self.lengths) != len(self.open):
            raise ValueError("word lengths must add up to the number of open tiles")
        self.paths = self._word_paths()
        self.by_cell = {cell: [] for cell in self.open}
        for i, p in enumerate(self.paths):
            for cell in p:
                self.by_cell[cell].append(i)

    def to_json(self):
        return {"kind": self.name, "rows": self.rows, "lengths": self.lengths, "extended": self.extended}

    @classmethod
    def from_json(cls, d):
        return cls(d["rows"], d["lengths"], d.get("extended", False))

    def _word_paths(self):
        _, trie = lexicon(self.extended)
        allowed = set(self.lengths) if self.lengths else set(range(3, 9))
        longest, found = max(allowed), []

        def dfs(cell, node, path):
            if "$" in node and len(path) in allowed:
                found.append(tuple(path))
            if len(path) == longest:
                return
            r, c = cell
            for dr, dc in DIRS:
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < self.n and 0 <= nxt[1] < self.n and nxt not in path:
                    ch = self.rows[nxt[0]][nxt[1]]
                    if ch in node:
                        path.append(nxt)
                        dfs(nxt, node[ch], path)
                        path.pop()

        for cell in self.open:
            ch = self.rows[cell[0]][cell[1]]
            if ch in trie:
                dfs(cell, trie[ch], [cell])
        return found

    def word(self, path):
        return "".join(self.rows[r][c] for r, c in path)

    # state: tuple of chosen path indices (sorted)
    def initial(self):
        return ()

    def assign(self, state, var, value):
        return tuple(sorted(state + (value,)))

    def _covered(self, state):
        return {cell for i in state for cell in self.paths[i]}

    def _remaining(self, state):
        if self.lengths is None:
            return None
        left = list(self.lengths)
        for i in state:
            left.remove(len(self.paths[i]))
        return left

    def complete(self, state):
        return len(self._covered(state)) == len(self.open) and not self._remaining(state)

    def answer(self, state):
        return [[list(c) for c in self.paths[i]] for i in state]

    def domains(self, state, prune=True):
        covered = self._covered(state)
        rem = self._remaining(state)
        left = set(rem) if rem is not None else set(range(3, 9))
        out = {}
        for cell in self.open:
            if cell in covered:
                continue
            out[cell] = [i for i in self.by_cell[cell] if len(self.paths[i]) in left
                         and not any(c in covered for c in self.paths[i])]
        if prune and any(not v for v in out.values()):
            return None
        return out

    def verify(self, answer):
        words, _ = lexicon(self.extended)
        vocab = set(words)
        seen = []
        for path in answer:
            path = [tuple(c) for c in path]
            if any(c not in self.by_cell for c in path):
                return False
            if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(path, path[1:])):
                return False
            if self.word(path) not in vocab:
                return False
            seen.extend(path)
        if sorted(seen) != sorted(self.open):
            return False
        return self.lengths is None or sorted(len(p) for p in answer) == self.lengths

    def encode(self, state):
        n = self.n
        x = np.zeros((30, n, n), np.float32)
        covered = self._covered(state)
        for r in range(n):
            for c in range(n):
                ch = self.rows[r][c]
                if ch == "#":
                    x[26, r, c] = 1
                else:
                    x[ord(ch) - 97, r, c] = 1
                x[27, r, c] = (r, c) in covered
        for k, L in enumerate(self.lengths or []):
            x[28, k // n, k % n] = L / 10  # word-length list, written into spare canvas cells
        x[29] = len(self.lengths or []) / 10
        return x

    def labels(self, solution, state):
        n = self.n
        t = np.full((n, n), 4, np.int64)
        m = np.zeros((n, n), np.float32)
        for path in solution:
            path = [tuple(c) for c in path]
            for a, b in zip(path, path[1:]):
                t[a] = DIRS.index((b[0] - a[0], b[1] - a[1]))
            for cell in path:
                m[cell] = 1
        return t, m

    def path_score(self, logp, path):
        total = 0.0
        for a, b in zip(path, path[1:]):
            total += logp[DIRS.index((b[0] - a[0], b[1] - a[1])), a[0], a[1]]
        total += logp[4, path[-1][0], path[-1][1]]
        return total / len(path)

    def scores(self, logp, state, var, values):
        return [float(self.path_score(logp, self.paths[i])) for i in values]

    def oneshot(self, logp):
        """Follow the network's arrows from every tile; read off the chains as words."""
        nxt = {}
        for cell in self.open:
            d = int(logp[:, cell[0], cell[1]].argmax())
            if d < 4:
                tgt = (cell[0] + DIRS[d][0], cell[1] + DIRS[d][1])
                nxt[cell] = tgt
        targets = set(nxt.values())
        chains = []
        for cell in self.open:
            if cell in targets:
                continue
            path, cur = [cell], cell
            while cur in nxt and nxt[cur] not in path and len(path) < 12:
                cur = nxt[cur]
                path.append(cur)
            chains.append([list(c) for c in path])
        return chains

    def random_partial(self, solution, rng):
        keep = rng.random() * 0.7
        chosen = []
        for path in solution:
            if rng.random() < keep:
                p = tuple(tuple(c) for c in path)
                try:
                    chosen.append(self.paths.index(p))
                except ValueError:
                    pass
        return tuple(sorted(chosen))


def _carve(n, walls, lengths_range, rng):
    open_cells = {(r, c) for r in range(n) for c in range(n)} - walls
    free = set(open_cells)
    paths = []
    while free:
        # start from the most cornered free tile so no tile gets stranded
        start = min(free, key=lambda cell: (sum((cell[0] + dr, cell[1] + dc) in free for dr, dc in DIRS), rng.random()))
        target = rng.randint(*lengths_range)
        path = [start]
        free.discard(start)
        while len(path) < target:
            r, c = path[-1]
            options = [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in free]
            if not options:
                break
            nxt = min(options, key=lambda cell: (sum((cell[0] + dr, cell[1] + dc) in free for dr, dc in DIRS), rng.random()))
            path.append(nxt)
            free.discard(nxt)
        if len(path) < 3:
            return None
        paths.append(path)
    return paths


def generate(seed, n=None):
    from ..search import search
    words, _ = lexicon()
    by_len = {}
    for w in words[:6000]:  # answers come from the most common words
        by_len.setdefault(len(w), []).append(w)
    rng = random.Random(seed)
    n = n or rng.choice([5, 6])
    while True:
        walls = set()
        for _ in range(rng.randint(1, n - 3)):
            walls.add((rng.randrange(n), rng.randrange(n)))
        paths = _carve(n, walls, (3, 7), rng)
        if paths is None or any(len(p) not in by_len for p in paths):
            continue
        for _ in range(5):
            grid = [["#"] * n for _ in range(n)]
            for p in paths:
                w = rng.choice(by_len[len(p)])
                for (r, c), ch in zip(p, w):
                    grid[r][c] = ch
            puzzle = Wend(["".join(row) for row in grid], [len(p) for p in paths])
            found, _, exhausted = search(puzzle, None, 20_000, count_limit=2)
            if len(found) == 1 and not exhausted:
                return puzzle, found[0]
