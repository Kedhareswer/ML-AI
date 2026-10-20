"""Deterministic synthetic mazes and an independent path verifier."""
from collections import deque
from dataclasses import dataclass
import random

MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

@dataclass(frozen=True)
class Maze:
    rows: tuple[str, ...]

    def __post_init__(self):
        n = len(self.rows)
        if not 2 <= n <= 8 or any(len(row) != n for row in self.rows):
            raise ValueError("Maze must be square, 2–8 cells per side")
        flat = "".join(self.rows)
        if set(flat) - set(".XSG") or flat.count("S") != 1 or flat.count("G") != 1:
            raise ValueError("Use '.', 'X', exactly one S and one G")

    def locate(self, char):
        return next((r, c) for r, row in enumerate(self.rows) for c, x in enumerate(row) if x == char)

    def shortest(self):
        start, goal = self.locate("S"), self.locate("G")
        queue, seen = deque([(start, "")]), {start}
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for move, (dr, dc) in MOVES.items():
                nxt = r + dr, c + dc
                if self.open(*nxt) and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + move))
        return None

    def open(self, r, c):
        return 0 <= r < len(self.rows) and 0 <= c < len(self.rows) and self.rows[r][c] != "X"

    def prompt(self, previous=None, feedback=None):
        text = "maze:\n" + "\n".join(self.rows) + "\n"
        if previous is not None:
            text += f"attempt:{previous}\nfeedback:{feedback}\n"
        return text + "path:"

def verify(maze, path):
    r, c = maze.locate("S")
    for i, move in enumerate(path):
        if move not in MOVES:
            return {"valid": False, "optimal": False, "reason": "invalid_symbol", "at": i}
        dr, dc = MOVES[move]
        r, c = r + dr, c + dc
        if not maze.open(r, c):
            return {"valid": False, "optimal": False, "reason": "blocked", "at": i}
    valid = (r, c) == maze.locate("G")
    shortest = maze.shortest()
    return {"valid": valid, "optimal": valid and shortest is not None and len(path) == len(shortest),
            "reason": "solved" if valid else "incomplete", "at": len(path)}

def generate(seed, size=4):
    if not 2 <= size <= 8:
        raise ValueError("size must be between 2 and 8")
    rng = random.Random(seed)
    for _ in range(1000):
        cells = ["X" if rng.random() < .22 else "." for _ in range(size * size)]
        start, goal = rng.sample(range(size * size), 2)
        cells[start], cells[goal] = "S", "G"
        maze = Maze(tuple("".join(cells[i:i+size]) for i in range(0, len(cells), size)))
        if maze.shortest() is not None:
            return maze
    raise RuntimeError("Unable to generate a connected maze")

def example(seed, size=4):
    maze = generate(seed, size)
    rng = random.Random(seed ^ 0xBEEF)
    if rng.random() < .5:
        failed = "".join(rng.choices(list(MOVES), k=rng.randint(1, size * size)))
        result = verify(maze, failed)
        if result["valid"]:
            failed, result = "", verify(maze, "")
        prompt = maze.prompt(failed, result["reason"])
    else:
        prompt = maze.prompt()
    return prompt, maze.shortest()
