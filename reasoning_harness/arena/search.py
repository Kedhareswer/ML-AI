"""Solvers shared by every grid puzzle.

Modes, from strictest to most helped:
  oneshot  - one network pass, argmax every open cell, no rules applied (Sudoku/Tango/Queens).
  neural   - the network repeatedly picks its most confident legal move; no backtracking.
  search   - depth-first backtracking; the network orders choices; rule pruning removes dead ends.
  exact    - the same backtracking with no network (classical baseline).
Every returned answer is checked by the puzzle's independent verifier.
"""
import time
import numpy as np
import torch

from .nets import GridNet


class Budget(Exception):
    pass


def _logits(net, puzzle, state):
    if net is None:
        return None
    x, mask, regions = puzzle.tensors(state)
    with torch.inference_mode():
        out = net(torch.from_numpy(x)[None], torch.from_numpy(mask)[None], torch.from_numpy(regions)[None])
    return puzzle.normalise(out[0].numpy())


def _ordered(puzzle, logp, state, var, values):
    if logp is None:
        return list(values), [0.0] * len(values)
    scores = puzzle.scores(logp, state, var, values)
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [values[i] for i in order], [scores[i] for i in order]


def search(puzzle, net=None, budget=100_000, count_limit=1):
    """Backtracking search. Returns (solutions, expanded_states)."""
    found, expanded, trail, trails = [], 0, [], []

    def rec(state):
        nonlocal expanded
        doms = puzzle.domains(state, prune=True)
        if doms is None:
            return
        if not doms:
            if puzzle.complete(state):
                found.append(puzzle.answer(state))
                trails.append(list(trail))
            return
        logp = _logits(net, puzzle, state)
        # Most constrained variable first; ties go to the network's most confident one.
        smallest = min(len(v) for v in doms.values())
        if smallest == 0:
            return
        best = None
        for var, values in doms.items():
            if len(values) != smallest:
                continue
            ordered, scores = _ordered(puzzle, logp, state, var, values)
            if best is None or scores[0] > best[0]:
                best = (scores[0], var, ordered)
            if logp is None:
                break
        _, var, ordered = best
        for value in ordered:
            expanded += 1
            if expanded > budget:
                raise Budget
            trail.append((var, value))
            rec(puzzle.assign(state, var, value))
            trail.pop()
            if len(found) >= count_limit:
                return

    try:
        rec(puzzle.initial())
        exhausted = False
    except Budget:
        exhausted = True
    search.last_trail = trails[0] if trails else []
    return found, expanded, exhausted


def count_solutions(puzzle, limit=2, budget=200_000):
    found, _, exhausted = search(puzzle, None, budget, count_limit=limit)
    return len(found), exhausted


def greedy(puzzle, net):
    """Neural decisions only: most confident legal move each step, never undone."""
    state, steps = puzzle.initial(), 0
    greedy.last_trail = []
    while True:
        doms = puzzle.domains(state, prune=False)
        if doms is None:
            return None, steps
        if not doms:
            return (puzzle.answer(state) if puzzle.complete(state) else None), steps
        logp = _logits(net, puzzle, state)
        best = None
        for var, values in doms.items():
            if not values:
                continue
            ordered, scores = _ordered(puzzle, logp, state, var, values)
            if best is None or scores[0] > best[0]:
                best = (scores[0], var, ordered[0])
        if best is None:
            return None, steps
        state = puzzle.assign(state, best[1], best[2])
        greedy.last_trail.append((best[1], best[2]))
        steps += 1


def oneshot(puzzle, net):
    if not hasattr(puzzle, "oneshot"):
        return None
    return puzzle.oneshot(_logits(net, puzzle, puzzle.initial()))


def solve(puzzle, net=None, mode="search", budget=100_000):
    t = time.perf_counter()
    expanded, exhausted, answer = 0, False, None
    if mode in ("search", "exact"):
        found, expanded, exhausted = search(puzzle, net if mode == "search" else None, budget)
        answer = found[0] if found else None
    elif mode == "neural":
        answer, expanded = greedy(puzzle, net)
    elif mode == "oneshot":
        answer = oneshot(puzzle, net)
    else:
        raise ValueError(f"unknown mode {mode}")
    ok = answer is not None and puzzle.verify(answer)
    trail = getattr(search, "last_trail", []) if mode in ("search", "exact") else getattr(greedy, "last_trail", []) if mode == "neural" else []
    return {"mode": mode, "solved": bool(ok), "answer": answer, "expanded_states": expanded,
            "budget_exhausted": exhausted, "ms": round(1000 * (time.perf_counter() - t), 2),
            "trail": [[_plain(v), _plain(x)] for v, x in trail]}


def _plain(v):
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def untrained_like(net, seed=0):
    torch.manual_seed(seed)
    return GridNet(**net.config).eval()
