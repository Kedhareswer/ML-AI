"""American checkers (English draughts) with a value network trained from scratch by self-play.

Rules: 8x8 dark squares, men move diagonally forward, kings both ways (one
square), captures are compulsory and multi-jumps must be completed, a man
that reaches the far row is crowned and the move ends. A side with no legal
move loses. 80 plies without a capture or crowning is a draw.

Boards are always stored from the side to move's point of view: +1 own man,
+2 own king, -1/-2 opponent, own men move toward row 0.
"""
import random
import time

import numpy as np
import torch
from torch import nn

DRAW_PLIES = 80


def initial():
    b = np.zeros((8, 8), np.int8)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                if r < 3:
                    b[r, c] = -1
                elif r > 4:
                    b[r, c] = 1
    return b


def flip(b):
    return -b[::-1, ::-1].copy()


def _jumps(b, r, c, piece, captured):
    dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1)) if piece == 2 else ((-1, -1), (-1, 1))
    found = []
    for dr, dc in dirs:
        mr, mc, lr, lc = r + dr, c + dc, r + 2 * dr, c + 2 * dc
        if 0 <= lr < 8 and 0 <= lc < 8 and b[mr, mc] < 0 and (mr, mc) not in captured and b[lr, lc] == 0:
            caps = captured + [(mr, mc)]
            if piece == 1 and lr == 0:  # crowned: move ends
                found.append(([(lr, lc)], caps))
                continue
            b2 = b.copy()
            b2[r, c], b2[lr, lc] = 0, piece
            more = _jumps(b2, lr, lc, piece, caps)
            if more:
                found += [([(lr, lc)] + p, cp) for p, cp in more]
            else:
                found.append(([(lr, lc)], caps))
    return found


def legal_moves(b):
    """Moves as (path, captured): path starts at the moving piece's square."""
    caps, quiet = [], []
    for r, c in zip(*np.nonzero(b > 0)):
        r, c = int(r), int(c)
        piece = int(b[r, c])
        for path, cp in _jumps(b, r, c, piece, []):
            caps.append(([(r, c)] + path, cp))
        dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1)) if piece == 2 else ((-1, -1), (-1, 1))
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and b[nr, nc] == 0:
                quiet.append(([(r, c), (nr, nc)], []))
    return caps if caps else quiet


def apply(b, move):
    """Returns the next board (flipped to the opponent's view) and whether it was progress."""
    path, captured = move
    b = b.copy()
    (r0, c0), (r1, c1) = path[0], path[-1]
    piece = b[r0, c0]
    b[r0, c0] = 0
    for r, c in captured:
        b[r, c] = 0
    crowned = piece == 1 and r1 == 0
    b[r1, c1] = 2 if crowned else piece
    return flip(b), bool(captured) or crowned


def absolute(b, side):
    """Board in black's orientation (black positive, moving up the screen)."""
    return b if side == 0 else flip(b)


def material(b):
    return float((b == 1).sum() + 1.5 * (b == 2).sum() - (b == -1).sum() - 1.5 * (b == -2).sum())


def encode(boards):
    arr = np.stack(boards)
    x = np.stack([arr == 1, arr == 2, arr == -1, arr == -2], 1).astype(np.float32)
    return torch.from_numpy(x)


class ValueNet(nn.Module):
    """Predicts the final result (+1 win, -1 loss) for the side to move."""

    def __init__(self, width=64):
        super().__init__()
        self.config = {"width": width}
        self.body = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(width * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        return self.body(x).squeeze(1)


def net_values(net, boards):
    with torch.inference_mode():
        return net(encode(boards)).numpy()


# --------------------------------------------------------------------- players
def choose(b, player, rng, plies_since_progress=0):
    moves = legal_moves(b)
    if not moves:
        return None
    kind = player["kind"]
    if kind == "random" or len(moves) == 1:
        return rng.choice(moves)
    depth = player.get("depth", 2)
    score = (lambda bs: net_values(player["net"], bs)) if kind == "net" else (lambda bs: np.array([material(x) / 12 for x in bs]))

    def negamax(board, d, alpha, beta):
        ms = legal_moves(board)
        if not ms:
            return -1.0
        children = [apply(board, m)[0] for m in ms]
        if d == 1:
            return float(np.max(-score(children)))
        best = -2.0
        for child in children:
            v = -negamax(child, d - 1, -beta, -alpha)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    vals = []
    children = [apply(b, m)[0] for m in moves]
    if depth == 1:
        vals = list(-score(children))
    else:
        for child in children:
            vals.append(-negamax(child, depth - 1, -2.0, 2.0))
    vals = np.array(vals)
    if player.get("explore", 0) > 0 and rng.random() < player["explore"]:
        return rng.choice(moves)
    tau = player.get("temperature", 0)
    if tau > 0:  # self-play diversity: sample moves in proportion to exp(value / tau)
        p = np.exp((vals - vals.max()) / tau)
        return moves[int(np.searchsorted(np.cumsum(p / p.sum()), rng.random()))]
    best = np.flatnonzero(vals >= vals.max() - 1e-6)
    return moves[int(rng.choice(list(best)))]


def play(p_black, p_white, rng, opening_random=2, max_plies=200, record=False):
    """Black moves first. Returns +1 black win, -1 white win, 0 draw (and the positions)."""
    # Internal view: side to move's pieces are positive. The absolute board is black's view.
    b, quiet, history = initial(), 0, []
    players = (p_black, p_white)
    for ply in range(max_plies):
        side = ply % 2
        if record:
            history.append((b.copy(), side))
        player = {"kind": "random"} if ply < opening_random else players[side]
        move = choose(b, player, rng, quiet)
        if move is None:
            return (-1 if side == 0 else 1), history, ply
        b, progress = apply(b, move)
        quiet = 0 if progress else quiet + 1
        if quiet >= DRAW_PLIES:
            return 0, history, ply
    return 0, history, max_plies


def td_targets(net, hist, result, lam):
    """TD(lambda) targets for one game, each from the mover's point of view.

    The last position takes the game result; earlier ones blend the (negated)
    value of the next position with the target after it.
    """
    boards = [b for b, _ in hist]
    sides = [s for _, s in hist]
    v = net_values(net, boards) if boards else np.zeros(0)
    out = np.zeros(len(boards), np.float32)
    nxt = None
    for t in range(len(boards) - 1, -1, -1):
        z = result if sides[t] == 0 else -result
        if nxt is None:
            out[t] = z
        else:
            out[t] = (1 - lam) * (-v[t + 1]) + lam * (-nxt)
        nxt = out[t]
    return out


def train_selfplay(run, generations=12, games_per_gen=300, epochs=4, seed=0, log=print,
                   explore=0.15, temperature=0.0, td_lambda=None):
    from pathlib import Path
    import json
    torch.manual_seed(seed)
    rng = random.Random(seed)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    net = ValueNet().eval()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer_x, buffer_y = [], []
    t0 = time.time()
    metrics = open(run / "metrics.jsonl", "w")
    for gen in range(1, generations + 1):
        player = {"kind": "net", "net": net, "depth": 1, "explore": explore, "temperature": temperature}
        results = []
        for _ in range(games_per_gen):
            result, hist, _ = play(player, player, rng, opening_random=4, record=True)
            results.append(result)
            if td_lambda is None:
                targets = [result if side == 0 else -result for _, side in hist]  # result for the mover
            else:
                targets = td_targets(net, hist, result, td_lambda)
            for (board, _), z in zip(hist, targets):
                buffer_x.append(board)
                buffer_y.append(float(z))
        buffer_x, buffer_y = buffer_x[-200_000:], buffer_y[-200_000:]
        net.train()
        y_all = torch.tensor(buffer_y, dtype=torch.float32)
        losses = []
        for _ in range(epochs):
            perm = torch.randperm(len(buffer_x))
            for i in range(0, len(perm), 256):
                idx = perm[i:i + 256]
                x = encode([buffer_x[j] for j in idx.tolist()])
                loss = ((net(x) - y_all[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        net.eval()
        rec = {"generation": gen, "positions": len(buffer_x), "mse": round(float(np.mean(losses[-50:])), 4),
               "draw_rate": round(results.count(0) / len(results), 3), "elapsed_s": round(time.time() - t0, 1)}
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        log(json.dumps(rec))
        torch.save({"config": net.config, "state": net.state_dict(), "generation": gen}, run / "weights.pt")
    return net


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    net = ValueNet(**data["config"])
    net.load_state_dict(data["state"])
    return net.eval()
