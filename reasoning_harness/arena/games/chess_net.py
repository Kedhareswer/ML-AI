"""Chess move-policy network trained from scratch on the Lichess puzzle database (CC0).

python-chess supplies the rules (legal moves, check, mate); every move choice
comes from the network or from the explicit search below. Boards are encoded
from the side to move's point of view, so one network plays both colours.
"""
import hashlib
import io
import time

import chess
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PIECES = "PNBRQKpnbrqk"


def board_codes(board):
    """64 int8 codes (0 empty, 1-12 pieces) oriented to side to move, plus 6 flags."""
    codes = np.zeros(64, np.int8)
    flip = board.turn == chess.BLACK
    for sq, piece in board.piece_map().items():
        s = chess.square_mirror(sq) if flip else sq
        sym = piece.symbol()
        if flip:
            sym = sym.swapcase()
        codes[s] = PIECES.index(sym) + 1
    us, them = board.turn, not board.turn
    flags = np.array([board.has_kingside_castling_rights(us), board.has_queenside_castling_rights(us),
                      board.has_kingside_castling_rights(them), board.has_queenside_castling_rights(them),
                      board.ep_square is not None,
                      (chess.square_mirror(board.ep_square) if flip else board.ep_square) % 8 if board.ep_square is not None else 0],
                     np.int8)
    return codes, flags


def planes(codes, flags):
    """Batch (B,64) codes + (B,6) flags -> (B,18,8,8) float tensor."""
    codes = torch.as_tensor(codes, dtype=torch.long)
    flags = torch.as_tensor(flags, dtype=torch.float32)
    b = codes.shape[0]
    onehot = F.one_hot(codes, 13)[..., 1:].permute(0, 2, 1).float().view(b, 12, 8, 8)
    extra = flags[:, :5, None, None].expand(b, 5, 8, 8)
    ep = torch.zeros(b, 1, 8, 8)
    has_ep = flags[:, 4] > 0
    if has_ep.any():
        ep[has_ep, 0, 5, flags[has_ep, 5].long()] = 1
    return torch.cat([onehot, extra, ep], 1)


def move_index(move, turn):
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(w, w, 3, padding=1), nn.Conv2d(w, w, 3, padding=1)
        self.n1, self.n2 = nn.BatchNorm2d(w), nn.BatchNorm2d(w)

    def forward(self, x):
        return F.relu(x + self.n2(self.c2(F.relu(self.n1(self.c1(x))))))


class ChessNet(nn.Module):
    def __init__(self, width=96, blocks=6):
        super().__init__()
        self.config = {"width": width, "blocks": blocks}
        self.stem = nn.Sequential(nn.Conv2d(18, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.policy = nn.Conv2d(width, 64, 1)  # for each from-square, logits over 64 to-squares

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.policy(h).flatten(2).transpose(1, 2).reshape(x.shape[0], 4096)  # [from*64 + to]


def policy(net, board):
    """log-probabilities over the legal moves of `board`."""
    codes, flags = board_codes(board)
    with torch.inference_mode():
        logits = net(planes(codes[None], flags[None]))
    moves = list(board.legal_moves)
    idx = torch.tensor([move_index(m, board.turn) for m in moves])
    lp = torch.log_softmax(logits[0, idx], 0).numpy()
    # a promotion and its under-promotions share a from/to index: prefer the queen
    adj = [lp[i] - (0.5 if m.promotion not in (None, chess.QUEEN) else 0) for i, m in enumerate(moves)]
    return dict(zip(moves, adj))


def split_of(game_url):
    """Deterministic split by source game, so test puzzles never share a game with training."""
    h = int(hashlib.sha1(game_url.split("#")[0].encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 2 else "train"


def build_dataset(zst_path, out_path, limit_positions=1_200_000):
    """Parse the Lichess puzzle CSV into compact arrays of (position, best move)."""
    import csv
    import zstandard
    reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(zst_path, "rb"))))
    codes, flags, moves = [], [], []
    test_rows, daily_rows = [], []
    t = time.time()
    for row in reader:
        split = split_of(row["GameUrl"])
        if row.get("DailyDate"):
            daily_rows.append(row)
            continue
        if split == "test":
            test_rows.append(row)
            continue
        if len(moves) >= limit_positions:
            if len(test_rows) > 20000:
                break
            continue
        board = chess.Board(row["FEN"])
        line = row["Moves"].split()
        board.push_uci(line[0])
        for k, uci in enumerate(line[1:]):
            if k % 2 == 0:  # the solver's moves
                c, f = board_codes(board)
                codes.append(c), flags.append(f)
                moves.append(move_index(chess.Move.from_uci(uci), board.turn))
            board.push_uci(uci)
        if len(moves) % 100000 < 3:
            print(len(moves), round(time.time() - t), flush=True)
    np.savez_compressed(out_path, codes=np.array(codes), flags=np.array(flags), moves=np.array(moves, np.int64))
    return test_rows, daily_rows
