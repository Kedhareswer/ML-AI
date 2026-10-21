"""Chess players and the Lichess-puzzle benchmark.

Players:
  neural  - plays the network's highest-probability legal move (no search).
  search  - alpha-beta search (material + mate) whose move ordering and root
            tie-breaks come from the network.
  classic - the same alpha-beta with capture-first ordering and no network.
  random  - uniformly random legal move.
"""
import json
import math
import random
import time

import chess
import numpy as np
import torch

from .chess_net import ChessNet, policy

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE = 1000


class NodeBudget(Exception):
    pass


def material(board):
    s = 0
    for piece in board.piece_map().values():
        s += VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
    return s


def mvv_lva(board, move):
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        v = VALUES[victim.piece_type] if victim else 1
        a = VALUES[board.piece_at(move.from_square).piece_type]
        return 10 * v - a + 100
    return 50 if board.gives_check(move) else 0


class Searcher:
    def __init__(self, net=None, budget=6000, depth=3, prior_weight=0.3, net_plies=2):
        self.net, self.budget, self.depth = net, budget, depth
        self.prior_weight, self.net_plies = prior_weight, net_plies
        self.nodes = 0

    def order(self, board, ply):
        moves = list(board.legal_moves)
        if self.net is not None and ply < self.net_plies:
            pri = policy(self.net, board)
            return sorted(moves, key=lambda m: -pri[m]), pri
        return sorted(moves, key=lambda m: -mvv_lva(board, m)), None

    def quiesce(self, board, alpha, beta, qdepth=0):
        self.nodes += 1
        if self.nodes > self.budget:
            raise NodeBudget
        if board.is_checkmate():
            return -MATE
        stand = material(board)
        if stand >= beta or qdepth >= 4:
            return stand
        alpha = max(alpha, stand)
        caps = sorted((m for m in board.legal_moves if board.is_capture(m)), key=lambda m: -mvv_lva(board, m))
        for m in caps:
            board.push(m)
            v = -self.quiesce(board, -beta, -alpha, qdepth + 1)
            board.pop()
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if self.nodes > self.budget:
            raise NodeBudget
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
            return 0
        if depth == 0:
            return self.quiesce(board, alpha, beta)
        moves, _ = self.order(board, ply)
        best = -math.inf
        for m in moves:
            board.push(m)
            v = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def choose(self, board):
        """Iterative deepening; returns the best move of the deepest finished iteration."""
        board = board.copy()  # a budget stop can leave moves pushed; never touch the caller's board
        self.nodes = 0
        moves, pri = self.order(board, 0)
        best_move = moves[0]
        try:
            for d in range(1, self.depth + 1):
                scored = []
                for m in moves:
                    board.push(m)
                    v = -self.negamax(board, d - 1, -math.inf, math.inf, 1)
                    board.pop()
                    bonus = self.prior_weight * pri[m] if pri is not None else 0.0
                    scored.append((v + bonus, m))
                scored.sort(key=lambda x: -x[0])
                best_move = scored[0][1]
                moves = [m for _, m in scored]
                if scored[0][0] >= MATE - 50:
                    break
        except NodeBudget:
            pass
        return best_move


def make_player(kind, net=None, budget=6000, depth=3, rng=None):
    rng = rng or random.Random(0)
    if kind == "random":
        return lambda b: rng.choice(list(b.legal_moves))
    if kind == "neural":
        return lambda b: max(policy(net, b).items(), key=lambda kv: kv[1])[0]
    if kind == "search":
        s = Searcher(net, budget, depth)
        return s.choose
    if kind == "classic":
        s = Searcher(None, budget, depth)
        return s.choose
    raise ValueError(kind)


def solve_puzzle(row, player):
    """Lichess rules: every solver move must match, except any mating move is accepted."""
    board = chess.Board(row["FEN"])
    line = row["Moves"].split()
    board.push_uci(line[0])
    played = []
    for k, uci in enumerate(line[1:]):
        if k % 2 == 0:
            move = player(board)
            played.append(move.uci())
            expected = chess.Move.from_uci(uci)
            board.push(move)
            if move != expected and not board.is_checkmate():
                return False, played
            if board.is_checkmate():
                return True, played
            board.pop()
            board.push(expected)
        else:
            board.push_uci(uci)
    return True, played


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    net = ChessNet(**data["config"])
    net.load_state_dict(data["state"])
    return net.eval()


def play_game(white, black, max_plies=160, opening=None):
    board = chess.Board()
    for uci in opening or []:
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mover = white if board.turn == chess.WHITE else black
        board.push(mover(board))
    out = board.outcome(claim_draw=True)
    if out is None:  # adjudicate unfinished games by material
        m = material(board) * (1 if board.turn == chess.WHITE else -1)
        result = "1-0" if m >= 3 else ("0-1" if m <= -3 else "1/2-1/2")
        return result, board, "adjudicated"
    return out.result(), board, out.termination.name.lower()
