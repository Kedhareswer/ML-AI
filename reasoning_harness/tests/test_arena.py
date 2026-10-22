"""Rules, verifiers, generators, solvers and UI handlers for the puzzle arena."""
import json
import random
import unittest

import numpy as np
import torch

from arena.nets import GridNet
from arena.registry import GENERATORS, from_json, identity
from arena.search import count_solutions, solve
from arena.puzzles.sudoku import Sudoku, MiniSudoku
from arena.puzzles.tango import Tango
from arena.puzzles.queens import Queens
from arena.puzzles.zip import Zip
from arena.puzzles.patches import Patches

# A real Sudoku-Extreme test row (rating 3) and its published answer.
SUDOKU_Q = ".9......3....8.9...6.7.32....4..5..85.....63......9.5..8..1...41....782..239...7."
SUDOKU_A = "297456183345281967861793245934625718512874639678139452786512394159347826423968571"


class GeneratorTests(unittest.TestCase):
    def test_every_generator_gives_unique_verified_puzzles(self):
        for kind, gen in GENERATORS.items():
            for seed in range(3):
                puzzle, solution = gen(seed)
                self.assertTrue(puzzle.verify(solution), kind)
                self.assertEqual(count_solutions(puzzle)[0], 1, kind)
                again = from_json(json.loads(json.dumps(puzzle.to_json())))
                self.assertTrue(again.verify(solution), kind)

    def test_generation_is_deterministic(self):
        for kind, gen in GENERATORS.items():
            a, _ = gen(42)
            b, _ = gen(42)
            self.assertEqual(identity(a.to_json()), identity(b.to_json()), kind)


class VerifierTests(unittest.TestCase):
    def test_sudoku_real_row(self):
        p = Sudoku.parse(SUDOKU_Q)
        answer = [int(c) for c in SUDOKU_A]
        self.assertTrue(p.verify(answer))
        bad = answer[:]
        bad[0], bad[1] = bad[1], bad[0]
        self.assertFalse(p.verify(bad))
        result = solve(p, mode="exact")
        self.assertTrue(result["solved"])
        self.assertEqual(result["answer"], answer)

    def test_verifiers_reject_corrupted_answers(self):
        rng = random.Random(0)
        for kind, gen in GENERATORS.items():
            puzzle, solution = gen(5)
            corrupt = json.loads(json.dumps(solution))
            if kind in ("mini_sudoku",):
                i = next(i for i, v in enumerate(puzzle.clues.reshape(-1)) if v == 0)
                corrupt[i] = corrupt[i] % 6 + 1
            elif kind == "tango":
                r, c = next((r, c) for r in range(6) for c in range(6) if (r, c) not in puzzle.givens)
                corrupt[r][c] ^= 1
            elif kind == "queens":
                corrupt[0], corrupt[1] = corrupt[1], corrupt[0]
            elif kind == "zip":
                corrupt[1], corrupt[2] = corrupt[2], corrupt[1]
            elif kind == "patches":
                corrupt[0] = [corrupt[0][0], corrupt[0][1], corrupt[0][2], corrupt[0][3] + 1]
            elif kind == "wend":
                corrupt = corrupt[1:]
            self.assertFalse(puzzle.verify(corrupt), kind)

    def test_tango_rules(self):
        t = Tango({}, {((0, 0), (0, 1)): "x"})
        s = t.initial()
        s = t.assign(s, (0, 0), 0)
        self.assertEqual(t.domains(s)[(0, 1)], [1])

    def test_queens_touching_rejected(self):
        regions = [[r for _ in range(5)] for r in range(5)]
        q = Queens(regions)
        self.assertFalse(q.verify([0, 1, 3, 4, 2]))  # rows 0/1 touch diagonally

    def test_zip_walls_block(self):
        z = Zip(2, {(0, 0): 1, (1, 0): 2}, walls=[((0, 0), (0, 1))])
        self.assertFalse(z.verify([[0, 0], [0, 1], [1, 1], [1, 0]]))

    def test_patches_shape_rules(self):
        p = Patches(2, [(0, 0, 2, "tall"), (0, 1, 2, "any")])
        self.assertTrue(p.verify([[0, 0, 2, 1], [0, 1, 2, 1]]))
        self.assertFalse(Patches(2, [(0, 0, 2, "wide"), (1, 0, 2, "any")]).verify([[0, 0, 2, 1], [0, 1, 2, 1]]))


class SolverTests(unittest.TestCase):
    def test_modes_run_with_a_network_and_verify(self):
        torch.manual_seed(0)
        for kind, gen in GENERATORS.items():
            puzzle, _ = gen(1)
            net = GridNet(puzzle.in_ch, puzzle.out_ch, 16, 2).eval()
            for mode in ("neural", "search", "exact"):
                r = solve(puzzle, net, mode, budget=20000)
                if mode != "neural":
                    self.assertTrue(r["solved"], (kind, mode))
                if r["solved"]:
                    self.assertTrue(puzzle.verify(r["answer"]))

    def test_budget_is_respected(self):
        p = Sudoku.parse(SUDOKU_Q)
        r = solve(p, None, "exact", budget=5)
        self.assertLessEqual(r["expanded_states"], 6)

    def test_network_sees_padding_mask(self):
        puzzle, _ = GENERATORS["queens"](3)
        net = GridNet(puzzle.in_ch, puzzle.out_ch, 16, 2).eval()
        x, mask, regions = puzzle.tensors(puzzle.initial())
        out = net(torch.from_numpy(x)[None], torch.from_numpy(mask)[None], torch.from_numpy(regions)[None])
        self.assertTrue(torch.all(out[0, :, puzzle.n:, :] == net.head.bias[:, None, None]))


class GameTests(unittest.TestCase):
    def test_checkers_forced_capture_and_crowning(self):
        from arena.games import checkers as ck
        b = np.zeros((8, 8), np.int8)
        b[5, 2], b[4, 3] = 1, -1
        b[6, 7] = 1
        moves = ck.legal_moves(b)
        self.assertTrue(all(m[1] for m in moves))  # capture is compulsory
        b = np.zeros((8, 8), np.int8)
        b[1, 2] = 1
        nb, progress = ck.apply(b, ([(1, 2), (0, 1)], []))
        self.assertTrue(progress)
        self.assertEqual(ck.flip(nb)[0, 1], 2)

    def test_checkers_double_jump(self):
        from arena.games import checkers as ck
        b = np.zeros((8, 8), np.int8)
        b[6, 1], b[5, 2], b[3, 4] = 1, -1, -1
        moves = ck.legal_moves(b)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][0][-1], (2, 5))
        self.assertEqual(len(moves[0][1]), 2)

    def test_chess_puzzle_checker_accepts_alternative_mate(self):
        import chess
        from arena.games.chess_engine import solve_puzzle
        # after the setup move ...a6, Rd8# is the listed answer and Re8# is an equally valid mate
        row = {"FEN": "6k1/p4ppp/8/8/8/8/5PPP/3RR1K1 b - - 0 1", "Moves": "a7a6 d1d8"}
        self.assertTrue(solve_puzzle(row, lambda b: chess.Move.from_uci("d1d8"))[0])
        self.assertTrue(solve_puzzle(row, lambda b: chess.Move.from_uci("e1e8"))[0])
        self.assertFalse(solve_puzzle(row, lambda b: chess.Move.from_uci("g1f1"))[0])

    def test_chess_search_finds_mate_in_one(self):
        import chess
        from arena.games.chess_engine import Searcher
        board = chess.Board("6k1/p4ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
        before = board.fen()
        Searcher(None, 30, 3).choose(board)  # budget stop mid-search must not corrupt the board
        self.assertEqual(board.fen(), before)
        move = Searcher(None, 5000, 2).choose(board)
        board.push(move)
        self.assertTrue(board.is_checkmate())


@unittest.skipUnless(__import__("pathlib").Path("artifacts/arena/pinpoint/bigrams.npz").exists(), "needs Pinpoint link counts")
class PinpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from arena.games.pinpoint import Guesser, LinkGuesser
        root = "artifacts/arena/pinpoint"
        cls.g = LinkGuesser(root, Guesser(root, open(root + "/candidates.txt").read().split()))

    def test_phrase_compound_and_category_links(self):
        self.assertEqual(self.g.rank(["baby", "poker"])[0][0], "face")
        self.assertEqual(self.g.rank(["butter", "dragon", "fire", "house", "horse"])[0][0], "fly")
        self.assertEqual(self.g.rank(["mercury", "venus", "mars", "jupiter", "saturn"])[0][0], "planet")
        self.assertIn("poker face", self.g.explain(["baby", "poker"], "face")[1])

    def test_guesses_are_distinct_and_never_a_clue(self):
        out = [w for w, _ in self.g.rank(["storm", "snow", "thunder"], 5)]
        self.assertEqual(len(out), len(set(out)))
        self.assertFalse({"storm", "snow", "thunder"} & set(out))


if __name__ == "__main__":
    unittest.main()
