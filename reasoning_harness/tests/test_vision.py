"""Image reading: board finding, grid size, and full screenshot -> puzzle round trips."""
import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arena.vision import board as B

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "artifacts" / "arena" / "vision"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def draw_sudoku(grid, n=9, cell=48, margin=(130, 70), canvas=(900, 700)):
    """A plain screenshot-like image: board somewhere on a page, digits in each cell."""
    img = Image.new("RGB", canvas, (245, 244, 240))
    d = ImageDraw.Draw(img)
    x0, y0 = margin
    d.rectangle([x0 - 3, y0 - 3, x0 + n * cell + 3, y0 + n * cell + 3], fill=(20, 20, 20))
    d.rectangle([x0, y0, x0 + n * cell, y0 + n * cell], fill=(255, 255, 255))
    for k in range(1, n):
        w = 3 if k % 3 == 0 else 1
        d.line([x0 + k * cell, y0, x0 + k * cell, y0 + n * cell], fill=(40, 40, 40), width=w)
        d.line([x0, y0 + k * cell, x0 + n * cell, y0 + k * cell], fill=(40, 40, 40), width=w)
    font = ImageFont.truetype(FONT, int(cell * 0.6))
    for i, v in enumerate(grid):
        if v:
            r, c = divmod(i, n)
            d.text((x0 + c * cell + cell / 2, y0 + r * cell + cell / 2), str(v), font=font, fill=(0, 0, 0), anchor="mm")
    d.text((20, 20), "Some page title and buttons around the board", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class BoardTests(unittest.TestCase):
    def test_finds_board_and_grid_size(self):
        grid = [0] * 81
        board, quad, n = B.find_board(B.load(draw_sudoku(grid)), [9])
        self.assertEqual(n, 9)
        x = sorted(p[0] for p in quad)
        self.assertLess(abs(x[0] - 130), 8)
        self.assertLess(abs(x[-1] - (130 + 9 * 48)), 8)

    def test_grid_size_choice_prefers_true_size(self):
        img = Image.new("RGB", (700, 600), (250, 250, 250))
        d = ImageDraw.Draw(img)
        n, cell, x0, y0 = 7, 60, 100, 50
        for k in range(n + 1):
            d.line([x0 + k * cell, y0, x0 + k * cell, y0 + n * cell], fill=(30, 30, 30), width=2)
            d.line([x0, y0 + k * cell, x0 + n * cell, y0 + k * cell], fill=(30, 30, 30), width=2)
        _, _, found = B.find_board(np.asarray(img), list(range(4, 9)))
        self.assertEqual(found, 7)


@unittest.skipUnless((VISION / "digits.pt").exists(), "vision classifiers not trained")
class ReadTests(unittest.TestCase):
    def test_sudoku_round_trip(self):
        from arena.vision.read import read
        grid = [int(c) for c in "000260701680070090190004500820100040004602900050003028009300074040050036703018000"]
        out = read("sudoku", draw_sudoku(grid), root=VISION)
        self.assertEqual(out["n"], 9)
        self.assertEqual(out["puzzle"]["grid"], grid)

    def test_fen_from_labels(self):
        from arena.vision.read import _fen
        grid = [["empty"] * 8 for _ in range(8)]
        grid[7][4], grid[0][4], grid[7][7] = "wK", "bK", "wR"
        self.assertEqual(_fen(grid, "w"), "4k3/8/8/8/8/8/8/4K2R w K - 0 1")


if __name__ == "__main__":
    unittest.main()
