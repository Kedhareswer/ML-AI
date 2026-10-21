# Solving from screenshots

Recorded by `scripts/capture_vision.py`: Playwright uploads an image to the
running `arena ui`, which reads it with the trained classifiers and solves it.
The uploaded images (`*-input.*`) are kept next to the results
(`*-from-image.png`), and `captures.json` holds what was read.

- [image-solving-demo.mp4](image-solving-demo.mp4): 3-minute recording of all six uploads.
  It is built from browser frames captured about 5 times per second.
- Inputs were made fresh for the demo:
  - Sudoku, Queens, Zip and Tango: held-out puzzles screenshotted in dark mode, then rescaled and JPEG-compressed.
  - Chess: a real Lichess daily puzzle rendered by Lichess in the *california* piece set, which the classifiers never saw. The board was read exactly, and the engine's move (Nf3+) is the puzzle's solution.
  - Checkers: a mid-game position.

| Input | Result |
|---|---|
| [sudoku-input.jpg](sudoku-input.jpg) | [read and solved](sudoku-from-image.png) |
| [queens-input.jpg](queens-input.jpg) | [read and solved](queens-from-image.png) |
| [zip-input.jpg](zip-input.jpg) | [read and solved](zip-from-image.png) |
| [tango-input.jpg](tango-input.jpg) | [read and solved](tango-from-image.png) |
| [chess-input.gif](chess-input.gif) | [best move Nf3+](chess-from-image.png) |
| [checkers-input.jpg](checkers-input.jpg) | [best move](checkers-from-image.png) |

The chess read was flagged "low confidence" (one square at 47%) but was
exactly right. The flag asks the user to check the grid; it does not mean the
read is wrong. Solve times shown in these captures are inflated: a benchmark
was running on the same 4 CPU cores. Use the held-out test in
[ARENA.md](../../../ARENA.md#solving-from-a-screenshot) to judge accuracy.
