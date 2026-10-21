# Arena evidence

Captured by `scripts/capture_arena.py`: Playwright drives Chromium against the
locally running `arena ui` server with the trained models. Every screenshot is
a real model response; the right-hand panel shows the verifier's verdict.
`captures.json` holds the API response behind each screenshot.

- [arena-demo.mp4](arena-demo.mp4): 2-minute recording of the whole session.
  It is built from browser frames captured about 5 times per second, so it
  plays slightly faster than real time.
- Screenshots: [Sudoku](sudoku-search.png) (real Sudoku-Extreme puzzle,
  network + backtracking), [Mini Sudoku](mini-sudoku-neural.png),
  [Tango](tango-neural.png), [Queens](queens-neural.png) and
  [Patches](patches-neural.png) (network only), [Zip](zip-search.png) and
  [Wend](wend-search.png) (network + backtracking),
  [chess](chess-neural.png) (network only, a real held-out Lichess puzzle),
  [checkers](checkers-game.png) (full game against a material engine),
  [Pinpoint](pinpoint.png).
- [benchmark-summary.png](benchmark-summary.png): chart of the held-out
  benchmarks.

Puzzles are chosen at random, so the recording includes honest failures: the
chess network missed its puzzle (it played e4e5; the solution was e3f4),
Pinpoint guessed "hour" for Monday…Friday, and the checkers game was a
200-ply draw. Use the benchmark tables, not single screenshots, to judge
accuracy.
