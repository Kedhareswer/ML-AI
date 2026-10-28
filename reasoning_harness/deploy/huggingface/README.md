---
title: Puzzle Arena
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Scratch-trained solvers for Sudoku, Queens, Zip, chess
---

# Puzzle Arena

Small neural networks trained **from scratch** on a CPU for Sudoku (real
Sudoku-Extreme puzzles), Mini Sudoku, Tango, Queens, Zip, Patches, Wend,
chess puzzles (real Lichess puzzles), checkers and a Pinpoint experiment.

- Pick a game and solve held-out puzzles with the network alone, network +
  backtracking, or a classical solver. Every answer is checked by an
  independent rule verifier.
- **Solve from a screenshot:** upload or paste an image of a board. The app
  finds the board, reads every cell with classifiers trained from scratch,
  checks the reading against the rules, and solves it.

Results, methods and limitations (including what does *not* work well, such
as Pinpoint) are documented in the source repository's `ARENA.md`.

Runs on the free CPU tier. The hardest Sudoku-Extreme puzzles can take
10–30 seconds with network-guided search.
