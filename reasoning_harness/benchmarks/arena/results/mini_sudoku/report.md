# mini_sudoku held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| One network pass, no rules | 123/300 (41.0%) | 35.58–46.65% | 0.0 | 3.92 | 13.19 |
| Network picks every move, rules block illegal moves, no backtracking | 298/300 (99.33%) | 97.6–99.82% | 22.97 | 103.02 | 263.14 |
| Same, random-initialised network (control) | 2/300 (0.67%) | 0.18–2.4% | 17.49 | 82.67 | 194.97 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 23.0 | 110.12 | 245.72 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 22.99 | 105.94 | 273.76 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 23.0 | 5.0 | 14.54 |
