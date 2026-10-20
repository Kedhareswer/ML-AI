# queens held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| One network pass, no rules | 180/300 (60.0%) | 54.36–65.38% | 0.0 | 5.38 | 11.37 |
| Network picks every move, rules block illegal moves, no backtracking | 271/300 (90.33%) | 86.46–93.19% | 7.9 | 46.29 | 85.22 |
| Same, random-initialised network (control) | 1/300 (0.33%) | 0.06–1.86% | 5.9 | 40.24 | 70.0 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 13.96 | 48.41 | 175.49 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 66.23 | 150.91 | 708.2 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 69.78 | 5.19 | 34.45 |
