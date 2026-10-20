# wend held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| One network pass, no rules | 1/300 (0.33%) | 0.06–1.86% | 0.0 | 6.64 | 9.73 |
| Network picks every move, rules block illegal moves, no backtracking | 47/300 (15.67%) | 11.99–20.21% | 4.78 | 34.11 | 51.96 |
| Same, random-initialised network (control) | 11/300 (3.67%) | 2.06–6.45% | 4.1 | 31.29 | 49.56 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 6.64 | 38.43 | 63.54 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 6.98 | 37.83 | 61.86 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 7.39 | 0.81 | 1.36 |
