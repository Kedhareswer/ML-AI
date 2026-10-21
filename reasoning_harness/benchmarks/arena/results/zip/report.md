# zip held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Network picks every move, rules block illegal moves, no backtracking | 211/300 (70.33%) | 64.93–75.22% | 36.1 | 127.87 | 195.15 |
| Same, random-initialised network (control) | 0/300 (0.0%) | 0–1.26% | 9.84 | 30.6 | 80.62 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 53.26 | 155.56 | 375.88 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 117.21 | 278.77 | 985.09 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 105.26 | 8.55 | 37.67 |
