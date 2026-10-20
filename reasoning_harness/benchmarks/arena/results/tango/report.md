# tango held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| One network pass, no rules | 85/300 (28.33%) | 23.53–33.68% | 0.0 | 3.23 | 9.06 |
| Network picks every move, rules block illegal moves, no backtracking | 252/300 (84.0%) | 79.43–87.71% | 29.74 | 120.45 | 183.44 |
| Same, random-initialised network (control) | 0/300 (0.0%) | 0–1.26% | 23.67 | 98.22 | 153.38 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 35.27 | 121.39 | 244.04 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 64.37 | 161.53 | 823.8 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 71.12 | 12.57 | 56.95 |
