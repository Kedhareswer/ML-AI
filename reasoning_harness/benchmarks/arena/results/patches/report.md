# patches held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Network picks every move, rules block illegal moves, no backtracking | 217/300 (72.33%) | 67.01–77.09% | 8.63 | 101.44 | 177.63 |
| Same, random-initialised network (control) | 7/300 (2.33%) | 1.13–4.74% | 7.56 | 95.17 | 166.4 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 8.83 | 106.27 | 175.14 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 13.96 | 106.42 | 199.67 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 13.87 | 0.45 | 8.85 |
