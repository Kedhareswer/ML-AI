# sudoku held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| One network pass, no rules | 8/300 (2.67%) | 1.36–5.17% | 0.0 | 16.44 | 22.23 |
| Network picks every move, rules block illegal moves, no backtracking | 62/300 (20.67%) | 16.47–25.61% | 54.06 | 886.0 | 1038.59 |
| Same, random-initialised network (control) | 0/300 (0.0%) | 0–1.26% | 42.98 | 706.54 | 856.5 |
| Backtracking ordered by the trained network | 285/300 (95.0%) | 91.92–96.95% | 687.38 | 8807.81 | 28803.7 |
| Backtracking ordered by a random network (control) | 294/300 (98.0%) | 95.71–99.08% | 544.45 | 6625.39 | 24858.21 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 586.55 | 99.05 | 718.64 |
