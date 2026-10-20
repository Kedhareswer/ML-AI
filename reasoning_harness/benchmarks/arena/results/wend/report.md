# wend held-out benchmark

300 puzzles never seen in training. Every answer is checked by an independent rule verifier.

| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| One network pass, no rules | 0/300 (0.0%) | 0–1.26% | 0.0 | 11.38 | 20.02 |
| Network picks every move, rules block illegal moves, no backtracking | 66/300 (22.0%) | 17.68–27.03% | 4.97 | 56.28 | 99.16 |
| Same, random-initialised network (control) | 7/300 (2.33%) | 1.13–4.74% | 4.16 | 49.92 | 86.62 |
| Backtracking ordered by the trained network | 300/300 (100.0%) | 98.74–100.0% | 6.58 | 60.09 | 105.29 |
| Backtracking ordered by a random network (control) | 300/300 (100.0%) | 98.74–100.0% | 6.89 | 60.41 | 111.25 |
| Classical backtracking, no network (baseline) | 300/300 (100.0%) | 98.74–100.0% | 7.39 | 0.95 | 9.57 |
