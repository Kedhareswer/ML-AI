# maze-planning-v1

Ready: **False**. Cases: 1200. Device: cpu.

| Size | Greedy | Feedback retries | No-feedback retries | Untrained | Random | BFS |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 99.3% | 99.7% | 99.7% | 93.3% | 5.3% | 100.0% |
| 5 | 90.3% | 96.0% | 96.0% | 86.0% | 2.0% | 100.0% |
| 6 | 88.3% | 94.7% | 94.7% | 78.0% | 1.3% | 100.0% |
| 8 | 84.0% | 92.3% | 92.3% | 60.0% | 0.3% | 100.0% |

Provisional gates:

- PASS: weight_size_30_to_100_MB
- PASS: verifier_exhaustive_audit
- PASS: bfs_all_cases_valid_and_optimal
- PASS: no_training_overlap
- PASS: multiple_evaluation_seeds
- PASS: id_quality_target
- PASS: larger_maze_quality_target
- FAIL / UNRUN: positive_feedback_effect_95_CI
- PASS: bounded_failure_on_impossible_mazes
- FAIL / UNRUN: gpu_runtime_measured

See report.json for confidence intervals, latency, resource measurements, provenance and limitations. Raw paths and verdicts are in records.jsonl.
