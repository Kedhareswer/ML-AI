# maze-planning-v1

Ready: **False**. Cases: 270. Device: cpu.

| Size | Greedy | Feedback retries | No-feedback retries | Untrained | Random | BFS |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 94.4% | 98.9% | 98.9% | 92.2% | 5.6% | 100.0% |
| 6 | 88.9% | 95.6% | 95.6% | 80.0% | 0.0% | 100.0% |
| 8 | 78.9% | 85.6% | 85.6% | 54.4% | 0.0% | 100.0% |

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
