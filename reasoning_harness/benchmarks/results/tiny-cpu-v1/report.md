# maze-planning-v1

Ready: **False**. Cases: 900. Device: cpu.

| Size | Greedy | Feedback retries | No-feedback retries | Untrained | Random | BFS |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 9.7% | 13.0% | 15.7% | 10.3% | 10.3% | 100.0% |
| 4 | 4.7% | 11.3% | 11.0% | 6.7% | 2.7% | 100.0% |
| 5 | 4.0% | 9.3% | 9.7% | 7.0% | 0.7% | 100.0% |

Provisional gates:

- FAIL / UNRUN: weight_size_30_to_100_MB
- PASS: verifier_exhaustive_audit
- PASS: bfs_all_cases_valid_and_optimal
- PASS: no_training_overlap
- PASS: multiple_evaluation_seeds
- FAIL / UNRUN: id_quality_target
- FAIL / UNRUN: larger_maze_quality_target
- FAIL / UNRUN: positive_feedback_effect_95_CI
- PASS: bounded_failure_on_impossible_mazes
- FAIL / UNRUN: gpu_runtime_measured

See report.json for confidence intervals, latency, resource measurements, provenance and limitations. Raw paths and verdicts are in records.jsonl.
