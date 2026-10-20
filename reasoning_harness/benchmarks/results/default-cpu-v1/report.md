# maze-planning-v1

Ready: **False**. Cases: 180. Device: cpu.

| Size | Greedy | Feedback retries | No-feedback retries | Untrained | Random | BFS |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 0.0% | 10.0% | 8.9% | 12.2% | 2.2% | 100.0% |
| 5 | 0.0% | 10.0% | 8.9% | 10.0% | 0.0% | 100.0% |

Provisional gates:

- PASS: weight_size_30_to_100_MB
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
