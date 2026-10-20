# Historical prototype benchmarks

These results describe the earlier failed autoregressive prototypes. The current trained policy, strict holdout results, three training repetitions and local solver readiness are documented in [POLICY_RESULTS.md](POLICY_RESULTS.md).

**Current model status: NOT READY.** Successful execution of the harness does not establish model quality. Neither saved checkpoint meets the proposed quality gates. The default-size checkpoint has only two training steps; it is a training/export smoke artifact.

## Saved evidence

- [38 MB checkpoint report](results/default-cpu-v1/report.md): 180 unique held-out mazes, 90 each at sizes 4 and 5, three evaluation seeds. [Machine-readable metrics](results/default-cpu-v1/report.json), [all candidates and verdicts](results/default-cpu-v1/records.jsonl), [frozen cases](results/default-cpu-v1/cases.json), [stress cases and validation loss](results/default-cpu-v1/supplement.json).
- [Tiny test checkpoint report](results/tiny-cpu-v1/report.md): 900 unique held-out mazes, 300 each at sizes 3, 4 and 5, three evaluation seeds. This checkpoint is below the requested weight size and serves as a training-system diagnostic. [Machine-readable metrics](results/tiny-cpu-v1/report.json), [raw outcomes](results/tiny-cpu-v1/records.jsonl), [stress and validation loss](results/tiny-cpu-v1/supplement.json).
- [Nine passing tests](results/unit-tests.txt), including exhaustive verifier checks, measurable training loss reduction, exact checkpoint resume, export equality, malformed-input rejection, causal attention and HTTP inference.

The complete updated notebook also executed locally with explicit CPU/tiny-model overrides: installation, two training steps, 30-case evaluation, inference and its new 90-case benchmark, including stress, validation loss and resource collection. This extra integration run is separate from the 1,080 quality cases above and does not validate remote Colab or GPU execution. The CLI's `--require-ready` behavior was exercised: it saved evidence and exited 2 for a failing candidate.

Primary benchmark runs finished before stress/teacher-forced diagnostics were integrated into the command, so those diagnostics are saved separately in `supplement.json`. Future invocations include them in `report.json`. Final source fingerprints are also saved; latency measurements cover earlier core inference code with identical finite-temperature behavior. Training configuration, manifests and per-step logs accompany each report. Model weights and optimizer states remain outside the repository under `/workspace/ml-ai-environment/harness-default-smoke` and `harness-tiny-pilot`; they are not published trained releases.

## Results

| Checkpoint / maze size | Greedy | Four feedback attempts | Four attempts without feedback | Untrained control | Exact BFS |
|---|---:|---:|---:|---:|---:|
| 38 MB, 4x4 | 0.0% | 10.0% | 8.9% | 12.2% | 100% |
| 38 MB, 5x5 | 0.0% | 10.0% | 8.9% | 10.0% | 100% |
| Tiny, 3x3 | 9.7% | 13.0% | 15.7% | 10.3% | 100% |
| Tiny, 4x4 | 4.7% | 11.3% | 11.0% | 6.7% | 100% |
| Tiny, 5x5 | 4.0% | 9.3% | 9.7% | 7.0% | 100% |

The 38 MB model's pooled feedback gain was 1.11 percentage points, with exploratory paired bootstrap 95% interval **−1.11 to +3.33 points**. This does not establish a feedback advantage. On the 100-example teacher-forced validation set, answer-token NLL was 3.429, perplexity 30.84, and unconstrained next-token accuracy 21.0%. These are diagnostic token metrics, not task success rates.

Both checkpoints failed the fixed adjacent-goal, open-grid, forced-detour and long-corridor fixtures under the tested sampling seed. Both correctly returned no verified solution for an impossible fixture, which demonstrates bounded failure of the wrapper, not learned impossibility recognition.

The default model's feedback-path latency was p50 **0.221 s**, p95 **0.432 s**. Process peak RSS was about **354.5 MB**, including both trained/untrained model copies, loading and audit. This is not isolated inference or training memory. Timing used two CPU threads on shared cloud hardware; other processes were not controlled. CUDA memory was **unmeasured**, not zero. The FP32 weight file was **38,351,058 bytes**.

## Run the full suite

From `reasoning_harness/`, with the package installed:

```bash
python -m unittest discover -s tests -v
sys2 benchmark --run runs/maze4 --output runs/maze4/benchmark-v1 \
  --count 100 --sizes 4 5 6 --seeds 1000000 2000000 3000000 \
  --attempts 4 --temperature 0.8 --require-ready
```

This evaluates 900 mazes plus 100 teacher-forced examples and fixed stress fixtures. `--output` must be a new directory. The command saves all evidence before exiting **2** when a gate fails. Without `--require-ready`, zero exit status means the evaluation completed, not that the model passed.

The coverage is specific to maze planning:

| Dimension | Coverage |
|---|---|
| Correctness | Independent verifier simulation over all 48 2x2 layouts and 16,512 candidate paths; BFS length checked against exhaustive enumeration |
| Model quality | Greedy, single stochastic attempt, bounded feedback retries, optimal route rate, error categories |
| Controls | Same initial RNG/first candidate for no-feedback retries; same-architecture random initialization; random paths; exact BFS |
| Generalization | Exact training/test deduplication; multiple maze sizes and evaluation seeds; fixed detours/corridors; future runs also group by shortest-route length |
| Uncertainty | Wilson rate intervals, paired bootstrap feedback differences, per-size and per-seed reports |
| Validation loss | Held-out answer NLL, perplexity and unconstrained token accuracy; no grammar masking |
| Reliability | Export/reload, exact CPU resume, invalid inputs, context bounds, finite loss and bounded failure |
| Resources | Serialized size, latency p50/p95, moves generated, attempt counts, process peak RSS, CUDA peak allocation when available |
| Provenance | Weight/config/corpus SHA-256, raw cases and paths, software/runtime metadata; future runs also hash source files |

MMLU, GSM8K and multimodal benchmarks do not validate this maze-only action model. They require different tasks, model interfaces and training. No general intelligence or broad language capability is claimed.

## Completion criteria and outstanding work

The default engineering gates propose **80%** in-distribution success, **60%** on larger mazes, and a positive lower bound for the feedback-minus-resampling 95% interval, alongside size, verifier, leak prevention, bounded-failure and GPU measurement checks. These are explicit project targets, not published industry standards. Set any different targets before evaluating a new candidate; do not lower them after seeing test scores.

**Failed:** trained planning quality and measurable benefit from feedback. **Passed:** weight size for the default model, verifier audit, zero exact training overlap, BFS reference, checkpoint/export/inference functionality. **Unrun:** CUDA training, CUDA performance and a real Colab disconnect/resume.

Before a trained-model release:

1. Train the full-size model beyond the two-step smoke run. Use validation-only experiments to tune data/training; reserve fresh test seeds after inspecting these results.
2. Run the full benchmark on the intended GPU runtime and reach the predeclared quality gates. Re-run stress fixtures and inspect actual paths, not only aggregate loss.
3. Repeat training with independent initialization seeds. Current evaluation seeds measure problem/sampling variation, not training-run variation.
4. Verify GPU mixed-precision backward passes, checkpoint reload, UI and a real Colab runtime recovery. GPU inference alone is insufficient.
5. Record target-hardware training memory and throughput. Set deployment-specific latency/memory budgets before claiming performance readiness.

The benchmark's `ready` field covers its single-checkpoint gates only. It does not replace these release checks, multimodal support, or a review of a trained artifact. The present result is a working evaluation harness and a failed model candidate, not a complete reasoning model.
