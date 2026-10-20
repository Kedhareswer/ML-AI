# Tiny System 2 policy v1

**Status:** ready for the tested local CPU maze-solving workflow, with explicit neural-only and hybrid-search modes. Not a general reasoning/chat model. Remote Colab GPU execution remains unvalidated.

## Artifact

- Architecture: four-hidden-layer MLP, width 1600, 207 inputs, four move logits.
- Parameters: **8,024,004**.
- Inference weights: **32,099,618 bytes** (32.10 decimal MB), unquantized FP32.
- Packaged files: [weights](artifacts/policy-v1/weights.pt), [configuration](artifacts/policy-v1/config.json), [training manifest](artifacts/policy-v1/manifest.json), [training log](artifacts/policy-v1/metrics.jsonl), [checksums](artifacts/policy-v1/checksums.json).
- No pretrained model, external dataset or API is used.

## Training

Seed 314; 1,500 optimizer steps; batch size 64; AdamW at learning rate 0.0005; weight decay 0.001; gradient clipping at 1.0; FP32; two CPU threads; PyTorch 2.6.0+cpu / Python 3.12.14. The original training-loop log reached 1,500 steps in approximately 89.5 seconds, excluding dataset construction and final artifact export. This is one observed shared-machine run, not a hardware guarantee.

The corpus comprises 3,000 synthetic reachable mazes of sizes 3x3 and 4x4 and 27,392 reachable decision states. Reverse BFS supplies shortest-path next-action labels offline, with equal probability for tied optimal actions. Input features are padded wall/current/goal planes, coordinates, neighbor legality and neighbor Manhattan distances to the goal. The input features and legality/no-revisit constraints are identical for trained and untrained baselines.

Two additional independent runs used seeds 315 and 316 with the same recipe. All three models were trained; the released model is the preselected primary seed 314.

## Inference and the source of guarantees

Neural-only inference follows model move scores while masking illegal/revisited cells. It can fail in a dead end. Stochastic retries resample the policy; this model does **not** read or learn textual failure feedback.

Hybrid mode uses learned priorities to order depth-first backtracking over legal states, expanding at most 64 distinct states. This can recover from an incorrect branch. State-space search supplies completeness on supported reachable grids within that budget. An untrained ordering also solves all benchmark cases with enough search, but visits more states. No BFS solution is passed to the policy or hybrid search at inference. The final verifier uses BFS to score optimality.

Neither rollout nor DFS guarantees shortest routes. BFS is the exact optimal baseline and is often more appropriate than a learned model for real maze solving.

## Final evaluation

The primary test comprises **1,200 unique wall-layout/goal combinations**, 300 each at 4x4, 5x5, 6x6 and 8x8, spanning three fresh evaluation seeds. All combinations appearing in training are excluded regardless of start position, since training supervises every reachable start. Two independent training repetitions contribute 270 final held-out tasks each.

| Maze | Neural only, greedy | Up to four neural rollouts | Neural + backtracking |
|---|---:|---:|---:|
| 4x4 | 99.3% | 99.7% | 100% |
| 5x5 | 90.3% | 96.0% | 100% |
| 6x6 | 88.3% | 94.7% | 100% |
| 8x8 | 84.0% | 92.3% | 100% |

Across the three independent training runs, all **1,740/1,740** final hybrid routes were verified. The primary hybrid solved 1,200/1,200, with a Wilson 95% interval of 99.68–100%. This finite evaluation is not a universal success guarantee. The primary neural-only aggregate was 90.5%, and the primary guided routes were shortest on 86.2% of tasks. Exact BFS was shortest on all tasks.

The primary held-out expert decision test achieved 97/100 optimal next moves. Greedy improvement over untrained policy was positive on all four sizes, with paired-bootstrap intervals reported in the [complete results](benchmarks/POLICY_RESULTS.md).

Primary guided latency on this shared CPU: p50 9.6 ms, p95 31.8 ms. Peak process RSS was 340.9 MB, including two models, loading and audit; this is not isolated model/training memory. GPU resource use was not measured.

## Tests and demonstrations

Twelve tests passed, including independent verifier audit, supervised trainability, checkpoint resume equality, export/reload, search completeness/bounds, canonical holdout handling and actual HTTP inference. The complete notebook also ran locally with explicit tiny/CPU overrides. Actual [screenshots, screen recording and API responses](evidence/README.md) show the primary neural policy solving a forced detour and the hybrid solving an 8x8 corridor.

## Limits

- Synthetic square mazes of sizes 2–8 only; training used sizes 3–4. No natural-language or multimodal reasoning capability is claimed.
- A larger state space, different input representation or new task requires new data, tests and a measured budget.
- The model alone is imperfect, and the hybrid's search guarantee must be credited to the algorithm.
- Two easy stress fixtures overlap training layout/goal combinations and are marked in the raw report. They do not contribute to held-out accuracy.
- The original neural-feedback/GPU research readiness flag stays false. The applicable release status is `local_hybrid_solver_ready`, which all three final models pass.
- Real GPU training, GPU latency and Colab runtime deletion/recovery have not been tested. The supplied notebook supports those workflows without claiming validation.
- Source, model and evidence are saved locally in this checkout; no GitHub push or publication was performed.
