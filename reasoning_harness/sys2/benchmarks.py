"""Versioned planning benchmark: quality, controls, robustness and resources."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import torch

from .engine import batch, load, solve
from .model import Planner
from .tasks import MOVES, Maze, example, generate, verify

VERSION = "maze-planning-v1"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def wilson(successes, total):
    if not total:
        return [None, None]
    p, z = successes/total, 1.959963984540054
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]

def percentile(values, p):
    ordered = sorted(values)
    point = (len(ordered)-1)*p
    low = int(point)
    return ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(point-low)

def paired_difference(a, b, repetitions=2000):
    differences = [int(x)-int(y) for x, y in zip(a, b)]
    rng = random.Random(731)
    samples = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences) for _ in range(repetitions))
    return {"difference": statistics.mean(differences), "bootstrap_95_ci": [percentile(samples, .025), percentile(samples, .975)],
            "discordant_wins": sum(x == 1 for x in differences), "discordant_losses": sum(x == -1 for x in differences)}

def independent_valid(rows, path):
    """Audit oracle: separate complex-coordinate simulation, no task helpers."""
    cells = {complex(c, r): char for r, row in enumerate(rows) for c, char in enumerate(row)}
    position = next(p for p, char in cells.items() if char == "S")
    for char in path:
        delta = {"U": -1j, "D": 1j, "L": -1, "R": 1}.get(char)
        if delta is None:
            return False
        position += delta
        if cells.get(position, "X") == "X":
            return False
    return cells[position] == "G"

def audit_verifier():
    checked, mazes = 0, 0
    # Exhaust all 2x2 layouts and every path of length 0..4, plus malformed moves.
    paths = ["".join(p) for n in range(5) for p in itertools.product("UDLR", repeat=n)] + ["?", "D?", "\n"]
    for start, goal in itertools.permutations(range(4), 2):
        other = [i for i in range(4) if i not in (start, goal)]
        for walls in itertools.product(".X", repeat=2):
            cells = ["."]*4
            cells[start], cells[goal] = "S", "G"
            for i, char in zip(other, walls):
                cells[i] = char
            maze = Maze(("".join(cells[:2]), "".join(cells[2:])))
            valid_lengths = []
            for path in paths:
                expected = independent_valid(maze.rows, path)
                actual = verify(maze, path)
                if actual["valid"] != expected:
                    raise AssertionError(f"Verifier disagrees with independent oracle: {maze.rows}, {path}")
                if expected:
                    valid_lengths.append(len(path))
                checked += 1
            target = maze.shortest()
            expected_length = min(valid_lengths) if valid_lengths else None
            if (len(target) if target is not None else None) != expected_length:
                raise AssertionError("BFS oracle length disagrees with exhaustive path enumeration")
            mazes += 1
    return {"passed": True, "layouts": mazes, "path_checks": checked}

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def timed_solve(model, maze, attempts, temperature, seed, feedback=True):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    synchronize(device)
    start = time.perf_counter()
    result = solve(model, maze, attempts, temperature, use_feedback=feedback)
    synchronize(device)
    result["seconds"] = time.perf_counter()-start
    result["generated_moves"] = sum(len(item["path"]) for item in result["history"])
    return result

METHODS = ("greedy", "feedback", "no_feedback", "untrained", "random", "bfs")

def stress_suite(model, forbidden=(), attempts=4, temperature=.8):
    if hasattr(model,"search"):
        forbidden = {s.replace("S",".") for s in forbidden}
    fixtures = {
        "adjacent_goal": ("SG..", "....", "....", "...."),
        "open_diagonal": ("S...", "....", "....", "...G"),
        "forced_detour": ("S...", "XXX.", "....", "GXXX"),
        "long_corridor_8x8": ("S.......", "XXXXXXX.", "........", ".XXXXXXX", "........", "XXXXXXX.", "........", "GXXXXXXX"),
        "impossible": ("SXX", "XXX", "XXG"),
    }
    records = []
    for name, rows in fixtures.items():
        maze = Maze(rows)
        target = maze.shortest()
        result = {"name": name, "rows": list(rows), "reachable": target is not None,
                  "shortest_moves": len(target) if target is not None else None,
                  "training_overlap": ("\n".join(rows).replace("S",".") if hasattr(model,"search") else "\n".join(rows)) in forbidden}
        for method, budget, temp, feedback in [("greedy",1,0,True), ("feedback",attempts,temperature,True),
                                                ("no_feedback",attempts,temperature,False)]:
            result[method] = timed_solve(model, maze, budget, temp, 321, feedback)
        records.append(result)
    return records

@torch.inference_mode()
def validation_loss(model, run, count=100, seed=4_000_000):
    """Answer-only held-out NLL/accuracy, without generation grammar masking."""
    manifest = json.loads((Path(run)/"manifest.json").read_text())
    if hasattr(model, "search"):
        from .policy import distances, features
        losses, correct, selected_seeds = [], 0, []
        forbidden = {s.replace("S",".") for s in manifest["train_mazes"]}
        candidate, skipped = seed, 0
        while len(losses) < count and candidate < seed+10000:
            maze = generate(candidate, manifest["training"]["size"])
            candidate += 1
            signature = "\n".join(maze.rows).replace("S",".")
            if signature in forbidden:
                skipped += 1
                continue
            forbidden.add(signature)
            selected_seeds.append(candidate-1)
            d = distances(maze)
            position = maze.locate("S")
            x, legal = features(maze, position)
            target = torch.tensor([float(d.get((position[0]+dr,position[1]+dc)) == d[position]-1) for dr,dc in MOVES.values()])
            target = target/target.sum()
            scores = model(x.to(next(model.parameters()).device)).cpu().masked_fill(~legal,-1e4)
            losses.append(float(-(target*torch.log_softmax(scores,-1)).sum()))
            correct += int(target[int(scores.argmax())] > 0)
        if len(losses) != count:
            raise ValueError("Insufficient unique held-out validation mazes")
        return {"count":count,"seed":seed,"maze_seeds":selected_seeds,"overlaps_skipped":skipped,"training_overlap":0,
                "decision_NLL":statistics.mean(losses),"optimal_next_move_accuracy":correct/count,
                "note":"Held-out expert decision accuracy, not complete-route success"}
    size = manifest["training"]["size"]
    forbidden = set(manifest["train_mazes"])
    samples, candidate = [], seed
    while len(samples) < count and candidate < seed+10000:
        signature = "\n".join(generate(candidate,size).rows)
        if signature not in forbidden:
            forbidden.add(signature)
            samples.append(example(candidate,size))
        candidate += 1
    if len(samples) != count:
        raise ValueError("Insufficient held-out validation examples")
    total_loss, correct, tokens = 0., 0, 0
    model.eval()
    for offset in range(0, len(samples), 8):
        x, y = batch(samples[offset:offset+8], next(model.parameters()).device, model.config.context)
        logits = model(x)
        total_loss += float(torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten(), ignore_index=-100, reduction="sum"))
        mask = y != -100
        tokens += int(mask.sum())
        correct += int(((logits.argmax(-1) == y) & mask).sum())
    loss = total_loss/tokens
    return {"count": count, "seed": seed, "supervised_tokens": tokens, "answer_NLL": loss,
            "answer_perplexity": math.exp(loss), "unconstrained_next_token_accuracy": correct/tokens,
            "note": "Teacher-forced answer-token metrics, not free-generation planning accuracy"}

def summarize(records):
    metrics = {}
    methods = METHODS + tuple(name for name in ("untrained_greedy","guided_search","untrained_guided_search") if records and name in records[0])
    for method in methods:
        values = [r[method] for r in records]
        successes = sum(v["solved"] for v in values)
        latencies = [v["seconds"] for v in values]
        errors = Counter(h["reason"] for v in values for h in v["history"] if not h["valid"])
        metrics[method] = {"solved": successes, "count": len(values), "success_rate": successes/len(values),
                           "success_wilson_95_ci": wilson(successes, len(values)),
                           "optimal_rate": sum(v["history"][-1]["optimal"] for v in values)/len(values),
                           "mean_attempts": statistics.mean(v["attempts_used"] for v in values),
                           "latency_p50_seconds": statistics.median(latencies),
                           "latency_p95_seconds": percentile(latencies, .95),
                           "mean_generated_moves": statistics.mean(v["generated_moves"] for v in values),
                           "mean_expanded_states": statistics.mean(v.get("expanded_states",0) for v in values),
                           "failures": dict(errors)}
    metrics["single_stochastic_attempt"] = {"success_rate": sum(r["feedback"]["history"][0]["valid"] for r in records)/len(records)}
    metrics["feedback_minus_resampling"] = paired_difference([r["feedback"]["solved"] for r in records], [r["no_feedback"]["solved"] for r in records])
    return metrics

def benchmark(run, output, count=100, sizes=(3,4,5), seeds=(1_000_000,2_000_000,3_000_000),
              attempts=4, temperature=.8, device="auto", id_target=.8, ood_target=.6):
    if count < 1 or not seeds or not sizes or len(set(sizes)) != len(sizes) or len(set(seeds)) != len(seeds):
        raise ValueError("Positive count and nonempty unique sizes/seeds required")
    if not all(2 <= n <= 8 for n in sizes) or not 1 <= attempts <= 32 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Invalid size, attempt budget or temperature")
    if not 0 <= id_target <= 1 or not 0 <= ood_target <= 1:
        raise ValueError("Targets must be rates between zero and one")
    output, run = Path(output), Path(run)
    if output.exists():
        raise ValueError("Benchmark output directory must be new")
    manifest = json.loads((run/"manifest.json").read_text())
    train_size = manifest["training"]["size"]
    model = load(run, device)
    device = next(model.parameters()).device
    torch.manual_seed(0)
    untrained = type(model)(model.config).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True)
    audit = audit_verifier()
    identity = (lambda rows: "\n".join(rows).replace("S",".")) if hasattr(model,"search") else (lambda rows: "\n".join(rows))
    training_keys = {s.replace("S",".") if hasattr(model,"search") else s for s in manifest["train_mazes"]}
    seen = set(training_keys)
    records, cases = [], []
    begin = time.perf_counter()
    # Warm up kernels; these calls are excluded from latency measurements.
    warmup = generate(987654321, train_size)
    for net in (model, untrained):
        net.generate(warmup.prompt(), max_moves=train_size**2, temperature=0)
    with (output/"records.jsonl").open("w") as raw:
        for size in sizes:
            for seed in seeds:
                added, candidate = 0, seed
                while added < count and candidate < seed+max(10000, count*100):
                    maze = generate(candidate, size)
                    candidate += 1
                    signature = identity(maze.rows)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    sample_seed = (seed*1_000_003 + added) % (2**63)
                    case = {"size": size, "split": "in_distribution" if size == train_size else "different_size",
                            "cohort_seed": seed, "maze_seed": candidate-1, "sampling_seed": sample_seed, "rows": list(maze.rows),
                            "task_key":signature}
                    row = dict(case)
                    row["greedy"] = timed_solve(model, maze, 1, 0, sample_seed)
                    row["feedback"] = timed_solve(model, maze, attempts, temperature, sample_seed)
                    row["no_feedback"] = timed_solve(model, maze, attempts, temperature, sample_seed, feedback=False)
                    if row["feedback"]["history"][0] != row["no_feedback"]["history"][0]:
                        raise AssertionError("Unmatched first candidate in feedback control")
                    row["untrained"] = timed_solve(untrained, maze, attempts, temperature, sample_seed, feedback=False)
                    if hasattr(model,"search"):
                        row["untrained_greedy"] = timed_solve(untrained,maze,1,0,sample_seed,feedback=False)
                        for key,net in [("guided_search",model),("untrained_guided_search",untrained)]:
                            synchronize(device)
                            started = time.perf_counter()
                            search = net.search(maze,budget=64)
                            synchronize(device)
                            search["seconds"] = time.perf_counter()-started
                            search["generated_moves"] = sum(len(h["path"]) for h in search["history"])
                            row[key] = search
                    start = time.perf_counter()
                    histories = []
                    rng = random.Random(sample_seed)
                    for _ in range(attempts):
                        path = "".join(rng.choices("UDLR", k=rng.randint(1, size*size)))
                        histories.append({"path": path, **verify(maze, path)})
                        if histories[-1]["valid"]:
                            break
                    row["random"] = {"solved": histories[-1]["valid"], "attempts_used": len(histories), "history": histories,
                                     "generated_moves": sum(len(h["path"]) for h in histories), "seconds": time.perf_counter()-start}
                    start = time.perf_counter()
                    path = maze.shortest()
                    row["bfs"] = {"solved": True, "attempts_used": 1, "history": [{"path": path, **verify(maze,path)}],
                                  "generated_moves": len(path), "seconds": time.perf_counter()-start}
                    records.append(row)
                    cases.append(case)
                    raw.write(json.dumps(row)+"\n")
                    raw.flush()
                    added += 1
                if added != count:
                    raise ValueError("Unique test space exhausted; choose larger mazes or fewer examples")
                print(json.dumps({"size": size, "seed": seed, "completed": len(records)}), flush=True)
    # Impossible tasks exercise bounded failure, not learned impossibility detection.
    impossible = [Maze(("SX", "XG")), Maze(("SXX", "XXX", "XXG"))]
    robustness = []
    for maze in impossible:
        result = timed_solve(model, maze, attempts, temperature, 17)
        robustness.append({"rows": list(maze.rows), "correctly_unsolved": not result["solved"], **result})
    stress = stress_suite(model, manifest["train_mazes"], attempts, temperature)
    held_out_loss = validation_loss(model, run)
    groups = {str(size): summarize([r for r in records if r["size"] == size]) for size in sizes}
    by_seed = {f"{size}:{seed}": summarize([r for r in records if r["size"] == size and r["cohort_seed"] == seed])
               for size in sizes for seed in seeds}
    by_difficulty = {}
    for name, low, high in [("short_1_to_2",1,2), ("medium_3_to_5",3,5), ("long_6_plus",6,1000)]:
        subset = [r for r in records if low <= len(r["bfs"]["history"][0]["path"]) <= high]
        if subset:
            by_difficulty[name] = summarize(subset)
    overall = summarize(records)
    in_dist = [r for r in records if r["size"] == train_size]
    ood = [r for r in records if r["size"] > train_size]
    feedback_ci = paired_difference([r["feedback"]["solved"] for r in in_dist], [r["no_feedback"]["solved"] for r in in_dist]) if in_dist else None
    gates = {
        "weight_size_30_to_100_MB": 30e6 <= (run/"weights.pt").stat().st_size <= 100e6,
        "verifier_exhaustive_audit": audit["passed"],
        "bfs_all_cases_valid_and_optimal": all(r["bfs"]["history"][-1]["optimal"] for r in records),
        "no_training_overlap": not (training_keys & {identity(c["rows"]) for c in cases}),
        "multiple_evaluation_seeds": len(seeds) >= 3,
        "id_quality_target": bool(in_dist) and sum(r["feedback"]["solved"] for r in in_dist)/len(in_dist) >= id_target,
        "larger_maze_quality_target": bool(ood) and sum(r["feedback"]["solved"] for r in ood)/len(ood) >= ood_target,
        "positive_feedback_effect_95_CI": feedback_ci is not None and feedback_ci["bootstrap_95_ci"][0] > 0,
        "bounded_failure_on_impossible_mazes": all(r["correctly_unsolved"] for r in robustness),
        "gpu_runtime_measured": device.type == "cuda",
    }
    local_solver_gates = None
    if hasattr(model,"search"):
        local_solver_gates = {"weight_size":gates["weight_size_30_to_100_MB"], "verifier_audit":audit["passed"],
                              "all_held_out_routes_verified":all(r["guided_search"]["solved"] for r in records),
                              "bounded_expansions":all(r["guided_search"]["expanded_states"] <= 64 for r in records),
                              "all_reachable_stress_solved":all(model.search(Maze(tuple(r["rows"])),64)["solved"] for r in stress if r["reachable"]),
                              "impossible_maze_rejected":not model.search(Maze(("SX","XG")),64)["solved"],
                              "no_training_overlap":gates["no_training_overlap"],
                              "neural_greedy_id_quality":bool(in_dist) and sum(r['greedy']['solved'] for r in in_dist)/len(in_dist)>=id_target,
                              "neural_greedy_larger_quality":bool(ood) and sum(r['greedy']['solved'] for r in ood)/len(ood)>=ood_target,
                              "learned_order_reduces_expansions":overall['guided_search']['mean_expanded_states'] < overall['untrained_guided_search']['mean_expanded_states']}
    (output/"cases.json").write_text(json.dumps(cases, indent=2))
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    except ImportError:
        peak = None
    report = {"benchmark_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "weights_sha256": sha256(run/"weights.pt"), "config_sha256": sha256(run/"config.json"),
              "case_sha256": sha256(output/"cases.json"), "model": model.info(),
              "task_identity":"walls_and_goal_all_start_positions_excluded" if hasattr(model,"search") else "exact_maze_with_start",
              "training": manifest["training"], "weights_file_bytes": (run/"weights.pt").stat().st_size,
              "settings": {"count_per_size_seed": count, "sizes": list(sizes), "seeds": list(seeds), "attempts": attempts, "temperature": temperature},
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(),
                          "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "cpu_threads": torch.get_num_threads(), "process_peak_RSS_bytes": peak,
                          "memory_scope": "Entire benchmark process including both models, loading and audit; not isolated model memory",
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                          "wall_seconds": time.perf_counter()-begin},
              "verifier_audit": audit, "count": len(records), "overall": overall, "by_size": groups, "by_seed": by_seed,
              "by_shortest_path_length": by_difficulty, "held_out_loss": held_out_loss, "stress": stress,
              "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              "robustness": robustness, "thresholds": {"id_success": id_target, "larger_maze_success": ood_target,
                  "note": "Provisional engineering acceptance targets, not external benchmark standards; set before evaluating a candidate"},
              "gates": gates, "ready": all(gates.values()),
              "local_hybrid_solver_gates":local_solver_gates,
              "local_hybrid_solver_ready":all(local_solver_gates.values()) if local_solver_gates else None,
              "limitations": ["Synthetic maze benchmark only; not a natural-language, multimodal or general reasoning benchmark",
                              "Evaluation seeds measure sampling/task variability, not independent training runs",
                              "Bootstrap intervals are exploratory, not corrected for multiple comparisons",
                              "Latency is sequential eager inference; hardware-specific and includes verifier work",
                              "Cloud CPU load is uncontrolled; measurements are not isolated performance guarantees",
                              "GPU inference measurement does not validate GPU mixed-precision training or remote Colab execution",
                              "No training result may be called complete solely because benchmark execution succeeded"]}
    (output/"report.json").write_text(json.dumps(report, indent=2))
    lines = [f"# {VERSION}", "", f"Original neural-feedback research gates: **{report['ready']}**. Cases: {len(records)}. Device: {device}.", "",
             "| Size | Greedy | Feedback retries | No-feedback retries | Untrained | Random | BFS |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for size, summary in groups.items():
        rates = [f"{summary[m]['success_rate']:.1%}" for m in METHODS]
        lines.append("| "+size+" | "+" | ".join(rates)+" |")
    if local_solver_gates is not None:
        lines += ["",f"Local hybrid solver ready: **{report['local_hybrid_solver_ready']}**.","",
                  "| Size | Policy-guided success | Untrained-guided success | Learned mean states | Untrained mean states |",
                  "|---|---:|---:|---:|---:|"]
        for size,summary in groups.items():
            trained,random_net=summary['guided_search'],summary['untrained_guided_search']
            lines.append(f"| {size} | {trained['success_rate']:.1%} | {random_net['success_rate']:.1%} | {trained['mean_expanded_states']:.2f} | {random_net['mean_expanded_states']:.2f} |")
        lines += ["","Search completeness is supplied by bounded backtracking. The network supplies ordering, not the completeness guarantee."]
    lines += ["", "Provisional gates:", ""] + [f"- {'PASS' if passed else 'FAIL / UNRUN'}: {name}" for name, passed in gates.items()]
    lines += ["", "See report.json for confidence intervals, latency, resource measurements, provenance and limitations. Raw paths and verdicts are in records.jsonl."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return report
