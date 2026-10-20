"""Held-out benchmark for one puzzle network, with untrained and classical controls."""
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .nets import load_net
from .registry import from_json
from .search import solve, untrained_like

MODES = (
    ("oneshot", "trained", "One network pass, no rules"),
    ("neural", "trained", "Network picks every move, rules block illegal moves, no backtracking"),
    ("neural", "untrained", "Same, random-initialised network (control)"),
    ("search", "trained", "Backtracking ordered by the trained network"),
    ("search", "untrained", "Backtracking ordered by a random network (control)"),
    ("exact", None, "Classical backtracking, no network (baseline)"),
)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0, c - h), 2), round(100 * min(1, c + h), 2)]


def run_bench(run, rows, output, budget=2000, threads=2, exact_budget=200_000):
    torch.set_num_threads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    net, extra = load_net(Path(run) / "weights.pt")
    nets = {"trained": net, "untrained": untrained_like(net, seed=1234), None: None}
    records = []
    with open(output / "records.jsonl", "w") as f:
        for i, row in enumerate(rows):
            puzzle = from_json(row)
            rec = {"index": i, "kind": row["kind"], "n": puzzle.n, "rating": row.get("rating"),
                   "source": row.get("source"), "results": {}}
            for mode, which, _ in MODES:
                if mode == "oneshot" and not hasattr(puzzle, "oneshot"):
                    continue
                key = f"{mode}:{which or 'none'}"
                res = solve(puzzle, nets[which], mode, exact_budget if mode == "exact" else budget)
                res["matches_reference"] = res["answer"] == row["solution"] if res["answer"] is not None else False
                res.pop("trail", None)
                rec["results"][key] = res
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)

    def summarise(recs):
        out = {}
        for mode, which, desc in MODES:
            key = f"{mode}:{which or 'none'}"
            rs = [r["results"][key] for r in recs if key in r["results"]]
            if not rs:
                continue
            k = sum(r["solved"] for r in rs)
            ms = np.array([r["ms"] for r in rs])
            out[key] = {"description": desc, "solved": k, "total": len(rs), "rate": round(100 * k / len(rs), 2),
                        "wilson95": wilson(k, len(rs)),
                        "mean_expanded_states": round(float(np.mean([r["expanded_states"] for r in rs])), 2),
                        "budget_exhausted": sum(r["budget_exhausted"] for r in rs),
                        "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2)}
        return out

    groups = {}
    for r in records:
        label = f"{r['n']}x{r['n']}"
        if r["rating"] is not None:
            label = "rating 0 (easy)" if r["rating"] == 0 else ("rating 1-20" if r["rating"] <= 20 else "rating >20 (hard)")
        groups.setdefault(label, []).append(r)
    report = {
        "kind": rows[0]["kind"], "cases": len(records), "run": str(run), "model": extra,
        "parameters": net.parameters_count(), "search_budget_states": budget, "exact_budget_states": exact_budget,
        "summary": summarise(records), "by_group": {g: summarise(rs) for g, rs in sorted(groups.items())},
        "environment": {"torch": torch.__version__, "python": platform.python_version(), "threads": threads,
                        "device": "cpu", "date": time.strftime("%Y-%m-%d")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    lines = [f"# {report['kind']} held-out benchmark", "",
             f"{len(records)} puzzles never seen in training. Every answer is checked by an independent rule verifier.", "",
             "| Method | Solved | 95% CI | Mean states | p50 ms | p95 ms |", "|---|---:|---:|---:|---:|---:|"]
    for key, s in report["summary"].items():
        lines.append(f"| {s['description']} | {s['solved']}/{s['total']} ({s['rate']}%) | {s['wilson95'][0]}–{s['wilson95'][1]}% "
                     f"| {s['mean_expanded_states']} | {s['p50_ms']} | {s['p95_ms']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
