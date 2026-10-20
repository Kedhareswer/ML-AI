"""Collect every arena benchmark report into one chart and one Markdown table."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/arena/results")
out_png = Path(sys.argv[2] if len(sys.argv) > 2 else "evidence/arena/benchmark-summary.png")
NAMES = {"sudoku": "Sudoku (real, Sudoku-Extreme)", "mini_sudoku": "Mini Sudoku", "tango": "Tango", "queens": "Queens",
         "zip": "Zip", "patches": "Patches", "wend": "Wend"}

rows = []
for kind, label in NAMES.items():
    rep = root / kind / "report.json"
    if not rep.exists():
        continue
    s = json.loads(rep.read_text())["summary"]
    rows.append((label, s.get("oneshot:trained", {}).get("rate"), s["neural:trained"]["rate"], s["neural:untrained"]["rate"],
                 s["search:trained"]["rate"], s["exact:none"]["rate"],
                 s["search:trained"]["mean_expanded_states"], s["search:untrained"]["mean_expanded_states"],
                 s["neural:trained"]["total"]))
chess = json.loads((root / "chess" / "puzzles-report.json").read_text())["sets"] if (root / "chess" / "puzzles-report.json").exists() else {}
for name, label in (("test", "Chess (held-out Lichess)"), ("daily", "Chess (Lichess daily)")):
    if name in chess:
        c = chess[name]
        rows.append((label, None, c["neural"]["rate"], c["neural_untrained"]["rate"], c["search"]["rate"], c["classic"]["rate"],
                     None, None, c["neural"]["total"]))

lines = ["| Game | Cases | Network only | Untrained network (control) | Network + search | Classical solver | Search states: trained / untrained ordering |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for label, one, neu, unt, srch, exact, st, su, n in rows:
    states = f"{st} / {su}" if st is not None else "—"
    lines.append(f"| {label} | {n} | {neu}% | {unt}% | {srch}% | {exact}% | {states} |")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
labels = [r[0].replace(" (", "\n(") for r in rows]
x = range(len(rows))
w = 0.27
ax.bar([i - w for i in x], [r[3] for r in rows], w, label="Untrained network (control)", color="#c9c4bb")
ax.bar(list(x), [r[2] for r in rows], w, label="Trained network only", color="#2f6fde")
ax.bar([i + w for i in x], [r[4] for r in rows], w, label="Trained network + search", color="#1f8a4c")
ax.set_ylabel("Held-out puzzles solved (%)")
ax.set_ylim(0, 105)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 1.13), ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png)
print("wrote", out_png)
