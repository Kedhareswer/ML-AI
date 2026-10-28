"""Draw docparse comparison charts from saved results; no hard-coded scores.

Inputs: benchmarks/parsebench/leaderboard.csv (ParseBench's published leaderboard),
benchmarks/parsebench/report.json and benchmarks/extractbench/report.json (our runs).
usage: python scripts/plot_docparse.py   -> evidence/docparse/*.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "docparse"
DIMS = ["Tables", "Charts", "Content_Faithfulness", "Semantic_Formatting", "Visual_Grounding"]
LABELS = ["Tables", "Charts", "Content\nfaithfulness", "Semantic\nformatting", "Visual\ngrounding"]
US, LOCAL, OTHER = "#e4572e", "#5b8bd6", "#b9c2cf"


def leaderboard():
    rows = list(csv.DictReader(open(ROOT / "benchmarks/parsebench/leaderboard.csv", encoding="utf-8")))
    ours = json.loads((ROOT / "benchmarks/parsebench/report.json").read_text())["final"]
    rows.append({"Provider": "docparse (ours)", "Category": "Open Source - Local", "Overall": ours["Overall"],
                 **{d: ours[d]["score"] for d in DIMS}})
    for r in rows:
        for k in ["Overall", *DIMS]:
            r[k] = float(r[k] or 0)
    return rows


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e9ef", zorder=0)


def overall_local(rows):
    """Every 'Open Source - Local' parser plus a few open-weight VLMs for scale."""
    pick = [r for r in rows if r["Category"] == "Open Source - Local"]
    vlm = sorted((r for r in rows if r["Category"] == "VLM - Open Weight"), key=lambda r: -r["Overall"])
    pick += vlm[:3] + [max(rows, key=lambda r: r["Overall"])]
    pick.sort(key=lambda r: r["Overall"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(pick) + 1.4))
    colors = [US if "ours" in r["Provider"] else LOCAL if r["Category"] == "Open Source - Local" else OTHER for r in pick]
    ax.barh([f'{r["Provider"]}' for r in pick], [r["Overall"] for r in pick], color=colors, zorder=3)
    for i, r in enumerate(pick):
        ax.text(r["Overall"] + 0.8, i, f'{r["Overall"]:.1f}', va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("ParseBench overall (mean of 5 dimensions, higher is better)")
    ax.set_title("ParseBench: docparse vs every local parser (blue);\ngrey = top 3 open-weight VLMs and the #1 system", fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "parsebench_overall.png", dpi=130)
    plt.close(fig)


def dimensions(rows):
    by = {r["Provider"]: r for r in rows}
    names = ["docparse (ours)", "LiteParse (no OCR)", "Warp Ingest", "PyMuPDF4LLM", "Nemotron-3.5-Parse",
             max(rows, key=lambda r: r["Overall"])["Provider"]]
    names = [n for n in names if n in by]
    colors = [US, "#7fa7e0", "#5b8bd6", "#2f5fa8", "#9aa3b0", "#5f6874"]
    x = np.arange(len(DIMS))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, n in enumerate(names):
        vals = [by[n][d] for d in DIMS]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=f'{n} ({by[n]["Overall"]:.1f})', color=colors[i], zorder=3)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("ParseBench by dimension (legend: overall score)", fontsize=11, loc="left")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e9ef", zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "parsebench_dimensions.png", dpi=130)
    plt.close(fig)


def size(rows):
    """Weights on disk: docparse vs open-weight document models (sizes read from the Hub)."""
    sizes = json.loads((ROOT / "benchmarks/parsebench/model_sizes.json").read_text())["models"]
    ours = json.loads((ROOT / "artifacts/docparse/manifest.json").read_text())["weights_bytes"]
    items = sorted([(v["weights_bytes"], n) for n, v in sizes.items()], reverse=True) + [(ours, "docparse (ours)")]
    gb = [b / 1e9 for b, _ in items]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    # linear axis on purpose: the bar lengths are the true size ratios, so 6 MB is a sliver
    ax.barh([n for _, n in items], gb, color=[US if "ours" in n else OTHER for _, n in items], zorder=3)
    for i, ((b, n), g) in enumerate(zip(items, gb)):
        label = f"{g:.1f} GB" if g >= 1 else f"{b / 1e6:.0f} MB"
        ours_bar = {"fontweight": "bold", "color": US} if "ours" in n else {}
        ax.text(g + 0.03, i, label, va="center", fontsize=9, **ours_bar)
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("Model weights on disk (GB, linear scale)")
    big = max(b for b, _ in items)
    ax.set_title(f"docparse is {ours / 1e6:.0f} MB: {big / ours:.0f}x smaller than the largest here, and runs on a CPU",
                 fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "model_size.png", dpi=130)
    plt.close(fig)


def extractbench():
    arms = json.loads((ROOT / "benchmarks/extractbench/report.json").read_text())["arms"]
    vals = [("pypdf text", arms["pypdf_extract_local"]["mean_value_f1"], LOCAL),
            ("docparse (ours)", arms["docparse_extract_local"]["mean_value_f1"], US)]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals], zorder=3, height=0.55)
    for i, (_, v, _) in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Value F1 (%), higher is better")
    ax.set_title("ExtractBench: which parser feeds the LLM better? (same small LLM, 6 test documents)",
                 fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "extractbench.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = leaderboard()
    overall_local(rows)
    dimensions(rows)
    size(rows)
    extractbench()
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
