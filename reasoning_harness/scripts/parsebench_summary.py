"""Summarise a ParseBench run the way the official leaderboard does, next to local open-source parsers.

    python scripts/parsebench_summary.py output/docparse [leaderboard.csv]

Headline metric per dimension (parse_bench.analysis.aggregation_report._DEFAULT_METRICS);
Overall is the mean of the five dimensions.
"""
import csv
import json
import sys
from pathlib import Path

HEADLINE = {"table": ("Tables", "grits_trm_composite"), "chart": ("Charts", "rule_pass_rate"),
            "text_content": ("Content_Faithfulness", "content_faithfulness"),
            "text_formatting": ("Semantic_Formatting", "semantic_formatting"),
            "layout": ("Visual_Grounding", "layout_element_rule_pass_rate")}


def summarise(run_dir):
    run_dir = Path(run_dir)
    out = {}
    for cat, (col, metric) in HEADLINE.items():
        report = run_dir / cat / "_evaluation_report.json"
        if report.exists():
            d = json.loads(report.read_text())
            m = d["aggregate_metrics"]
            out[col] = {"score": round(100 * m.get(f"avg_{metric}", m.get("avg_rule_pass_rate", 0.0)), 2),
                        "documents": d["total_examples"], "failed": d.get("failed", 0)}
    if len(out) == 5:
        out["Overall"] = round(sum(v["score"] for v in out.values()) / 5, 2)
    return out


def main():
    res = summarise(sys.argv[1])
    print(json.dumps(res, indent=2))
    if len(sys.argv) > 2:
        cols = [c for c, _ in HEADLINE.values()]
        print("\n| Parser | Overall | " + " | ".join(cols) + " |\n|---|" + "---:|" * 6)
        print(f"| **docparse (this run)** | **{res.get('Overall', '-')}** | "
              + " | ".join(str(res[c]["score"]) if c in res else "-" for c in cols) + " |")
        for row in csv.DictReader(open(sys.argv[2])):
            if row["Category"] == "Open Source - Local":
                print(f"| {row['Provider']} | {row['Overall']} | " + " | ".join(row[c] for c in cols) + " |")


if __name__ == "__main__":
    main()
