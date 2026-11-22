
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("data/processed/experiments/search_results")

INPUT_PATH = BASE / "search_strategy_summary_top5.csv"
OUTPUT_DIR = BASE / "plots" / "top5_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

objectives = {
    "mean_answer_f1": ("Answer F1", "Higher is better"),
    "mean_answer_em": ("Answer EM", "Higher is better"),
    "mean_latency": ("Latency (seconds)", "Lower is better"),
    "mean_total_tokens": ("Total Tokens", "Lower is better"),
}

for target, (metric_name, direction_text) in objectives.items():

    subset = df[df["target"] == target].copy()

    subset = subset.sort_values("strategy")

    strategies = subset["strategy"].str.replace(
        "_guided", "", regex=False
    )

    x = range(len(subset))
    width = 0.25

    plt.figure(figsize=(13, 6))

    plt.bar(
        [i - width for i in x],
        subset["rank1_actual"],
        width,
        label="Rank-1 actual"
    )

    plt.bar(
        list(x),
        subset["best_actual_of_5"],
        width,
        label="Best actual of Top-5"
    )

    plt.bar(
        [i + width for i in x],
        subset["average_actual_of_5"],
        width,
        label="Average actual of Top-5"
    )

    plt.xticks(list(x), strategies, rotation=25, ha="right")
    plt.ylabel(metric_name)
    plt.xlabel("Search Strategy")
    plt.title(f"Top-1 vs Top-5: {metric_name}\n{direction_text}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filename = target.replace("mean_", "")
    plt.savefig(
        OUTPUT_DIR / f"{filename}_top1_vs_top5.png",
        dpi=300
    )

    plt.close()

print("TOP-1 VS TOP-5 PLOTS GENERATED")
print(f"Saved in: {OUTPUT_DIR}")
print("Total plots: 4")
