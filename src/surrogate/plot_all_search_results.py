
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

BASE = Path("data/processed/experiments/search_results")
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INPUT = BASE / "search_strategy_detailed_comparison.csv"

df = pd.read_csv(INPUT)

objectives = {
    "mean_answer_f1": {
        "label": "Answer F1",
        "direction": "max",
        "filename": "answer_f1"
    },
    "mean_answer_em": {
        "label": "Answer EM",
        "direction": "max",
        "filename": "answer_em"
    },
    "mean_latency": {
        "label": "Latency (seconds)",
        "direction": "min",
        "filename": "latency"
    },
    "mean_total_tokens": {
        "label": "Total Tokens",
        "direction": "min",
        "filename": "tokens"
    }
}

strategies = sorted(df["strategy"].unique())

# --------------------------------------------------
# 1. Best actual result vs evaluation budget
# Four separate plots: F1, EM, latency, tokens.
# --------------------------------------------------

for target, info in objectives.items():
    subset = df[df["target"] == target].copy()

    plt.figure(figsize=(11, 6))

    for strategy in strategies:
        group = subset[
            subset["strategy"] == strategy
        ].sort_values("selection_rank")

        values = group["actual_value"]

        if info["direction"] == "max":
            best_so_far = values.cummax()
        else:
            best_so_far = values.cummin()

        plt.plot(
            group["selection_rank"],
            best_so_far,
            marker="o",
            linewidth=2,
            label=strategy
        )

    plt.xlabel("Actual RAG Evaluations per Strategy")
    plt.ylabel(f"Best Actual {info['label']} Found")
    plt.title(f"Best {info['label']} vs Evaluation Budget")
    plt.xticks([1, 2])
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = PLOT_DIR / f"01_best_{info['filename']}_vs_budget.png"
    plt.savefig(path, dpi=300)
    plt.close()

# --------------------------------------------------
# 2. Best actual result among two selections
# Strategy comparison for each objective.
# --------------------------------------------------

for target, info in objectives.items():
    subset = df[df["target"] == target].copy()

    summary = (
        subset.groupby("strategy")["actual_value"]
        .agg(
            lambda x: (
                x.max() if info["direction"] == "max"
                else x.min()
            )
        )
        .reindex(strategies)
    )

    plt.figure(figsize=(11, 6))
    summary.plot(kind="bar")

    plt.xlabel("Search Strategy")
    plt.ylabel(f"Best Actual {info['label']}")
    plt.title(f"Search Strategy Comparison: {info['label']}")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = PLOT_DIR / f"02_strategy_comparison_{info['filename']}.png"
    plt.savefig(path, dpi=300)
    plt.close()

# --------------------------------------------------
# 3. Top-1 vs Top-2 actual performance.
# --------------------------------------------------

for target, info in objectives.items():
    subset = df[df["target"] == target].copy()

    pivot = subset.pivot(
        index="strategy",
        columns="selection_rank",
        values="actual_value"
    ).reindex(strategies)

    pivot = pivot.rename(columns={
        1: "Predicted Rank 1",
        2: "Predicted Rank 2"
    })

    plt.figure(figsize=(11, 6))
    pivot.plot(kind="bar", ax=plt.gca())

    plt.xlabel("Search Strategy")
    plt.ylabel(f"Actual {info['label']}")
    plt.title(f"Top-1 vs Top-2 Selection: {info['label']}")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = PLOT_DIR / f"03_top1_vs_top2_{info['filename']}.png"
    plt.savefig(path, dpi=300)
    plt.close()

# --------------------------------------------------
# 4. Quality vs latency for the 41 evaluated configs.
# --------------------------------------------------

unique_configs = df.drop_duplicates("config_id").copy()

plt.figure(figsize=(9, 6))
plt.scatter(
    unique_configs["total_latency_seconds_mean"],
    unique_configs["answer_f1_mean"],
    alpha=0.8
)

plt.xlabel("Actual Mean Latency (seconds)")
plt.ylabel("Actual Mean Answer F1")
plt.title("Search Experiment: Answer F1 vs Latency")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "04_search_quality_vs_latency.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 5. Quality vs token usage for the 41 evaluated configs.
# --------------------------------------------------

plt.figure(figsize=(9, 6))
plt.scatter(
    unique_configs["total_tokens_mean"],
    unique_configs["answer_f1_mean"],
    alpha=0.8
)

plt.xlabel("Actual Mean Total Tokens")
plt.ylabel("Actual Mean Answer F1")
plt.title("Search Experiment: Answer F1 vs Token Usage")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "05_search_quality_vs_tokens.png",
    dpi=300
)
plt.close()

print("\nALL SEARCH EXPERIMENT PLOTS GENERATED")
print(f"Unique evaluated configurations: {unique_configs['config_id'].nunique()}")
print(f"Strategies: {len(strategies)}")
print(f"Objectives: {len(objectives)}")
print(f"Total plots generated: {len(list(PLOT_DIR.glob('*.png')))}")
print(f"Plot folder: {PLOT_DIR}")
