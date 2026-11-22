
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("data/processed/experiments/surrogate_results")

INPUT = BASE / "sample_efficiency_results.csv"
OUTPUT = BASE / "plots/final_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Use all training budgets for the first three plots.
metrics = [
    ("RMSE", "Prediction Error (RMSE)", "01_final_rmse_comparison.png"),
    ("Spearman", "Ranking Performance (Spearman)", "02_final_spearman_comparison.png"),
    ("Regret", "Configuration Discovery Regret", "03_final_regret_comparison.png"),
]

for metric, ylabel, filename in metrics:
    plt.figure(figsize=(11, 6))

    for target in df["target"].unique():
        subset = df[df["target"] == target]

        summary = subset.groupby("budget")[metric].mean()

        plt.plot(
            summary.index,
            summary.values,
            marker="o",
            label=target
        )

    plt.xlabel("Number of Training Configurations")
    plt.ylabel(ylabel)
    plt.title(f"{metric} Across Training Budgets")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=300)
    plt.close()

# Detailed comparison at maximum budget.
max_budget = df["budget"].max()
final_df = df[df["budget"] == max_budget]

for metric, ylabel, filename in [
    ("RMSE", "RMSE", "04_model_rmse_at_320.png"),
    ("Spearman", "Spearman Correlation", "05_model_spearman_at_320.png"),
]:
    pivot = final_df.pivot(
        index="model",
        columns="target",
        values=metric
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_xlabel("Surrogate Model")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: Model Comparison at {max_budget} Configurations")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=300)
    plt.close()

print("FINAL SURROGATE COMPARISON PLOTS GENERATED")
print(f"Plots: {len(list(OUTPUT.glob('*.png')))}")
print(f"Saved in: {OUTPUT}")
