
import pandas as pd
from pathlib import Path

INPUT = Path(
    "data/processed/experiments/surrogate_results/"
    "sample_efficiency_results.csv"
)

OUTPUT_DIR = Path(
    "data/processed/experiments/surrogate_results/"
)

df = pd.read_csv(INPUT)

# Use the maximum training budget.
max_budget = df["budget"].max()
final_df = df[df["budget"] == max_budget].copy()

# Define which direction is better for each metric.
metric_directions = {
    "RMSE": "min",
    "MAE": "min",
    "R2": "max",
    "Spearman": "max",
    "TopK_overlap": "max",
    "Regret": "min",
}

# Create a complete comparison table.
comparison = final_df[
    [
        "budget",
        "target",
        "model",
        "RMSE",
        "MAE",
        "R2",
        "Spearman",
        "TopK_overlap",
        "Regret",
        "actual_best_config_id",
        "predicted_best_config_id",
        "actual_best_value",
        "actual_value_at_predicted_best",
    ]
].copy()

comparison.to_csv(
    OUTPUT_DIR / "final_surrogate_comparison.csv",
    index=False
)

# Identify the winning model for every objective and metric.
winner_rows = []

for target in comparison["target"].unique():
    subset = comparison[comparison["target"] == target]

    for metric, direction in metric_directions.items():
        if direction == "min":
            best_value = subset[metric].min()
        else:
            best_value = subset[metric].max()

        winners = subset[subset[metric] == best_value]

        for _, winner in winners.iterrows():
            winner_rows.append({
                "budget": max_budget,
                "target": target,
                "metric": metric,
                "winning_model": winner["model"],
                "best_value": best_value,
            })

winners_df = pd.DataFrame(winner_rows)

winners_df.to_csv(
    OUTPUT_DIR / "final_surrogate_winners.csv",
    index=False
)

print("\nFINAL SURROGATE COMPARISON COMPLETE")
print(f"Maximum training budget: {max_budget}")
print(f"Models compared: {comparison['model'].nunique()}")
print(f"Objectives compared: {comparison['target'].nunique()}")

print("\nFULL COMPARISON:")
print(comparison.to_string(index=False))

print("\nWINNING MODELS:")
print(winners_df.to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_DIR / "final_surrogate_comparison.csv")
print(OUTPUT_DIR / "final_surrogate_winners.csv")
