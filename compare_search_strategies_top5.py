
import pandas as pd
from pathlib import Path

BASE = Path("data/processed/experiments/search_results")

SELECTIONS_PATH = BASE / "selected_search_configurations_top5.csv"
RESULTS_PATH = BASE / "hotpotqa_search_config_results_top5.csv"

DETAIL_PATH = BASE / "search_strategy_detailed_comparison_top5.csv"
SUMMARY_PATH = BASE / "search_strategy_summary_top5.csv"

selections = pd.read_csv(SELECTIONS_PATH)
results = pd.read_csv(RESULTS_PATH)

# Map research objectives to actual measured metrics.
metric_map = {
    "mean_answer_f1": "answer_f1_mean",
    "mean_answer_em": "answer_em_mean",
    "mean_latency": "total_latency_seconds_mean",
    "mean_total_tokens": "total_tokens_mean",
}

# Check required columns.
for col in metric_map.values():
    if col not in results.columns:
        raise ValueError(
            f"Missing metric column: {col}\n"
            f"Available columns: {results.columns.tolist()}"
        )

# Check that each strategy selected exactly 5 configurations per objective.
counts = selections.groupby(["target", "strategy"]).size()

if not (counts == 5).all():
    raise ValueError(
        f"Expected exactly 5 selections per strategy/objective:\n"
        f"{counts[counts != 5]}"
    )

# Join each selected configuration with its actual RAG result.
merged = selections.merge(
    results[["config_id", *metric_map.values()]],
    on="config_id",
    how="left",
    validate="many_to_one"
)

if merged[list(metric_map.values())].isna().any().any():
    missing = merged[merged[list(metric_map.values())].isna().any(axis=1)]
    raise ValueError(
        f"Some selected configurations have no actual results:\n"
        f"{missing[['strategy', 'target', 'config_id']]}"
    )

merged["actual_value"] = merged.apply(
    lambda row: row[metric_map[row["target"]]],
    axis=1
)

# Sort selections and save detailed results.
merged = merged.sort_values(
    ["target", "strategy", "selection_rank"]
)

merged.to_csv(DETAIL_PATH, index=False)

# Summarize each strategy for each objective.
summary_rows = []

for (target, strategy), group in merged.groupby(["target", "strategy"]):

    group = group.sort_values("selection_rank")
    direction = group["direction"].iloc[0]

    values = group["actual_value"].tolist()

    if direction == "max":
        best_value = max(values)
    else:
        best_value = min(values)

    row = {
        "target": target,
        "direction": direction,
        "strategy": strategy,
    }

    # Actual performance of individual predicted ranks.
    for rank, value in enumerate(values, start=1):
        row[f"rank{rank}_actual"] = value

    # Performance across all five selected configurations.
    row["best_actual_of_5"] = best_value
    row["average_actual_of_5"] = sum(values) / len(values)

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

# Rank strategies separately for each objective.
# Rank 1 means the best actual performance.
summary["strategy_rank"] = 0

for target in summary["target"].unique():

    mask = summary["target"] == target
    direction = summary.loc[mask, "direction"].iloc[0]

    summary.loc[mask, "strategy_rank"] = (
        summary.loc[mask, "best_actual_of_5"]
        .rank(
            ascending=(direction == "min"),
            method="min"
        )
    )

summary = summary.sort_values(
    ["target", "strategy_rank"]
)

summary.to_csv(SUMMARY_PATH, index=False)

print("\nTOP-5 SEARCH STRATEGY COMPARISON COMPLETE")
print(f"Selection entries compared: {len(merged)}")
print(f"Unique configurations evaluated: {merged['config_id'].nunique()}")
print(f"Strategies: {merged['strategy'].nunique()}")
print(f"Objectives: {merged['target'].nunique()}")

print("\nSUMMARY:")
print(summary.to_string(index=False))

print(f"\nDetailed results: {DETAIL_PATH}")
print(f"Summary results: {SUMMARY_PATH}")
