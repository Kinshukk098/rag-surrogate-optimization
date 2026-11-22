
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("data/processed/experiments/search_results")

INPUT = BASE / "search_strategy_detailed_comparison.csv"
OUTPUT = BASE / "search_best_quality_vs_budget.png"

df = pd.read_csv(INPUT)

# Focus on answer F1 objective.
df = df[df["target"] == "mean_answer_f1"].copy()

strategies = sorted(df["strategy"].unique())

plt.figure(figsize=(10, 6))

for strategy in strategies:
    group = df[df["strategy"] == strategy].sort_values("selection_rank")

    # Best actual F1 discovered up to each evaluation.
    best_so_far = group["actual_value"].cummax()

    evaluations = group["selection_rank"]

    plt.plot(
        evaluations,
        best_so_far,
        marker="o",
        label=strategy
    )

plt.xlabel("Actual RAG Evaluations per Strategy")
plt.ylabel("Best Actual Answer F1 Found")
plt.title("Best Quality Found vs Actual RAG Evaluation Budget")

plt.xticks([1, 2])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=300)
plt.close()

print("SEARCH PROGRESS PLOT GENERATED")
print(f"Saved to: {OUTPUT}")
