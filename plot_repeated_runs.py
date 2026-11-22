
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("data/processed/experiments/repeated_runs")
INPUT = BASE / "repeated_run_summary.csv"
PLOT_DIR = BASE / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

metrics = [
    {
        "column": "answer_f1",
        "title": "Answer F1 Stability Across Five Runs",
        "ylabel": "Mean Answer F1",
        "filename": "01_f1_stability.png",
        "higher_is_better": True,
    },
    {
        "column": "answer_em",
        "title": "Answer EM Stability Across Five Runs",
        "ylabel": "Mean Answer EM",
        "filename": "02_em_stability.png",
        "higher_is_better": True,
    },
    {
        "column": "total_latency_seconds",
        "title": "Latency Stability Across Five Runs",
        "ylabel": "Mean Latency (seconds)",
        "filename": "03_latency_stability.png",
        "higher_is_better": False,
    },
    {
        "column": "total_tokens",
        "title": "Token Usage Stability Across Five Runs",
        "ylabel": "Mean Tokens",
        "filename": "04_token_stability.png",
        "higher_is_better": False,
    },
]

for item in metrics:

    metric = item["column"]
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Sort each graph according to its own objective.
    # Quality: highest first.
    # Latency and tokens: lowest first.
    plot_df = df.sort_values(
        mean_col,
        ascending=not item["higher_is_better"]
    ).reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    plt.bar(
        plot_df["config_id"],
        plot_df[mean_col],
        yerr=plot_df[std_col],
        capsize=6,
    )

    plt.title(item["title"], fontsize=13)
    plt.xlabel("Configuration ID")
    plt.ylabel(item["ylabel"])

    plt.xticks(rotation=25)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    output_file = PLOT_DIR / item["filename"]
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated: {output_file}")

print("\nREPEATED-RUN PLOTS GENERATED SUCCESSFULLY")
print("Saved in:", PLOT_DIR)
print("Total plots:", len(metrics))
