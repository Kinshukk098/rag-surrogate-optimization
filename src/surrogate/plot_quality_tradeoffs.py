
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/experiments/hotpotqa_config_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/experiments/surrogate_results/plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Keep only valid measured configurations
df = df.dropna(
    subset=["mean_answer_f1", "mean_latency", "mean_total_tokens"]
)

# Separate training and test configurations visually
colors = df["split"].map({
    "train": "tab:blue",
    "test": "tab:orange"
})

# --------------------------------------------------
# PLOT 1: Quality vs Latency
# --------------------------------------------------

plt.figure(figsize=(9, 6))

for split in df["split"].unique():
    subset = df[df["split"] == split]

    plt.scatter(
        subset["mean_latency"],
        subset["mean_answer_f1"],
        label=split,
        alpha=0.75
    )

plt.xlabel("Mean RAG Latency")
plt.ylabel("Mean Answer F1")
plt.title("RAG Quality vs Latency")
plt.legend(title="Configuration Split")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_quality_vs_latency.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# PLOT 2: Quality vs Token Usage
# --------------------------------------------------

plt.figure(figsize=(9, 6))

for split in df["split"].unique():
    subset = df[df["split"] == split]

    plt.scatter(
        subset["mean_total_tokens"],
        subset["mean_answer_f1"],
        label=split,
        alpha=0.75
    )

plt.xlabel("Mean Total Tokens")
plt.ylabel("Mean Answer F1")
plt.title("RAG Quality vs Token Usage")
plt.legend(title="Configuration Split")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_quality_vs_token_usage.png",
    dpi=300
)
plt.close()


print("\nTrade-off plots generated successfully!")

print("Configurations plotted:", len(df))
print("Output directory:", OUTPUT_DIR)

print("\nGenerated:")
print("06_quality_vs_latency.png")
print("07_quality_vs_token_usage.png")
