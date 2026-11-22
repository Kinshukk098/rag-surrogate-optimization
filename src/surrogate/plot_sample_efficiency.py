
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data/processed/experiments/surrogate_results/sample_efficiency_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/experiments/surrogate_results/plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

# Focus on answer quality (F1) for the main plots
quality_df = df[df["target"] == "mean_answer_f1"].copy()

models = quality_df["model"].unique()

# --------------------------------------------------
# PLOT 1: Prediction error vs training configurations
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    model_df = quality_df[quality_df["model"] == model]
    plt.plot(
        model_df["budget"],
        model_df["RMSE"],
        marker="o",
        label=model
    )

plt.xlabel("Number of Training Configurations")
plt.ylabel("RMSE")
plt.title("Prediction Error vs Training Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_rmse_vs_budget.png", dpi=300)
plt.close()


# --------------------------------------------------
# PLOT 2: MAE vs training configurations
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    model_df = quality_df[quality_df["model"] == model]
    plt.plot(
        model_df["budget"],
        model_df["MAE"],
        marker="o",
        label=model
    )

plt.xlabel("Number of Training Configurations")
plt.ylabel("MAE")
plt.title("MAE vs Training Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_mae_vs_budget.png", dpi=300)
plt.close()


# --------------------------------------------------
# PLOT 3: Spearman correlation vs training configurations
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    model_df = quality_df[quality_df["model"] == model]
    plt.plot(
        model_df["budget"],
        model_df["Spearman"],
        marker="o",
        label=model
    )

plt.xlabel("Number of Training Configurations")
plt.ylabel("Spearman Rank Correlation")
plt.title("Ranking Performance vs Training Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_spearman_vs_budget.png", dpi=300)
plt.close()


# --------------------------------------------------
# PLOT 4: Top-K overlap vs training configurations
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    model_df = quality_df[quality_df["model"] == model]
    plt.plot(
        model_df["budget"],
        model_df["TopK_overlap"],
        marker="o",
        label=model
    )

plt.xlabel("Number of Training Configurations")
plt.ylabel("Top-K Overlap")
plt.title("Top-K Ranking Overlap vs Training Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_topk_overlap_vs_budget.png", dpi=300)
plt.close()


# --------------------------------------------------
# PLOT 5: Regret vs training configurations
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    model_df = quality_df[quality_df["model"] == model]
    plt.plot(
        model_df["budget"],
        model_df["Regret"],
        marker="o",
        label=model
    )

plt.xlabel("Number of Training Configurations")
plt.ylabel("Regret (Lower is Better)")
plt.title("Best-Configuration Selection Regret vs Training Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_regret_vs_budget.png", dpi=300)
plt.close()


print("\nAll available sample-efficiency plots generated!")
print("Plots saved in:", OUTPUT_DIR)

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file.name)
