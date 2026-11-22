
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/experiments/surrogate_results/predicted_vs_actual_values.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/experiments/surrogate_results/plots/predicted_vs_actual"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Use the largest training budget for the main model comparison
df = df[df["budget"] == 320].copy()

models = sorted(df["model"].unique())

# Common axis limits for fair visual comparison
min_value = min(df["actual_f1"].min(), df["predicted_f1"].min())
max_value = max(df["actual_f1"].max(), df["predicted_f1"].max())

# Individual plots for all five models
for model in models:

    model_df = df[df["model"] == model]

    actual = model_df["actual_f1"].to_numpy()
    predicted = model_df["predicted_f1"].to_numpy()

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.75
    )

    # Perfect prediction reference line
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect Prediction"
    )

    plt.xlabel("Actual Answer F1")
    plt.ylabel("Predicted Answer F1")
    plt.title(f"Predicted vs Actual Quality: {model}")

    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)

    plt.text(
        0.05,
        0.95,
        f"RMSE = {rmse:.4f}\nR² = {r2:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"{model}_predicted_vs_actual.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# Combined comparison of all five models
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, model in enumerate(models):

    model_df = df[df["model"] == model]

    actual = model_df["actual_f1"].to_numpy()
    predicted = model_df["predicted_f1"].to_numpy()

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    axes[i].scatter(actual, predicted, alpha=0.7)

    axes[i].plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    axes[i].set_title(
        f"{model}\nRMSE={rmse:.4f}, R²={r2:.4f}"
    )

    axes[i].set_xlabel("Actual F1")
    axes[i].set_ylabel("Predicted F1")
    axes[i].set_xlim(min_value, max_value)
    axes[i].set_ylim(min_value, max_value)
    axes[i].grid(True)

# Hide unused sixth panel
axes[5].axis("off")

fig.suptitle(
    "Predicted vs Actual Quality - All Surrogate Models (Budget 320)",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "all_models_comparison.png",
    dpi=300
)
plt.close()

print("\nPredicted vs Actual plots generated successfully!")
print("Output directory:", OUTPUT_DIR)

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file.name)
