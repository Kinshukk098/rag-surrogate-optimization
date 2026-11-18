
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    create_preprocessor,
    load_data,
)

from .models import create_models
from .evaluation import evaluate_regression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "research_config.json"


def load_experiment_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config["surrogate_experiment"]


def create_nested_subsets(train_df, budgets, random_state=42):

    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(len(train_df))

    subsets = {}

    for budget in budgets:

        if budget > len(train_df):
            raise ValueError(
                f"Budget {budget} exceeds training data size."
            )

        indices = permutation[:budget]

        subsets[budget] = (
            train_df.iloc[indices]
            .copy()
            .reset_index(drop=True)
        )

    return subsets


def run_experiments():

    experiment_config = load_experiment_config()

    random_state = experiment_config["random_seed"]
    budgets = experiment_config["training_budgets"]

    target_directions = experiment_config["targets"]
    ranking_config = experiment_config["ranking"]

    actual_top_k = ranking_config["actual_top_k"]
    predicted_top_k = ranking_config["predicted_top_k"]

    data_path = (
        PROJECT_ROOT
        / "data/processed/experiments/hotpotqa_config_results.csv"
    )

    output_dir = (
        PROJECT_ROOT
        / "data/processed/experiments/surrogate_results"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fixed train and test configurations
    train_df, test_df = load_data(data_path)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\n========== DATA SUMMARY ==========")
    print("Training configurations:", len(train_df))
    print("Test configurations:", len(test_df))
    print("Training budgets:", budgets)

    # Generate nested training subsets
    subsets = create_nested_subsets(
        train_df,
        budgets,
        random_state,
    )

    # Initialize five surrogate models
    model_templates = create_models(
        random_state=random_state,
        model_config=experiment_config["models"],
    )

    results = []

    total_runs = (
        len(budgets)
        * len(model_templates)
        * len(TARGET_COLUMNS)
    )

    run_number = 0

    # Sample-efficiency experiment
    for budget in budgets:

        current_train = subsets[budget]

        X_train = current_train[FEATURE_COLUMNS]
        X_test = test_df[FEATURE_COLUMNS]

        print(f"\n========== BUDGET: {budget} ==========")

        for model_name, model_template in model_templates.items():

            for target in TARGET_COLUMNS:

                run_number += 1

                print(
                    f"\n[{run_number}/{total_runs}] "
                    f"Model: {model_name} | Target: {target}"
                )

                y_train = current_train[target].to_numpy()
                y_test = test_df[target].to_numpy()

                # Preprocessing + surrogate model
                pipeline = Pipeline([
                    ("preprocessor", create_preprocessor()),
                    ("model", clone(model_template)),
                ])

                # Train directly using the current budget
                pipeline.fit(X_train, y_train)

                # Predict on the fixed unseen test set
                predictions = np.asarray(
                    pipeline.predict(X_test)
                ).ravel()

                # Evaluate regression and ranking performance
                metrics = evaluate_regression(
                    y_true=y_test,
                    y_pred=predictions,
                    target_name=target,
                    optimize=target_directions[target],
                    actual_top_k=actual_top_k,
                    predicted_top_k=predicted_top_k,
                )

                # Identify actual and predicted best configurations
                if target_directions[target] == "max":

                    actual_best_index = np.argmax(y_test)
                    predicted_best_index = np.argmax(predictions)

                else:

                    actual_best_index = np.argmin(y_test)
                    predicted_best_index = np.argmin(predictions)

                actual_best_config_id = test_df.iloc[
                    actual_best_index
                ]["config_id"]

                predicted_best_config_id = test_df.iloc[
                    predicted_best_index
                ]["config_id"]

                actual_best_value = y_test[actual_best_index]

                actual_value_at_predicted_best = (
                    y_test[predicted_best_index]
                )

                # Store experiment result
                record = {
                    "budget": budget,
                    "model": model_name,
                    "target": target,

                    **metrics,

                    "actual_best_config_id": actual_best_config_id,
                    "predicted_best_config_id": predicted_best_config_id,

                    "actual_best_value": actual_best_value,
                    "actual_value_at_predicted_best":
                        actual_value_at_predicted_best,
                }

                results.append(record)

                print(
                    f"RMSE: {metrics['RMSE']:.4f} | "
                    f"MAE: {metrics['MAE']:.4f} | "
                    f"R2: {metrics['R2']:.4f} | "
                    f"Spearman: {metrics['Spearman']:.4f} | "
                    f"Top-K: {metrics['TopK_overlap']:.4f} | "
                    f"Regret: {metrics['Regret']:.4f}"
                )

    # Save all results
    results_df = pd.DataFrame(results)

    output_path = (
        output_dir / "sample_efficiency_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print("Total experiment runs:", len(results_df))
    print("Expected experiment runs:", total_runs)
    print("Results saved to:", output_path)

    print("\nResults per budget:")
    print(results_df.groupby("budget").size())

    print("\nResults per model:")
    print(results_df.groupby("model").size())

    print("\nResults per target:")
    print(results_df.groupby("target").size())


if __name__ == "__main__":
    run_experiments()
