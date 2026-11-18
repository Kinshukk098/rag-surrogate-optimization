
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .preprocessing import (
    FEATURE_COLUMNS,
    create_preprocessor,
    load_data,
)
from .models import create_models
from .experiments import (
    load_experiment_config,
    create_nested_subsets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/experiments/hotpotqa_config_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/experiments/surrogate_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "mean_answer_f1"


def export_predictions():

    config = load_experiment_config()

    random_state = config["random_seed"]
    budgets = config["training_budgets"]

    train_df, test_df = load_data(DATA_PATH)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    subsets = create_nested_subsets(
        train_df,
        budgets,
        random_state,
    )

    models = create_models(
        random_state=random_state,
        model_config=config["models"],
    )

    all_predictions = []

    total_runs = len(budgets) * len(models)
    run_number = 0

    for budget in budgets:

        current_train = subsets[budget]

        X_train = current_train[FEATURE_COLUMNS]
        y_train = current_train[TARGET].to_numpy()

        X_test = test_df[FEATURE_COLUMNS]
        y_actual = test_df[TARGET].to_numpy()

        for model_name, model_template in models.items():

            run_number += 1

            print(
                f"[{run_number}/{total_runs}] "
                f"Budget: {budget} | Model: {model_name}"
            )

            pipeline = Pipeline([
                ("preprocessor", create_preprocessor()),
                ("model", clone(model_template)),
            ])

            pipeline.fit(X_train, y_train)

            y_predicted = np.asarray(
                pipeline.predict(X_test)
            ).ravel()

            for i in range(len(test_df)):

                all_predictions.append({
                    "budget": budget,
                    "model": model_name,
                    "config_id": test_df.iloc[i]["config_id"],
                    "actual_f1": y_actual[i],
                    "predicted_f1": y_predicted[i],
                })

    predictions_df = pd.DataFrame(all_predictions)

    output_path = (
        OUTPUT_DIR / "predicted_vs_actual_values.csv"
    )

    predictions_df.to_csv(output_path, index=False)

    print("\n========== EXPORT COMPLETED ==========")
    print("Total prediction rows:", len(predictions_df))
    print("Expected rows:", len(budgets) * len(models) * len(test_df))
    print("Saved to:", output_path)

    print("\nRows per budget:")
    print(predictions_df.groupby("budget").size())

    print("\nRows per model:")
    print(predictions_df.groupby("model").size())


if __name__ == "__main__":
    export_predictions()
