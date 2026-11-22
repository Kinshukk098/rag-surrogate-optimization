
import numpy as np
import pandas as pd
import sklearn.base

from .preprocessing import (
    load_data,
    create_preprocessor,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

from .models import create_models


def main():

    data_path = (
        "data/processed/experiments/"
        "hotpotqa_config_results.csv"
    )

    train_df, test_df = load_data(data_path)

    preprocessor = create_preprocessor()

    X_train = preprocessor.fit_transform(
        train_df[FEATURE_COLUMNS]
    )

    X_test = preprocessor.transform(
        test_df[FEATURE_COLUMNS]
    )

    models = create_models(random_state=42)

    print("\nMODEL VALIDATION")
    print("=" * 60)

    successful = 0
    total = len(models) * len(TARGET_COLUMNS)

    for model_name, model_template in models.items():

        for target in TARGET_COLUMNS:

            print(
                f"\nTesting {model_name} | {target}"
            )

            model = sklearn.base.clone(
                model_template
            )

            y_train = train_df[target].to_numpy()
            y_test = test_df[target].to_numpy()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            predictions = np.asarray(
                predictions
            ).ravel()

            assert predictions.shape == (80,)

            assert np.isfinite(
                predictions
            ).all(), "Non-finite predictions found!"

            print(
                "Prediction shape:",
                predictions.shape
            )

            print(
                "Prediction range:",
                round(predictions.min(), 4),
                "to",
                round(predictions.max(), 4)
            )

            print(
                "Actual mean:",
                round(y_test.mean(), 4)
            )

            successful += 1

    print("\n" + "=" * 60)
    print(f"Successful model-target fits: {successful}/{total}")
    print("Model validation completed!")


if __name__ == "__main__":
    main()
