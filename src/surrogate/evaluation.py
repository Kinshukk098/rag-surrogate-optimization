
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression(
    y_true,
    y_pred,
    target_name,
    optimize="max",
    actual_top_k=5,
    predicted_top_k=10,
):
    """
    Evaluate regression accuracy and configuration ranking.

    Parameters
    ----------
    y_true : array
        Actual target values.

    y_pred : array
        Predicted target values.

    target_name : str
        Name of the target being evaluated.

    optimize : str
        'max' for targets where higher is better.
        'min' for targets where lower is better.

    actual_top_k : int
        Number of actual best configurations.

    predicted_top_k : int
        Number of predicted best configurations.
    """

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted lengths differ.")

    if len(y_true) < 2:
        raise ValueError("At least two samples are required.")

    if not np.isfinite(y_true).all():
        raise ValueError("Actual values contain NaN or infinity.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain NaN or infinity.")

    if optimize not in ("max", "min"):
        raise ValueError("optimize must be 'max' or 'min'.")

    # --------------------------------------------------
    # 1. Regression accuracy
    # --------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true, y_pred
    )

    r2 = r2_score(
        y_true, y_pred
    )

    # --------------------------------------------------
    # 2. Ranking accuracy
    # --------------------------------------------------

    spearman = spearmanr(
        y_true, y_pred
    ).statistic

    if not np.isfinite(spearman):
        spearman = np.nan

    # --------------------------------------------------
    # 3. Top-K overlap
    # --------------------------------------------------

    n = len(y_true)

    k_actual = min(actual_top_k, n)
    k_predicted = min(predicted_top_k, n)

    if optimize == "max":
        actual_best_indices = np.argsort(-y_true)[:k_actual]
        predicted_best_indices = np.argsort(-y_pred)[:k_predicted]

    else:
        actual_best_indices = np.argsort(y_true)[:k_actual]
        predicted_best_indices = np.argsort(y_pred)[:k_predicted]

    overlap = len(
        set(actual_best_indices)
        & set(predicted_best_indices)
    )

    top_k_overlap = overlap / k_actual

    # --------------------------------------------------
    # 4. Best configuration regret
    # --------------------------------------------------

    predicted_best_index = predicted_best_indices[0]

    if optimize == "max":
        actual_optimum = np.max(y_true)

        actual_at_predicted_best = y_true[
            predicted_best_index
        ]

        regret = (
            actual_optimum - actual_at_predicted_best
        )

    else:
        actual_optimum = np.min(y_true)

        actual_at_predicted_best = y_true[
            predicted_best_index
        ]

        regret = (
            actual_at_predicted_best - actual_optimum
        )

    return {
        "target": target_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Spearman": spearman,
        "TopK_overlap": top_k_overlap,
        "Regret": regret,
    }
