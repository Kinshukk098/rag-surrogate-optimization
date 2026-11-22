
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


# ==========================================================
# INPUT FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "context_budget",
    "retriever",
    "reranker",
]


# ==========================================================
# PREDICTION TARGETS
# ==========================================================

TARGET_COLUMNS = [
    "mean_answer_f1",
    "mean_answer_em",
    "mean_latency",
    "mean_total_tokens",
]


NUMERIC_FEATURES = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "context_budget",
]


CATEGORICAL_FEATURES = [
    "retriever",
    "reranker",
]


# ==========================================================
# LOAD AND VALIDATE DATA
# ==========================================================

def load_data(path):

    df = pd.read_csv(path)

    required_columns = (
        ["split", "config_id"]
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["config_id"].duplicated().any():
        raise ValueError(
            "Configuration IDs are not unique."
        )

    train_df = (
        df[df["split"] == "train"]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df[df["split"] == "test"]
        .copy()
        .reset_index(drop=True)
    )

    if len(train_df) != 320:
        raise ValueError(
            f"Expected 320 training configurations, "
            f"found {len(train_df)}"
        )

    if len(test_df) != 80:
        raise ValueError(
            f"Expected 80 test configurations, "
            f"found {len(test_df)}"
        )

    if not train_df[TARGET_COLUMNS].notna().all().all():
        raise ValueError("Missing training target values.")

    if not test_df[TARGET_COLUMNS].notna().all().all():
        raise ValueError("Missing test target values.")

    return train_df, test_df


# ==========================================================
# PREPROCESSING PIPELINE
# ==========================================================

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ],
        sparse_threshold=0.0
    )

    return preprocessor
