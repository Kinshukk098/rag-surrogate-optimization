
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "data/processed/experiments/"
    "hotpotqa_question_results_final_v2.csv"
)

OUTPUT_PATH = Path(
    "data/processed/experiments/"
    "hotpotqa_config_results.csv"
)

# --------------------------------------------------
# Load question-level results
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Loaded records:", len(df))

# --------------------------------------------------
# Validate input data
# --------------------------------------------------

required_columns = [
    "split",
    "config_id",
    "question_id",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "context_budget",
    "retriever",
    "reranker",
    "answer_f1",
    "answer_em",
    "supporting_title_recall",
    "supporting_context_title_recall",
    "total_latency_seconds",
    "total_tokens",
    "context_tokens",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df.duplicated(
    ["split", "config_id", "question_id"]
).any():
    raise ValueError("Duplicate question evaluations found!")

counts = df.groupby(
    ["split", "config_id"]
).size()

if not (counts == 100).all():
    raise ValueError(
        "Some configurations do not have exactly 100 questions!"
    )

# --------------------------------------------------
# Configuration parameters
# --------------------------------------------------

config_columns = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "context_budget",
    "retriever",
    "reranker",
]

# Confirm parameters remain fixed per configuration
for col in config_columns:
    variation = df.groupby(
        ["split", "config_id"]
    )[col].nunique()

    if not (variation == 1).all():
        raise ValueError(
            f"Inconsistent parameter values: {col}"
        )

# --------------------------------------------------
# Aggregate question-level metrics
# --------------------------------------------------

group_columns = ["split", "config_id"]

aggregated = df.groupby(
    group_columns,
    as_index=False
).agg(
    **{
        col: (col, "first")
        for col in config_columns
    },

    mean_answer_f1=("answer_f1", "mean"),
    std_answer_f1=("answer_f1", "std"),

    mean_answer_em=("answer_em", "mean"),

    mean_retrieval_recall=(
        "supporting_title_recall", "mean"
    ),

    mean_context_recall=(
        "supporting_context_title_recall", "mean"
    ),

    mean_latency=(
        "total_latency_seconds", "mean"
    ),

    std_latency=(
        "total_latency_seconds", "std"
    ),

    mean_total_tokens=("total_tokens", "mean"),

    mean_context_tokens=("context_tokens", "mean"),

    mean_input_tokens=("input_tokens", "mean"),

    mean_output_tokens=("output_tokens", "mean"),

    question_count=("question_id", "count"),
)

# --------------------------------------------------
# Save aggregated dataset
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

aggregated.to_csv(OUTPUT_PATH, index=False)

# --------------------------------------------------
# Final validation and summary
# --------------------------------------------------

print("\nAggregation completed!")
print("Output:", OUTPUT_PATH)
print("Total configurations:", len(aggregated))

print("\nSplit summary:")
print(
    aggregated.groupby("split").agg(
        configurations=("config_id", "count"),
        questions_per_config=("question_count", "mean"),
    )
)

print("\nAggregated columns:")
print(aggregated.columns.tolist())

print("\nFirst five rows:")
print(aggregated.head())

print("\nSaved successfully!")
