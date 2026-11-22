
import pandas as pd
from pathlib import Path

INPUT_PATH = Path(
    "data/processed/experiments/search_results/"
    "hotpotqa_search_question_results_top5.csv"
)

OUTPUT_PATH = Path(
    "data/processed/experiments/search_results/"
    "hotpotqa_search_config_results_top5.csv"
)

CONFIG_COLS = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "context_budget",
    "retriever",
    "reranker",
]

METRIC_COLS = [
    "answer_f1",
    "answer_em",
    "supporting_title_recall",
    "supporting_context_title_recall",
    "total_latency_seconds",
    "total_tokens",
    "context_tokens",
    "input_tokens",
    "output_tokens",
]

df = pd.read_csv(INPUT_PATH)

required = [
    "dataset", "split", "config_id", "question_id",
    *CONFIG_COLS, *METRIC_COLS
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["config_id"].isna().any():
    raise ValueError("Missing config_id values")

duplicates = df.duplicated(
    subset=["dataset", "split", "config_id", "question_id"]
)
if duplicates.any():
    raise ValueError(f"Found {duplicates.sum()} duplicate question rows")

counts = df.groupby("config_id")["question_id"].nunique()
if not (counts == 100).all():
    raise ValueError(
        f"Expected 100 questions per config. Counts:\n{counts[counts != 100]}"
    )

# Check that each configuration has consistent parameter values.
for col in CONFIG_COLS:
    inconsistent = df.groupby("config_id")[col].nunique()
    if (inconsistent > 1).any():
        raise ValueError(f"Inconsistent configuration values in {col}")

agg = {
    col: "first" for col in CONFIG_COLS
}

for col in METRIC_COLS:
    agg[col] = "mean"

agg["total_latency_seconds"] = ["mean", "std"]
agg["answer_f1"] = ["mean", "std"]
agg["answer_em"] = ["mean", "std"]

result = (
    df.groupby(["dataset", "split", "config_id"])
      .agg(agg)
)

result.columns = [
    "_".join(str(part) for part in col if part)
    if isinstance(col, tuple) else col
    for col in result.columns
]

result = result.reset_index()
result["question_count"] = (
    df.groupby(["dataset", "split", "config_id"])
      .size()
      .values
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("SEARCH AGGREGATION COMPLETE")
print(f"Question-level rows: {len(df)}")
print(f"Unique configurations: {result['config_id'].nunique()}")
print(f"Aggregated rows: {len(result)}")
print(f"Questions per config: {result['question_count'].value_counts().to_dict()}")
print(f"Saved to: {OUTPUT_PATH}")
