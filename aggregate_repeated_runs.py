
import pandas as pd
from pathlib import Path

BASE = Path("data/processed/experiments/repeated_runs")

INPUT_PATH = BASE / "repeated_question_results.csv"
RUN_LEVEL_PATH = BASE / "repeated_run_level_metrics.csv"
SUMMARY_PATH = BASE / "repeated_run_summary.csv"

df = pd.read_csv(INPUT_PATH)

metrics = [
    "answer_f1",
    "answer_em",
    "total_latency_seconds",
    "total_tokens",
]

# Calculate average metrics for each configuration in each repeat.
run_level = (
    df.groupby(["config_id", "repeat_id"], as_index=False)
    .agg(
        **{
            metric: (metric, "mean")
            for metric in metrics
        },
        question_count=("question_id", "nunique"),
        chunk_size=("chunk_size", "first"),
        chunk_overlap=("chunk_overlap", "first"),
        top_k=("top_k", "first"),
        similarity_threshold=("similarity_threshold", "first"),
        context_budget=("context_budget", "first"),
        retriever=("retriever", "first"),
        reranker=("reranker", "first"),
    )
)

run_level.to_csv(RUN_LEVEL_PATH, index=False)

# Calculate mean and standard deviation across the five repeats.
summary = (
    run_level.groupby("config_id")
    .agg(
        repeats=("repeat_id", "nunique"),
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in metrics
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in metrics
        },
        chunk_size=("chunk_size", "first"),
        chunk_overlap=("chunk_overlap", "first"),
        top_k=("top_k", "first"),
        similarity_threshold=("similarity_threshold", "first"),
        context_budget=("context_budget", "first"),
        retriever=("retriever", "first"),
        reranker=("reranker", "first"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

print("=" * 60)
print("REPEATED-RUN AGGREGATION COMPLETE")
print("=" * 60)

print("\nRun-level rows:", len(run_level))
print("Summary rows:", len(summary))

print("\nREPEATED-RUN SUMMARY:")
print(
    summary[
        [
            "config_id",
            "answer_f1_mean",
            "answer_f1_std",
            "answer_em_mean",
            "answer_em_std",
            "total_latency_seconds_mean",
            "total_latency_seconds_std",
            "total_tokens_mean",
            "total_tokens_std",
        ]
    ].round(5).to_string(index=False)
)

print("\nSaved:")
print(RUN_LEVEL_PATH)
print(SUMMARY_PATH)
