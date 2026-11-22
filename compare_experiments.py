
import pandas as pd

files = {
    "Dense + Reranker": "data/processed/experiment_results_dense_reranker.csv",
    "Dense without Reranker": "data/processed/experiment_results_dense_no_reranker.csv"
}

metrics = [
    "answer_em",
    "answer_f1",
    "supporting_title_recall",
    "total_latency_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens"
]

for name, path in files.items():
    df = pd.read_csv(path)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Questions:", len(df))
    print("Question IDs:", sorted(df["question_id"].tolist()))

    print("\nMean metrics:")
    print(df[metrics].mean().round(4))

    print("\nIndividual question results:")
    print(df[["question_id", "answer_em", "answer_f1",
              "supporting_title_recall"]].to_string(index=False))
