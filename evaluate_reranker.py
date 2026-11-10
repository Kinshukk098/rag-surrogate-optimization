
import json
import csv
import time

from src.rag.pipeline import RAGPipeline


# Load the same fixed 100-question dataset
with open(
    "data/processed/hotpotqa_100_corpus.json",
    "r",
    encoding="utf-8"
) as f:
    documents = json.load(f)

with open(
    "data/splits/hotpotqa_100_questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


config = {
    "chunk_size": 256,
    "chunk_overlap": 32,
    "retriever": "dense",
    "top_k": 5,
    "retrieval_k": 20,
    "reranker": True
}

pipeline = RAGPipeline(config, documents)


def supporting_title_recall(results, supporting_facts):

    required_titles = set(supporting_facts["title"])

    if not required_titles:
        return 0.0

    retrieved_titles = {
        result["title"]
        for result in results
    }

    return len(
        required_titles.intersection(retrieved_titles)
    ) / len(required_titles)

    if not required_titles:
        return 0.0

    retrieved_titles = {
        result["title"]
        for result in results
    }

    return len(
        required_titles.intersection(retrieved_titles)
    ) / len(required_titles)


baseline_scores = []
candidate_scores = []
reranked_scores = []
reranking_times = []

output_rows = []

for index, question_data in enumerate(questions):

    question = question_data["question"]
    supporting_facts = question_data["supporting_facts"]

    # Retrieve 20 candidates once
    candidates = pipeline.retriever.retrieve(
        question,
        top_k=20
    )

    # Baseline: first 5 from dense retrieval
    baseline_top5 = candidates[:5]

    # Candidate pool recall
    candidate_recall = supporting_title_recall(
        candidates,
        supporting_facts
    )

    # Rerank the same 20 candidates
    start_time = time.perf_counter()

    reranked_top5 = pipeline.reranker.rerank(
        question,
        candidates,
        top_k=5
    )

    elapsed = time.perf_counter() - start_time

    baseline_recall = supporting_title_recall(
        baseline_top5,
        supporting_facts
    )

    reranked_recall = supporting_title_recall(
        reranked_top5,
        supporting_facts
    )

    baseline_scores.append(baseline_recall)
    candidate_scores.append(candidate_recall)
    reranked_scores.append(reranked_recall)
    reranking_times.append(elapsed)

    output_rows.append({
        "question_id": question_data["question_id"],
        "baseline_recall_at_5": baseline_recall,
        "candidate_recall_at_20": candidate_recall,
        "reranked_recall_at_5": reranked_recall,
        "reranking_latency_seconds": elapsed
    })

    print(
        f"Question {index + 1}/{len(questions)} | "
        f"Dense@5: {baseline_recall:.3f} | "
        f"Candidates@20: {candidate_recall:.3f} | "
        f"Reranked@5: {reranked_recall:.3f}"
    )


def average(values):
    return sum(values) / len(values) if values else 0.0


print("\n========== RERANKER EVALUATION ==========")

print(
    "Dense Recall@5:",
    round(average(baseline_scores), 4)
)

print(
    "Dense Candidate Recall@20:",
    round(average(candidate_scores), 4)
)

print(
    "Reranked Recall@5:",
    round(average(reranked_scores), 4)
)

print(
    "Average Reranking Latency (seconds):",
    round(average(reranking_times), 4)
)

with open(
    "data/processed/reranker_evaluation_100.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=output_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(output_rows)

print("\nSaved evaluation results:")
print("data/processed/reranker_evaluation_100.csv")
