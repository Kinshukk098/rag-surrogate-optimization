
import json
from pathlib import Path

from src.rag.chunking import chunk_documents
from src.rag.bm25_retriever import BM25Retriever

BASE = Path(__file__).parent

with open("data/processed/hotpotqa_100_corpus.json", encoding="utf-8") as f:
    documents = json.load(f)

with open("data/splits/hotpotqa_100_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

# Build BM25 index
chunks = chunk_documents(
    documents,
    chunk_size=256,
    chunk_overlap=32
)

retriever = BM25Retriever(chunks)

total_recall = 0
question_results = []

for question in questions:

    retrieved = retriever.retrieve(
        question["question"],
        top_k=5
    )

    retrieved_titles = {
        item["title"] for item in retrieved
    }

    supporting_titles = set(
        question["supporting_facts"]["title"]
    )

    overlap = retrieved_titles & supporting_titles

    recall = (
        len(overlap) / len(supporting_titles)
        if supporting_titles else 0
    )

    total_recall += recall

    question_results.append({
        "id": question["question_id"],
        "question": question["question"],
        "supporting_titles": list(supporting_titles),
        "retrieved_titles": list(retrieved_titles),
        "matched_titles": list(overlap),
        "recall_at_5": recall
    })

average_recall = total_recall / len(questions)

print("\nBM25 EVALUATION")
print("------------------------")
print("Questions:", len(questions))
print("Average Supporting Title Recall@5:",
      round(average_recall, 4))

for result in question_results:
    print(
        result["id"],
        "Recall@5:",
        round(result["recall_at_5"], 3)
    )

output = BASE / "data/processed/bm25_evaluation.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(question_results, f, indent=2)

print("\nSaved results to:", output)
