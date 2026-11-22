
import json

from src.rag.chunking import chunk_documents
from src.rag.dense_retriever import DenseRetriever


with open("data/processed/hotpotqa_100_corpus.json", encoding="utf-8") as f:
    documents = json.load(f)

with open("data/splits/hotpotqa_100_questions.json", encoding="utf-8") as f:
    questions = json.load(f)


chunks = chunk_documents(
    documents,
    chunk_size=256,
    chunk_overlap=32
)

retriever = DenseRetriever(chunks)

all_recalls = []
results_data = []

for question in questions:

    query = question["question"]

    supporting_titles = set(
        question["supporting_facts"]["title"]
    )

    retrieved = retriever.retrieve(query, top_k=5)

    retrieved_titles = set(
        doc["title"] for doc in retrieved
    )

    if supporting_titles:
        recall = (
            len(supporting_titles & retrieved_titles)
            / len(supporting_titles)
        )
    else:
        recall = 0.0

    all_recalls.append(recall)

    results_data.append({
    "question_id": len(results_data) + 1,
    "question": query,
    "recall_at_5": recall,
    "supporting_titles": list(supporting_titles),
    "retrieved_titles": list(retrieved_titles)
})


average_recall = sum(all_recalls) / len(all_recalls)

print("\nDENSE RETRIEVAL EVALUATION")
print("--------------------------")
print("Questions:", len(questions))
print(f"Average Supporting Title Recall@5: {average_recall:.4f}")

for item in results_data:
    print(
        item["question_id"],
        "Recall@5:",
        item["recall_at_5"]
    )


output = {
    "retriever": "Dense",
    "model": "all-MiniLM-L6-v2",
    "questions": len(questions),
    "top_k": 5,
    "average_recall_at_5": average_recall,
    "results": results_data
}

with open(
    "data/processed/dense_evaluation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("\nSaved results to data/processed/dense_evaluation.json")
