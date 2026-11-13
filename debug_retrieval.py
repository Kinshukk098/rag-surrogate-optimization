
import json

from src.rag.chunking import chunk_documents
from src.rag.bm25_retriever import BM25Retriever

with open("data/processed/hotpotqa_100_corpus.json", encoding="utf-8") as f:
    documents = json.load(f)

with open("data/splits/hotpotqa_100_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

question = questions[0]["question"]

chunks = chunk_documents(
    documents,
    chunk_size=256,
    chunk_overlap=0
)

retriever = BM25Retriever(chunks)
results = retriever.retrieve(question, top_k=20)

scores = [doc["score"] for doc in results]
min_score = min(scores)
max_score = max(scores)

print("\nQUESTION:", question)
print("Gold answer:", questions[0]["answer"])
print("Min score:", min_score)
print("Max score:", max_score)

print("\nTOP 20 AFTER NORMALIZATION")
print("=" * 90)

for rank, doc in enumerate(results, 1):
    score = doc["score"]

    if max_score - min_score < 1e-12:
        normalized = 1.0
    else:
        normalized = (score - min_score) / (max_score - min_score)

    status = "KEEP" if normalized >= 0.35 else "FILTERED"

    print(
        f"{rank:2}. Raw={score:9.4f} | "
        f"Normalized={normalized:.4f} | "
        f"{status:8} | {doc.get('title')}"
    )
