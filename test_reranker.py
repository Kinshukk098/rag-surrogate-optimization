
import json

from src.rag.pipeline import RAGPipeline


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

question = questions[0]["question"]

print("\nQuestion:")
print(question)

results = pipeline.retrieve(question)

print("\nFinal reranked results:")

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Title:", result["title"])
    print("Retrieval score:", result["score"])
    print("Reranker score:", result["rerank_score"])
    print("Text:", result["text"][:300])
