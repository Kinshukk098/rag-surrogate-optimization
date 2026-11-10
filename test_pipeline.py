
import json

from src.rag.pipeline import RAGPipeline


# Load our existing 100-question dataset
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


# First test: BM25
config = {
    "chunk_size": 256,
    "chunk_overlap": 32,
    "retriever": "dense",
    "top_k": 5,
    "reranker": False
}

pipeline = RAGPipeline(config, documents)

# Test retrieval for the first question
question = questions[0]["question"]

print("\nQuestion:")
print(question)

results = pipeline.retrieve(question)

print("\nRetrieved chunks:")

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Title:", result["title"])
    print("Score:", result["score"])
    print("Text:", result["text"][:300])
