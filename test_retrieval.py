
import json

from src.rag.chunking import chunk_documents
from src.rag.bm25_retriever import BM25Retriever

with open("data/processed/hotpotqa_corpus.json",
          encoding="utf-8") as f:
    documents = json.load(f)

with open("data/splits/hotpotqa_20_questions.json",
          encoding="utf-8") as f:
    questions = json.load(f)

# Create chunks
chunks = chunk_documents(
    documents,
    chunk_size=256,
    chunk_overlap=32
)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

# Build BM25
retriever = BM25Retriever(chunks)

# Test first question
question = questions[0]["question"]

results = retriever.retrieve(
    question,
    top_k=5
)

print("\nQUESTION:")
print(question)

print("\nRETRIEVED DOCUMENTS:")

for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['title']}")
    print("Score:", round(result["score"], 4))
    print(result["text"][:500])
