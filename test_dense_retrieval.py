
import json

from src.rag.chunking import chunk_documents
from src.rag.dense_retriever import DenseRetriever


with open("data/processed/hotpotqa_corpus.json", encoding="utf-8") as f:
    documents = json.load(f)

with open("data/splits/hotpotqa_20_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

chunks = chunk_documents(
    documents,
    chunk_size=256,
    chunk_overlap=32
)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

retriever = DenseRetriever(chunks)

question = questions[0]["question"]

results = retriever.retrieve(question, top_k=5)

print("\nQUESTION:")
print(question)

print("\nDENSE RETRIEVAL RESULTS:")

for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc['title']}")
    print(f"Score: {doc['score']:.4f}")
    print(doc["text"][:300])
