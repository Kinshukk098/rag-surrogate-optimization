
import json
from src.rag.metrics import exact_match, token_f1
from src.rag.pipeline import RAGPipeline
from src.rag.llm import GroqAnswerGenerator


# Load the existing 100-question dataset and corpus
with open("data/splits/hotpotqa_100_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

with open("data/processed/hotpotqa_100_corpus.json", encoding="utf-8") as f:
    documents = json.load(f)


# Use one real question
question = questions[0]["question"]

print("Question:", question)

# Initialize a dense retrieval pipeline
config = {
    "chunk_size": 256,
    "chunk_overlap": 32,
    "retriever": "dense",
    "top_k": 5,
    "reranker": False
}

pipeline = RAGPipeline(config, documents)

# Retrieve relevant context
retrieved_documents = pipeline.retrieve(question)

print("\nRetrieved documents:", len(retrieved_documents))

# Generate an answer using Groq
generator = GroqAnswerGenerator()
result = generator.generate(question, retrieved_documents)

print("\nGenerated answer:")
print(result["answer"])
# Compare generated answer with HotpotQA gold answer
gold_answer = questions[0]["answer"]
predicted_answer = result["answer"]

print("\nAnswer evaluation:")
print("Gold answer:", gold_answer)
print("Predicted answer:", predicted_answer)
print("Exact Match:", exact_match(predicted_answer, gold_answer))
print("Token F1:", round(token_f1(predicted_answer, gold_answer), 4))
print("\nGeneration metrics:")
print("Model:", result["model"])
print("Input tokens:", result["input_tokens"])
print("Output tokens:", result["output_tokens"])
print("Total tokens:", result["total_tokens"])
print("Generation latency:", round(result["generation_latency_seconds"], 4), "seconds")
