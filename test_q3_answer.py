
import json
from src.rag.pipeline import RAGPipeline
from src.rag.llm import GroqAnswerGenerator

with open("data/splits/hotpotqa_100_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

with open("data/processed/hotpotqa_100_corpus.json", encoding="utf-8") as f:
    data = json.load(f)

documents = data if isinstance(data, list) else data["documents"]

with open("data/configurations/selected_balanced_v2/train_configs_320.json", encoding="utf-8") as f:
    configs = json.load(f)

if isinstance(configs, dict):
    configs = configs.get("configurations", configs.get("configs", []))

config = next(c for c in configs if c["config_id"] == "config_02785")

test_config = {
    k: v for k, v in config.items()
    if k not in ("config_id", "selection_id", "split")
}

test_config["top_k"] = 10
test_config["retrieval_k"] = 20
test_config["context_budget"] = 1536

pipeline = RAGPipeline(test_config, documents)
generator = GroqAnswerGenerator()

q = questions[2]
retrieved = pipeline.retrieve(q["question"])

result = generator.generate(q["question"], retrieved, 1536)

print("\nQUESTION:", q["question"])
print("GOLD ANSWER:", q["answer"])
print("PREDICTED ANSWER:", result["answer"])

print("CONTEXT TOKENS:", result["context_tokens"])
print("DOCUMENTS USED:", result["context_documents_used"])
print("CONTEXT TITLES:", result["context_document_titles"])

print("API PROMPT TOKENS:", result["input_tokens"])
print("API OUTPUT TOKENS:", result["output_tokens"])
print("API TOTAL TOKENS:", result["total_tokens"])

print("CONTEXT BUILD TIME:",
      result["context_construction_latency_seconds"])

print("GENERATION LATENCY:",
      result["generation_latency_seconds"])
