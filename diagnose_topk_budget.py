
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

pipeline = RAGPipeline(test_config, documents)
generator = GroqAnswerGenerator()

q = questions[2]
retrieved = pipeline.retrieve(q["question"])

print("Retrieved titles:")
for i, doc in enumerate(retrieved, 1):
    print(i, doc["title"])

print("\nContext budget experiment:")

for budget in [1024, 1536, 2048, 3072, 4096]:
    context, tokens, count, titles = generator.build_context(
        retrieved, budget
    )

    supporting = q["supporting_facts"]["title"]
    found = [title for title in supporting if title in titles]

    print(
        f"Budget={budget:4d} | "
        f"Actual={tokens:4d} | "
        f"Documents={count:2d} | "
        f"Supporting docs={len(found)}/2: {found}"
    )
