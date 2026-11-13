
import json

from src.rag.pipeline import RAGPipeline
from src.rag.llm import GroqAnswerGenerator

QUESTIONS_PATH = "data/splits/hotpotqa_100_questions.json"
CORPUS_PATH = "data/processed/hotpotqa_100_corpus.json"
CONFIG_PATH = "data/configurations/selected_balanced_v2/train_configs_320.json"

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)

with open(CORPUS_PATH, encoding="utf-8") as f:
    data = json.load(f)

documents = data if isinstance(data, list) else data["documents"]

with open(CONFIG_PATH, encoding="utf-8") as f:
    configs = json.load(f)

if isinstance(configs, dict):
    configs = configs.get("configurations", configs.get("configs", []))

targets = ["config_02785", "config_10601"]
budgets = [256, 512, 1024, 2048, 4096]

generator = GroqAnswerGenerator()

for config_id in targets:
    config = next(c for c in configs if c["config_id"] == config_id)

    config = {
        k: v for k, v in config.items()
        if k not in ("config_id", "selection_id", "split")
    }
    config["retrieval_k"] = 20

    print("\n" + "=" * 80)
    print("CONFIG:", config_id)

    pipeline = RAGPipeline(config, documents)

    for q in [questions[0], questions[2]]:
        retrieved = pipeline.retrieve(q["question"])

        print("\nQUESTION:", q["question_id"])
        print("Retrieved titles:", [d["title"] for d in retrieved])
        print("Supporting titles:", q["supporting_facts"]["title"])

        for budget in budgets:
            context, tokens, count, titles = generator.build_context(
                retrieved, budget
            )

            supporting_in_context = [
                title for title in q["supporting_facts"]["title"]
                if title in titles
            ]

            print(
                f"Budget={budget:4} | "
                f"Actual tokens={tokens:4} | "
                f"Documents={count} | "
                f"Supporting docs={supporting_in_context}"
            )
