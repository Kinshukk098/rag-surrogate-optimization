
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

generator = GroqAnswerGenerator()

config = next(c for c in configs if c["config_id"] == "config_02785")

for top_k in [7, 10, 15, 20]:
    test_config = {
        k: v for k, v in config.items()
        if k not in ("config_id", "selection_id", "split")
    }

    test_config["top_k"] = top_k
    test_config["retrieval_k"] = 20

    pipeline = RAGPipeline(test_config, documents)

    print(f"\n{'='*60}")
    print(f"TOP_K = {top_k}")

    for question_index in [0, 2]:
        q = questions[question_index]
        retrieved = pipeline.retrieve(q["question"])

        context, tokens, count, titles = generator.build_context(
            retrieved, 1024
        )

        supporting = q["supporting_facts"]["title"]
        retrieved_support = [t for t in supporting if t in
                             [d["title"] for d in retrieved]]
        context_support = [t for t in supporting if t in titles]

        print(f"\nQuestion {question_index + 1}")
        print("Retrieved supporting docs:", retrieved_support)
        print("Context supporting docs:", context_support)
        print("Context tokens:", tokens)
        print("Documents in context:", count)
