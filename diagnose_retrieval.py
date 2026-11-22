
import json
from src.rag.chunking import chunk_documents
from src.rag.bm25_retriever import BM25Retriever

QUESTIONS_PATH = "data/splits/hotpotqa_100_questions.json"
CORPUS_PATH = "data/processed/hotpotqa_100_corpus.json"
CONFIG_PATH = "data/configurations/selected_balanced_v2/train_configs_320.json"

TARGET_CONFIGS = {"config_02785", "config_10601"}

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus_data = json.load(f)

documents = (
    corpus_data if isinstance(corpus_data, list)
    else corpus_data["documents"]
)

with open(CONFIG_PATH, encoding="utf-8") as f:
    configs = json.load(f)

if isinstance(configs, dict):
    configs = configs.get("configurations", configs.get("configs", []))

selected_configs = [
    c for c in configs if c.get("config_id") in TARGET_CONFIGS
]

if len(selected_configs) != 2:
    raise ValueError(
        f"Expected 2 configs, found {len(selected_configs)}. "
        "Check the configuration JSON structure."
    )

for config in selected_configs:
    config_id = config["config_id"]

    print("\n" + "#" * 80)
    print("CONFIG:", config_id)
    print(json.dumps(config, indent=2))

    chunks = chunk_documents(
        documents,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

    retriever = BM25Retriever(chunks)

    for question_data in questions[:3]:
        question = question_data["question"]
        support_titles = question_data["supporting_facts"]["title"]

        query_tokens = retriever.tokenize(question)
        all_scores = retriever.bm25.get_scores(query_tokens)

        top_indices = all_scores.argsort()[::-1][:20]
        top_scores = [float(all_scores[i]) for i in top_indices]

        min_score = min(top_scores)
        max_score = max(top_scores)

        print("\n" + "-" * 70)
        print("QUESTION:", question_data["question_id"], question)

        print("\nSUPPORTING DOCUMENT RANKS (across ALL chunks):")

        for title in support_titles:
            matching = [
                (i, float(all_scores[i]))
                for i, chunk in enumerate(chunks)
                if chunk["title"].strip().casefold()
                == title.strip().casefold()
            ]

            if not matching:
                print(f"  {title}: NO CHUNKS")
                continue

            best_index, best_score = max(
                matching, key=lambda x: x[1]
            )

            rank = 1 + sum(
                score > best_score for score in all_scores
            )

            print(
                f"  {title}: best_score={best_score:.4f}, "
                f"global_chunk_rank={rank}, "
                f"best_chunk_id={chunks[best_index]['chunk_id']}"
            )
            print(
                "    Best chunk preview:",
                chunks[best_index]["text"][:220].replace("\n", " ")
            )

        print("\nTOP 20 RETRIEVED CHUNKS:")

        for rank, (idx, score) in enumerate(
            zip(top_indices, top_scores), start=1
        ):
            if max_score - min_score < 1e-12:
                normalized = 1.0
            else:
                normalized = (
                    (score - min_score) / (max_score - min_score)
                )

            chunk = chunks[idx]
            is_support = any(
                chunk["title"].strip().casefold()
                == title.strip().casefold()
                for title in support_titles
            )

            marker = " <-- SUPPORT" if is_support else ""

            print(
                f"{rank:2}. score={score:9.4f} "
                f"norm={normalized:.4f} "
                f"title={chunk['title']}{marker}"
            )
