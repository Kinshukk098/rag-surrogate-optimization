
import csv
import json
import os
import time

import torch

from src.rag.pipeline import RAGPipeline
from src.rag.llm import GroqAnswerGenerator
from src.rag.metrics import exact_match, token_f1


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def supporting_title_recall(retrieved_titles, supporting_facts):
    required_titles = set(supporting_facts.get("title", []))

    if not required_titles:
        return 0.0

    retrieved_titles = set(retrieved_titles)

    return len(required_titles & retrieved_titles) / len(required_titles)


def main():

    with open("experiment_config.json", "r", encoding="utf-8") as file:
        settings = json.load(file)

    questions = load_json(settings["questions_path"])
    documents = load_json(settings["corpus_path"])

    limit = settings.get("question_limit")
    if limit is not None:
        questions = questions[:limit]

    config = settings["rag_config"]
    llm_settings = settings["llm"]
    run_id = settings["experiment"]["run_id"]

    config_id = config.get("config_id", run_id)
    output_path = settings["output_path"]

    # Never overwrite an existing experiment result.
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Choose a new output_path in experiment_config.json."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("Dataset:", settings["dataset"])
    print("Questions to evaluate:", len(questions))
    print("Configuration ID:", config_id)
    print("Configuration:", config)

    # 1. Initialize retrieval pipeline.
    initialization_start = time.perf_counter()

    pipeline = RAGPipeline(config, documents)

    initialization_seconds = (
        time.perf_counter() - initialization_start
    )

    # 2. Initialize answer generator.
    generator = GroqAnswerGenerator(
        model_name=llm_settings["model"],
        temperature=llm_settings.get("temperature", 0),
        max_completion_tokens=llm_settings.get(
            "max_completion_tokens", 256
        ),
        seed=llm_settings.get("seed", 42)
    )

    print(f"\nInitialization time: {initialization_seconds:.3f} seconds")
    print("\nStarting question evaluation...\n")

    rows = []

    for index, question_data in enumerate(questions, start=1):

        question_id = question_data["question_id"]
        question_text = question_data["question"]
        gold_answer = question_data["answer"]

        supporting_facts = question_data.get("supporting_facts", {})

        print(f"[{index}/{len(questions)}] Question ID: {question_id}")

        # 3. Retrieval, threshold filtering and optional reranking.
        sync_gpu()
        retrieval_start = time.perf_counter()

        retrieved_documents = pipeline.retrieve(question_text)

        sync_gpu()

        retrieval_pipeline_latency = (
            time.perf_counter() - retrieval_start
        )

        retrieval_stats = pipeline.last_retrieval_stats

        retrieved_titles = [
            doc.get("title", "")
            for doc in retrieved_documents
        ]

        # 4. Generate answer and measure context construction separately.
        if retrieved_documents:

            generation_result = generator.generate(
                question_text,
                retrieved_documents,
                context_budget=config["context_budget"]
            )

            predicted_answer = generation_result["answer"]

            context_construction_latency = generation_result[
                "context_construction_latency_seconds"
            ]

            generation_latency = generation_result[
                "generation_latency_seconds"
            ]

            input_tokens = generation_result["input_tokens"]
            output_tokens = generation_result["output_tokens"]
            total_tokens = generation_result["total_tokens"]

            context_tokens = generation_result["context_tokens"]

            context_documents_used = generation_result[
                "context_documents_used"
            ]

            context_document_titles = generation_result[
                "context_document_titles"
            ]

            actual_context = generation_result["context_text"]

        else:

            predicted_answer = "Insufficient information."

            context_construction_latency = 0.0
            generation_latency = 0.0

            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            context_tokens = 0
            context_documents_used = 0

            context_document_titles = []
            actual_context = ""

        # 5. Answer quality evaluation.
        em = exact_match(predicted_answer, gold_answer)
        f1 = token_f1(predicted_answer, gold_answer)

        # 6. Retrieval-stage and context-stage supporting-title recall.
        retrieval_recall = supporting_title_recall(
            retrieved_titles,
            supporting_facts
        )

        context_recall = supporting_title_recall(
            context_document_titles,
            supporting_facts
        )

        # 7. Total measured pipeline latency.
        total_latency = (
            retrieval_pipeline_latency
            + context_construction_latency
            + generation_latency
        )

        # 8. Store question-level experiment results.
        row = {
            "run_id": run_id,
            "dataset": settings["dataset"],
            "config_id": config_id,
            "question_id": question_id,
            "question": question_text,

            # RAG configuration.
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "top_k": config["top_k"],
            "similarity_threshold": config["similarity_threshold"],
            "context_budget": config["context_budget"],
            "retriever": config["retriever"],
            "reranker": config["reranker"],

            # Retrieval statistics.
            "candidate_count": retrieval_stats["candidate_count"],
            "filtered_count": retrieval_stats["filtered_count"],
            "returned_count": retrieval_stats["returned_count"],

            # Context statistics.
            "context_documents_used": context_documents_used,
            "context_tokens": context_tokens,

            # Model information.
            "llm_model": llm_settings["model"],

            # Quality metrics.
            "answer_em": em,
            "answer_f1": f1,
            "supporting_title_recall": retrieval_recall,
            "supporting_context_title_recall": context_recall,

            # Latency measurements.
            "retrieval_pipeline_latency_seconds": (
                retrieval_pipeline_latency
            ),
            "context_construction_latency_seconds": (
                context_construction_latency
            ),
            "generation_latency_seconds": generation_latency,
            "total_latency_seconds": total_latency,

            # Token usage.
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,

            # Answer and context details.
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,

            "retrieved_titles": json.dumps(
                retrieved_titles,
                ensure_ascii=False
            ),

            "context_document_titles": json.dumps(
                context_document_titles,
                ensure_ascii=False
            ),

            "actual_context": actual_context
        }

        rows.append(row)

        print("  Gold:", gold_answer)
        print("  Predicted:", predicted_answer)

        print(
            f"  EM: {em:.3f} | F1: {f1:.3f} | "
            f"Retrieval Recall: {retrieval_recall:.3f} | "
            f"Context Recall: {context_recall:.3f}"
        )

        print(
            f"  Retrieved: {retrieval_stats['candidate_count']} "
            f"candidates -> {retrieval_stats['filtered_count']} "
            f"filtered -> {retrieval_stats['returned_count']} returned"
        )

        print(
            f"  Documents in context: {context_documents_used}"
        )

        print(f"  Context tokens: {context_tokens}")

        print(
            f"  Retrieval latency: "
            f"{retrieval_pipeline_latency:.3f}s"
        )

        print(
            f"  Context construction latency: "
            f"{context_construction_latency:.3f}s"
        )

        print(
            f"  Generation latency: {generation_latency:.3f}s"
        )

        print(
            f"  Total latency: {total_latency:.3f}s\n"
        )

    # 9. Save question-level results.
    if rows:

        with open(
            output_path, "w", newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

    # 10. Print aggregate summary.
    count = len(rows)

    if count:

        def mean(field):
            return sum(row[field] for row in rows) / count

        print("\n========== EXPERIMENT SUMMARY ==========")

        print("Run ID:", run_id)
        print("Configuration ID:", config_id)
        print("Questions evaluated:", count)

        print("Mean Exact Match:", round(mean("answer_em"), 4))
        print("Mean Answer F1:", round(mean("answer_f1"), 4))

        print(
            "Mean Supporting-title Retrieval Recall:",
            round(mean("supporting_title_recall"), 4)
        )

        print(
            "Mean Supporting-title Context Recall:",
            round(mean("supporting_context_title_recall"), 4)
        )

        print(
            "Mean Retrieval Latency:",
            round(mean("retrieval_pipeline_latency_seconds"), 4)
        )

        print(
            "Mean Context Construction Latency:",
            round(mean("context_construction_latency_seconds"), 4)
        )

        print(
            "Mean Generation Latency:",
            round(mean("generation_latency_seconds"), 4)
        )

        print(
            "Mean Total Latency:",
            round(mean("total_latency_seconds"), 4)
        )

        print(
            "Mean Context Tokens:",
            round(mean("context_tokens"), 2)
        )

        print(
            "Mean API Input Tokens:",
            round(mean("input_tokens"), 2)
        )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
