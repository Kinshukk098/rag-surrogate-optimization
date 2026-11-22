
import argparse
import csv
import json
import os
import time
import random
import numpy as np
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


def append_csv(path, row):
    """Append one row, writing the header only for a new file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def load_completed_keys(path):
    """Identify completed dataset/split/config/question combinations."""
    completed = set()

    if not os.path.exists(path):
        return completed

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            completed.add((
                row["dataset"],
                row["split"],
                row["config_id"],
                row["question_id"]
            ))

    return completed


def record_failure(path, row):
    append_csv(path, row)


def evaluate_question(
    pipeline,
    generator,
    question_data,
    config,
    llm_settings,
    settings,
    dataset_name,
    split,
    config_id,
    run_id
):
    question_id = question_data["question_id"]
    question_text = question_data["question"]
    gold_answer = question_data["answer"]
    supporting_facts = question_data.get("supporting_facts", {})

    # Retrieval measurement
    sync_gpu()
    retrieval_start = time.perf_counter()

    retrieved_documents = pipeline.retrieve(question_text)

    sync_gpu()
    retrieval_latency = time.perf_counter() - retrieval_start

    retrieval_stats = pipeline.last_retrieval_stats

    retrieved_titles = [
        doc.get("title", "") for doc in retrieved_documents
    ]

    # Context construction and LLM generation
    if retrieved_documents:
        result = generator.generate(
            question_text,
            retrieved_documents,
            context_budget=config["context_budget"]
        )

        predicted_answer = result["answer"]
        context_latency = result["context_construction_latency_seconds"]
        generation_latency = result["generation_latency_seconds"]

        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        total_tokens = result["total_tokens"]

        context_tokens = result["context_tokens"]
        context_documents_used = result["context_documents_used"]
        context_document_titles = result["context_document_titles"]
        actual_context = result["context_text"]

    else:
        predicted_answer = "Insufficient information."
        context_latency = 0.0
        generation_latency = 0.0

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        context_tokens = 0
        context_documents_used = 0
        context_document_titles = []
        actual_context = ""

    # Quality metrics
    em = exact_match(predicted_answer, gold_answer)
    f1 = token_f1(predicted_answer, gold_answer)

    retrieval_recall = supporting_title_recall(
        retrieved_titles, supporting_facts
    )

    context_recall = supporting_title_recall(
        context_document_titles, supporting_facts
    )

    total_latency = (
        retrieval_latency + context_latency + generation_latency
    )

    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "split": split,
        "config_id": config_id,
        "question_id": question_id,
        "question": question_text,

        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
        "retrieval_k": config["retrieval_k"],
        "similarity_threshold": config["similarity_threshold"],
        "context_budget": config["context_budget"],
        "retriever": config["retriever"],
        "reranker": config["reranker"],

        "candidate_count": retrieval_stats["candidate_count"],
        "filtered_count": retrieval_stats["filtered_count"],
        "returned_count": retrieval_stats["returned_count"],

        "context_documents_used": context_documents_used,
        "context_tokens": context_tokens,

        "llm_model": llm_settings["model"],

        "answer_em": em,
        "answer_f1": f1,
        "supporting_title_recall": retrieval_recall,
        "supporting_context_title_recall": context_recall,

        "retrieval_pipeline_latency_seconds": retrieval_latency,
        "context_construction_latency_seconds": context_latency,
        "generation_latency_seconds": generation_latency,
        "total_latency_seconds": total_latency,

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,

        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,

        "retrieved_titles": json.dumps(
            retrieved_titles, ensure_ascii=False
        ),
        "context_document_titles": json.dumps(
            context_document_titles, ensure_ascii=False
        ),
        "actual_context": actual_context
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train"
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limit number of configurations for a small test"
    )

    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Limit questions per configuration for a small test"
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output CSV path override"
    )

    parser.add_argument(
        "--failed-path",
        default=None,
        help="Optional failed-runs CSV path override"
    ) 

    args = parser.parse_args()

    settings = load_json("research_config.json")
    # Reproducibility: initialize random seeds
    random_seed = settings["experiment"].get("random_seed", 42)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"Random seed: {random_seed}")
    dataset_name = "hotpotqa"
    dataset_settings = settings["datasets"][dataset_name]

    if not dataset_settings["enabled"]:
        raise ValueError(f"{dataset_name} is disabled in research_config.json")

    questions = load_json(dataset_settings["questions_path"])
    documents = load_json(dataset_settings["corpus_path"])

    # CLI argument overrides the central configuration
    question_limit = (
        args.question_limit
        if args.question_limit is not None
        else settings["experiment"].get("question_limit")
    )

    if question_limit is not None:
        questions = questions[:question_limit]

    config_path = settings["configuration_files"][args.split]
    configurations = load_json(config_path)

    if args.max_configs is not None:
        configurations = configurations[:args.max_configs]

    llm_settings = settings["llm"]
    experiment_settings = settings["experiment"]
    output_settings = settings["outputs"]

    output_path = (
        args.output_path
        if args.output_path
        else output_settings["hotpotqa_results"]
    )

    failed_path = (
        args.failed_path
        if args.failed_path
        else output_settings["failed_runs"]
    )

    os.makedirs(output_settings["directory"], exist_ok=True)

    completed = load_completed_keys(output_path)

    print("=" * 65)
    print("RAG BATCH EXPERIMENT")
    print("Dataset:", dataset_name)
    print("Split:", args.split)
    print("Configurations:", len(configurations))
    print("Questions per configuration:", len(questions))
    print("Existing completed records:", len(completed))
    print("Output:", output_path)
    print("=" * 65)

    # Initialize the answer generator once for this process.
    generator = GroqAnswerGenerator(
        model_name=llm_settings["model"],
        temperature=llm_settings.get("temperature", 0),
        max_completion_tokens=llm_settings.get(
            "max_completion_tokens", 256
        ),
        seed=llm_settings.get("seed", 42)
    )

    max_retries = experiment_settings.get("max_retries", 3)
    retry_delay = experiment_settings.get("retry_delay_seconds", 5)
    retrieval_k = experiment_settings.get("retrieval_k", 20)

    for config_index, selected_config in enumerate(configurations, start=1):
        config_id = selected_config["config_id"]

        config = {
            key: value
            for key, value in selected_config.items()
            if key not in ["config_id", "selection_id", "split"]
        }

        config["retrieval_k"] = retrieval_k

        run_id = (
            f"{experiment_settings['experiment_id']}_"
            f"{dataset_name}_{args.split}_{config_id}"
        )

        pending_questions = [
            question for question in questions
            if (
                dataset_name,
                args.split,
                config_id,
                str(question["question_id"])
            ) not in completed
        ]

        print("\n" + "=" * 65)
        print(
            f"Configuration {config_index}/{len(configurations)}: "
            f"{config_id}"
        )
        print("Parameters:", config)
        print("Remaining questions:", len(pending_questions))

        if not pending_questions:
            print("Already completed. Skipping.")
            continue

        try:
            initialization_start = time.perf_counter()
            pipeline = RAGPipeline(config, documents)

            initialization_seconds = (
                time.perf_counter() - initialization_start
            )

            print(
                f"Pipeline initialization: "
                f"{initialization_seconds:.3f}s"
            )

        except Exception as error:
            print("CONFIGURATION INITIALIZATION FAILED:", repr(error))

            record_failure(failed_path, {
                "dataset": dataset_name,
                "split": args.split,
                "config_id": config_id,
                "question_id": "",
                "error": repr(error),
                "stage": "pipeline_initialization"
            })
            continue

        for question_index, question_data in enumerate(
            pending_questions, start=1
        ):
            question_id = str(question_data["question_id"])

            print(
                f"[{question_index}/{len(pending_questions)}] "
                f"Question: {question_id}"
            )

            success = False

            for attempt in range(1, max_retries + 1):
                try:
                    row = evaluate_question(
                        pipeline=pipeline,
                        generator=generator,
                        question_data=question_data,
                        config=config,
                        llm_settings=llm_settings,
                        settings=settings,
                        dataset_name=dataset_name,
                        split=args.split,
                        config_id=config_id,
                        run_id=run_id
                    )

                    append_csv(output_path, row)

                    completed.add((
                        dataset_name,
                        args.split,
                        config_id,
                        question_id
                    ))

                    print(
                        f"  F1={row['answer_f1']:.3f} | "
                        f"EM={row['answer_em']:.3f} | "
                        f"Latency={row['total_latency_seconds']:.3f}s | "
                        f"Tokens={row['total_tokens']}"
                    )

                    success = True
                    break

                except Exception as error:
                    print(
                        f"  Attempt {attempt}/{max_retries} failed: "
                        f"{repr(error)}"
                    )

                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)

            if not success:
                record_failure(failed_path, {
                    "dataset": dataset_name,
                    "split": args.split,
                    "config_id": config_id,
                    "question_id": question_id,
                    "error": "All retry attempts failed",
                    "stage": "question_evaluation"
                })

                print("  Recorded as failed. Continuing.")

    print("\n" + "=" * 65)
    print("BATCH FINISHED")
    print("Split:", args.split)
    print("Output:", output_path)
    print("Total completed records:", len(load_completed_keys(output_path)))
    print("=" * 65)


if __name__ == "__main__":
    main()
