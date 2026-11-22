
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

from run_batch_experiment import (
    evaluate_question,
    append_csv,
    load_completed_keys,
    record_failure,
)


SELECTIONS_PATH = "data/processed/experiments/search_results/selected_search_configurations_reduced.csv"

OUTPUT_PATH = "data/processed/experiments/search_results/hotpotqa_search_question_results_reduced.csv"

FAILED_PATH = "data/processed/experiments/search_results/hotpotqa_search_failed_runs_reduced.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_unique_configurations(path):
    """Load 560 strategy selections as unique RAG configurations."""

    configurations = {}

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            config_id = row["config_id"]

            config = {
                "config_id": config_id,
                "chunk_size": int(row["chunk_size"]),
                "chunk_overlap": int(row["chunk_overlap"]),
                "top_k": int(row["top_k"]),
                "similarity_threshold": float(
                    row["similarity_threshold"]
                ),
                "context_budget": int(row["context_budget"]),
                "retriever": row["retriever"],
                "reranker": row["reranker"].strip().lower() == "true",
            }

            if config_id in configurations:
                if configurations[config_id] != config:
                    raise ValueError(
                        f"Inconsistent configuration: {config_id}"
                    )
            else:
                configurations[config_id] = config

    return list(configurations.values())


def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limit configurations for a small test",
    )

    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Limit questions per configuration for a small test",
    )

    args = parser.parse_args()

    settings = load_json("research_config.json")

    random_seed = settings["experiment"].get("random_seed", 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dataset_name = "hotpotqa"
    split = "search"

    dataset_settings = settings["datasets"][dataset_name]

    questions = load_json(dataset_settings["questions_path"])
    documents = load_json(dataset_settings["corpus_path"])

    if args.question_limit is not None:
        questions = questions[:args.question_limit]

    configurations = load_unique_configurations(SELECTIONS_PATH)

    if args.max_configs is not None:
        configurations = configurations[:args.max_configs]

    llm_settings = settings["llm"]
    experiment_settings = settings["experiment"]

    retrieval_k = experiment_settings.get("retrieval_k", 20)
    max_retries = experiment_settings.get("max_retries", 3)
    retry_delay = experiment_settings.get("retry_delay_seconds", 5)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    completed = load_completed_keys(OUTPUT_PATH)

    print("=" * 65)
    print("RAG SEARCH EVALUATION")
    print("Dataset:", dataset_name)
    print("Unique configurations:", len(configurations))
    print("Questions per configuration:", len(questions))
    print("Existing completed records:", len(completed))
    print("Output:", OUTPUT_PATH)
    print("=" * 65)

    generator = GroqAnswerGenerator(
        model_name=llm_settings["model"],
        temperature=llm_settings.get("temperature", 0),
        max_completion_tokens=llm_settings.get(
            "max_completion_tokens", 256
        ),
        seed=llm_settings.get("seed", 42),
    )

    for config_index, config in enumerate(configurations, start=1):
        config_id = config["config_id"]
        config["retrieval_k"] = retrieval_k

        run_id = (
            f"{experiment_settings['experiment_id']}_"
            f"{dataset_name}_search_{config_id}"
        )

        pending_questions = [
            question
            for question in questions
            if (
                dataset_name,
                split,
                config_id,
                str(question["question_id"]),
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
            start = time.perf_counter()
            pipeline = RAGPipeline(config, documents)
            initialization_time = time.perf_counter() - start

            print(
                f"Pipeline initialization: "
                f"{initialization_time:.2f}s"
            )

        except Exception as error:
            print("PIPELINE INITIALIZATION FAILED:", repr(error))

            record_failure(FAILED_PATH, {
                "dataset": dataset_name,
                "split": split,
                "config_id": config_id,
                "question_id": "",
                "error": repr(error),
                "stage": "pipeline_initialization",
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
                        split=split,
                        config_id=config_id,
                        run_id=run_id,
                    )

                    append_csv(OUTPUT_PATH, row)

                    completed.add((
                        dataset_name,
                        split,
                        config_id,
                        question_id,
                    ))

                    print(
                        f"  F1={row['answer_f1']:.3f} | "
                        f"EM={row['answer_em']:.3f} | "
                        f"Latency={row['total_latency_seconds']:.2f}s | "
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
                record_failure(FAILED_PATH, {
                    "dataset": dataset_name,
                    "split": split,
                    "config_id": config_id,
                    "question_id": question_id,
                    "error": "All retry attempts failed",
                    "stage": "question_evaluation",
                })

                print("  Recorded as failed. Continuing.")

    print("\n" + "=" * 65)
    print("SEARCH EVALUATION FINISHED")
    print("Output:", OUTPUT_PATH)
    print("Completed records:", len(load_completed_keys(OUTPUT_PATH)))
    print("=" * 65)


if __name__ == "__main__":
    main()
