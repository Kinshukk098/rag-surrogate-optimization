
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
    record_failure,
)

BASE = "data/processed/experiments/repeated_runs"

SELECTIONS_PATH = (
    "data/processed/experiments/search_results/"
    "selected_search_configurations_top5.csv"
)

OUTPUT_PATH = f"{BASE}/repeated_question_results.csv"
FAILED_PATH = f"{BASE}/repeated_failed_runs.csv"

SELECTED_CONFIG_IDS = [
    "config_00538",
    "config_09514",
    "config_04265",
    "config_00095",
    "config_10289",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_configurations(path):
    configurations = {}

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            config_id = row["config_id"]

            if config_id not in SELECTED_CONFIG_IDS:
                continue

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
                "reranker": (
                    row["reranker"].strip().lower() == "true"
                ),
            }

            if config_id in configurations:
                if configurations[config_id] != config:
                    raise ValueError(
                        f"Inconsistent configuration: {config_id}"
                    )
            else:
                configurations[config_id] = config

    missing = set(SELECTED_CONFIG_IDS) - set(configurations)

    if missing:
        raise ValueError(
            f"Selected configurations not found: {sorted(missing)}"
        )

    return [configurations[cid] for cid in SELECTED_CONFIG_IDS]


def load_completed_keys(path):
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
                row["question_id"],
                row["repeat_id"],
            ))

    return completed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-repeats", type=int, default=None)

    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--failed-path", default=FAILED_PATH)

    args = parser.parse_args()

    settings = load_json("research_config.json")

    repeat_settings = settings["surrogate_experiment"]["repeated_runs"]

    if not repeat_settings["enabled"]:
        raise ValueError("Repeated runs are disabled in research_config.json")

    n_repeats = repeat_settings["n_repeats"]

    if args.max_repeats is not None:
        n_repeats = min(n_repeats, args.max_repeats)

    random_seed = settings["experiment"].get("random_seed", 42)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dataset_name = "hotpotqa"
    split = "repeated"

    dataset_settings = settings["datasets"][dataset_name]

    questions = load_json(dataset_settings["questions_path"])
    documents = load_json(dataset_settings["corpus_path"])

    if args.max_questions is not None:
        questions = questions[:args.max_questions]

    configurations = load_configurations(SELECTIONS_PATH)

    if args.max_configs is not None:
        configurations = configurations[:args.max_configs]

    llm_settings = settings["llm"]
    experiment_settings = settings["experiment"]

    max_retries = experiment_settings.get("max_retries", 3)
    retry_delay = experiment_settings.get("retry_delay_seconds", 5)
    retrieval_k = experiment_settings.get("retrieval_k", 20)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    completed = load_completed_keys(args.output_path)

    print("=" * 65)
    print("REPEATED-RUN EXPERIMENT")
    print("Dataset:", dataset_name)
    print("Configurations:", len(configurations))
    print("Repeats:", n_repeats)
    print("Questions per configuration per repeat:", len(questions))
    print("Existing completed records:", len(completed))
    print("Expected records:", len(configurations) * n_repeats * len(questions))
    print("Output:", args.output_path)
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

        print("\n" + "#" * 65)
        print(
            f"CONFIGURATION {config_index}/{len(configurations)}: "
            f"{config_id}"
        )
        print("Parameters:", config)

        for repeat_number in range(1, n_repeats + 1):

            repeat_id = f"repeat_{repeat_number}"

            pending_questions = [
                question
                for question in questions
                if (
                    dataset_name,
                    split,
                    config_id,
                    str(question["question_id"]),
                    repeat_id,
                ) not in completed
            ]

            print("\n" + "-" * 60)
            print(f"{config_id} | {repeat_id}")
            print("Remaining questions:", len(pending_questions))

            if not pending_questions:
                print("Already completed. Skipping.")
                continue

            run_id = (
                f"{experiment_settings['experiment_id']}_"
                f"{dataset_name}_{config_id}_{repeat_id}"
            )

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

                record_failure(args.failed_path, {
                    "dataset": dataset_name,
                    "split": split,
                    "config_id": config_id,
                    "repeat_id": repeat_id,
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

                        row["repeat_id"] = repeat_id

                        append_csv(args.output_path, row)

                        completed.add((
                            dataset_name,
                            split,
                            config_id,
                            question_id,
                            repeat_id,
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
                    record_failure(args.failed_path, {
                        "dataset": dataset_name,
                        "split": split,
                        "config_id": config_id,
                        "repeat_id": repeat_id,
                        "question_id": question_id,
                        "error": "All retry attempts failed",
                        "stage": "question_evaluation",
                    })

                    print("  Recorded as failed. Continuing.")

    print("\n" + "=" * 65)
    print("REPEATED-RUN EVALUATION FINISHED")
    print("Completed records:", len(load_completed_keys(args.output_path)))
    print("Output:", args.output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()
