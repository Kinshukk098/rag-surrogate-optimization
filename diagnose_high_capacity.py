
import os
import re
import json
import csv
import string
import time

from src.rag.pipeline import RAGPipeline
from src.rag.llm import GroqAnswerGenerator


# --------------------------------------------------
# PATHS
# --------------------------------------------------

QUESTIONS_PATH = "data/splits/hotpotqa_100_questions.json"
CORPUS_PATH = "data/processed/hotpotqa_100_corpus.json"

OUTPUT_PATH = (
    "data/processed/experiments/high_capacity_diagnostic.csv"
)


# --------------------------------------------------
# HIGH-CAPACITY DIAGNOSTIC CONFIGURATION
# --------------------------------------------------

config = {
    "chunk_size": 256,
    "chunk_overlap": 32,
    "top_k": 10,
    "retrieval_k": 100,
    "similarity_threshold": 0.0,
    "context_budget": 2048,
    "retriever": "bm25",
    "reranker": False
}


# --------------------------------------------------
# HOTPOTQA METRICS
# --------------------------------------------------

def normalize_answer(text):
    text = str(text).lower()
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def calculate_em(prediction, gold):
    prediction = normalize_answer(prediction)
    return float(prediction == normalize_answer(gold))


def calculate_f1(prediction, gold):
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)

    common = {}

    for token in prediction_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0

    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading HotpotQA questions and corpus...")

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)
questions = [questions[2]]

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus_data = json.load(f)

documents = (
    corpus_data
    if isinstance(corpus_data, list)
    else corpus_data["documents"]
)

print("Questions:", len(questions))
print("Corpus documents:", len(documents))


# --------------------------------------------------
# INITIALIZE RAG PIPELINE AND LLM
# --------------------------------------------------

print("\nInitializing high-capacity RAG pipeline...")

pipeline = RAGPipeline(config, documents)

generator = GroqAnswerGenerator(
    model_name="qwen/qwen3.8-27b",
    temperature=0,
    max_completion_tokens=256,
    seed=42
)


# --------------------------------------------------
# RUN FIRST THREE QUESTIONS
# --------------------------------------------------

results = []

for index, question_data in enumerate(questions[:3]):

    question = question_data["question"]

    support_titles = set(
        question_data["supporting_facts"]["title"]
    )

    gold_answer = question_data["answer"]

    print("\n" + "=" * 70)
    print(f"QUESTION {index + 1}: {question}")

    try:
        # Retrieval
        retrieval_start = time.perf_counter()

        retrieved_docs = pipeline.retrieve(question)

        retrieval_latency = (
            time.perf_counter() - retrieval_start
        )

        retrieved_titles = [
            doc.get("title", "")
            for doc in retrieved_docs
        ]

        retrieved_title_set = set(retrieved_titles)

        found_retrieval = (
            support_titles.intersection(retrieved_title_set)
        )

        retrieval_recall = (
            len(found_retrieval) / len(support_titles)
            if support_titles else 0.0
        )

        # LLM generation and context construction
        generated = generator.generate(
            question=question,
            documents=retrieved_docs,
            context_budget=config["context_budget"]
        )

        predicted_answer = generated["answer"].strip()

        context_titles = set(
            generated["context_document_titles"]
        )

        found_context = support_titles.intersection(
            context_titles
        )

        context_recall = (
            len(found_context) / len(support_titles)
            if support_titles else 0.0
        )

        # Answer metrics
        gold_answers = (
            gold_answer
            if isinstance(gold_answer, list)
            else [gold_answer]
        )

        answer_em = max(
            calculate_em(predicted_answer, gold)
            for gold in gold_answers
        )

        answer_f1 = max(
            calculate_f1(predicted_answer, gold)
            for gold in gold_answers
        )

        stats = pipeline.last_retrieval_stats

        row = {
            "question_id": question_data.get(
                "question_id", index + 1
            ),
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,

            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "retrieval_k": config["retrieval_k"],
            "top_k": config["top_k"],
            "context_budget": config["context_budget"],
            "similarity_threshold": config["similarity_threshold"],

            "candidate_count": stats["candidate_count"],
            "filtered_count": stats["filtered_count"],
            "returned_count": stats["returned_count"],

            "supporting_title_recall": retrieval_recall,
            "supporting_context_title_recall": context_recall,

            "context_documents_used": generated[
                "context_documents_used"
            ],
            "context_tokens": generated["context_tokens"],

            "answer_em": answer_em,
            "answer_f1": answer_f1,

            "retrieval_latency_seconds": retrieval_latency,
            "context_construction_latency_seconds": generated[
                "context_construction_latency_seconds"
            ],
            "generation_latency_seconds": generated[
                "generation_latency_seconds"
            ],

            "input_tokens": generated["input_tokens"],
            "output_tokens": generated["output_tokens"],
            "total_tokens": generated["total_tokens"],

            "retrieved_titles": json.dumps(retrieved_titles),
            "context_document_titles": json.dumps(
                generated["context_document_titles"]
            ),
            "actual_context": generated["context_text"]
        }

        results.append(row)

        print("Retrieved documents:", retrieved_titles)
        print("Context documents:", generated["context_document_titles"])
        print("Retrieval recall:", retrieval_recall)
        print("Context recall:", context_recall)
        print("Prediction:", predicted_answer)
        print("Gold:", gold_answer)
        print("F1:", answer_f1)
        print("EM:", answer_em)
        print("API tokens:", generated["total_tokens"])

    except Exception as error:
        print("ERROR:", repr(error))


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

if results:

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("Successful questions:", len(results))
    print("Average F1:", sum(r["answer_f1"] for r in results) / len(results))
    print("Average EM:", sum(r["answer_em"] for r in results) / len(results))
    print("Average retrieval recall:", sum(
        r["supporting_title_recall"] for r in results
    ) / len(results))
    print("Average context recall:", sum(
        r["supporting_context_title_recall"] for r in results
    ) / len(results))
    print("Saved:", OUTPUT_PATH)

else:
    print("No successful evaluations. Check the errors above.")
