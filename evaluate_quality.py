
import os
import csv
import json
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.rag.metrics import exact_match, token_f1


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

INPUT_PATH = "data/processed/ragas_pilot_input.csv"
OUTPUT_PATH = "data/processed/quality_evaluation.csv"

JUDGE_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ABSTENTION_PHRASES = [
    "insufficient information",
    "cannot be determined",
    "not enough information",
    "i don't know",
    "unable to determine",
    "information is insufficient",
    "not enough evidence"
]


# ============================================================
# 2. INITIALIZE MODELS
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

print("Loading semantic similarity model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")


# ============================================================
# 3. SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(prediction, reference):

    embeddings = embedding_model.encode(
        [prediction, reference],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    score = float(np.dot(embeddings[0], embeddings[1]))

    return max(0.0, min(1.0, score))


# ============================================================
# 4. ABSTENTION DETECTION
# ============================================================

def is_abstention(answer):

    normalized = answer.strip().lower()

    return any(
        phrase in normalized
        for phrase in ABSTENTION_PHRASES
    )


# ============================================================
# 5. PARSE AND VALIDATE JUDGE RESPONSE
# ============================================================

def parse_judge_response(raw_response):

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        raw_response,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace("```", "").strip()

    match = re.search(
        r"\{.*\}",
        cleaned,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            f"Invalid judge response: {raw_response}"
        )

    result = json.loads(match.group())

    required_keys = [
        "context_sufficiency",
        "answer_support",
        "reason"
    ]

    for key in required_keys:
        if key not in result:
            raise ValueError(
                f"Missing key '{key}' in judge response."
            )

    valid_sufficiency = [
        "sufficient",
        "insufficient",
        "partial"
    ]

    valid_support = [
        "supported",
        "unsupported",
        "partially_supported",
        "not_applicable"
    ]

    if result["context_sufficiency"] not in valid_sufficiency:
        raise ValueError("Invalid context_sufficiency.")

    if result["answer_support"] not in valid_support:
        raise ValueError("Invalid answer_support.")

    return result


# ============================================================
# 6. PROGRAMMATIC FAITHFULNESS SCORING
# ============================================================

def calculate_faithfulness(
    context_sufficiency,
    answer_support,
    abstained
):

    # Case A: Generated answer is an abstention.

    if abstained:

        if answer_support != "not_applicable":
            raise ValueError(
                "Abstention must have answer_support='not_applicable'."
            )

        # Partial means relevant information exists,
        # but the context cannot establish a complete answer.
        # Therefore abstention is justified.

        if context_sufficiency in [
            "insufficient",
            "partial"
        ]:

            return True, 1.0

        # If sufficient evidence exists, abstaining is
        # considered an inappropriate refusal.

        return False, 0.0

    # Case B: Generated answer contains a factual claim.

    if answer_support == "supported":
        return None, 1.0

    if answer_support == "partially_supported":
        return None, 0.5

    if answer_support == "unsupported":
        return None, 0.0

    raise ValueError(
        "Invalid answer_support for a non-abstention answer."
    )


# ============================================================
# 7. LLM-BASED EVIDENCE EVALUATION
# ============================================================

def evaluate_faithfulness(question, answer, contexts):

    abstained = is_abstention(answer)

    if not contexts:

        return {
            "context_sufficiency": "insufficient",
            "answer_support": (
                "not_applicable" if abstained else "unsupported"
            ),
            "abstention_appropriate": (
                True if abstained else None
            ),
            "faithfulness_score": (
                1.0 if abstained else 0.0
            ),
            "faithfulness_reason": "No retrieved contexts.",
            "judge_raw_response": ""
        }

    context_text = "\n\n".join(
        f"Context {i + 1}: {context}"
        for i, context in enumerate(contexts)
    )

    prompt = f"""
You are a strict evidence evaluator for a
Retrieval-Augmented Generation system.

Evaluate the generated answer using ONLY the question
and retrieved contexts.

QUESTION:
{question}

RETRIEVED CONTEXTS:
{context_text}

GENERATED ANSWER:
{answer}

Evaluate these fields:

1. context_sufficiency:

- sufficient:
  The retrieved contexts contain enough information
  to establish the answer to the question.

- insufficient:
  The retrieved contexts do not contain the information
  needed to answer the question.

- partial:
  The contexts contain some relevant information,
  but do not establish a complete answer.

2. answer_support:

- supported:
  The generated factual answer agrees with and is
  supported by the retrieved evidence.

- unsupported:
  The answer contradicts the evidence or makes
  an unsupported factual claim.

- partially_supported:
  Some claims are supported but others are not.

- not_applicable:
  Use this only if the generated answer abstains.

IMPORTANT RULES:

1. Evaluate the actual conclusion of the generated answer,
   not merely whether its words or entities appear in context.

2. For numerical questions, compare the actual numbers.

3. For comparison questions, determine which option
   the evidence establishes as correct.

4. If the generated answer contradicts the conclusion
   established by the contexts, mark it unsupported.

5. An entity being mentioned in a context does not mean
   selecting that entity is a supported answer.

6. If the answer abstains, use not_applicable for
   answer_support. Do not decide abstention appropriateness;
   that will be calculated separately.

7. Do not use external knowledge.

8. Do not use the gold answer.

9. Keep the reason under 25 words.

Return ONLY valid JSON in this format:

{{
    "context_sufficiency": "sufficient",
    "answer_support": "supported",
    "reason": "The answer agrees with the retrieved evidence."
}}
"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evidence evaluator. "
                    "Return valid JSON only. "
                    "Evaluate evidence and answer consistency."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_completion_tokens=300,
        seed=42
    )

    raw_response = response.choices[0].message.content or ""

    result = parse_judge_response(raw_response)

    context_sufficiency = result["context_sufficiency"]
    answer_support = result["answer_support"]

    # Enforce consistency between answer type and support label.

    if abstained and answer_support != "not_applicable":
        raise ValueError(
            "Judge must mark abstention as not_applicable."
        )

    if not abstained and answer_support == "not_applicable":
        raise ValueError(
            "Non-abstention answer cannot be not_applicable."
        )

    # Calculate final faithfulness in Python,
    # rather than trusting the LLM's numerical score.

    abstention_appropriate, faithfulness_score = (
        calculate_faithfulness(
            context_sufficiency,
            answer_support,
            abstained
        )
    )

    return {
        "context_sufficiency": context_sufficiency,
        "answer_support": answer_support,
        "abstention_appropriate": abstention_appropriate,
        "faithfulness_score": faithfulness_score,
        "faithfulness_reason": result["reason"],
        "judge_raw_response": raw_response
    }


# ============================================================
# 8. MAIN EVALUATION
# ============================================================

def main():

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "run_id",
        "question_id",
        "question",
        "gold_answer",
        "predicted_answer",
        "retrieved_contexts"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing input columns: {missing_columns}"
        )

    results = []

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    print("\nStarting answer quality evaluation...")
    print("Questions:", len(df))

    for index, row in df.iterrows():

        question = str(row["question"])
        gold_answer = str(row["gold_answer"])
        predicted_answer = str(row["predicted_answer"])

        contexts = json.loads(row["retrieved_contexts"])

        if not isinstance(contexts, list):
            raise ValueError(
                f"Retrieved contexts must be a list for question "
                f"{row['question_id']}"
            )

        print(
            f"\n[{index + 1}/{len(df)}] "
            f"Question ID: {row['question_id']}"
        )

        # Answer correctness metrics.

        em = exact_match(
            predicted_answer,
            gold_answer
        )

        f1 = token_f1(
            predicted_answer,
            gold_answer
        )

        similarity = semantic_similarity(
            predicted_answer,
            gold_answer
        )

        # Evidence and faithfulness evaluation.

        faithfulness = evaluate_faithfulness(
            question,
            predicted_answer,
            contexts
        )

        result = {
            "run_id": row["run_id"],
            "question_id": row["question_id"],
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "answer_em": em,
            "answer_f1": f1,
            "semantic_similarity": similarity,
            "context_sufficiency": faithfulness["context_sufficiency"],
            "answer_support": faithfulness["answer_support"],
            "abstention_appropriate": faithfulness["abstention_appropriate"],
            "faithfulness_score": faithfulness["faithfulness_score"],
            "faithfulness_reason": faithfulness["faithfulness_reason"],
            "judge_raw_response": faithfulness["judge_raw_response"],
            "llm_judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL
        }

        results.append(result)

        # Save after each completed question.

        output_df = pd.DataFrame(results)

        output_df.to_csv(
            OUTPUT_PATH,
            index=False,
            quoting=csv.QUOTE_ALL
        )

        print(f"  EM: {em:.3f}")
        print(f"  F1: {f1:.3f}")
        print(f"  Semantic similarity: {similarity:.3f}")

        print(
            "  Context sufficiency:",
            faithfulness["context_sufficiency"]
        )

        print(
            "  Answer support:",
            faithfulness["answer_support"]
        )

        print(
            "  Abstention appropriate:",
            faithfulness["abstention_appropriate"]
        )

        print(
            "  Faithfulness:",
            faithfulness["faithfulness_score"]
        )

    # Final summary.

    output_df = pd.DataFrame(results)

    print("\n========== QUALITY EVALUATION SUMMARY ==========")

    for metric in [
        "answer_em",
        "answer_f1",
        "semantic_similarity",
        "faithfulness_score"
    ]:

        print(
            f"Mean {metric}: "
            f"{output_df[metric].mean():.4f}"
        )

    print("\nResults saved to:", OUTPUT_PATH)


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
