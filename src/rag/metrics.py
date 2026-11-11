
import re
import string
from collections import Counter


def normalize_answer(text):
    """Normalize an answer using standard QA evaluation conventions."""
    text = text.lower()
    text = "".join(
        char for char in text if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, ground_truth):
    return float(
        normalize_answer(prediction) ==
        normalize_answer(ground_truth)
    )


def token_f1(prediction, ground_truth):
    predicted_tokens = normalize_answer(prediction).split()
    actual_tokens = normalize_answer(ground_truth).split()

    common = Counter(predicted_tokens) & Counter(actual_tokens)
    common_count = sum(common.values())

    if not predicted_tokens or not actual_tokens:
        return float(predicted_tokens == actual_tokens)

    if common_count == 0:
        return 0.0

    precision = common_count / len(predicted_tokens)
    recall = common_count / len(actual_tokens)

    return 2 * precision * recall / (precision + recall)
