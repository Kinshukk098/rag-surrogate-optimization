
import json
import random
from pathlib import Path
from datasets import load_dataset

# Reproducibility
SEED = 42
NUM_QUESTIONS = 100

random.seed(SEED)

# Directories
SPLIT_DIR = Path("data/splits")
PROCESSED_DIR = Path("data/processed")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading HotpotQA validation dataset...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation"
)

print("Total validation questions:", len(dataset))

# Select 100 fixed questions
selected_indices = random.sample(
    range(len(dataset)),
    NUM_QUESTIONS
)

selected_questions = [
    dataset[i] for i in selected_indices
]

# Convert dataset records to JSON-compatible dictionaries
questions = []

for index, question in enumerate(selected_questions, start=1):
    questions.append({
        "question_id": index,
        "question": question["question"],
        "answer": question["answer"],
        "supporting_facts": {
            "title": question["supporting_facts"]["title"],
            "sent_id": question["supporting_facts"]["sent_id"]
        }
    })

# Save selected questions
questions_path = SPLIT_DIR / "hotpotqa_100_questions.json"

with open(questions_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

print("Saved questions:", len(questions))

# Build corpus from candidate contexts of ALL selected questions
corpus = []
seen_documents = set()

for question in selected_questions:

    context = question["context"]

    titles = context["title"]
    sentences = context["sentences"]

    for title, sentence_list in zip(titles, sentences):

        text = " ".join(sentence_list).strip()

        if not text:
            continue

        # Deduplicate identical title-text pairs
        document_key = (title, text)

        if document_key in seen_documents:
            continue

        seen_documents.add(document_key)

        corpus.append({
            "title": title,
            "text": text
        })

# Save corpus
corpus_path = PROCESSED_DIR / "hotpotqa_100_corpus.json"

with open(corpus_path, "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)

print("Unique corpus documents:", len(corpus))
print("Saved corpus:", corpus_path)
print("Saved questions:", questions_path)
print("Dataset preparation completed.")
