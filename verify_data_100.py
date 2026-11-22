
import json

with open(
    "data/splits/hotpotqa_100_questions.json",
    encoding="utf-8"
) as f:
    questions = json.load(f)

with open(
    "data/processed/hotpotqa_100_corpus.json",
    encoding="utf-8"
) as f:
    corpus = json.load(f)

corpus_titles = {
    doc["title"] for doc in corpus
}

supporting_titles = set()

for question in questions:
    supporting_titles.update(
        question["supporting_facts"]["title"]
    )

available_titles = supporting_titles & corpus_titles

missing_titles = supporting_titles - corpus_titles

coverage = (
    len(available_titles) / len(supporting_titles) * 100
    if supporting_titles else 0
)

print("\nHOTPotQA 100 DATA VERIFICATION")
print("--------------------------------")
print("Questions:", len(questions))
print("Corpus documents:", len(corpus))
print("Unique supporting titles:", len(supporting_titles))
print("Available supporting titles:", len(available_titles))
print("Missing supporting titles:", len(missing_titles))
print(f"Supporting title coverage: {coverage:.2f}%")

if missing_titles:
    print("\nMissing titles:")
    for title in sorted(missing_titles):
        print("-", title)
