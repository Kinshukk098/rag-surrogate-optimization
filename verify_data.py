
import json
from pathlib import Path

base = Path(__file__).parent

with open(base / "data/splits/hotpotqa_20_questions.json",
          encoding="utf-8") as f:
    questions = json.load(f)

with open(base / "data/processed/hotpotqa_corpus.json",
          encoding="utf-8") as f:
    documents = json.load(f)

document_titles = {doc["title"] for doc in documents}

total_supporting_facts = 0
found_supporting_titles = 0

for question in questions:
    facts = question["supporting_facts"]

    titles = facts["title"]

    for title in titles:
        total_supporting_facts += 1

        if title in document_titles:
            found_supporting_titles += 1

print("QUESTIONS:", len(questions))
print("CORPUS DOCUMENTS:", len(documents))
print("SUPPORTING FACT TITLES:", total_supporting_facts)
print("TITLES PRESENT IN CORPUS:", found_supporting_titles)

print(
    "Supporting-title coverage:",
    round(found_supporting_titles / max(total_supporting_facts, 1) * 100, 2),
    "%"
)
