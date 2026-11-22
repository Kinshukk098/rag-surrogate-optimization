
import csv
import json
from itertools import product
from pathlib import Path

PARAMETERS = {
    "chunk_size": [128, 256, 384, 512, 768],
    "chunk_overlap": [0, 32, 64, 96],
    "top_k": [1, 3, 5, 7, 10],
    "similarity_threshold": [0.00, 0.05, 0.10, 0.15, 0.20, 0.35],
    "context_budget": [256, 512, 1024, 2048, 4096],
    "retriever": ["bm25", "dense"],
    "reranker": [False, True],
}

OUTPUT_DIR = Path("data/configurations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

keys = list(PARAMETERS.keys())
all_configs = []

for values in product(*(PARAMETERS[key] for key in keys)):
    config = dict(zip(keys, values))

    # Overlap must be smaller than chunk size.
    if config["chunk_overlap"] >= config["chunk_size"]:
        continue

    config["config_id"] = f"config_{len(all_configs) + 1:05d}"
    all_configs.append(config)

# Save complete candidate space.
json_path = OUTPUT_DIR / "all_candidate_configs.json"
csv_path = OUTPUT_DIR / "all_candidate_configs.csv"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_configs, f, indent=2)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["config_id"] + keys)
    writer.writeheader()
    writer.writerows(all_configs)

print(f"Total valid configurations: {len(all_configs)}")
print(f"JSON saved: {json_path}")
print(f"CSV saved: {csv_path}")
print("\nFirst 5 configurations:")

for config in all_configs[:5]:
    print(config)
