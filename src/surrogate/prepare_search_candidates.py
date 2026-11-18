
import json
import random
from pathlib import Path

SEED = 42
POOL_SIZE = 1000

all_path = Path("data/configurations/all_candidate_configs.json")
train_path = Path(
    "data/configurations/selected_balanced_v2/train_configs_320.json"
)
test_path = Path(
    "data/configurations/selected_balanced_v2/test_configs_80.json"
)

output_path = Path(
    "data/configurations/search_candidate_pool_1000.json"
)

with open(all_path, "r", encoding="utf-8") as f:
    all_configs = json.load(f)

with open(train_path, "r", encoding="utf-8") as f:
    train_configs = json.load(f)

with open(test_path, "r", encoding="utf-8") as f:
    test_configs = json.load(f)

# Exclude all previously evaluated configurations.
evaluated_ids = {
    config["config_id"]
    for config in train_configs + test_configs
}

unevaluated = [
    config for config in all_configs
    if config["config_id"] not in evaluated_ids
]

assert len(unevaluated) == 11600, (
    f"Expected 11600 candidates, found {len(unevaluated)}"
)

# Select a reproducible candidate pool.
rng = random.Random(SEED)
candidate_pool = rng.sample(unevaluated, POOL_SIZE)

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(candidate_pool, f, indent=2)

print("Total configuration space:", len(all_configs))
print("Previously evaluated:", len(evaluated_ids))
print("Unevaluated configurations:", len(unevaluated))
print("Candidate pool size:", len(candidate_pool))
print("Saved to:", output_path)
print("First candidate:", candidate_pool[0])
