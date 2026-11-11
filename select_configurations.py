
import csv
import json
from pathlib import Path

import numpy as np


# ==================================================
# EXPERIMENT SETTINGS
# ==================================================

SEED = 42

N_CONFIGS = 400
N_TRAIN = 320
N_TEST = 80

INPUT_PATH = Path(
    "data/configurations/all_candidate_configs.json"
)

OUTPUT_DIR = Path(
    "data/configurations/selected_balanced"
)

PARAMETERS = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "context_budget",
    "retriever",
    "reranker",
]

LEVELS = {
    "chunk_size": [128, 256, 384, 512, 768],
    "chunk_overlap": [0, 32, 64, 96],
    "top_k": [1, 3, 5, 7, 10],
    "similarity_threshold": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
    "context_budget": [256, 512, 1024, 2048, 4096],
    "retriever": ["bm25", "dense"],
    "reranker": [False, True],
}


# ==================================================
# LOAD CANDIDATE SPACE
# ==================================================

def load_candidates():

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        candidates = json.load(file)

    if len(candidates) != 12000:
        raise ValueError(
            f"Expected 12000 candidates, found {len(candidates)}"
        )

    candidate_map = {}

    for config in candidates:

        key = tuple(config[p] for p in PARAMETERS)

        if key in candidate_map:
            raise ValueError("Duplicate configurations in candidate space.")

        candidate_map[key] = config

    expected_count = (
        len(LEVELS["chunk_size"])
        * len(LEVELS["chunk_overlap"])
        * len(LEVELS["top_k"])
        * len(LEVELS["similarity_threshold"])
        * len(LEVELS["context_budget"])
        * len(LEVELS["retriever"])
        * len(LEVELS["reranker"])
    )

    if len(candidate_map) != expected_count:
        raise ValueError(
            "Candidate space does not match the expected parameter grid."
        )

    return candidates, candidate_map


# ==================================================
# CREATE BALANCED MARGINAL LEVELS
# ==================================================

def create_balanced_columns(rng):

    columns = {}

    for parameter, levels in LEVELS.items():

        base_count = N_CONFIGS // len(levels)
        remainder = N_CONFIGS % len(levels)

        counts = np.full(len(levels), base_count, dtype=int)

        # Randomly distribute the remainder.
        if remainder:
            extra_indices = rng.permutation(len(levels))[:remainder]
            counts[extra_indices] += 1

        values = []

        for level, count in zip(levels, counts):
            values.extend([level] * int(count))

        values = np.asarray(values, dtype=object)
        rng.shuffle(values)

        columns[parameter] = values

    return columns


# ==================================================
# CHECK UNIQUENESS
# ==================================================

def find_duplicate_indices(rows):

    seen = {}
    duplicates = []

    for index, row in enumerate(rows):

        if row in seen:
            duplicates.append(index)
        else:
            seen[row] = index

    return duplicates


# ==================================================
# RESOLVE DUPLICATES WITHOUT CHANGING MARGINALS
# ==================================================

def resolve_duplicates(columns, rng):

    rows = [
        tuple(columns[p][i] for p in PARAMETERS)
        for i in range(N_CONFIGS)
    ]

    max_attempts = 100000

    for attempt in range(max_attempts):

        duplicates = find_duplicate_indices(rows)

        if not duplicates:
            return columns

        duplicate_index = duplicates[0]

        other_indices = rng.permutation(N_CONFIGS)

        resolved = False

        for other_index in other_indices:

            other_index = int(other_index)

            if other_index == duplicate_index:
                continue

            for column_index in rng.permutation(len(PARAMETERS)):

                parameter = PARAMETERS[int(column_index)]

                value_a = columns[parameter][duplicate_index]
                value_b = columns[parameter][other_index]

                if value_a == value_b:
                    continue

                candidate_a = list(rows[duplicate_index])
                candidate_b = list(rows[other_index])

                candidate_a[column_index] = value_b
                candidate_b[column_index] = value_a

                candidate_a = tuple(candidate_a)
                candidate_b = tuple(candidate_b)

                if candidate_a == candidate_b:
                    continue

                existing = set(rows)
                existing.discard(rows[duplicate_index])
                existing.discard(rows[other_index])

                if (
                    candidate_a not in existing
                    and candidate_b not in existing
                ):

                    columns[parameter][duplicate_index] = value_b
                    columns[parameter][other_index] = value_a

                    rows[duplicate_index] = candidate_a
                    rows[other_index] = candidate_b

                    resolved = True
                    break

            if resolved:
                break

        if not resolved:
            raise RuntimeError(
                "Could not resolve duplicate configurations. "
                "Try regenerating the randomized design."
            )

    raise RuntimeError("Maximum duplicate-resolution attempts exceeded.")


# ==================================================
# GENERATE SELECTED CONFIGURATIONS
# ==================================================

def select_configurations(candidate_map):

    rng = np.random.default_rng(SEED)

    for restart in range(100):

        columns = create_balanced_columns(rng)

        columns = resolve_duplicates(columns, rng)

        selected = []

        for i in range(N_CONFIGS):

            key = tuple(
                columns[p][i]
                for p in PARAMETERS
            )

            if key not in candidate_map:
                break

            config = candidate_map[key].copy()
            config["selection_id"] = f"balanced_lhs_{i + 1:03d}"

            selected.append(config)

        if len(selected) == N_CONFIGS:
            return selected

    raise RuntimeError(
        "Could not create 400 valid unique configurations."
    )


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

def create_split(selected):

    rng = np.random.default_rng(SEED)

    indices = rng.permutation(N_CONFIGS)

    train = []
    test = []

    for index in indices[:N_TRAIN]:

        config = selected[int(index)].copy()
        config["split"] = "train"
        train.append(config)

    for index in indices[N_TRAIN:]:

        config = selected[int(index)].copy()
        config["split"] = "test"
        test.append(config)

    return train, test


# ==================================================
# SAVE OUTPUTS
# ==================================================

def save_json(data, path):

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_csv(data, path):

    with open(path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(data[0].keys())
        )

        writer.writeheader()
        writer.writerows(data)


# ==================================================
# VALIDATE DISTRIBUTIONS
# ==================================================

def validate_selection(selected, train, test):

    keys = [
        tuple(config[p] for p in PARAMETERS)
        for config in selected
    ]

    assert len(keys) == N_CONFIGS
    assert len(set(keys)) == N_CONFIGS

    assert len(train) == N_TRAIN
    assert len(test) == N_TEST

    train_ids = {c["config_id"] for c in train}
    test_ids = {c["config_id"] for c in test}

    assert not train_ids.intersection(test_ids)

    for parameter, levels in LEVELS.items():

        counts = {
            str(level): sum(
                config[parameter] == level
                for config in selected
            )
            for level in levels
        }

        expected_max_difference = 1

        if max(counts.values()) - min(counts.values()) > expected_max_difference:
            raise AssertionError(
                f"Unbalanced distribution for {parameter}: {counts}"
            )

        print(f"\n{parameter}: {counts}")


# ==================================================
# MAIN
# ==================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading candidate configurations...")

    candidates, candidate_map = load_candidates()

    print("Candidate space:", len(candidates))

    print("\nGenerating balanced discrete LHS design...")

    selected = select_configurations(candidate_map)

    train, test = create_split(selected)

    validate_selection(selected, train, test)

    save_json(
        selected,
        OUTPUT_DIR / "selected_400_configs.json"
    )

    save_csv(
        selected,
        OUTPUT_DIR / "selected_400_configs.csv"
    )

    save_json(
        train,
        OUTPUT_DIR / "train_configs_320.json"
    )

    save_json(
        test,
        OUTPUT_DIR / "test_configs_80.json"
    )

    save_csv(
        train,
        OUTPUT_DIR / "train_configs_320.csv"
    )

    save_csv(
        test,
        OUTPUT_DIR / "test_configs_80.csv"
    )

    manifest = {
        "sampling_method": "Balanced randomized discrete LHS",
        "random_seed": SEED,
        "candidate_space_size": len(candidates),
        "selected_configurations": N_CONFIGS,
        "training_configurations": N_TRAIN,
        "test_configurations": N_TEST,
        "parameters": PARAMETERS,
        "questions_per_configuration": 100,
        "expected_question_configuration_evaluations": 40000
    }

    save_json(
        manifest,
        OUTPUT_DIR / "selection_manifest.json"
    )

    print("\n===================================")
    print("BALANCED LHS SELECTION SUCCESSFUL")
    print("===================================")

    print("Selected:", len(selected))
    print("Training:", len(train))
    print("Testing:", len(test))
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
