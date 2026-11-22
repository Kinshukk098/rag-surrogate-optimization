
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import qmc
from scipy.spatial.distance import cdist

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    create_preprocessor,
    load_data,
)
from .models import create_models
from .experiments import load_experiment_config


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data/processed/experiments/hotpotqa_config_results.csv"
)

CANDIDATE_PATH = (
    ROOT / "data/configurations/search_candidate_pool_1000.json"
)

OUTPUT_PATH = (
    ROOT / "data/processed/experiments/search_results"
    / "selected_search_configurations_top5.csv"
)

SEED = 42
SELECTION_BUDGET = 5

TARGET_DIRECTIONS = {
    "mean_answer_f1": "max",
    "mean_answer_em": "max",
    "mean_latency": "min",
    "mean_total_tokens": "min",
}


def select_configurations():

    random.seed(SEED)
    np.random.seed(SEED)

    # Load evaluated configurations.
    train_df, _ = load_data(DATA_PATH)

    # Load unevaluated candidate pool.
    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
        candidates = pd.DataFrame(json.load(f))

    if len(candidates) < SELECTION_BUDGET:
        raise ValueError("Candidate pool is smaller than selection budget.")

    X_train = train_df[FEATURE_COLUMNS]
    X_candidates = candidates[FEATURE_COLUMNS]

    experiment_config = load_experiment_config()

    model_templates = create_models(
        random_state=SEED,
        model_config=experiment_config["models"],
    )

    results = []

    # Process all four optimization targets independently.
    for target in TARGET_COLUMNS:

        direction = TARGET_DIRECTIONS[target]

        print(f"\n========== TARGET: {target} ==========")

        y_train = train_df[target].to_numpy()

        # --------------------------------------------------
        # 1. Random Search
        # --------------------------------------------------

        random_indices = random.sample(
            range(len(candidates)),
            SELECTION_BUDGET
        )

        # --------------------------------------------------
        # 2. True 7-dimensional LHS selection
        # --------------------------------------------------

        lhs_features = candidates[FEATURE_COLUMNS].copy()

        lhs_features["retriever"] = (
            lhs_features["retriever"].map({
                "bm25": 0,
                "dense": 1
            })
        )

        lhs_features["reranker"] = (
            lhs_features["reranker"].astype(int)
        )

        # Normalize every feature to [0, 1].
        mins = lhs_features.min()
        maxs = lhs_features.max()

        X_normalized = (
            (lhs_features - mins) /
            (maxs - mins).replace(0, 1)
        ).to_numpy(dtype=float)

        sampler = qmc.LatinHypercube(
            d=len(FEATURE_COLUMNS),
            seed=SEED
        )

        lhs_samples = sampler.random(n=SELECTION_BUDGET)

        # Map LHS points to distinct real configurations.
        distances = cdist(lhs_samples, X_normalized)

        lhs_indices = []
        available = set(range(len(candidates)))

        for row in range(SELECTION_BUDGET):

            ordered = np.argsort(distances[row])

            chosen = next(
                (
                    int(idx)
                    for idx in ordered
                    if int(idx) in available
                ),
                None
            )

            if chosen is None:
                raise RuntimeError(
                    "Could not find enough unique LHS candidates."
                )

            lhs_indices.append(chosen)
            available.remove(chosen)

        # Store baseline selections.
        strategies = {
            "random_search": random_indices,
            "lhs": lhs_indices,
        }

        # --------------------------------------------------
        # 3. Five surrogate-guided strategies
        # --------------------------------------------------

        for model_name, model_template in model_templates.items():

            print(f"Training {model_name} for {target}...")

            pipeline = Pipeline([
                ("preprocessor", create_preprocessor()),
                ("model", clone(model_template)),
            ])

            pipeline.fit(X_train, y_train)

            predictions = np.asarray(
                pipeline.predict(X_candidates)
            ).ravel()

            # Rank according to optimization direction.
            if direction == "max":
                ranked_indices = np.argsort(predictions)[::-1]
            else:
                ranked_indices = np.argsort(predictions)

            strategies[f"{model_name}_guided"] = (
                ranked_indices[:SELECTION_BUDGET].tolist()
            )

        # --------------------------------------------------
        # 4. Save selected configurations
        # --------------------------------------------------

        for strategy, indices in strategies.items():

            for rank, index in enumerate(indices, start=1):

                config = candidates.iloc[index].to_dict()

                results.append({
                    "target": target,
                    "direction": direction,
                    "strategy": strategy,
                    "selection_rank": rank,
                    **config,
                })

        print("Strategies:", len(strategies))
        print("Selections per strategy:", SELECTION_BUDGET)

    # --------------------------------------------------
    # 5. Export results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== SELECTION COMPLETED ==========")
    print("Total selections:", len(results_df))
    print("Expected selections:", 4 * 7 *  SELECTION_BUDGET)
    print("Unique configurations:", results_df["config_id"].nunique())
    print("Saved to:", OUTPUT_PATH)

    print("\nSelections per target and strategy:")

    print(
        results_df.groupby(
            ["target", "strategy"]
        ).size()
    )


if __name__ == "__main__":
    select_configurations()
