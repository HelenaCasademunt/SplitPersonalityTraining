"""
Shapley value analysis for cross-topic probe transfer.

Trains probes on all 2^n - 1 topic subsets and computes Shapley values
to measure each topic's marginal contribution to predicting other topics.
"""

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import wandb

from scripts.probes.probes import ProbeTrainer, CLASSIFICATION_THRESHOLD
import scripts.probes.extract_activations as extract_activations

# Override to use new activations
extract_activations.ACTIVATIONS_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "activations_new"


# =============================================================================
# CONFIG - Edit these values
# =============================================================================

EXPERIMENT_NAME = "qwen"
LAYERS = [20]
REG_STRENGTH = 100.0
ACTIVATION_SOURCE = "response"  # Options: "response", "response_last", "intervention", "intervention_last"

WANDB_PROJECT = "probe-shapley"
OUTPUT_DIR = Path(__file__).parent / "shapley_results" / ACTIVATION_SOURCE


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_topics() -> List[str]:
    """Get all topics from the activation metadata."""
    metadata_path = extract_activations.ACTIVATIONS_OUTPUT_DIR / EXPERIMENT_NAME / "train" / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    topics = set(s["topic"] for s in metadata["samples"] if s.get("topic"))
    return sorted(topics)


def generate_all_subsets(topics: List[str]) -> List[FrozenSet[str]]:
    """Generate all non-empty subsets of topics."""
    subsets = []
    for r in range(1, len(topics) + 1):
        for combo in combinations(topics, r):
            subsets.append(frozenset(combo))
    return subsets


def subset_to_str(subset: FrozenSet[str]) -> str:
    """Convert subset to a short string for logging."""
    return "+".join(sorted(subset))


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate_subset(
    train_topics: FrozenSet[str],
    all_topics: List[str],
) -> Dict[str, float]:
    """Train probe on subset and evaluate on all topics.

    Returns:
        Dict mapping topic -> accuracy
    """
    trainer = ProbeTrainer(
        experiment_name=EXPERIMENT_NAME,
        layers=LAYERS,
        reg_strength=REG_STRENGTH,
        activation_source=ACTIVATION_SOURCE,
    )

    # Train on the subset
    train_acc = trainer.train(include_topics=list(train_topics), save=False)

    # Evaluate on each topic
    results = {}
    for topic in all_topics:
        if topic in train_topics:
            # Topic was in training set - only use val set
            acts, labels = trainer.load_activations_and_labels("val", include_topics=[topic])
        else:
            # Topic was NOT in training set - use train + val combined
            train_acts, train_labels = trainer.load_activations_and_labels("train", include_topics=[topic])
            val_acts, val_labels = trainer.load_activations_and_labels("val", include_topics=[topic])
            acts = np.concatenate([train_acts, val_acts])
            labels = np.concatenate([train_labels, val_labels])

        preds = trainer.probe.predict(acts)
        acc = np.mean((preds > CLASSIFICATION_THRESHOLD) == (labels > CLASSIFICATION_THRESHOLD))
        results[topic] = float(acc)

    return results, train_acc


def run_all_subsets() -> Dict[Tuple[FrozenSet[str], str], float]:
    """Train and evaluate all topic subsets.

    Returns:
        Dict mapping (subset, eval_topic) -> accuracy
    """
    topics = get_all_topics()
    subsets = generate_all_subsets(topics)

    print(f"Found {len(topics)} topics: {topics}")
    print(f"Generated {len(subsets)} non-empty subsets")
    print()

    all_results = {}

    for i, subset in enumerate(subsets):
        subset_str = subset_to_str(subset)
        print(f"[{i+1}/{len(subsets)}] Training on: {subset_str}")

        wandb.init(
            project=WANDB_PROJECT,
            name=f"subset_{len(subset)}_{subset_str[:50]}",
            config={
                "experiment_name": EXPERIMENT_NAME,
                "train_topics": sorted(subset),
                "n_train_topics": len(subset),
                "layers": LAYERS,
                "reg_strength": REG_STRENGTH,
                "activation_source": ACTIVATION_SOURCE,
            },
            reinit=True,
        )

        eval_results, train_acc = train_and_evaluate_subset(subset, topics)

        # Log results
        log_dict = {"train_acc": train_acc}
        for topic, acc in eval_results.items():
            log_dict[f"eval_{topic}"] = acc
            all_results[(subset, topic)] = acc

        wandb.log(log_dict)
        wandb.finish()

        # Print summary
        avg_acc = np.mean(list(eval_results.values()))
        print(f"  Train acc: {train_acc:.4f}, Avg eval acc: {avg_acc:.4f}")

    return all_results, topics


# =============================================================================
# SHAPLEY VALUE COMPUTATION
# =============================================================================

def compute_shapley_matrix(
    results: Dict[Tuple[FrozenSet[str], str], float],
    topics: List[str],
) -> np.ndarray:
    """Compute Shapley value matrix.

    Returns:
        9x9 matrix where entry [i,j] = Shapley value of topic i for predicting topic j
    """
    n = len(topics)
    topic_to_idx = {t: i for i, t in enumerate(topics)}
    shapley_matrix = np.zeros((n, n))

    # For each (train_topic, eval_topic) pair
    for train_topic in topics:
        train_idx = topic_to_idx[train_topic]

        for eval_topic in topics:
            eval_idx = topic_to_idx[eval_topic]

            # Compute Shapley value: average marginal contribution over all coalitions
            shapley_value = 0.0
            other_topics = [t for t in topics if t != train_topic]

            # Iterate over all subsets S of other topics
            for r in range(len(other_topics) + 1):
                for combo in combinations(other_topics, r):
                    S = frozenset(combo)
                    S_with_A = S | {train_topic}

                    # Get performance with and without train_topic
                    perf_with = results.get((S_with_A, eval_topic), 0.0)
                    perf_without = results.get((S, eval_topic), 0.5) if S else 0.5  # baseline 0.5 for empty set

                    marginal = perf_with - perf_without

                    # Shapley weight: |S|! * (n-|S|-1)! / n!
                    s = len(S)
                    weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)

                    shapley_value += weight * marginal

            shapley_matrix[train_idx, eval_idx] = shapley_value

    return shapley_matrix


def compute_ceiling(
    results: Dict[Tuple[FrozenSet[str], str], float],
    topics: List[str],
) -> Dict[str, float]:
    """Compute ceiling performance for each topic (train on all, eval on val)."""
    all_topics = frozenset(topics)
    ceiling = {}
    for topic in topics:
        ceiling[topic] = results.get((all_topics, topic), 0.0)
    return ceiling


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run all subset trainings
    results, topics = run_all_subsets()

    # Save raw results
    # Convert frozenset keys to strings for JSON
    results_json = {
        f"{subset_to_str(k[0])}|{k[1]}": v
        for k, v in results.items()
    }
    with open(OUTPUT_DIR / "raw_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Compute Shapley matrix
    shapley_matrix = compute_shapley_matrix(results, topics)
    np.save(OUTPUT_DIR / "shapley_matrix.npy", shapley_matrix)

    # Save topic order for reference
    with open(OUTPUT_DIR / "topics.json", "w") as f:
        json.dump(topics, f, indent=2)

    # Compute ceiling
    ceiling = compute_ceiling(results, topics)
    with open(OUTPUT_DIR / "ceiling.json", "w") as f:
        json.dump(ceiling, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SHAPLEY VALUE SUMMARY")
    print("=" * 70)

    print("\nCeiling performance (train on all, eval on val):")
    for topic, acc in sorted(ceiling.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {acc:.4f}")

    print("\nShapley matrix saved to:", OUTPUT_DIR / "shapley_matrix.npy")
    print("Run plot_shapley.py to generate visualizations.")

    return results, shapley_matrix, ceiling, topics


if __name__ == "__main__":
    main()
