"""
Cross-topic generalization sweep: train on all-but-one topic, evaluate on held-out topic.
"""

import json
from pathlib import Path

import numpy as np
import wandb

from scripts.probes.probes import ProbeTrainer, CLASSIFICATION_THRESHOLD
import scripts.probes.extract_activations as extract_activations

# Override to use new activations
extract_activations.ACTIVATIONS_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "activations_new"
ACTIVATIONS_OUTPUT_DIR = extract_activations.ACTIVATIONS_OUTPUT_DIR


# =============================================================================
# CONFIG
# =============================================================================

EXPERIMENT_NAME = "qwen"

# Fixed hyperparams (use best from previous sweep, or reasonable defaults)
LAYERS = [20]
REG_STRENGTH = 100.0
ACTIVATION_SOURCE = "intervention"

WANDB_PROJECT = "probe-cross-topic"


# =============================================================================
# GET TOPICS FROM DATA
# =============================================================================

def get_all_topics(experiment_name: str) -> list:
    metadata_path = ACTIVATIONS_OUTPUT_DIR / experiment_name / "train" / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    topics = set(s["topic"] for s in metadata["samples"] if s.get("topic"))
    return sorted(topics)


# =============================================================================
# SWEEP LOGIC
# =============================================================================

def run_cross_topic_sweep():
    topics = get_all_topics(EXPERIMENT_NAME)
    print(f"Found {len(topics)} topics: {topics}")
    print()

    results = []

    for held_out_topic in topics:
        train_topics = [t for t in topics if t != held_out_topic]

        run_name = f"holdout_{held_out_topic}"
        print(f"Training with {held_out_topic} held out ({len(train_topics)} train topics)")

        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "experiment_name": EXPERIMENT_NAME,
                "held_out_topic": held_out_topic,
                "train_topics": train_topics,
                "layers": LAYERS,
                "reg_strength": REG_STRENGTH,
                "activation_source": ACTIVATION_SOURCE,
            },
            reinit=True,
        )

        trainer = ProbeTrainer(
            experiment_name=EXPERIMENT_NAME,
            layers=LAYERS,
            reg_strength=REG_STRENGTH,
            activation_source=ACTIVATION_SOURCE,
            logger=wandb.log,
        )

        # Train on all except held-out topic
        train_acc = trainer.train(exclude_topics=[held_out_topic])

        # Evaluate on held-out topic - use BOTH train and val splits since we didn't train on it
        held_out_train_acts, held_out_train_labels = trainer.load_activations_and_labels(
            "train", include_topics=[held_out_topic]
        )
        held_out_val_acts, held_out_val_labels = trainer.load_activations_and_labels(
            "val", include_topics=[held_out_topic]
        )

        # Combine train + val for held-out topic
        held_out_acts = np.concatenate([held_out_train_acts, held_out_val_acts])
        held_out_labels = np.concatenate([held_out_train_labels, held_out_val_labels])

        preds = trainer.probe.predict(held_out_acts)
        held_out_acc = np.mean((preds > CLASSIFICATION_THRESHOLD) == (held_out_labels > CLASSIFICATION_THRESHOLD))

        # Also evaluate on train topics val split for comparison
        train_topics_val_acc = trainer.evaluate(exclude_topics=[held_out_topic])

        wandb.log({
            "held_out_acc": held_out_acc,
            "held_out_n_samples": len(held_out_acts),
            "train_topics_val_acc": train_topics_val_acc,
        })

        result = {
            "held_out_topic": held_out_topic,
            "train_acc": train_acc,
            "held_out_acc": held_out_acc,
            "held_out_n_samples": len(held_out_acts),
            "train_topics_val_acc": train_topics_val_acc,
            "generalization_gap": train_topics_val_acc - held_out_acc,
        }
        results.append(result)

        print(f"  Train acc: {train_acc:.4f}")
        print(f"  Held-out ({held_out_topic}) acc: {held_out_acc:.4f} (n={len(held_out_acts)})")
        print(f"  Train topics val acc: {train_topics_val_acc:.4f}")
        print(f"  Generalization gap: {result['generalization_gap']:.4f}")
        print()

        wandb.finish()

    # Print summary
    print("=" * 70)
    print("CROSS-TOPIC GENERALIZATION SUMMARY")
    print("=" * 70)

    results_sorted = sorted(results, key=lambda x: x["held_out_acc"], reverse=True)
    for r in results_sorted:
        print(f"{r['held_out_topic']:30s}: held_out={r['held_out_acc']:.3f}, "
              f"train_val={r['train_topics_val_acc']:.3f}, gap={r['generalization_gap']:+.3f}")

    # Summary stats
    avg_held_out = sum(r["held_out_acc"] for r in results) / len(results)
    avg_train_val = sum(r["train_topics_val_acc"] for r in results) / len(results)
    avg_gap = sum(r["generalization_gap"] for r in results) / len(results)

    print()
    print(f"Average held-out acc: {avg_held_out:.3f}")
    print(f"Average train topics val acc: {avg_train_val:.3f}")
    print(f"Average generalization gap: {avg_gap:+.3f}")

    return results


if __name__ == "__main__":
    run_cross_topic_sweep()
