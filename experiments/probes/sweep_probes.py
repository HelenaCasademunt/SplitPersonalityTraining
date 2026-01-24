"""
Sweep script for training and evaluating probes across hyperparameters.
"""

import itertools

import wandb

from probes import ProbeTrainer


# =============================================================================
# SWEEP PARAMETERS - Edit these lists
# =============================================================================

EXPERIMENT_NAME = "qwen"

# Each entry is a list of layers to concatenate
LAYERS = [[10], [20], [30], [10, 20, 30]]

REG_STRENGTHS = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

LABEL_KEY = "flag"
ACTIVATION_SOURCES = ["response", "response_last", "intervention", "intervention_last"]

WANDB_PROJECT = "probe-sweep"


# =============================================================================
# SWEEP LOGIC
# =============================================================================

def run_sweep():
    all_combinations = list(itertools.product(LAYERS, REG_STRENGTHS, ACTIVATION_SOURCES))

    print(f"Running sweep with {len(all_combinations)} configurations")
    print(f"  Layers: {LAYERS}")
    print(f"  Reg strengths: {REG_STRENGTHS}")
    print(f"  Activation sources: {ACTIVATION_SOURCES}")
    print()

    results = []

    for layers, reg_strength, activation_source in all_combinations:
        run_name = f"layers_{layers}_reg_{reg_strength}_{activation_source}"
        print(f"Training: {run_name}")

        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "experiment_name": EXPERIMENT_NAME,
                "layers": layers,
                "reg_strength": reg_strength,
                "label_key": LABEL_KEY,
                "activation_source": activation_source,
            },
            reinit=True,
        )

        trainer = ProbeTrainer(
            experiment_name=EXPERIMENT_NAME,
            layers=layers,
            reg_strength=reg_strength,
            label_key=LABEL_KEY,
            activation_source=activation_source,
            logger=wandb.log,
        )

        train_acc = trainer.train()
        val_acc = trainer.evaluate()

        result = {
            "layers": layers,
            "reg_strength": reg_strength,
            "activation_source": activation_source,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        results.append(result)

        print(f"  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

        wandb.finish()

    # Print summary sorted by val_acc
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY (sorted by val_acc)")
    print("=" * 60)

    results_sorted = sorted(results, key=lambda x: x["val_acc"], reverse=True)
    for r in results_sorted:
        print(f"layers={r['layers']}, reg={r['reg_strength']}, src={r['activation_source']}: "
              f"train={r['train_acc']:.4f}, val={r['val_acc']:.4f}")

    best = results_sorted[0]
    print(f"\nBest config: layers={best['layers']}, reg={best['reg_strength']}, src={best['activation_source']}")
    print(f"Best val acc: {best['val_acc']:.4f}")

    return results


if __name__ == "__main__":
    run_sweep()
