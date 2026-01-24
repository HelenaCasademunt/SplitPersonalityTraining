"""
Plot sweep results from W&B.
"""

import json
from pathlib import Path

import wandb
import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = Path(__file__).parent / "plots"

# =============================================================================
# CONFIG
# =============================================================================

WANDB_PROJECT = "probe-sweep"
WANDB_ENTITY = None  # Set if needed


# =============================================================================
# FETCH DATA
# =============================================================================

def fetch_sweep_data():
    api = wandb.Api()
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    runs = api.runs(project_path)

    data = []
    for run in runs:
        # Config may be a JSON string
        config = run.config
        if isinstance(config, str):
            config = json.loads(config)

        # Extract values (W&B wraps them in {"value": ...})
        def get_val(d, key):
            v = d[key]
            if isinstance(v, dict) and "value" in v:
                return v["value"]
            return v

        # Summary needs special handling - _json_dict may be a string
        summary_raw = run.summary._json_dict
        if isinstance(summary_raw, str):
            summary = json.loads(summary_raw)
        else:
            summary = summary_raw

        # Skip runs that don't have results yet
        if "train_acc_linear" not in summary:
            continue

        data.append({
            "layers": str(get_val(config, "layers")),
            "reg_strength": get_val(config, "reg_strength"),
            "activation_source": get_val(config, "activation_source"),
            "train_acc": summary["train_acc_linear"],
            "val_acc": summary["test_acc_linear"],
        })

    df = pd.DataFrame(data)
    df["overfit_gap"] = df["train_acc"] - df["val_acc"]
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_train_val_grouped(df, param, title, filename):
    """Plot train and val accuracy as grouped bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_grouped = df.groupby(param).agg({
        "train_acc": ["mean", "std"],
        "val_acc": ["mean", "std"],
    }).reset_index()

    x = range(len(df_grouped))
    width = 0.35

    train_means = df_grouped[("train_acc", "mean")]
    train_stds = df_grouped[("train_acc", "std")]
    val_means = df_grouped[("val_acc", "mean")]
    val_stds = df_grouped[("val_acc", "std")]

    ax.bar([i - width/2 for i in x], train_means, width, yerr=train_stds, label="Train", capsize=3)
    ax.bar([i + width/2 for i in x], val_means, width, yerr=val_stds, label="Val", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped[param], rotation=45, ha="right")
    ax.set_xlabel(param)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_best_per_source(df, filename):
    """Plot best val_acc for each activation source."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Readable source names
    source_labels = {
        "response": "Response [S] mean",
        "response_last": "Response [S] last token",
        "intervention": "Intervention [I] mean",
        "intervention_last": "Intervention [I] last token",
    }

    # Get best run per source
    best_per_source = df.loc[df.groupby("activation_source")["val_acc"].idxmax()]
    best_per_source = best_per_source.sort_values("val_acc", ascending=False)

    x = range(len(best_per_source))

    ax.bar(x, best_per_source["train_acc"], width=0.35, label="Train", align="edge")
    ax.bar([i + 0.35 for i in x], best_per_source["val_acc"], width=0.35, label="Val", align="edge")

    labels = [source_labels.get(s, s) for s in best_per_source["activation_source"]]
    ax.set_xticks([i + 0.35/2 for i in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Activation Source")
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Ridge Probe per Activation Source")
    ax.legend()

    # Add annotations with config details and val acc above the plot
    for i, row in enumerate(best_per_source.itertuples()):
        ax.annotate(f"layers={row.layers}, reg={row.reg_strength}\nval={row.val_acc:.3f}",
                    xy=(i + 0.35/2, 1.02), fontsize=8, ha="center")

    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def make_plots(df):
    PLOTS_DIR.mkdir(exist_ok=True)

    plot_train_val_grouped(df, "layers", "Train vs Val Accuracy by Layer", PLOTS_DIR / "acc_by_layer.png")
    plot_train_val_grouped(df, "reg_strength", "Train vs Val Accuracy by Reg Strength", PLOTS_DIR / "acc_by_reg.png")
    plot_train_val_grouped(df, "activation_source", "Train vs Val Accuracy by Activation Source", PLOTS_DIR / "acc_by_source.png")
    plot_best_per_source(df, PLOTS_DIR / "best_per_source.png")


if __name__ == "__main__":
    df = fetch_sweep_data()
    print(f"Fetched {len(df)} runs")
    print(df)
    make_plots(df)
