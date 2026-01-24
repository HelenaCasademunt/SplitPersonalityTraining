"""
Plot cross-topic generalization results (leave-one-out) extracted from Shapley data.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

ACTIVATION_SOURCE = "response"

RESULTS_DIR = Path(__file__).parent / "shapley_results" / ACTIVATION_SOURCE
PLOTS_DIR = Path(__file__).parent / "plots"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_leave_one_out_results():
    """Extract leave-one-out results from Shapley data."""
    with open(RESULTS_DIR / "raw_results.json") as f:
        raw = json.load(f)

    with open(RESULTS_DIR / "topics.json") as f:
        topics = json.load(f)

    with open(RESULTS_DIR / "ceiling.json") as f:
        ceiling = json.load(f)

    results = []
    for held_out in topics:
        train_topics = sorted([t for t in topics if t != held_out])
        train_key = "+".join(train_topics)
        lookup_key = f"{train_key}|{held_out}"

        loo_acc = raw.get(lookup_key)
        ceil_acc = ceiling.get(held_out)

        if loo_acc is not None and ceil_acc is not None:
            results.append({
                "topic": held_out,
                "leave_one_out_acc": loo_acc,
                "ceiling_acc": ceil_acc,
                "gap": ceil_acc - loo_acc,
            })

    return results, topics


# =============================================================================
# PLOTTING
# =============================================================================

def plot_leave_one_out(results: list, filename: Path):
    """Plot leave-one-out accuracy vs ceiling."""
    fig, ax = plt.subplots(figsize=(12, 6))

    topics = [r["topic"] for r in results]
    loo_accs = [r["leave_one_out_acc"] for r in results]
    ceil_accs = [r["ceiling_acc"] for r in results]

    x = np.arange(len(topics))
    width = 0.35

    bars1 = ax.bar(x - width/2, ceil_accs, width, label="Ceiling (train on all)", color="steelblue")
    bars2 = ax.bar(x + width/2, loo_accs, width, label="Leave-one-out", color="coral")

    ax.set_xlabel("Held-out Topic")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Topic Generalization: Leave-One-Out vs Ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar, val in zip(bars1, ceil_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, loo_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Add average lines
    avg_ceil = np.mean(ceil_accs)
    avg_loo = np.mean(loo_accs)
    ax.axhline(y=avg_ceil, color="steelblue", linestyle="--", alpha=0.7)
    ax.axhline(y=avg_loo, color="coral", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_generalization_gap(results: list, filename: Path):
    """Plot generalization gap (ceiling - leave-one-out) by topic."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by gap
    sorted_results = sorted(results, key=lambda x: -x["gap"])
    topics = [r["topic"] for r in sorted_results]
    gaps = [r["gap"] for r in sorted_results]

    colors = ["red" if g > 0.1 else "orange" if g > 0 else "green" for g in gaps]
    bars = ax.bar(range(len(topics)), gaps, color=colors)

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_xlabel("Held-out Topic")
    ax.set_ylabel("Generalization Gap (Ceiling - Leave-one-out)")
    ax.set_title("Cross-Topic Generalization Gap")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, gaps)):
        y_pos = val + 0.01 if val >= 0 else val - 0.03
        ax.text(i, y_pos, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    results, topics = load_leave_one_out_results()

    print("Leave-one-out results:")
    print("-" * 70)
    print(f"{'Topic':30s} {'Leave-1-out':>12s} {'Ceiling':>12s} {'Gap':>12s}")
    print("-" * 70)
    for r in results:
        print(f"{r['topic']:30s} {r['leave_one_out_acc']:12.4f} {r['ceiling_acc']:12.4f} {r['gap']:+12.4f}")

    avg_loo = np.mean([r["leave_one_out_acc"] for r in results])
    avg_ceil = np.mean([r["ceiling_acc"] for r in results])
    print("-" * 70)
    print(f"{'AVERAGE':30s} {avg_loo:12.4f} {avg_ceil:12.4f} {avg_ceil - avg_loo:+12.4f}")

    plot_leave_one_out(results, PLOTS_DIR / "cross_topic_leave_one_out.png")
    plot_generalization_gap(results, PLOTS_DIR / "cross_topic_gap.png")

    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
