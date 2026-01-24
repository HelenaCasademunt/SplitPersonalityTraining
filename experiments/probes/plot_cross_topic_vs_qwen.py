"""
Compare probe cross-topic generalization vs Qwen leave-one-out results.
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
QWEN_RESULTS_DIR = Path(__file__).parent.parent.parent / "qwen_eval_results"
PLOTS_DIR = Path(__file__).parent / "plots" / "probe_vs_qwen"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_probe_leave_one_out():
    """Extract leave-one-out results from Shapley data."""
    with open(RESULTS_DIR / "raw_results.json") as f:
        raw = json.load(f)

    with open(RESULTS_DIR / "topics.json") as f:
        topics = json.load(f)

    results = {}
    for held_out in topics:
        train_topics = sorted([t for t in topics if t != held_out])
        train_key = "+".join(train_topics)
        lookup_key = f"{train_key}|{held_out}"
        results[held_out] = raw.get(lookup_key)

    return results, topics


def load_qwen_leave_one_out():
    """Load Qwen leave-one-out results from eval files."""
    results = {}

    for filepath in QWEN_RESULTS_DIR.glob("eval_results_qwen_leave_out_*.json"):
        with open(filepath) as f:
            data = json.load(f)

        # Extract topic from filename
        topic = filepath.stem.replace("eval_results_qwen_leave_out_", "")
        results[topic] = data["metadata"]["overall_accuracy"]

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_cross_topic_comparison(probe_results: dict, qwen_results: dict, topics: list, filename: Path):
    """Plot probe vs Qwen leave-one-out accuracy side by side."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Filter to topics that exist in both
    common_topics = [t for t in topics if t in probe_results and t in qwen_results]

    probe_accs = [probe_results[t] for t in common_topics]
    qwen_accs = [qwen_results[t] for t in common_topics]

    x = np.arange(len(common_topics))
    width = 0.35

    bars1 = ax.bar(x - width/2, qwen_accs, width, label="Qwen (leave-one-out)", color="steelblue")
    bars2 = ax.bar(x + width/2, probe_accs, width, label="Probe (leave-one-out)", color="coral")

    ax.set_xlabel("Held-out Topic")
    ax.set_ylabel("Accuracy on Held-out Topic")
    ax.set_title("Cross-Topic Generalization: Qwen vs Probe (Leave-One-Out)")
    ax.set_xticks(x)
    ax.set_xticklabels(common_topics, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar, val in zip(bars1, qwen_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, probe_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Add average lines
    avg_qwen = np.mean(qwen_accs)
    avg_probe = np.mean(probe_accs)
    ax.axhline(y=avg_qwen, color="steelblue", linestyle="--", alpha=0.7, label=f"Qwen avg: {avg_qwen:.3f}")
    ax.axhline(y=avg_probe, color="coral", linestyle="--", alpha=0.7, label=f"Probe avg: {avg_probe:.3f}")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

    return common_topics, probe_accs, qwen_accs


def plot_gap_comparison(probe_results: dict, qwen_results: dict, topics: list, filename: Path):
    """Plot the gap (Qwen - Probe) for each topic."""
    fig, ax = plt.subplots(figsize=(12, 6))

    common_topics = [t for t in topics if t in probe_results and t in qwen_results]
    gaps = [qwen_results[t] - probe_results[t] for t in common_topics]

    # Sort by gap
    sorted_pairs = sorted(zip(common_topics, gaps), key=lambda x: -x[1])
    sorted_topics = [p[0] for p in sorted_pairs]
    sorted_gaps = [p[1] for p in sorted_pairs]

    colors = ["steelblue" if g > 0 else "coral" for g in sorted_gaps]
    bars = ax.bar(range(len(sorted_topics)), sorted_gaps, color=colors)

    ax.set_xticks(range(len(sorted_topics)))
    ax.set_xticklabels(sorted_topics, rotation=45, ha="right")
    ax.set_xlabel("Held-out Topic")
    ax.set_ylabel("Accuracy Gap (Qwen - Probe)")
    ax.set_title("Cross-Topic Generalization Gap: Qwen vs Probe")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_gaps)):
        y_pos = val + 0.01 if val >= 0 else val - 0.02
        ax.text(i, y_pos, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    # Legend
    ax.text(0.02, 0.98, "Blue = Qwen better\nOrange = Probe better", transform=ax.transAxes,
            fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    probe_results, topics = load_probe_leave_one_out()
    qwen_results = load_qwen_leave_one_out()

    print("Cross-topic generalization comparison:")
    print("-" * 70)
    print(f"{'Topic':30s} {'Probe':>12s} {'Qwen':>12s} {'Gap (Q-P)':>12s}")
    print("-" * 70)

    common_topics = [t for t in topics if t in probe_results and t in qwen_results]
    for topic in common_topics:
        p = probe_results[topic]
        q = qwen_results[topic]
        print(f"{topic:30s} {p:12.4f} {q:12.4f} {q-p:+12.4f}")

    avg_probe = np.mean([probe_results[t] for t in common_topics])
    avg_qwen = np.mean([qwen_results[t] for t in common_topics])
    print("-" * 70)
    print(f"{'AVERAGE':30s} {avg_probe:12.4f} {avg_qwen:12.4f} {avg_qwen-avg_probe:+12.4f}")

    plot_cross_topic_comparison(probe_results, qwen_results, topics, PLOTS_DIR / "cross_topic_comparison.png")
    plot_gap_comparison(probe_results, qwen_results, topics, PLOTS_DIR / "cross_topic_gap_qwen_vs_probe.png")

    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
