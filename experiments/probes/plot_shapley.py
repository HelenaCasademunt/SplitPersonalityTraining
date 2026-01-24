"""
Visualize Shapley value results from cross-topic probe analysis.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIG - Should match sweep_shapley.py
# =============================================================================

ACTIVATION_SOURCE = "response"  # Options: "response", "response_last", "intervention", "intervention_last"

RESULTS_DIR = Path(__file__).parent / "shapley_results" / ACTIVATION_SOURCE
PLOTS_DIR = Path(__file__).parent / "plots" / f"shapley_{ACTIVATION_SOURCE}"


def load_results():
    """Load Shapley analysis results."""
    shapley_matrix = np.load(RESULTS_DIR / "shapley_matrix.npy")

    with open(RESULTS_DIR / "topics.json") as f:
        topics = json.load(f)

    with open(RESULTS_DIR / "ceiling.json") as f:
        ceiling = json.load(f)

    return shapley_matrix, topics, ceiling


def plot_shapley_heatmap(shapley_matrix: np.ndarray, topics: list, filename: Path):
    """Plot heatmap of Shapley values (raw, not normalized)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(shapley_matrix, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Shapley Value (raw accuracy contribution)", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(range(len(topics)))
    ax.set_yticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_yticklabels(topics)

    # Labels
    ax.set_xlabel("Evaluation Topic")
    ax.set_ylabel("Training Topic")
    ax.set_title("Shapley Values (Raw)\nTraining Topic Contribution to Evaluation Topic Accuracy")

    # Add text annotations
    for i in range(len(topics)):
        for j in range(len(topics)):
            val = shapley_matrix[i, j]
            color = "white" if abs(val) > 0.1 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_shapley_heatmap_normalized(shapley_matrix: np.ndarray, topics: list, ceiling: dict, filename: Path):
    """Plot heatmap of Shapley values normalized by ceiling."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize each column by its ceiling
    normalized_matrix = shapley_matrix.copy()
    for j, topic in enumerate(topics):
        ceil = ceiling[topic]
        if ceil > 0:
            normalized_matrix[:, j] = shapley_matrix[:, j] / ceil

    im = ax.imshow(normalized_matrix, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Shapley Value / Ceiling", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(range(len(topics)))
    ax.set_yticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_yticklabels(topics)

    # Labels
    ax.set_xlabel("Evaluation Topic")
    ax.set_ylabel("Training Topic")
    ax.set_title("Shapley Values (Normalized by Ceiling)\nContribution as Fraction of Max Achievable Accuracy")

    # Add text annotations
    for i in range(len(topics)):
        for j in range(len(topics)):
            val = normalized_matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_ceiling(ceiling: dict, filename: Path):
    """Plot ceiling performance per topic."""
    fig, ax = plt.subplots(figsize=(10, 6))

    topics = list(ceiling.keys())
    values = [ceiling[t] for t in topics]

    # Sort by performance
    sorted_pairs = sorted(zip(topics, values), key=lambda x: -x[1])
    topics = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    bars = ax.bar(range(len(topics)), values)

    # Color bars by value
    for bar, val in zip(bars, values):
        bar.set_color(plt.cm.RdYlGn(val))

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ceiling Performance (Train on All Topics, Eval on Val Set)")
    ax.set_ylim(0, 1)

    # Add value labels
    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_donor_receiver(shapley_matrix: np.ndarray, topics: list, filename: Path):
    """Plot donor quality (row means) and receiver ease (column means). Raw values, not normalized."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Donor quality: how much does training on topic A help OTHER topics (exclude diagonal)
    donor_quality = []
    for i in range(len(topics)):
        row = shapley_matrix[i, :]
        # Exclude diagonal (self-contribution)
        other_contributions = [row[j] for j in range(len(topics)) if j != i]
        donor_quality.append(np.mean(other_contributions))

    # Receiver ease: how much do OTHER topics help predict topic B (exclude diagonal)
    receiver_ease = []
    for j in range(len(topics)):
        col = shapley_matrix[:, j]
        # Exclude diagonal
        other_contributions = [col[i] for i in range(len(topics)) if i != j]
        receiver_ease.append(np.mean(other_contributions))

    # Plot donor quality
    ax1 = axes[0]
    sorted_idx = np.argsort(donor_quality)[::-1]
    sorted_topics = [topics[i] for i in sorted_idx]
    sorted_values = [donor_quality[i] for i in sorted_idx]

    bars1 = ax1.bar(range(len(topics)), sorted_values)
    for bar, val in zip(bars1, sorted_values):
        bar.set_color(plt.cm.Blues(0.3 + 0.7 * (val - min(sorted_values)) / (max(sorted_values) - min(sorted_values) + 1e-6)))

    ax1.set_xticks(range(len(topics)))
    ax1.set_xticklabels(sorted_topics, rotation=45, ha="right")
    ax1.set_xlabel("Training Topic")
    ax1.set_ylabel("Mean Shapley Value (Raw)")
    ax1.set_title("Donor Quality (Raw, Not Normalized)\n(How much does training on this topic help others?)")

    for i, val in enumerate(sorted_values):
        ax1.text(i, val + 0.002, f"{val:.3f}", ha="center", fontsize=8)

    # Plot receiver ease
    ax2 = axes[1]
    sorted_idx = np.argsort(receiver_ease)[::-1]
    sorted_topics = [topics[i] for i in sorted_idx]
    sorted_values = [receiver_ease[i] for i in sorted_idx]

    bars2 = ax2.bar(range(len(topics)), sorted_values)
    for bar, val in zip(bars2, sorted_values):
        bar.set_color(plt.cm.Oranges(0.3 + 0.7 * (val - min(sorted_values)) / (max(sorted_values) - min(sorted_values) + 1e-6)))

    ax2.set_xticks(range(len(topics)))
    ax2.set_xticklabels(sorted_topics, rotation=45, ha="right")
    ax2.set_xlabel("Evaluation Topic")
    ax2.set_ylabel("Mean Shapley Value (Raw)")
    ax2.set_title("Transfer Receptivity (Raw, Not Normalized)\n(How much do other topics help predict this one?)")

    for i, val in enumerate(sorted_values):
        ax2.text(i, val + 0.002, f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_self_vs_transfer(shapley_matrix: np.ndarray, topics: list, filename: Path):
    """Plot self-contribution (diagonal) vs average transfer contribution. Raw values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    self_contrib = np.diag(shapley_matrix)

    # Average contribution from others
    transfer_contrib = []
    for j in range(len(topics)):
        col = shapley_matrix[:, j]
        other = [col[i] for i in range(len(topics)) if i != j]
        transfer_contrib.append(np.mean(other))

    x = np.arange(len(topics))
    width = 0.35

    bars1 = ax.bar(x - width/2, self_contrib, width, label="Self-contribution", color="steelblue")
    bars2 = ax.bar(x + width/2, transfer_contrib, width, label="Avg transfer from others", color="coral")

    ax.set_xlabel("Topic")
    ax.set_ylabel("Shapley Value (Raw)")
    ax.set_title("Self-Contribution vs Transfer from Other Topics (Raw, Not Normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    shapley_matrix, topics, ceiling = load_results()

    print(f"Loaded Shapley matrix: {shapley_matrix.shape}")
    print(f"Topics: {topics}")
    print()

    # Raw (not normalized) plots
    plot_shapley_heatmap(shapley_matrix, topics, PLOTS_DIR / "shapley_heatmap_raw.png")
    plot_donor_receiver(shapley_matrix, topics, PLOTS_DIR / "donor_receiver_raw.png")
    plot_self_vs_transfer(shapley_matrix, topics, PLOTS_DIR / "self_vs_transfer_raw.png")

    # Normalized by ceiling
    plot_shapley_heatmap_normalized(shapley_matrix, topics, ceiling, PLOTS_DIR / "shapley_heatmap_normalized.png")

    # Ceiling (reference)
    plot_ceiling(ceiling, PLOTS_DIR / "ceiling_performance.png")

    print("\nAll plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
