"""Plot evaluation results with error bars across folds."""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_eval_results(results_dir: Union[str, Path]) -> dict:
    """Load all eval results from a directory.

    Returns dict mapping (model, output_format) -> list of accuracies across folds.
    """
    results_dir = Path(results_dir)
    results = defaultdict(list)

    # Pattern: eval_results_{model}_{output_format}_fold{N}.json
    pattern = re.compile(r"eval_results_(\w+)_(flag_only|flag_then_review|review_then_flag)_fold(\d+)\.json")

    for json_file in sorted(results_dir.glob("eval_results_*.json")):
        match = pattern.match(json_file.name)
        if not match:
            print(f"Warning: skipping {json_file.name} - doesn't match expected pattern")
            continue

        model = match.group(1)
        output_format = match.group(2)
        fold = int(match.group(3))

        with open(json_file) as f:
            data = json.load(f)

        accuracy = data["metadata"]["overall_accuracy"]
        results[(model, output_format)].append(accuracy)

    return dict(results)


def plot_accuracies(results: dict, output_path: Optional[Union[str, Path]] = None):
    """Create a grouped bar chart of accuracies by model and output_format."""
    # Get unique models and output formats
    models = sorted(set(k[0] for k in results.keys()))
    output_formats = sorted(set(k[1] for k in results.keys()))

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar positioning
    x = np.arange(len(models)) # Add gap between models
    width = 0.2  # Thinner bars
    n_formats = len(output_formats)
    offsets = np.linspace(-width * (n_formats - 1) / 2, width * (n_formats - 1) / 2, n_formats)

    # Colors for each output format
    colors = plt.cm.Set2.colors

    # Plot bars for each output format
    for i, output_format in enumerate(output_formats):
        means = []
        stds = []
        for model in models:
            key = (model, output_format)
            if key in results:
                accs = results[key]
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            label=output_format.replace("_", " "),
            color=colors[i % len(colors)],
            capsize=5,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add mean accuracy labels above each bar
        for bar, mean, std in zip(bars, means, stds):
            height = mean + std
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Customize the plot
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Evaluation Accuracy by Model and Output Format\n(error bars = std across 3 folds)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(title="Output Format", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="train-project/eval_results",
        help="Directory containing eval result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (e.g., plot.png). If not specified, just displays.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_eval_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print("Loaded results:")
    for (model, fmt), accs in sorted(results.items()):
        print(f"  {model} / {fmt}: {accs} (mean={np.mean(accs):.3f}, std={np.std(accs):.3f})")

    # Default output path is inside the results directory
    output_path = args.output if args.output else results_dir / "accuracy_plot.png"
    plot_accuracies(results, output_path)


if __name__ == "__main__":
    main()
