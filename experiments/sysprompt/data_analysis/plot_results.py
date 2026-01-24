#!/usr/bin/env python3
"""Plot augmentation experiment results with split mismatch evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

from plot_utils import (
    load_all_results, MODEL_ORDER, MODEL_LABELS, EVAL_CONDITIONS,
    EVAL_LABELS, MODEL_COLORS
)


def calculate_accuracy(samples):
    """Calculate accuracy from samples."""
    if len(samples) == 0:
        return 0.0
    correct = sum(1 for s in samples if s.get('verdict') == 'CORRECT')
    return correct / len(samples)


def main():
    parser = argparse.ArgumentParser(description='Plot augmentation experiment results')
    parser.add_argument('eval_run', nargs='?', default='eval_run_5',
                       help='Eval run directory (default: eval_run_5)')
    parser.add_argument('--output', '-o', default='augmentation_results.png',
                       help='Output filename (default: augmentation_results.png)')
    parser.add_argument('--exclude', '-e', default='',
                       help='Comma-separated list of models to exclude (e.g., "baseline,15pct_swap")')
    args = parser.parse_args()

    # Navigate to parent directory to find results
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    eval_path = parent_dir / args.eval_run

    if not eval_path.exists():
        print(f"Error: {eval_path} does not exist")
        sys.exit(1)

    # Parse excluded models
    excluded_models = set()
    if args.exclude:
        excluded_models = set(m.strip() for m in args.exclude.split(','))
        print(f"Excluding models: {excluded_models}")

    # Filter models
    models_to_plot = [m for m in MODEL_ORDER if m not in excluded_models]
    model_labels_filtered = [MODEL_LABELS[i] for i, m in enumerate(MODEL_ORDER) if m not in excluded_models]
    model_colors_filtered = [MODEL_COLORS[i] for i, m in enumerate(MODEL_ORDER) if m not in excluded_models]

    if not models_to_plot:
        print("Error: No models remaining after exclusion")
        sys.exit(1)

    print(f"Loading results from {eval_path}")

    # Load all results
    all_results = load_all_results(eval_path, verbose=True)

    # Calculate accuracies for each model and eval condition
    accuracies = {}
    for model in MODEL_ORDER:
        if model not in all_results:
            accuracies[model] = {cond: 0.0 for cond in EVAL_CONDITIONS}
            continue

        accuracies[model] = {}
        for cond in EVAL_CONDITIONS:
            if cond in all_results[model]:
                samples = all_results[model][cond]['samples']
                accuracies[model][cond] = calculate_accuracy(samples)
            else:
                accuracies[model][cond] = 0.0

    # Create plot
    num_models = len(models_to_plot)
    # Dynamically adjust figure width based on number of models
    fig_width = max(16, 12 + num_models * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(EVAL_CONDITIONS))
    # Dynamically adjust bar width based on number of models
    width = min(0.11, 0.8 / num_models)

    # Adjust centering based on number of models
    center_offset = (num_models - 1) / 2

    for i, model in enumerate(models_to_plot):
        accs = [accuracies[model][cond] for cond in EVAL_CONDITIONS]
        offset = (i - center_offset) * width
        ax.bar(x + offset, accs, width, label=model_labels_filtered[i], color=model_colors_filtered[i])

    ax.set_xlabel('Evaluation Condition', fontsize=12)
    ax.set_ylabel('Accuracy (correct / valid)', fontsize=12)
    ax.set_title('System Prompt Dependency: Training Data Augmentation vs Training Approach',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_LABELS)
    # Adjust legend font size based on number of models
    legend_fontsize = max(7, 10 - num_models * 0.1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = script_dir / args.output
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Plot saved to {output_file}")

    # Print summary table
    print("\nResults Summary:")
    print(f"{'Model':<20} {'Eval 1':<12} {'Eval 2':<12} {'Eval 3':<12} {'Eval 4':<12}")
    print("-" * 72)
    for model in models_to_plot:
        row = [f"{accuracies[model][cond]:.3f}" for cond in EVAL_CONDITIONS]
        print(f"{model:<20} {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12}")


if __name__ == '__main__':
    main()
