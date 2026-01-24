#!/usr/bin/env python3
"""
Generate pairwise agreement heatmaps showing how often models agree with each other.
Creates one heatmap per evaluation condition (4 conditions including split mismatch).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from typing import Dict, List

from plot_utils import (
    load_all_results, MODEL_ORDER, MODEL_LABELS, EVAL_CONDITIONS, EVAL_LABELS,
    MODEL_COLORS
)
from matplotlib.patches import Rectangle


def calculate_agreement_matrix(all_results: Dict, models: List[str], eval_cond: str) -> np.ndarray:
    """Calculate pairwise agreement matrix between models for a specific eval condition.

    Args:
        all_results: Nested dict {model: {eval_cond: {'samples': [...]}}}
        models: List of model names in order
        eval_cond: Evaluation condition to analyze

    Returns:
        NxN matrix of agreement rates
    """
    n_models = len(models)
    agreement_matrix = np.zeros((n_models, n_models))

    # Get all sample keys (topic, sample_idx) for this eval condition
    all_keys = set()
    for model in models:
        if model in all_results and eval_cond in all_results[model]:
            samples = all_results[model][eval_cond]['samples']
            for sample in samples:
                key = (sample['topic'], sample['sample_idx'])
                all_keys.add(key)

    all_keys = sorted(all_keys)

    # Index samples by (topic, sample_idx) for each model
    indexed_results = {}
    for model in models:
        indexed = {}
        if model in all_results and eval_cond in all_results[model]:
            for sample in all_results[model][eval_cond]['samples']:
                key = (sample['topic'], sample['sample_idx'])
                indexed[key] = sample.get('verdict', 'UNKNOWN')
        indexed_results[model] = indexed

    # Calculate pairwise agreement
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                agreement_matrix[i, j] = 1.0  # 100% agreement with self
                continue

            # Count agreements
            agreements = 0
            total = 0

            for key in all_keys:
                verdict1 = indexed_results[model1].get(key)
                verdict2 = indexed_results[model2].get(key)

                if verdict1 is not None and verdict2 is not None:
                    total += 1
                    if verdict1 == verdict2:
                        agreements += 1

            agreement_matrix[i, j] = agreements / total if total > 0 else 0.0

    return agreement_matrix


def plot_heatmaps(agreement_matrices: Dict[str, np.ndarray],
                  model_labels: List[str],
                  eval_labels: List[str],
                  output_path: Path,
                  model_colors: List[str]):
    """Create side-by-side heatmaps for each evaluation condition."""

    # Create short labels for axes
    short_labels = [f"M{i+1}" for i in range(len(model_labels))]

    # Dynamically adjust figure size based on number of models
    num_models = len(model_labels)
    # Each heatmap needs room for the matrix (scales with num_models)
    # Base size 5 per heatmap, plus 0.3 per model for the matrix
    heatmap_size = 5 + num_models * 0.3
    fig_width = max(26, heatmap_size * 4 + 2)  # 4 heatmaps plus margins
    fig_height = max(7.5, heatmap_size + 1.5)  # Height for one row of heatmaps plus legend
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid: 4 heatmaps on top, legend below
    gs = fig.add_gridspec(2, 4, height_ratios=[6, 1], hspace=0.4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # Find global min/max for consistent color scale
    vmin = min(mat.min() for mat in agreement_matrices.values() if mat is not None)
    vmax = 1.0  # Always 100% max

    for idx, (eval_cond, eval_label) in enumerate(zip(EVAL_CONDITIONS, eval_labels)):
        ax = axes[idx]

        if eval_cond not in agreement_matrices or agreement_matrices[eval_cond] is None:
            ax.text(0.5, 0.5, f'No data for {eval_label}',
                   ha='center', va='center', fontsize=12)
            ax.set_title(eval_label, fontsize=14, fontweight='bold')
            continue

        matrix = agreement_matrices[eval_cond]

        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')

        # Set ticks and labels with short M1-M7 notation
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=0, ha='center', fontsize=10)
        ax.set_yticklabels(short_labels, fontsize=10)

        # Color the tick labels
        for i, (xtick, ytick) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
            xtick.set_color(model_colors[i])
            xtick.set_fontweight('bold')
            ytick.set_color(model_colors[i])
            ytick.set_fontweight('bold')

        # Add agreement percentages as text
        # Adjust font size based on number of models
        text_fontsize = max(6, 10 - num_models * 0.2)
        for i in range(len(model_labels)):
            for j in range(len(model_labels)):
                text_color = 'white' if matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{matrix[i, j]*100:.0f}%',
                       ha='center', va='center', color=text_color,
                       fontsize=text_fontsize, fontweight='bold')

        ax.set_title(eval_label, fontsize=14, fontweight='bold', pad=10)

        # Add colorbar for first plot only
        if idx == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Agreement Rate', rotation=270, labelpad=20)

    # Add legend at the bottom spanning all columns
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    # Create colored legend
    n_models = len(model_labels)
    spacing = 1.0 / n_models

    for i, (label, color) in enumerate(zip(model_labels, model_colors)):
        x_pos = (i + 0.5) * spacing

        # Add colored rectangle
        rect = Rectangle((x_pos - 0.06, 0.3), 0.03, 0.4,
                         facecolor=color, edgecolor='black', linewidth=1,
                         transform=legend_ax.transAxes)
        legend_ax.add_patch(rect)

        # Add text (label already contains "M1:", "M2:", etc.)
        legend_ax.text(x_pos, 0.5, label, ha='center', va='center',
                      fontsize=9, transform=legend_ax.transAxes)

    plt.suptitle('Model Agreement Matrices: Pairwise Verdict Agreement',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {output_path}")


def print_agreement_stats(agreement_matrices: Dict[str, np.ndarray],
                         model_labels: List[str],
                         eval_labels: List[str]):
    """Print summary statistics about agreement."""
    for eval_cond, eval_label in zip(EVAL_CONDITIONS, eval_labels):
        if eval_cond not in agreement_matrices or agreement_matrices[eval_cond] is None:
            continue

        matrix = agreement_matrices[eval_cond]

        print(f"\n{'='*60}")
        print(f"Agreement Statistics - {eval_label}")
        print('='*60)

        # Get off-diagonal elements (exclude self-agreement)
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diag = matrix[mask]

        print(f"Average agreement: {off_diag.mean()*100:.1f}%")
        print(f"Min agreement: {off_diag.min()*100:.1f}%")
        print(f"Max agreement: {off_diag.max()*100:.1f}%")
        print(f"Std deviation: {off_diag.std()*100:.1f}%")

        # Find most/least similar pairs
        n = len(model_labels)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((matrix[i,j], model_labels[i], model_labels[j]))

        pairs.sort(reverse=True)

        print(f"\nMost similar pairs:")
        for agreement, m1, m2 in pairs[:3]:
            print(f"  {m1} ↔ {m2}: {agreement*100:.1f}%")

        print(f"\nLeast similar pairs:")
        for agreement, m1, m2 in pairs[-3:]:
            print(f"  {m1} ↔ {m2}: {agreement*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Generate pairwise agreement heatmaps')
    parser.add_argument('eval_run', nargs='?', default='eval_run_4',
                       help='Eval run directory (default: eval_run_4)')
    parser.add_argument('--output', '-o', default='agreement_heatmap.png',
                       help='Output filename (default: agreement_heatmap.png)')
    parser.add_argument('--exclude', '-e', default='',
                       help='Comma-separated list of models to exclude (e.g., "baseline,15pct_mismatch")')
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

    print(f"Loading results from {eval_path}\n")

    # Load all results
    all_results = load_all_results(eval_path, verbose=True)

    # Calculate agreement matrices for each eval condition
    agreement_matrices = {}

    for eval_cond in EVAL_CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Calculating agreement matrix for: {eval_cond}")
        print('='*60)

        matrix = calculate_agreement_matrix(all_results, models_to_plot, eval_cond)
        agreement_matrices[eval_cond] = matrix

    if not agreement_matrices:
        print("❌ No data loaded!")
        return

    # Print statistics
    print_agreement_stats(agreement_matrices, model_labels_filtered, EVAL_LABELS)

    # Create visualization
    output_path = script_dir / args.output
    print(f"\n{'='*60}")
    print("Generating heatmap visualization...")
    print('='*60)
    plot_heatmaps(agreement_matrices, model_labels_filtered, EVAL_LABELS, output_path, model_colors_filtered)

    print(f"\n✓ Done! Open: {output_path}")


if __name__ == '__main__':
    main()
