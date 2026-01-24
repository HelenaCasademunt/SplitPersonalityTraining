#!/usr/bin/env python3
"""Plot breakdown of PASS/FAIL predictions within incorrect samples, with split mismatch evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse
from matplotlib.patches import Patch

from plot_utils import (
    load_all_results, MODEL_ORDER, MODEL_LABELS, EVAL_CONDITIONS,
    EVAL_LABELS, MODEL_COLORS
)


def calculate_error_breakdown(samples):
    """Calculate error breakdown from samples.

    Returns:
        dict with 'pass_pct', 'fail_pct', 'pass_count', 'fail_count', 'total_incorrect'
    """
    incorrect_samples = [s for s in samples if s.get('verdict') != 'CORRECT']

    if len(incorrect_samples) == 0:
        return {
            'pass_pct': 0.0,
            'fail_pct': 0.0,
            'pass_count': 0,
            'fail_count': 0,
            'total_incorrect': 0
        }

    pass_count = sum(1 for s in incorrect_samples if s.get('parsed_flag') == 'PASS')
    fail_count = sum(1 for s in incorrect_samples if s.get('parsed_flag') == 'FAIL')
    total = len(incorrect_samples)

    return {
        'pass_pct': pass_count / total if total > 0 else 0.0,
        'fail_pct': fail_count / total if total > 0 else 0.0,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'total_incorrect': total
    }


def main():
    parser = argparse.ArgumentParser(description='Plot error breakdown by PASS/FAIL predictions')
    parser.add_argument('eval_run', nargs='?', default='eval_run_4',
                       help='Eval run directory (default: eval_run_4)')
    parser.add_argument('--output', '-o', default='error_breakdown.png',
                       help='Output filename (default: error_breakdown.png)')
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

    print(f"Loading results from {eval_path}")

    # Load all results
    all_results = load_all_results(eval_path, verbose=True)

    # Calculate error breakdowns for each model and eval condition
    error_data = {}
    for model in models_to_plot:
        if model not in all_results:
            error_data[model] = {cond: calculate_error_breakdown([]) for cond in EVAL_CONDITIONS}
            continue

        error_data[model] = {}
        for cond in EVAL_CONDITIONS:
            if cond in all_results[model]:
                samples = all_results[model][cond]['samples']
                error_data[model][cond] = calculate_error_breakdown(samples)
            else:
                error_data[model][cond] = calculate_error_breakdown([])

    # Create plot
    num_models = len(models_to_plot)
    # Dynamically adjust figure width based on number of models
    fig_width = max(16, 12 + num_models * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    x = np.arange(len(EVAL_CONDITIONS))
    # Dynamically adjust bar width based on number of models
    width = min(0.11, 0.8 / num_models)

    # Adjust centering based on number of models
    center_offset = (num_models - 1) / 2

    for i, model in enumerate(models_to_plot):
        pass_pcts = [error_data[model][cond]['pass_pct'] for cond in EVAL_CONDITIONS]
        fail_pcts = [error_data[model][cond]['fail_pct'] for cond in EVAL_CONDITIONS]

        offset = (i - center_offset) * width

        # Stack PASS on bottom (lighter), FAIL on top (darker)
        ax.bar(x + offset, pass_pcts, width,
               color=model_colors_filtered[i], alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.bar(x + offset, fail_pcts, width, bottom=pass_pcts,
               color=model_colors_filtered[i], alpha=1.0, edgecolor='white', linewidth=0.5)

    # Add legend for models
    legend_elements = [Patch(facecolor=model_colors_filtered[i], label=model_labels_filtered[i])
                      for i in range(len(models_to_plot))]

    # Add PASS/FAIL explanation patches
    legend_elements.append(Patch(facecolor='none', label=''))
    legend_elements.append(Patch(facecolor='gray', alpha=0.7,
                                 label='Wrongly predicted PASS (lighter shade)'))
    legend_elements.append(Patch(facecolor='gray', alpha=1.0,
                                 label='Wrongly predicted FAIL (darker shade)'))

    # Adjust legend font size based on number of models
    legend_fontsize = max(7, 10 - num_models * 0.1)
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)

    ax.set_xlabel('Evaluation Condition', fontsize=12)
    ax.set_ylabel('Fraction of Incorrect Samples', fontsize=12)
    ax.set_title('Error Type Breakdown: PASS vs FAIL in Incorrect Predictions', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_LABELS)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = script_dir / args.output
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Plot saved to {output_file}")

    # Print summary
    print("\nError Breakdown Summary:")
    print(f"{'Model':<20} {'Condition':<18} {'Total Wrong':<12} {'PASS':<10} {'FAIL':<10}")
    print("-" * 85)
    for model in models_to_plot:
        for cond in EVAL_CONDITIONS:
            data = error_data[model][cond]
            if data['total_incorrect'] > 0:
                print(f"{model:<20} {cond:<18} {data['total_incorrect']:<12} "
                      f"{data['pass_count']:<10} {data['fail_count']:<10}")


if __name__ == '__main__':
    main()
