#!/usr/bin/env python3
"""
Compare model predictions across different training configurations.

Loads evaluation results from all training approaches and creates a comparison matrix
showing which models got each sample correct or incorrect.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_eval_results(results_dir: Path) -> Dict[str, Any]:
    """Load the evaluation results JSON from a directory."""
    json_files = list(results_dir.glob("eval_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No eval results found in {results_dir}")
    if len(json_files) > 1:
        print(f"Warning: Multiple eval results in {results_dir}, using most recent")

    json_file = sorted(json_files)[-1]  # Use most recent
    with open(json_file, 'r') as f:
        return json.load(f)


def create_comparison_matrix(base_dir: Path, eval_condition: str = "baseline") -> pd.DataFrame:
    """
    Create a comparison matrix for a specific evaluation condition.

    Args:
        base_dir: Base directory containing all results folders
        eval_condition: One of "baseline", "no_sysprompt", or "mismatch"

    Returns:
        DataFrame with columns for each model's predictions
    """
    # Define the training configurations to compare
    train_configs = {
        'baseline_model': f'results_baseline_{eval_condition}',
        '5pct_augmented': f'results_aug_5pct_{eval_condition}',
        '15pct_augmented': f'results_aug_15pct_{eval_condition}',
        'no_sysprompt': f'results_full_no_sysprompt_{eval_condition}',
        'mismatch_prompts': f'results_full_mismatch_{eval_condition}',
    }

    # Load all results
    all_data = {}
    for config_name, results_dir_name in train_configs.items():
        results_dir = base_dir / results_dir_name
        if not results_dir.exists():
            print(f"Warning: {results_dir} does not exist, skipping {config_name}")
            continue

        print(f"Loading {config_name} from {results_dir_name}...")
        data = load_eval_results(results_dir)
        all_data[config_name] = data

    if not all_data:
        raise ValueError("No data loaded from any configuration")

    # Create lookup dictionaries for each config: (topic, sample_idx) -> sample
    config_lookups = {}
    for config_name, data in all_data.items():
        lookup = {}
        for sample in data['samples']:
            key = (sample['topic'], sample['sample_idx'])
            lookup[key] = sample
        config_lookups[config_name] = lookup

    # Get all unique (topic, sample_idx) pairs from reference config
    reference_config = list(all_data.keys())[0]
    unique_keys = sorted(config_lookups[reference_config].keys())
    n_samples = len(unique_keys)

    print(f"Creating comparison matrix for {n_samples} samples...")

    # Build comparison dataframe
    rows = []
    for topic, sample_idx in unique_keys:
        row = {}

        # Get base information from reference sample
        ref_sample = config_lookups[reference_config][(topic, sample_idx)]
        row['topic'] = topic
        row['sample_idx'] = sample_idx
        row['task'] = ref_sample.get('task', '')
        row['system_prompt'] = ref_sample.get('system_prompt', '')
        row['expected_flag'] = ref_sample.get('expected_flag', '')
        row['intervention'] = ref_sample.get('intervention', '')
        row['system_prompt_variant'] = ref_sample.get('system_prompt_variant', '')

        # Get verdict from each config
        for config_name, lookup in config_lookups.items():
            sample = lookup.get((topic, sample_idx))

            if sample is None:
                print(f"Warning: Missing sample for {config_name} at {topic}/{sample_idx}")
                row[f'{config_name}_verdict'] = 'missing'
                row[f'{config_name}_correct'] = False
                row[f'{config_name}_parsed_flag'] = ''
                row[f'{config_name}_response'] = ''
            else:
                verdict = sample.get('verdict', 'unknown')
                row[f'{config_name}_verdict'] = verdict
                row[f'{config_name}_correct'] = verdict == 'CORRECT'
                row[f'{config_name}_parsed_flag'] = sample.get('parsed_flag', '')
                row[f'{config_name}_response'] = sample.get('response_evaluated', '')
                row[f'{config_name}_generated_output'] = sample.get('generated_output', '')

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add classification column
    df['classification'] = df.apply(classify_sample, axis=1)

    return df


def classify_sample(row: pd.Series) -> str:
    """Classify a sample based on which models got it correct."""
    correct_cols = [col for col in row.index if col.endswith('_correct')]
    correct_models = [col.replace('_correct', '') for col in correct_cols if row[col]]

    if len(correct_models) == len(correct_cols):
        return 'all_correct'
    elif len(correct_models) == 0:
        return 'all_wrong'
    elif 'baseline_model' in correct_models:
        if '5pct_augmented' not in correct_models or '15pct_augmented' not in correct_models:
            return 'baseline_correct_augmented_wrong'
        else:
            return 'baseline_and_augmented_correct'
    else:
        if '5pct_augmented' in correct_models or '15pct_augmented' in correct_models:
            return 'augmented_correct_baseline_wrong'
        else:
            return 'other_correct_baseline_wrong'


def main():
    # Path to eval_run_3 folder
    eval_run_dir = Path(__file__).parent.parent / 'eval_run_3'
    # Output to data_analysis folder
    output_dir = Path(__file__).parent

    # Create comparison for baseline evaluation condition
    print("=" * 80)
    print("Creating comparison matrix for BASELINE evaluation condition")
    print(f"Reading from: {eval_run_dir}")
    print(f"Writing to: {output_dir}")
    print("=" * 80)

    df = create_comparison_matrix(eval_run_dir, eval_condition='baseline')

    # Save full comparison matrix
    output_csv = output_dir / 'comparison_matrix_baseline_eval.csv'
    output_json = output_dir / 'comparison_matrix_baseline_eval.json'

    df.to_csv(output_csv, index=False)
    df.to_json(output_json, orient='records', indent=2)

    print(f"\nSaved comparison matrix to:")
    print(f"  - {output_csv}")
    print(f"  - {output_json}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal samples: {len(df)}")

    print("\nAccuracy by model:")
    for col in df.columns:
        if col.endswith('_correct'):
            model_name = col.replace('_correct', '')
            accuracy = df[col].mean()
            print(f"  {model_name}: {accuracy:.3f} ({df[col].sum()}/{len(df)})")

    print("\nClassification breakdown:")
    for classification, count in df['classification'].value_counts().items():
        print(f"  {classification}: {count} ({count/len(df)*100:.1f}%)")

    print("\nAccuracy by topic:")
    topic_accuracy = df.groupby('topic').agg({
        col: 'mean' for col in df.columns if col.endswith('_correct')
    }).round(3)
    print(topic_accuracy.to_string())


if __name__ == '__main__':
    main()
