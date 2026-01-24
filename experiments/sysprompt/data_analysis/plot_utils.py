#!/usr/bin/env python3
"""Common utilities for plotting system prompt experiment results."""

import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Model configuration
KNOWN_MODELS = [
    'baseline',
    '5pct_no_sysprompt_5pct_swap',
    '15pct_no_sysprompt_15pct_swap',
    '15pct_no_sysprompt',
    '15pct_swap',
    '33pct_no_sysprompt_33pct_swap',
    '33pct_no_sysprompt',
    '33pct_swap',
    '100pct_no_sysprompt',
    '100pct_swap',
]

MODEL_ORDER = [
    'baseline',
    '5pct_no_sysprompt_5pct_swap',
    '15pct_no_sysprompt_15pct_swap',
    '15pct_no_sysprompt',
    '15pct_swap',
    '33pct_no_sysprompt_33pct_swap',
    '33pct_no_sysprompt',
    '33pct_swap',
    '100pct_no_sysprompt',
    '100pct_swap',
]

MODEL_LABELS = [
    'M1: baseline',
    'M2: 5+5% mix',
    'M3: 15+15% mix',
    'M4: 15% w/o sys',
    'M5: 15% w/ swap',
    'M6: 33+33% mix',
    'M7: 33% w/o sys',
    'M8: 33% w/ swap',
    'M9: 100% w/o sys',
    'M10: 100% w/ swap',
]

# Evaluation conditions (eval_swapped gets split into swapped and a_only)
EVAL_CONDITIONS = ['eval_baseline', 'eval_no_sysprompt', 'eval_swapped', 'eval_a_only']

EVAL_LABELS = [
    'Eval 1:\nbaseline',
    'Eval 2:\nexclude sys',
    'Eval 3:\nswap A/B',
    'Eval 4:\nA-only'
]

# Color scheme for models
MODEL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#17becf',
    '#bcbd22', '#e377c2', '#8c564b', '#7f7f7f',
    '#d62728', '#9467bd', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5'
]


def load_results_from_dir(result_dir: Path) -> Optional[Dict]:
    """Load JSON results from a result directory.

    Args:
        result_dir: Path to results directory

    Returns:
        Dictionary with parsed JSON data, or None if not found
    """
    if not result_dir.exists():
        return None

    json_files = sorted(result_dir.glob("*.json"))
    if not json_files:
        return None

    with open(json_files[0]) as f:
        return json.load(f)


def parse_result_dir_name(dir_name: str) -> Optional[Tuple[str, str]]:
    """Parse model name and eval condition from result directory name.

    Args:
        dir_name: Directory name like 'results_baseline_eval_swapped'

    Returns:
        Tuple of (model_name, eval_condition) or None if unrecognized
    """
    # Try to match against known models (check longer names first to avoid partial matches)
    # Sort by length descending to check longest names first
    for known_model in sorted(KNOWN_MODELS, key=len, reverse=True):
        prefix = f"results_{known_model}_"
        if dir_name.startswith(prefix):
            model = known_model
            eval_condition = dir_name[len(prefix):]
            return model, eval_condition

    return None


def filter_augmentation_eligible(samples: List[Dict]) -> List[Dict]:
    """Filter samples to only those eligible for augmentation (no dependency issues).

    Removes samples where:
    - mentions_system_prompt == True (review explicitly references system prompt)
    - could_determine_without_prompt == False (verdict requires system prompt)

    This ensures all evaluation conditions are compared on the same filtered set.

    Args:
        samples: List of sample dictionaries

    Returns:
        Filtered list of samples
    """
    filtered = []
    for sample in samples:
        metadata = sample.get('metadata', {})
        if not metadata:
            # Keep samples without metadata
            filtered.append(sample)
            continue

        # Check for dependency issues
        mentions_system_prompt = metadata.get('mentions_system_prompt', False)
        could_determine_without_prompt = metadata.get('could_determine_without_prompt', True)

        has_dependency_issue = (
            mentions_system_prompt is True or
            could_determine_without_prompt is False
        )

        if not has_dependency_issue:
            filtered.append(sample)

    return filtered


def split_by_is_just_a(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into A/B swappable vs A-only.

    Args:
        samples: List of sample dictionaries

    Returns:
        Tuple of (swappable_samples, a_only_samples)
    """
    swappable = [s for s in samples if not s.get('is_just_A', False)]
    a_only = [s for s in samples if s.get('is_just_A', False)]
    return swappable, a_only


def load_all_results(eval_path: Path, verbose: bool = True) -> Dict[str, Dict[str, any]]:
    """Load all result directories and organize by model and eval condition.

    Automatically handles splitting eval_swapped into swapped vs A-only.

    Args:
        eval_path: Path to directory containing results_* subdirectories
        verbose: Whether to print loading progress

    Returns:
        Nested dict: {model_name: {eval_condition: data}}
        where eval_condition can be 'eval_baseline', 'eval_no_sysprompt',
        'eval_swapped', or 'eval_a_only'
    """
    results = {}

    for result_dir in sorted(glob.glob(str(eval_path / "results_*"))):
        dir_name = Path(result_dir).name

        parsed = parse_result_dir_name(dir_name)
        if parsed is None:
            if verbose:
                print(f"⚠️  Skipping unrecognized directory: {result_dir}")
            continue

        model, eval_condition = parsed

        data = load_results_from_dir(Path(result_dir))
        if data is None:
            if verbose:
                print(f"⚠️  No JSON files found in {result_dir}")
            continue

        samples = data.get('samples', [])

        # Filter to only augmentation-eligible samples
        # This ensures all eval conditions are compared on the same filtered dataset
        samples_filtered = filter_augmentation_eligible(samples)

        # Initialize model dict if needed
        if model not in results:
            results[model] = {}

        # Handle eval_swapped: split into swapped vs A-only
        if eval_condition == 'eval_swapped':
            swappable, a_only = split_by_is_just_a(samples_filtered)

            results[model]['eval_swapped'] = {
                'samples': swappable,
                'metadata': data.get('metadata', {})
            }
            results[model]['eval_a_only'] = {
                'samples': a_only,
                'metadata': data.get('metadata', {})
            }

            if verbose:
                print(f"Loaded: {dir_name} -> model={model}")
                print(f"  Swapped: {len(swappable)} samples (filtered: {len(samples)} → {len(samples_filtered)})")
                print(f"  A-only: {len(a_only)} samples")
        else:
            # For eval_baseline and eval_no_sysprompt: exclude A-only samples
            swappable, a_only = split_by_is_just_a(samples_filtered)

            results[model][eval_condition] = {
                'samples': swappable,
                'metadata': data.get('metadata', {})
            }

            if verbose:
                print(f"Loaded: {dir_name} -> model={model}, condition={eval_condition}")
                print(f"  Samples (excl. A-only): {len(swappable)} (filtered: {len(samples)} → {len(samples_filtered)})")

    return results
