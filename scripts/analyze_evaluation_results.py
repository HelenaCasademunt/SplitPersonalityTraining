#!/usr/bin/env python3
"""
Analyze quantitative SPT evaluation results.

This script reads all result files and produces:
1. A summary table with statistics for each run
2. Analysis of whether our hypotheses were confirmed
3. Breakdown by checkpoint, condition, and intervention type

Usage:
    python scripts/analyze_evaluation_results.py

Output includes:
- Results by checkpoint (accuracy for each condition)
- Aggregate results across all checkpoints
- Fair comparison (hp elicitation only) between lora-patch and no-lora-patch
"""

import json
from pathlib import Path
from collections import defaultdict

# Get the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation_results" / "quantitative"


def compute_statistics(results, expected_flag):
    """Compute accuracy statistics for a set of results."""
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "parsed": 0,
            "accuracy": 0.0,
            "pass_count": 0,
            "fail_count": 0,
        }

    parsed = sum(1 for r in results if r["flag_parsed_successfully"])
    pass_count = sum(1 for r in results if r["parsed_flag"] == "PASS")
    fail_count = sum(1 for r in results if r["parsed_flag"] == "FAIL")
    correct = sum(
        1
        for r in results
        if r["flag_parsed_successfully"] and r["parsed_flag"] == expected_flag
    )

    return {
        "total": total,
        "parsed": parsed,
        "accuracy": correct / parsed if parsed > 0 else 0.0,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }


def main():
    # Load all results
    all_data = []
    for result_file in sorted(RESULTS_DIR.glob("results_*.json")):
        with open(result_file) as f:
            data = json.load(f)

        if not data["metadata"]["processing_complete"]:
            print(f"WARNING: Incomplete file: {result_file.name}")
            continue

        all_data.append(
            {
                "filename": result_file.name,
                "metadata": data["metadata"],
                "results": data["results"],
            }
        )

    print(f"Loaded {len(all_data)} result files")
    print()

    # Organize data by checkpoint
    by_checkpoint_condition = defaultdict(lambda: defaultdict(list))
    for data in all_data:
        meta = data["metadata"]
        checkpoint = meta["checkpoint_name"]
        condition = meta["condition"]
        lora_patch = meta["lora_patch"]
        elicitation = meta["elicitation_type"]

        key = (checkpoint, lora_patch, elicitation)
        by_checkpoint_condition[key][condition].extend(data["results"])

    # Print results by checkpoint
    print("=" * 80)
    print("RESULTS BY CHECKPOINT")
    print("=" * 80)

    for (checkpoint, lora_patch, elicitation), conditions in sorted(
        by_checkpoint_condition.items()
    ):
        print(f"\n{checkpoint} (lora_patch={lora_patch}, elicitation={elicitation}):")

        for condition in ["trained_actual", "original_actual", "original_intended"]:
            if condition not in conditions:
                continue
            results = conditions[condition]
            expected = (
                "FAIL"
                if condition in ["trained_actual", "original_intended"]
                else "PASS"
            )
            stats = compute_statistics(results, expected)
            print(
                f'  {condition} (expect {expected}): {stats["accuracy"]:.1%} ({stats["parsed"]} samples)'
            )

    # Aggregate by condition (all checkpoints)
    print()
    print("=" * 80)
    print("AGGREGATE BY CONDITION (all checkpoints)")
    print("=" * 80)

    by_condition = defaultdict(list)
    for data in all_data:
        condition = data["metadata"]["condition"]
        by_condition[condition].extend(data["results"])

    for condition in ["trained_actual", "original_actual", "original_intended"]:
        expected = (
            "FAIL" if condition in ["trained_actual", "original_intended"] else "PASS"
        )
        stats = compute_statistics(by_condition[condition], expected)
        print(f'{condition}: {stats["accuracy"]:.1%} ({stats["parsed"]} samples)')

    # Compare hp elicitation only (fair comparison)
    print()
    print("=" * 80)
    print("FAIR COMPARISON: hp elicitation only (lora-patch vs no-lora-patch)")
    print("=" * 80)

    hp_with_lora = defaultdict(list)
    hp_without_lora = defaultdict(list)

    for data in all_data:
        meta = data["metadata"]
        if meta["elicitation_type"] != "hp":
            continue

        condition = meta["condition"]
        if meta["lora_patch"]:
            hp_with_lora[condition].extend(data["results"])
        else:
            hp_without_lora[condition].extend(data["results"])

    print("\nWith LoRA patch (hp elicitation):")
    for condition in ["trained_actual", "original_actual", "original_intended"]:
        expected = (
            "FAIL" if condition in ["trained_actual", "original_intended"] else "PASS"
        )
        stats = compute_statistics(hp_with_lora[condition], expected)
        print(f'  {condition}: {stats["accuracy"]:.1%} ({stats["parsed"]} samples)')

    print("\nWithout LoRA patch (hp elicitation):")
    for condition in ["trained_actual", "original_actual", "original_intended"]:
        expected = (
            "FAIL" if condition in ["trained_actual", "original_intended"] else "PASS"
        )
        stats = compute_statistics(hp_without_lora[condition], expected)
        print(f'  {condition}: {stats["accuracy"]:.1%} ({stats["parsed"]} samples)')

    # Summary comparison table
    print()
    print("=" * 80)
    print("SUMMARY TABLE (for README)")
    print("=" * 80)
    print()
    print("| Condition | With LoRA Patch | Without LoRA Patch |")
    print("|-----------|-----------------|---------------------|")

    for condition in ["trained_actual", "original_actual", "original_intended"]:
        expected = (
            "FAIL" if condition in ["trained_actual", "original_intended"] else "PASS"
        )
        with_lora = compute_statistics(hp_with_lora[condition], expected)
        without_lora = compute_statistics(hp_without_lora[condition], expected)

        # Bold the better one
        if without_lora["accuracy"] > with_lora["accuracy"]:
            print(
                f'| {condition} | {with_lora["accuracy"]:.1%} | **{without_lora["accuracy"]:.1%}** |'
            )
        else:
            print(
                f'| {condition} | **{with_lora["accuracy"]:.1%}** | {without_lora["accuracy"]:.1%} |'
            )

    print()
    print("Aggregate results (all elicitation types):")
    for condition in ["trained_actual", "original_actual", "original_intended"]:
        expected = (
            "FAIL" if condition in ["trained_actual", "original_intended"] else "PASS"
        )
        stats = compute_statistics(by_condition[condition], expected)
        print(f'- {condition}: {stats["accuracy"]:.1%} accuracy')


if __name__ == "__main__":
    main()
