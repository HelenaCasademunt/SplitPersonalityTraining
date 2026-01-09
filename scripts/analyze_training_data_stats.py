#!/usr/bin/env python3
"""
Analyze training data statistics using the same filtering logic as training.

This script reads the stage 3 tagged data and applies the same filters as
specified in training_config.json to report:
- Number of topics
- Number of models
- Total sample count after filtering

Usage:
    python scripts/analyze_training_data_stats.py
"""

import json
from pathlib import Path
from collections import defaultdict

# Get the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.json"
DATA_DIR = PROJECT_ROOT / "data" / "stage_3_tagged"


def load_config():
    """Load training configuration."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def passes_filters(intervention_data, tags_to_filter):
    """Check if an intervention passes all quality filters."""
    tags = intervention_data.get("tags", [])
    for tag in tags_to_filter:
        if tag in tags:
            return False
    return True


def main():
    config = load_config()

    # Extract filter settings
    intervention_types = config["intervention_types"]
    train_topics = config["train_topics"]
    tags_to_filter = config["tags_to_filter"]

    print("=" * 80)
    print("TRAINING DATA STATISTICS")
    print("=" * 80)
    print()
    print("Filter settings from training_config.json:")
    print(f"  Intervention types: {intervention_types}")
    print(f"  Train topics: {len(train_topics)} topics")
    print(f"  Tags to filter: {len(tags_to_filter)} tags")
    print()

    # Collect statistics
    total_samples = 0
    samples_by_topic = defaultdict(int)
    samples_by_model = defaultdict(int)
    samples_by_intervention = defaultdict(int)
    models_seen = set()

    for topic in train_topics:
        topic_dir = DATA_DIR / topic
        if not topic_dir.exists():
            print(f"WARNING: Topic directory not found: {topic}")
            continue

        for batch_file in topic_dir.glob("*.json"):
            with open(batch_file) as f:
                batch_data = json.load(f)

            for item in batch_data.get("data", []):
                if "inferences" not in item:
                    continue

                for model_name, model_data in item["inferences"].items():
                    for variant in ["A", "B"]:
                        if variant not in model_data:
                            continue
                        if "interventions" not in model_data[variant]:
                            continue

                        for intervention_key, intervention_data in model_data[variant][
                            "interventions"
                        ].items():
                            # Check if this intervention type is in our filter list
                            if intervention_key not in intervention_types:
                                continue

                            # Check if it passes quality filters
                            if not passes_filters(intervention_data, tags_to_filter):
                                continue

                            # Count this sample
                            total_samples += 1
                            samples_by_topic[topic] += 1
                            samples_by_model[model_name] += 1
                            samples_by_intervention[intervention_key] += 1
                            models_seen.add(model_name)

    # Print results
    print("=" * 80)
    print("RESULTS (after applying all filters)")
    print("=" * 80)
    print()

    print(f"Total samples: {total_samples:,}")
    print(f"Number of topics: {len(samples_by_topic)}")
    print(f"Number of models: {len(models_seen)}")
    print()

    print("Samples by topic:")
    for topic, count in sorted(samples_by_topic.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count:,}")
    print()

    print("Samples by model:")
    for model, count in sorted(samples_by_model.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count:,}")
    print()

    print("Samples by intervention type:")
    for intervention, count in sorted(
        samples_by_intervention.items(), key=lambda x: -x[1]
    ):
        print(f"  {intervention}: {count:,}")
    print()

    # Summary for the writeup
    print("=" * 80)
    print("SUMMARY FOR WRITEUP")
    print("=" * 80)
    print()
    print(f"- {len(train_topics)} topics")
    print(f"- {len(models_seen)} models: {', '.join(sorted(models_seen))}")
    print(f"- {total_samples:,} training samples (after quality filtering)")


if __name__ == "__main__":
    main()
