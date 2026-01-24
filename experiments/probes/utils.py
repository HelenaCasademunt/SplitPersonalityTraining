"""
Shared utilities for probe analysis scripts.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import scripts.probes.extract_activations as extract_activations

# Default paths
PROBES_DIR = Path(__file__).parent
PROBE_WEIGHTS_DIR = PROBES_DIR / "probe_weights"
PLOTS_DIR = PROBES_DIR / "plots"
EVAL_RESULTS_DIR = Path(__file__).parent.parent.parent / "qwen_eval_results"

# Override to use new activations
extract_activations.ACTIVATIONS_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "activations_new"
ACTIVATIONS_DIR = extract_activations.ACTIVATIONS_OUTPUT_DIR


def load_eval_results(eval_file: str) -> dict:
    """Load evaluation results from JSON file.

    Args:
        eval_file: Filename (not full path) of the eval results JSON.

    Returns:
        Parsed JSON data with metadata and samples.
    """
    path = EVAL_RESULTS_DIR / eval_file
    with open(path) as f:
        return json.load(f)


def load_activation_metadata(experiment_name: str, split: str = "val") -> dict:
    """Load activation metadata for an experiment.

    Args:
        experiment_name: Name of the experiment (e.g., "qwen").
        split: Data split ("train" or "val").

    Returns:
        Metadata dict with sample information.
    """
    path = ACTIVATIONS_DIR / experiment_name / split / "metadata.json"
    with open(path) as f:
        return json.load(f)


def load_activations(
    experiment_name: str,
    layers: List[int],
    split: str = "val",
    activation_source: str = "intervention",
) -> torch.Tensor:
    """Load and concatenate activations for specified layers.

    Args:
        experiment_name: Name of the experiment.
        layers: List of layer indices to load.
        split: Data split.
        activation_source: One of "response", "response_last", "intervention", "intervention_last".

    Returns:
        Tensor of shape (n_samples, hidden_dim * n_layers).
    """
    key_map = {
        "response": "response_activations",
        "response_last": "response_last_activations",
        "intervention": "intervention_activations",
        "intervention_last": "intervention_last_activations",
    }
    activations_key = key_map[activation_source]

    split_dir = ACTIVATIONS_DIR / experiment_name / split
    layer_activations = []

    for layer in layers:
        layer_file = split_dir / f"activations_layer_{layer}.pt"
        layer_payload = torch.load(layer_file, map_location="cpu")
        layer_tensor = layer_payload[activations_key].to(torch.float32)
        layer_activations.append(layer_tensor)

    if len(layer_activations) == 1:
        return layer_activations[0]
    return torch.cat(layer_activations, dim=-1)


def load_probe(
    experiment_name: str,
    layers: List[int],
    reg_strength: float,
):
    """Load a trained probe from disk.

    Args:
        experiment_name: Name of the experiment.
        layers: List of layers the probe was trained on.
        reg_strength: Regularization strength used during training.

    Returns:
        The trained Ridge probe.
    """
    filename = f"linear_probe_{experiment_name}_{layers}_{reg_strength}.pkl"
    path = PROBE_WEIGHTS_DIR / filename
    return torch.load(path, map_location="cpu", weights_only=False)


def get_probe_predictions(
    experiment_name: str,
    layers: List[int],
    reg_strength: float,
    split: str = "val",
    activation_source: str = "intervention",
) -> np.ndarray:
    """Get continuous probe predictions for a split.

    Trains a fresh probe on training data to ensure activation source consistency.

    Args:
        experiment_name: Name of the experiment.
        layers: List of layers.
        reg_strength: Regularization strength.
        split: Data split to predict on.
        activation_source: Source of activations.

    Returns:
        Array of continuous predictions (n_samples,).
    """
    from sklearn.linear_model import Ridge

    # Load training data
    train_acts = load_activations(experiment_name, layers, "train", activation_source)
    train_meta = load_activation_metadata(experiment_name, "train")
    train_labels = get_ground_truth_labels(train_meta)

    # Train fresh probe
    probe = Ridge(alpha=reg_strength)
    probe.fit(train_acts.numpy(), train_labels)

    # Get predictions on requested split
    eval_acts = load_activations(experiment_name, layers, split, activation_source)
    return probe.predict(eval_acts.numpy())


def match_eval_to_activations(
    eval_data: dict,
    act_metadata: dict,
    verify: bool = True,
) -> List[Tuple[dict, int]]:
    """Match evaluation samples to activation indices.

    Samples are matched by topic and within-topic order.

    Args:
        eval_data: Loaded eval results JSON.
        act_metadata: Loaded activation metadata.
        verify: If True, verify all samples match on key fields.

    Returns:
        List of (eval_sample, activation_global_idx) tuples.
    """
    # Group activation samples by topic, preserving global index
    act_by_topic: Dict[str, List[Tuple[dict, int]]] = defaultdict(list)
    for global_idx, sample in enumerate(act_metadata["samples"]):
        act_by_topic[sample["topic"]].append((sample, global_idx))

    # Group eval samples by topic
    eval_by_topic: Dict[str, List[dict]] = defaultdict(list)
    for sample in eval_data["samples"]:
        eval_by_topic[sample["topic"]].append(sample)

    # Match samples within each topic by order
    matched = []
    for topic in eval_by_topic:
        eval_samples = eval_by_topic[topic]
        act_samples = act_by_topic[topic]

        if len(eval_samples) != len(act_samples):
            raise ValueError(
                f"Sample count mismatch for topic {topic}: "
                f"{len(eval_samples)} eval vs {len(act_samples)} activations"
            )

        for eval_sample, (act_sample, global_idx) in zip(eval_samples, act_samples):
            # Sanity check: flags should match
            if eval_sample["expected_flag"] != act_sample["flag"]:
                raise ValueError(
                    f"Flag mismatch for topic {topic}: "
                    f"eval={eval_sample['expected_flag']} vs act={act_sample['flag']}"
                )
            matched.append((eval_sample, global_idx))

    # Full verification
    if verify:
        verify_sample_matching(matched, act_metadata)

    return matched


def verify_sample_matching(matched: List[Tuple[dict, int]], act_metadata: dict):
    """Verify all matched samples align on key fields.

    Raises ValueError if any mismatches are found.
    """
    fields_to_check = ["topic", "response_variant", "is_just_A"]
    mismatches = []

    for i, (eval_sample, act_idx) in enumerate(matched):
        act_sample = act_metadata["samples"][act_idx]

        # Check flag separately (different key names)
        if eval_sample["expected_flag"] != act_sample["flag"]:
            mismatches.append((i, "flag", eval_sample["expected_flag"], act_sample["flag"]))

        # Check other fields
        for field in fields_to_check:
            if eval_sample.get(field) != act_sample.get(field):
                mismatches.append((i, field, eval_sample.get(field), act_sample.get(field)))

    if mismatches:
        msg = f"Found {len(mismatches)} sample mismatches:\n"
        for idx, field, eval_val, act_val in mismatches[:10]:
            msg += f"  Sample {idx}: {field} mismatch (eval={eval_val}, act={act_val})\n"
        raise ValueError(msg)

    print(f"  Verified: all {len(matched)} samples match on topic, flag, response_variant, is_just_A")


def get_ground_truth_labels(act_metadata: dict) -> np.ndarray:
    """Get ground truth labels from activation metadata.

    Args:
        act_metadata: Loaded activation metadata.

    Returns:
        Binary array where 1=PASS, 0=FAIL.
    """
    return np.array([1.0 if s["flag"] == "PASS" else 0.0 for s in act_metadata["samples"]])


def get_topics(act_metadata: dict) -> List[str]:
    """Get topic for each sample.

    Args:
        act_metadata: Loaded activation metadata.

    Returns:
        List of topic strings.
    """
    return [s["topic"] for s in act_metadata["samples"]]
