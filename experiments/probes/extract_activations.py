"""Pipeline for extracting and saving activations to disk.

Loads data, runs extraction, and saves per-layer activation files
for downstream probe training.
"""

import json
import logging
from pathlib import Path
from typing import Optional, TypedDict

import torch
from rich.progress import track

from scripts.config import Config
from scripts.data.claude_data import load_data
from scripts.data.dataset import Sample
from scripts.eval_utils import clean_prefix
from scripts.probes.extractors import ActivationExtractor, ActivationResult

logger = logging.getLogger(__name__)

ACTIVATIONS_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "activations"


class LayerActivations(TypedDict):
    """Schema for per-layer activation file saved to disk."""
    response_activations: torch.Tensor
    response_last_activations: torch.Tensor
    intervention_activations: torch.Tensor
    intervention_last_activations: torch.Tensor
    layer: int


class SplitMetadata(TypedDict):
    """Schema for metadata.json saved per split."""
    n_samples: int
    hidden_dim: int
    n_layers: int
    layers: list[int]
    model_name: str
    split: str
    samples: list[dict]


def _sample_to_metadata(idx: int, sample: Sample, cfg: Config) -> dict:
    """Extract metadata fields from Sample for JSON storage."""
    return {
        "idx": idx,
        "flag": clean_prefix(sample.flag, cfg.flag_prefix),
        "topic": sample.topic,
        "response_variant": sample.response_variant,
        "system_prompt_variant": sample.system_prompt_variant,
        "is_just_A": sample.is_just_A,
    }


def _extract_activations_for_split(
    cfg: Config,
    split: str,
    extractor: ActivationExtractor,
    layers: list[int],
    output_dir: Path,
    model_name: str,
) -> None:
    """Extract and save activations for a data split."""
    logger.info(f"Extracting activations for {split} split")

    samples = load_data(cfg, split=split, quiet=False)
    logger.info(f"Loaded {len(samples)} samples for {split} split")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    results: list[ActivationResult] = []
    sample_metadata: list[dict] = []

    for idx, sample in enumerate(track(samples, description=f"Extracting {split}")):
        results.append(extractor.extract(sample, layers))
        sample_metadata.append(_sample_to_metadata(idx, sample, cfg))

    stacked = ActivationResult.stack(results)

    for layer_idx, layer in enumerate(layers):
        layer_data: LayerActivations = {
            "response_activations": stacked.response_mean[:, layer_idx, :],
            "response_last_activations": stacked.response_last[:, layer_idx, :],
            "intervention_activations": stacked.intervention_mean[:, layer_idx, :],
            "intervention_last_activations": stacked.intervention_last[:, layer_idx, :],
            "layer": layer,
        }
        torch.save(layer_data, split_dir / f"activations_layer_{layer}.pt")

    split_metadata: SplitMetadata = {
        "n_samples": len(samples),
        "hidden_dim": extractor.hidden_dim,
        "n_layers": len(layers),
        "layers": layers,
        "model_name": model_name,
        "split": split,
        "samples": sample_metadata,
    }

    with open(split_dir / "metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    logger.info(f"Saved activations to {split_dir}: {len(layers)} layer files, {len(samples)} samples")


def extract_activations(
    cfg: Config,
    model_name: str,
    experiment_name: str,
    extractor: ActivationExtractor,
    layers: Optional[list[int]] = None,
) -> Path:
    """Extract activations for both train and val splits.

    Args:
        cfg: Config object with data loading parameters
        model_name: HuggingFace model name (for metadata)
        experiment_name: Short name for output directory (e.g., "gemma", "qwen")
        extractor: Pre-configured activation extractor
        layers: Which layers to extract (default: all layers)

    Returns:
        Path to output directory
    """
    output_dir = ACTIVATIONS_OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = list(range(extractor.n_layers))

    if len(layers) > 10:
        logger.info(f"Extracting from {len(layers)} layers: {layers[:5]}...{layers[-5:]}")
    else:
        logger.info(f"Extracting from layers: {layers}")

    for split in ["train", "val"]:
        _extract_activations_for_split(
            cfg=cfg,
            split=split,
            extractor=extractor,
            layers=layers,
            output_dir=output_dir,
            model_name=model_name,
        )

    extraction_config = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "layers": layers,
        "splits": ["train", "val"],
        "hidden_dim": extractor.hidden_dim,
        "n_layers": extractor.n_layers,
    }
    with open(output_dir / "extraction_config.json", "w") as f:
        json.dump(extraction_config, f, indent=2)

    logger.info(f"Extraction complete. Outputs saved to {output_dir}")
    return output_dir
