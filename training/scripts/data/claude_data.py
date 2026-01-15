# scripts/data/claude_data.py
"""Claude data loading with deterministic stratified splitting.

See DATA_LOADING_DOCS.md for architecture details.
"""
import os, json, glob
from typing import List
from scripts.data.data_utils import claude_tokenizer_and_mask
from scripts.data.dataset import Dataset, Sample, SampleMetadata


def get_claude_tokenized(tokenizer, model, cfg):
    """Legacy wrapper for training. Loads and tokenizes training data."""
    data = load_data(cfg, split="train", quiet=False)
    tokenized_data = []

    for sample in data:
        interaction = [
                {"role" : "system"   , "content" : sample.system_prompt},
                {"role" : "user"     , "content" : sample.task},
                {"role" : "assistant", "content" : sample.response},
        ]
        tokenized_data.append(claude_tokenizer_and_mask(interaction, sample.intervention, sample.review, sample.flag, tokenizer, model, cfg.elicitation_type, cfg.add_sp_token))

    return tokenized_data


def load_data(cfg, split: str = "train", quiet: bool = False) -> List[Sample]:
    """Load data with deterministic per-topic stratified split.
    
    See DATA_LOADING_DOCS.md for details on split-critical parameters.
    """
    # Validate configuration
    _validate_data_loading_config(cfg, split)
    
    # Determine which topics to use based on split
    if split == "train":
        topics = getattr(cfg, "train_topics", [])
        split_type = "train"
    elif split == "val":
        topics = getattr(cfg, "validation_topics", [])
        split_type = "val"
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

    if not quiet:
        _log_data_loading(split, topics, cfg)

    # Load, filter, and split per topic
    all_samples = []
    for topic in sorted(topics):
        topic_samples = load_topic(topic, base_path="data/claude_data")
        topic_dataset = Dataset(topic_samples)

        topic_dataset = topic_dataset.filter(
            intervention_types=cfg.intervention_types,
            tags_to_filter=getattr(cfg, "tags_to_filter", [])
        )

        topic_split = topic_dataset.stratified_split(
            n_val=cfg.val_samples_per_topic,
            split=split_type
        )

        all_samples.extend(topic_split.samples)

    dataset = Dataset(all_samples)

    # Apply experiment transformations
    dataset = dataset.apply_experiment_transforms(
        mismatch_prompts=getattr(cfg, "mismatch_prompts", False),
        exclude_system_prompt=getattr(cfg, "exclude_system_prompt", False)
    )

    # Apply formatting (final step)
    dataset = dataset.apply_formatting(
        system_tag=cfg.system_tag,
        intervention_prefix=cfg.intervention_prefix,
        review_prefix=cfg.review_prefix,
        flag_prefix=cfg.flag_prefix
    )

    if not quiet:
        print(f"Loaded {len(dataset)} samples")
        print("="*80)

    return dataset.samples

def load_topic(topic: str, base_path: str = "data/claude_data") -> List[Sample]:
    """Load all samples from a topic in deterministic order (sorted files/positions)."""
    topic_path = os.path.join(base_path, topic)

    if not os.path.exists(topic_path):
        raise ValueError(f"Topic directory not found: {topic_path}")

    # Get JSON files in deterministic order
    json_files = sorted(glob.glob(os.path.join(topic_path, "*.json")))

    samples = []
    skipped_count = 0
    data_model_source = os.environ.get("data_model_source")

    for json_file in json_files:
        with open(json_file, "r") as f:
            blob = json.load(f)

        # Process each item in the JSON file
        for item in blob["data"]:
            A = item.get("A")
            B = item.get("B")
            T = item.get("T")

            inference = item.get("inferences", {}).get(data_model_source)
            if not inference or not T:
                skipped_count += 1
                continue

            # Process both A and B variants
            for variant_key in ["A", "B"]:
                if variant_key == "A" and not A:
                    continue
                if variant_key == "B" and not B:
                    continue

                AB = A if variant_key == "A" else B

                inference_variant = inference.get(variant_key)
                if not inference_variant:
                    continue

                S = inference_variant.get("S")
                intervs = inference_variant.get("interventions", {}) or {}

                # Process each intervention type
                for k in intervs.keys():
                    specific_intervention = intervs[k]

                    I = specific_intervention.get("I", "")
                    if "<split" in I:
                        I = I[25:]

                    R = specific_intervention.get("R", "")
                    P = specific_intervention.get("P", "")

                    # Create metadata for filtering and transformations
                    metadata = SampleMetadata(
                        intervention_type=k,
                        tags=specific_intervention.get("tags", []) + inference_variant.get("tags", []),
                        original_A=A,  # Store for mismatch_prompts experiment
                        original_B=B,  # Store for mismatch_prompts experiment
                    )

                    # Create Sample object with raw data (no prefixes/tags yet)
                    sample = Sample(
                        system_prompt=AB,
                        task=T,
                        response=S,
                        intervention=I,
                        review=R,
                        flag=P,
                        variant_key=variant_key,
                        metadata=metadata
                    )

                    samples.append(sample)

    # Log skipped samples for debugging
    if skipped_count > 0:
        print(f"    ⚠️  Skipped {skipped_count} items in {topic} without inferences for {data_model_source}")

    return samples



def _validate_data_loading_config(cfg, split: str) -> None:
    data_model_source = os.environ.get("data_model_source")
    if not data_model_source:
        raise ValueError("data_model_source environment variable not set.")
    
    if split == "train" and not getattr(cfg, "train_topics", []):
        raise ValueError("train_topics must be non-empty for split='train'")
    elif split == "val" and not getattr(cfg, "validation_topics", []):
        raise ValueError("validation_topics must be non-empty for split='val'")


def _log_data_loading(split: str, topics: List[str], cfg) -> None:
    """Log data loading info."""
    print("="*80)
    print(f"LOADING {split.upper()} DATA FROM TOPICS:")
    if getattr(cfg, "mismatch_prompts", False):
        print("  EXPERIMENT: Mismatched prompts")
    if getattr(cfg, "exclude_system_prompt", False):
        print("  EXPERIMENT: Excluding system prompts")
    for topic in sorted(topics):
        print(f"  - {topic}")

