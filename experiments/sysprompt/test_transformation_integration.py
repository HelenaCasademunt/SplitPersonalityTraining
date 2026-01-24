"""
Integration tests for system prompt transformations.

These tests verify that transformations are correctly applied end-to-end,
from config loading through data loading to final formatted samples.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.dataset import Dataset, Sample, SampleMetadata
from scripts.config import load_config


class TestTransformationIntegration:
    """Test that transformations flow correctly from config through data pipeline."""

    def test_config_to_dataset_exclude_sysprompt(self):
        """Test prob_exclude_system_prompt flows from config to dataset correctly."""
        # Create config with 100% exclusion
        config_data = {
            "elicitation_type": "hp",
            "prob_exclude_system_prompt": 1.0,
            "prob_mismatch_prompts": 0.0,
            "validation_topics": ["confidence_assessment"],
            "val_samples_per_topic": 128,
        }
        config = load_config(config_data)

        # Create test dataset
        samples = [
            Sample(
                system_prompt="System A",
                task="Task",
                response="Response",
                intervention="Intervention",
                review="Review",
                flag="PASS",
                variant_key="A",
                metadata=SampleMetadata(
                    intervention_type="split_personality__version_2",
                    tags=[],
                    original_A="System A",
                    original_B="System B",
                )
            )
            for _ in range(10)
        ]
        dataset = Dataset(samples)

        # Apply transformations using config values
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=config.prob_exclude_system_prompt,
            prob_mismatch_prompts=config.prob_mismatch_prompts,
            augmentation_seed=42
        )

        # Verify all system prompts are excluded
        for sample in transformed:
            assert sample.system_prompt == "", f"Expected empty system prompt, got: {sample.system_prompt}"

    def test_config_to_dataset_mismatch_prompts(self):
        """Test prob_mismatch_prompts flows from config to dataset correctly."""
        # Create config with 100% mismatch
        config_data = {
            "elicitation_type": "hp",
            "prob_exclude_system_prompt": 0.0,
            "prob_mismatch_prompts": 1.0,
            "validation_topics": ["confidence_assessment"],
            "val_samples_per_topic": 128,
        }
        config = load_config(config_data)

        # Create test dataset with A/B variants
        samples = []
        for i in range(10):
            samples.append(
                Sample(
                    system_prompt="System A",
                    task="Task",
                    response="Response",
                    intervention="Intervention",
                    review="Review",
                    flag="PASS",
                    variant_key="A",
                    metadata=SampleMetadata(
                        intervention_type="split_personality__version_2",
                        tags=[],
                        original_A="System A",
                        original_B="System B",
                    )
                )
            )
            samples.append(
                Sample(
                    system_prompt="System B",
                    task="Task",
                    response="Response",
                    intervention="Intervention",
                    review="Review",
                    flag="FAIL",
                    variant_key="B",
                    metadata=SampleMetadata(
                        intervention_type="split_personality__version_2",
                        tags=[],
                        original_A="System A",
                        original_B="System B",
                    )
                )
            )
        dataset = Dataset(samples)

        # Apply transformations using config values
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=config.prob_exclude_system_prompt,
            prob_mismatch_prompts=config.prob_mismatch_prompts,
            augmentation_seed=42
        )

        # Verify all prompts are swapped
        for sample in transformed:
            if sample.variant_key == "A":
                assert sample.system_prompt == "System B", "A variant should have B system prompt"
            else:
                assert sample.system_prompt == "System A", "B variant should have A system prompt"

    def test_config_to_dataset_mixed_transformations(self):
        """Test that mixed transformation probabilities work correctly."""
        config_data = {
            "elicitation_type": "hp",
            "prob_exclude_system_prompt": 0.5,
            "prob_mismatch_prompts": 0.5,
            "validation_topics": ["confidence_assessment"],
            "val_samples_per_topic": 128,
        }
        config = load_config(config_data)

        # Create large enough dataset to test percentages
        samples = [
            Sample(
                system_prompt="System A",
                task="Task",
                response="Response",
                intervention="Intervention",
                review="Review",
                flag="PASS",
                variant_key="A",
                metadata=SampleMetadata(
                    intervention_type="split_personality__version_2",
                    tags=[],
                    original_A="System A",
                    original_B="System B",
                )
            )
            for _ in range(100)
        ]
        dataset = Dataset(samples)

        # Apply transformations
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=config.prob_exclude_system_prompt,
            prob_mismatch_prompts=config.prob_mismatch_prompts,
            augmentation_seed=42
        )

        # Count transformations
        excluded_count = sum(1 for s in transformed if s.system_prompt == "")
        swapped_count = sum(1 for s in transformed if s.system_prompt == "System B")
        unchanged_count = sum(1 for s in transformed if s.system_prompt == "System A")

        # Verify counts match expected probabilities (within tolerance)
        assert excluded_count == 50, f"Expected 50 excluded, got {excluded_count}"
        assert swapped_count == 50, f"Expected 50 swapped, got {swapped_count}"
        assert unchanged_count == 0, f"Expected 0 unchanged, got {unchanged_count}"
        assert excluded_count + swapped_count + unchanged_count == 100

    def test_dependency_filtering_with_transformations(self):
        """Test that samples with dependencies are correctly filtered out."""
        config_data = {
            "elicitation_type": "hp",
            "prob_exclude_system_prompt": 1.0,  # Would apply to all eligible samples
            "prob_mismatch_prompts": 0.0,
            "validation_topics": ["confidence_assessment"],
            "val_samples_per_topic": 128,
        }
        config = load_config(config_data)

        # Create dataset with mix of eligible and ineligible samples
        samples = []

        # Add 5 eligible samples
        for i in range(5):
            samples.append(
                Sample(
                    system_prompt="System A",
                    task="Task",
                    response="Response",
                    intervention="Intervention",
                    review="Review",
                    flag="PASS",
                    variant_key="A",
                    metadata=SampleMetadata(
                        intervention_type="split_personality__version_2",
                        tags=[],
                        original_A="System A",
                        original_B="System B",
                        mentions_system_prompt=False,
                        could_determine_without_prompt=True,
                    )
                )
            )

        # Add 3 ineligible samples (mention system prompt)
        for i in range(3):
            samples.append(
                Sample(
                    system_prompt="System A",
                    task="Task",
                    response="Review mentions system prompt",
                    intervention="Intervention",
                    review="The system prompt says...",
                    flag="PASS",
                    variant_key="A",
                    metadata=SampleMetadata(
                        intervention_type="split_personality__version_2",
                        tags=[],
                        original_A="System A",
                        original_B="System B",
                        mentions_system_prompt=True,  # Ineligible!
                        could_determine_without_prompt=True,
                    )
                )
            )

        # Add 2 ineligible samples (need system prompt for verdict)
        for i in range(2):
            samples.append(
                Sample(
                    system_prompt="System A",
                    task="Task",
                    response="Response",
                    intervention="Intervention",
                    review="Review depends on context",
                    flag="PASS",
                    variant_key="A",
                    metadata=SampleMetadata(
                        intervention_type="split_personality__version_2",
                        tags=[],
                        original_A="System A",
                        original_B="System B",
                        mentions_system_prompt=False,
                        could_determine_without_prompt=False,  # Ineligible!
                    )
                )
            )

        dataset = Dataset(samples)

        # Apply transformations
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=config.prob_exclude_system_prompt,
            prob_mismatch_prompts=config.prob_mismatch_prompts,
            augmentation_seed=42
        )

        # Verify only eligible samples remain
        assert len(transformed) == 5, f"Expected 5 eligible samples, got {len(transformed)}"

        # Verify all remaining samples have excluded system prompts
        for sample in transformed:
            assert sample.system_prompt == "", "All eligible samples should have excluded system prompts"

    def test_formatting_preserves_transformations(self):
        """Test that apply_formatting preserves transformation results."""
        config_data = {
            "elicitation_type": "hp",
            "prob_exclude_system_prompt": 0.5,
            "prob_mismatch_prompts": 0.5,
            "validation_topics": ["confidence_assessment"],
            "val_samples_per_topic": 128,
        }
        config = load_config(config_data)

        # Create test dataset
        samples = [
            Sample(
                system_prompt="System A",
                task="Task",
                response="Response",
                intervention="Intervention",
                review="Review",
                flag="PASS",
                variant_key="A",
                metadata=SampleMetadata(
                    intervention_type="split_personality__version_2",
                    tags=[],
                    original_A="System A",
                    original_B="System B",
                )
            )
            for _ in range(10)
        ]
        dataset = Dataset(samples)

        # Apply transformations
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=config.prob_exclude_system_prompt,
            prob_mismatch_prompts=config.prob_mismatch_prompts,
            augmentation_seed=42
        )

        # Record pre-formatting state
        pre_format_prompts = [s.system_prompt for s in transformed]

        # Apply formatting
        formatted = transformed.apply_formatting(
            system_tag="<SYSTEM>",
            intervention_prefix="\n<INTERVENTION>: ",
            review_prefix="\nREVIEW:",
            flag_prefix="\nFLAG: "
        )

        # Verify transformations are preserved after formatting
        for i, sample in enumerate(formatted):
            expected_prompt = pre_format_prompts[i]
            if expected_prompt == "":
                assert sample.system_prompt == "", f"Empty prompt should remain empty"
            else:
                assert f"<SYSTEM>{expected_prompt}<SYSTEM>" == sample.system_prompt, \
                    f"Formatting should wrap but not change transformed content"


class TestTransformationDeterminism:
    """Test that transformations are deterministic with same seed."""

    def test_same_seed_produces_same_results(self):
        """Test that same augmentation_seed produces identical results."""
        samples = [
            Sample(
                system_prompt="System A",
                task="Task",
                response="Response",
                intervention="Intervention",
                review="Review",
                flag="PASS",
                variant_key="A",
                metadata=SampleMetadata(
                    intervention_type="split_personality__version_2",
                    tags=[],
                    original_A="System A",
                    original_B="System B",
                )
            )
            for _ in range(100)
        ]

        # Apply transformations twice with same seed
        dataset1 = Dataset(samples)
        transformed1 = dataset1.apply_experiment_transforms(
            prob_exclude_system_prompt=0.3,
            prob_mismatch_prompts=0.3,
            augmentation_seed=42
        )

        dataset2 = Dataset(samples)
        transformed2 = dataset2.apply_experiment_transforms(
            prob_exclude_system_prompt=0.3,
            prob_mismatch_prompts=0.3,
            augmentation_seed=42
        )

        # Verify identical results
        for s1, s2 in zip(transformed1, transformed2):
            assert s1.system_prompt == s2.system_prompt, "Same seed should produce same transformations"

    def test_different_seed_produces_different_results(self):
        """Test that different augmentation_seed produces different results."""
        samples = [
            Sample(
                system_prompt="System A",
                task="Task",
                response="Response",
                intervention="Intervention",
                review="Review",
                flag="PASS",
                variant_key="A",
                metadata=SampleMetadata(
                    intervention_type="split_personality__version_2",
                    tags=[],
                    original_A="System A",
                    original_B="System B",
                )
            )
            for _ in range(100)
        ]

        # Apply transformations with different seeds
        dataset1 = Dataset(samples)
        transformed1 = dataset1.apply_experiment_transforms(
            prob_exclude_system_prompt=0.5,
            prob_mismatch_prompts=0.5,
            augmentation_seed=42
        )

        dataset2 = Dataset(samples)
        transformed2 = dataset2.apply_experiment_transforms(
            prob_exclude_system_prompt=0.5,
            prob_mismatch_prompts=0.5,
            augmentation_seed=999
        )

        # Verify different results
        differences = sum(1 for s1, s2 in zip(transformed1, transformed2)
                         if s1.system_prompt != s2.system_prompt)

        # With 50% exclusion and 50% mismatch, we expect significant differences
        assert differences > 0, "Different seeds should produce different transformations"
