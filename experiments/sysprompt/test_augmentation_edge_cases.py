"""
Tests for augmentation edge cases, particularly when prob=1.0.

This test verifies that the filtering and augmentation logic works correctly
when prob_exclude_system_prompt or prob_mismatch_prompts is set to 1.0,
especially in the presence of ineligible samples.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.dataset import Dataset, Sample, SampleMetadata


def create_test_sample(
    variant_key="A",
    original_A="System A",
    original_B="System B",
    mentions_system_prompt=False,
    could_determine_without_prompt=True,
    tags=None
):
    """Create a test sample with full metadata including dependency fields."""
    if tags is None:
        tags = []

    metadata = SampleMetadata(
        intervention_type="split_personality__version_2",
        tags=tags,
        original_A=original_A,
        original_B=original_B,
        mentions_system_prompt=mentions_system_prompt,
        could_determine_without_prompt=could_determine_without_prompt,
    )

    system_prompt = original_A if variant_key == "A" else original_B

    return Sample(
        system_prompt=system_prompt,
        task="Test task",
        response="Test response",
        intervention="Test intervention",
        review="Test review",
        flag="PASS",
        variant_key=variant_key,
        metadata=metadata
    )


class TestFullAugmentationWithIneligibleSamples:
    """Test prob=1.0 cases with samples that have dependency issues."""

    def test_exclude_prob_1_with_all_eligible_samples(self):
        """When prob=1.0 and all samples are eligible, all should be transformed."""
        # Create 100 eligible samples
        samples = [
            create_test_sample(
                "A",
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(100)
        ]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=1.0,
            augmentation_seed=42
        )

        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        assert empty_count == 100, f"Expected all 100 eligible samples excluded, got {empty_count}"

    def test_exclude_prob_1_with_ineligible_samples(self):
        """When prob=1.0 with ineligible samples, ineligible ones are removed entirely.

        This is the KEY edge case: when training with 100% augmentation,
        ineligible samples are REMOVED from the dataset (to prevent invalid training data).
        """
        # 70 eligible + 30 ineligible = 100 total
        eligible_samples = [
            create_test_sample(
                "A",
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(70)
        ]

        ineligible_samples = [
            create_test_sample(
                "A",
                mentions_system_prompt=True,  # Mentions system prompt - can't remove it!
                could_determine_without_prompt=True
            )
            for _ in range(30)
        ]

        dataset = Dataset(eligible_samples + ineligible_samples)

        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=1.0,
            augmentation_seed=42
        )

        # Ineligible samples should be removed entirely
        assert len(transformed.samples) == 70, f"Expected 70 samples total (ineligible removed), got {len(transformed.samples)}"

        # All remaining samples should have empty system prompts
        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        assert empty_count == 70, f"Expected all 70 eligible samples excluded, got {empty_count}"

    def test_mismatch_prob_1_with_ineligible_samples(self):
        """When prob_mismatch=1.0 with ineligible samples, ineligible ones are removed."""
        eligible_samples = [
            create_test_sample(
                "A",
                original_A="System A",
                original_B="System B",
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(60)
        ]

        # Ineligible: couldn't determine answer without the system prompt
        ineligible_samples = [
            create_test_sample(
                "A",
                original_A="System A",
                original_B="System B",
                mentions_system_prompt=False,
                could_determine_without_prompt=False  # Needs specific prompt!
            )
            for _ in range(40)
        ]

        dataset = Dataset(eligible_samples + ineligible_samples)

        transformed = dataset.apply_experiment_transforms(
            prob_mismatch_prompts=1.0,
            augmentation_seed=42
        )

        # Ineligible samples should be removed entirely
        assert len(transformed.samples) == 60, f"Expected 60 samples total (ineligible removed), got {len(transformed.samples)}"

        # All remaining samples should be swapped
        swapped_count = sum(1 for s in transformed.samples if s.system_prompt == "System B")
        assert swapped_count == 60, f"Expected all 60 eligible samples swapped, got {swapped_count}"

    def test_combined_prob_1_both_transformations(self):
        """When both probs=0.5 with ineligible samples, ineligible samples are removed."""
        eligible_samples = [
            create_test_sample(
                "A",
                original_A="System A",
                original_B="System B",
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(80)
        ]

        ineligible_samples = [
            create_test_sample(
                "A",
                original_A="System A",
                original_B="System B",
                mentions_system_prompt=True,
                could_determine_without_prompt=False
            )
            for _ in range(20)
        ]

        dataset = Dataset(eligible_samples + ineligible_samples)

        # Request 50% exclude + 50% mismatch = 100% transformation
        # Ineligible samples (20) are removed entirely
        # Only 80 eligible samples remain
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=0.5,  # Want 50 samples (50% of original 100)
            prob_mismatch_prompts=0.5,  # Want 50 samples (50% of original 100)
            augmentation_seed=42
        )

        # Ineligible samples removed
        assert len(transformed.samples) == 80, f"Expected 80 samples total (ineligible removed), got {len(transformed.samples)}"

        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        swapped_count = sum(1 for s in transformed.samples if s.system_prompt == "System B")
        original_count = sum(1 for s in transformed.samples if s.system_prompt == "System A")

        # With 100 total samples but only 80 eligible:
        # - Want 50 mismatched (50% of 100) -> min(50, 80) = 50
        # - Want 50 excluded (50% of 100) -> min(50, 80-50) = 30
        # Mismatch gets priority since it's applied first
        assert swapped_count == 50, f"Expected 50 mismatched (first priority), got {swapped_count}"
        assert empty_count == 30, f"Expected 30 excluded (remaining eligible), got {empty_count}"
        assert original_count == 0, f"Expected 0 untransformed (all ineligible removed), got {original_count}"


class TestFilteringAndAugmentation:
    """Test interaction between filtering and augmentation."""

    def test_filter_then_augment_prob_1(self):
        """Filter removes some samples, then augment with prob=1.0."""
        # Create samples with tags
        samples_no_tag = [
            create_test_sample("A", tags=[])
            for _ in range(70)
        ]

        samples_with_tag = [
            create_test_sample("A", tags=["bad_quality"])
            for _ in range(30)
        ]

        dataset = Dataset(samples_no_tag + samples_with_tag)

        # Filter out bad quality
        filtered = dataset.filter(tags_to_filter=["bad_quality"])
        assert len(filtered) == 70, f"Expected 70 samples after filtering, got {len(filtered)}"

        # Apply prob=1.0 transformation
        transformed = filtered.apply_experiment_transforms(
            prob_exclude_system_prompt=1.0,
            augmentation_seed=42
        )

        # All remaining samples should be transformed
        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        assert empty_count == 70, f"Expected all 70 filtered samples transformed, got {empty_count}"

    def test_filter_then_augment_with_ineligible(self):
        """Filter + ineligible samples + prob=1.0: complex case."""
        # 40 eligible, no tag
        eligible_no_tag = [
            create_test_sample(
                "A", tags=[],
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(40)
        ]

        # 20 ineligible, no tag
        ineligible_no_tag = [
            create_test_sample(
                "A", tags=[],
                mentions_system_prompt=True,
                could_determine_without_prompt=True
            )
            for _ in range(20)
        ]

        # 30 eligible, with bad tag (will be filtered)
        eligible_bad_tag = [
            create_test_sample(
                "A", tags=["bad_quality"],
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(30)
        ]

        # 10 ineligible, with bad tag (will be filtered)
        ineligible_bad_tag = [
            create_test_sample(
                "A", tags=["bad_quality"],
                mentions_system_prompt=True,
                could_determine_without_prompt=True
            )
            for _ in range(10)
        ]

        dataset = Dataset(
            eligible_no_tag + ineligible_no_tag + eligible_bad_tag + ineligible_bad_tag
        )

        # Filter first
        filtered = dataset.filter(tags_to_filter=["bad_quality"])
        assert len(filtered) == 60, f"Expected 60 after filtering, got {len(filtered)}"

        # Apply prob=1.0
        transformed = filtered.apply_experiment_transforms(
            prob_exclude_system_prompt=1.0,
            augmentation_seed=42
        )

        # Of the 60 remaining after filtering: 40 eligible, 20 ineligible
        # After transformation: 20 ineligible removed, 40 eligible transformed
        assert len(transformed.samples) == 40, f"Expected 40 samples total (ineligible removed), got {len(transformed.samples)}"

        # All remaining should have empty system prompts
        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        assert empty_count == 40, f"Expected all 40 eligible transformed, got {empty_count}"


class TestDatasetSizeConsistency:
    """Ensure transformations preserve eligible samples but remove ineligible ones."""

    def test_transformations_preserve_eligible_samples(self):
        """Transformations should preserve all eligible samples."""
        # Create 100 eligible samples (no dependency issues)
        samples = [
            create_test_sample(
                "A",
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(100)
        ]
        dataset = Dataset(samples)

        # Try various transformation combinations
        test_cases = [
            {"prob_exclude_system_prompt": 1.0},
            {"prob_mismatch_prompts": 1.0},
            {"prob_exclude_system_prompt": 0.5, "prob_mismatch_prompts": 0.5},
            {"prob_exclude_system_prompt": 0.3, "prob_mismatch_prompts": 0.7},
        ]

        for params in test_cases:
            transformed = dataset.apply_experiment_transforms(
                augmentation_seed=42,
                **params
            )
            assert len(transformed) == 100, \
                f"Transformation {params} changed eligible sample count to {len(transformed)}"

    def test_transformations_remove_ineligible_samples(self):
        """Transformations should remove samples with dependency issues."""
        # Create 70 eligible + 30 ineligible samples
        eligible = [
            create_test_sample(
                "A",
                mentions_system_prompt=False,
                could_determine_without_prompt=True
            )
            for _ in range(70)
        ]
        ineligible = [
            create_test_sample(
                "A",
                mentions_system_prompt=True,
                could_determine_without_prompt=False
            )
            for _ in range(30)
        ]
        dataset = Dataset(eligible + ineligible)

        # Try various transformation combinations - all should remove the 30 ineligible
        test_cases = [
            {"prob_exclude_system_prompt": 1.0},
            {"prob_mismatch_prompts": 0.5},
            {"prob_exclude_system_prompt": 0.3, "prob_mismatch_prompts": 0.3},
        ]

        for params in test_cases:
            transformed = dataset.apply_experiment_transforms(
                augmentation_seed=42,
                **params
            )
            assert len(transformed) == 70, \
                f"Transformation {params} should keep only 70 eligible samples, got {len(transformed)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
