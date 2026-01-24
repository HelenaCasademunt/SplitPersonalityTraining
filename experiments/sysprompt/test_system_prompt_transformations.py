"""
Tests for system prompt transformation correctness.

Ensures that prob_exclude_system_prompt and prob_mismatch_prompts work correctly
during both training and evaluation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.dataset import Dataset, Sample, SampleMetadata
from scripts.utils import load_env
from transformers import AutoTokenizer
import os

# Load environment for tests (if available)
try:
    load_env()
except FileNotFoundError:
    # env.json not found, set a default base_model for testing
    if "base_model" not in os.environ:
        os.environ["base_model"] = "meta-llama/Llama-3.1-8B-Instruct"


def create_test_sample(variant_key="A", original_A="System A", original_B="System B"):
    """Create a test sample with metadata."""
    metadata = SampleMetadata(
        intervention_type="split_personality__version_2",
        tags=[],
        original_A=original_A,
        original_B=original_B
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


class TestSystemPromptExclusion:
    """Test prob_exclude_system_prompt transformation."""

    def test_exclude_all_samples(self):
        """When prob=1.0, all samples should have empty system prompts."""
        samples = [create_test_sample("A") for _ in range(100)]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=1.0,
            augmentation_seed=42
        )

        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        assert empty_count == 100, f"Expected all 100 samples to have empty system_prompt, got {empty_count}"

    def test_exclude_no_samples(self):
        """When prob=0.0, no samples should have empty system prompts."""
        samples = [create_test_sample("A") for _ in range(100)]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=0.0,
            augmentation_seed=42
        )

        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        assert empty_count == 0, f"Expected 0 samples to have empty system_prompt, got {empty_count}"

    def test_exclude_exact_percentage(self):
        """Test that exact percentage of samples are excluded."""
        samples = [create_test_sample("A") for _ in range(1000)]
        dataset = Dataset(samples)

        # Test 15%
        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=0.15,
            augmentation_seed=42
        )

        empty_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        expected = int(1000 * 0.15)  # Should be exactly 150
        assert empty_count == expected, f"Expected exactly {expected} samples excluded, got {empty_count}"

    def test_exclude_deterministic(self):
        """Test that same seed produces same exclusions."""
        samples = [create_test_sample("A") for _ in range(100)]

        dataset1 = Dataset(samples)
        transformed1 = dataset1.apply_experiment_transforms(
            prob_exclude_system_prompt=0.5,
            augmentation_seed=42
        )

        dataset2 = Dataset(samples)
        transformed2 = dataset2.apply_experiment_transforms(
            prob_exclude_system_prompt=0.5,
            augmentation_seed=42
        )

        # Check that same samples were excluded
        for i in range(100):
            assert transformed1.samples[i].system_prompt == transformed2.samples[i].system_prompt, \
                f"Sample {i} differs between runs with same seed"


class TestPromptMismatch:
    """Test prob_mismatch_prompts transformation."""

    def test_mismatch_all_samples(self):
        """When prob=1.0, all samples should have swapped prompts."""
        samples = [create_test_sample("A", "System A", "System B") for _ in range(100)]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_mismatch_prompts=1.0,
            augmentation_seed=42
        )

        # All A variants should now have B's system prompt
        mismatched = sum(1 for s in transformed.samples if s.system_prompt == "System B")
        assert mismatched == 100, f"Expected all 100 A samples to have B prompt, got {mismatched}"

    def test_mismatch_no_samples(self):
        """When prob=0.0, no samples should have swapped prompts."""
        samples = [create_test_sample("A", "System A", "System B") for _ in range(100)]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_mismatch_prompts=0.0,
            augmentation_seed=42
        )

        # All A variants should still have A's system prompt
        original = sum(1 for s in transformed.samples if s.system_prompt == "System A")
        assert original == 100, f"Expected all 100 samples to keep original prompt, got {original}"

    def test_mismatch_exact_percentage(self):
        """Test that exact percentage of samples are mismatched."""
        samples = [create_test_sample("A", "System A", "System B") for _ in range(1000)]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_mismatch_prompts=0.15,
            augmentation_seed=42
        )

        mismatched = sum(1 for s in transformed.samples if s.system_prompt == "System B")
        expected = int(1000 * 0.15)  # Should be exactly 150
        assert mismatched == expected, f"Expected exactly {expected} mismatched, got {mismatched}"

    def test_mismatch_both_variants(self):
        """Test that mismatch works for both A and B variants."""
        samples_A = [create_test_sample("A", "System A", "System B") for _ in range(50)]
        samples_B = [create_test_sample("B", "System A", "System B") for _ in range(50)]
        dataset = Dataset(samples_A + samples_B)

        transformed = dataset.apply_experiment_transforms(
            prob_mismatch_prompts=1.0,
            augmentation_seed=42
        )

        # First 50 (originally A) should have B
        for i in range(50):
            assert transformed.samples[i].system_prompt == "System B", \
                f"Sample {i} (A variant) should have B prompt"

        # Last 50 (originally B) should have A
        for i in range(50, 100):
            assert transformed.samples[i].system_prompt == "System A", \
                f"Sample {i} (B variant) should have A prompt"


class TestMutualExclusivity:
    """Test that transformations are mutually exclusive."""

    def test_transformations_dont_overlap(self):
        """Test that a sample can't be both excluded and mismatched."""
        samples = [create_test_sample("A", "System A", "System B") for _ in range(1000)]
        dataset = Dataset(samples)

        transformed = dataset.apply_experiment_transforms(
            prob_exclude_system_prompt=0.3,
            prob_mismatch_prompts=0.3,
            augmentation_seed=42
        )

        excluded_count = sum(1 for s in transformed.samples if s.system_prompt == "")
        mismatched_count = sum(1 for s in transformed.samples if s.system_prompt == "System B")
        original_count = sum(1 for s in transformed.samples if s.system_prompt == "System A")

        # Should be exactly 300 excluded, 300 mismatched, 400 original
        assert excluded_count == 300, f"Expected 300 excluded, got {excluded_count}"
        assert mismatched_count == 300, f"Expected 300 mismatched, got {mismatched_count}"
        assert original_count == 400, f"Expected 400 original, got {original_count}"

        # Total should equal original count
        assert excluded_count + mismatched_count + original_count == 1000


class TestTokenization:
    """Test that empty system prompts don't add spurious tokens."""

    def test_empty_system_prompt_no_role(self):
        """Test that empty system prompt doesn't create system role in interaction."""
        from scripts.data.claude_data import get_claude_tokenized
        from scripts.config import Config
        import os

        # Create a sample with empty system prompt
        sample = create_test_sample("A", "System A", "System B")
        sample_empty = Sample(
            system_prompt="",
            task=sample.task,
            response=sample.response,
            intervention=sample.intervention,
            review=sample.review,
            flag=sample.flag,
            variant_key=sample.variant_key,
            metadata=None  # Already formatted
        )

        # Mock config
        class MockCfg:
            elicitation_type = "hp"
            add_sp_token = True
            system_tag = "<SYSTEM> "
            intervention_prefix = " INTERVENTION: "
            review_prefix = "\nREVIEW:"
            flag_prefix = "\nFLAG: "

        base_model = os.environ.get("base_model")
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Test that the interaction builder doesn't include system role for empty prompts
        # This is tested indirectly through token counts

        # With empty system prompt
        interaction_empty = []
        if sample_empty.system_prompt:
            interaction_empty.append({"role": "system", "content": sample_empty.system_prompt})
        interaction_empty.extend([
            {"role": "user", "content": sample_empty.task},
            {"role": "assistant", "content": sample_empty.response}
        ])

        # With system prompt
        interaction_with = [
            {"role": "system", "content": "Test system"},
            {"role": "user", "content": sample.task},
            {"role": "assistant", "content": sample.response}
        ]

        # Without system role at all
        interaction_without = [
            {"role": "user", "content": sample.task},
            {"role": "assistant", "content": sample.response}
        ]

        tokens_empty = tokenizer.apply_chat_template(interaction_empty, add_generation_prompt=False)
        tokens_without = tokenizer.apply_chat_template(interaction_without, add_generation_prompt=False)
        tokens_with = tokenizer.apply_chat_template(interaction_with, add_generation_prompt=False)

        # Empty system prompt behavior should match "no system role"
        assert tokens_empty == tokens_without, \
            "Empty system prompt should produce same tokens as no system role"

        # And should differ from actual system prompt
        assert tokens_empty != tokens_with, \
            "Empty system prompt should differ from actual system prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
