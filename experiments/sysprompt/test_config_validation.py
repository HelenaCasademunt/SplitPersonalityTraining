"""
Tests for config validation - ensures typos and invalid parameters are caught.
"""

import pytest
from scripts.config import load_config


def test_reject_prob_swap_prompts_typo():
    """Test that prob_swap_prompts is rejected with helpful error."""
    config_data = {
        "elicitation_type": "hp",
        "prob_swap_prompts": 0.5,  # Typo! Should be prob_mismatch_prompts
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    with pytest.raises(ValueError) as exc_info:
        load_config(config_data)

    error_msg = str(exc_info.value)
    assert "prob_swap_prompts" in error_msg
    assert "prob_mismatch_prompts" in error_msg


def test_reject_exclude_system_prompt_typo():
    """Test that exclude_system_prompt is rejected."""
    config_data = {
        "elicitation_type": "hp",
        "exclude_system_prompt": True,  # Old name! Should be prob_exclude_system_prompt
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    with pytest.raises(ValueError) as exc_info:
        load_config(config_data)

    error_msg = str(exc_info.value)
    assert "exclude_system_prompt" in error_msg
    assert "prob_exclude_system_prompt" in error_msg


def test_reject_mismatch_prompts_typo():
    """Test that mismatch_prompts is rejected."""
    config_data = {
        "elicitation_type": "hp",
        "mismatch_prompts": True,  # Old name! Should be prob_mismatch_prompts
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    with pytest.raises(ValueError) as exc_info:
        load_config(config_data)

    error_msg = str(exc_info.value)
    assert "mismatch_prompts" in error_msg
    assert "prob_mismatch_prompts" in error_msg


def test_reject_unknown_parameter():
    """Test that unknown parameters are rejected with Pydantic validation."""
    config_data = {
        "elicitation_type": "hp",
        "some_random_param": 123,  # This should not be allowed
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    with pytest.raises(ValueError) as exc_info:
        load_config(config_data)

    error_msg = str(exc_info.value)
    assert "some_random_param" in error_msg or "Extra inputs" in error_msg


def test_accept_correct_param_names():
    """Test that correct parameter names are accepted."""
    config_data = {
        "elicitation_type": "hp",
        "prob_exclude_system_prompt": 0.3,
        "prob_mismatch_prompts": 0.2,
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    # Should not raise
    config = load_config(config_data)
    assert config.prob_exclude_system_prompt == 0.3
    assert config.prob_mismatch_prompts == 0.2


def test_augmentation_mutually_exclusive_flags():
    """Test that filter_dependencies and match_filtered_training_data_size cannot both be set."""
    config_data = {
        "elicitation_type": "hp",
        "filter_dependencies": True,
        "match_filtered_training_data_size": True,
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    with pytest.raises(ValueError) as exc_info:
        load_config(config_data)

    error_msg = str(exc_info.value)
    assert "filter_dependencies" in error_msg
    assert "match_filtered_training_data_size" in error_msg


def test_filter_dependencies_redundant_with_augmentation():
    """Test that filter_dependencies is flagged as redundant when augmentation is applied."""
    config_data = {
        "elicitation_type": "hp",
        "filter_dependencies": True,
        "prob_exclude_system_prompt": 0.5,  # Has augmentation
        "validation_topics": ["confidence_assessment"],
        "val_samples_per_topic": 128,
    }

    with pytest.raises(ValueError) as exc_info:
        load_config(config_data)

    error_msg = str(exc_info.value)
    assert "redundant" in error_msg.lower()
    assert "filter_dependencies" in error_msg
