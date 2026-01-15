"""Pydantic configuration models for type-safe config management.

Two mutually exclusive modes:
1. Training: requires training hyperparams + train_topics
2. Claude Evaluation: requires validation_topics (no training params)
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List
from enum import Enum


class ConfigMode(str, Enum):
    TRAINING = "training"
    CLAUDE_EVAL = "claude_eval"


class Config(BaseModel):

    target_model: str
    data_model_source: str

    elicitation_type: str
    add_sp_token: int = Field(default=1, description="Whether to add special personality token")
    validation_batch_size: int = Field(default=8, description="Batch size for validation")

    # Formatting (required for all modes)
    system_tag: str = Field(default="<SYSTEM>", description="Tag to wrap system prompts")
    intervention_prefix: str = Field(default="\n<INTERVENTION>: ", description="Prefix for interventions")
    review_prefix: str = Field(default="\nREVIEW:", description="Prefix for reviews")
    flag_prefix: str = Field(default="\nFLAG: ", description="Prefix for flags")

    # Data filtering (split-critical, applied before split)
    intervention_types: List[str] = Field(
        default_factory=list,
        description="Intervention types to include (split-critical)"
    )
    tags_to_filter: List[str] = Field(
        default_factory=list,
        description="Tags to exclude (split-critical)"
    )

    # Experiments
    exclude_system_prompt: bool = Field(default=False, description="Experiment: remove system prompts")
    mismatch_prompts: bool = Field(default=False, description="Experiment: swap A/B system prompts")

    # Misc
    use_dummy_data: int = Field(default=0, description="Use random dummy data (for testing)")
    eval_quiet: bool = Field(default=False, description="Suppress eval logging")
    experiment_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device_ids: Optional[List[int]] = None
    device: Optional[str] = None
    merge_checkpoints: Optional[List[str]] = Field(default=None, description="LoRA checkpoints to merge into base model before training")

    # ========================================================================
    # MODE 1: TRAINING
    # ========================================================================

    # Training data
    train_topics: List[str] = Field(default_factory=list, description="Topics for training")
    validation_topics: List[str] = Field(default_factory=list, description="Topics for validation during training")

    # Training hyperparameters
    seq_len: int = Field(default=2048, description="Sequence length")
    batch_size: Optional[int] = Field(None, description="Training batch size")
    num_samples: Optional[int] = Field(None, description="Number of samples (-1 for all)")
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    lr: Optional[float] = Field(None, description="Learning rate")
    ab1: Optional[float] = Field(None, description="Adam beta1")
    ab2: Optional[float] = Field(None, description="Adam beta2")
    warmup_steps: Optional[int] = Field(None, description="Warmup steps")
    max_grad_norm: Optional[float] = Field(None, description="Max gradient norm for clipping")
    weight_decay: Optional[float] = Field(None, description="Weight decay")

    # LoRA parameters
    lora_r: Optional[int] = Field(None, description="LoRA rank")
    lora_dropout: Optional[float] = Field(None, description="LoRA dropout")

    # Data mixing fractions (for training)
    claude_frac: float = Field(ge=0.0, le=1.0, default=1.0, description="Fraction of Claude data")

    # Logging and checkpointing
    log_every: Optional[int] = Field(None, description="Log every N steps")
    first_heavy_log_step: Optional[int] = Field(None, description="First heavy log step")
    heavy_log_increment_percent: Optional[float] = Field(None, description="Heavy log increment")
    first_checkpoint: Optional[int] = Field(None, description="First checkpoint step")
    checkpoint_increment_percent: Optional[float] = Field(None, description="Checkpoint increment")
    flag_loss_multiplier: Optional[float] = Field(None, description="Loss multiplier for flag tokens")

    # ========================================================================
    # MODE 2: CLAUDE EVALUATION (requires validation_topics, val_samples_per_topic)
    # ========================================================================

    val_samples_per_topic: Optional[int] = Field(
        None,
        gt=0,
        description="Per-topic validation split point (split-critical)"
    )
    
    eval_subsample_n: Optional[int] = Field(
        None,
        gt=0,
        description="Number of samples to evaluate (subsampled from validation split)"
    )

    class Config:
        extra = "allow"  # Backward compatibility
        populate_by_name = True

    def get_mode(self) -> ConfigMode:
        is_training = self.batch_size is not None or self.epochs is not None or self.lr is not None
        has_claude_eval = bool(self.validation_topics) and self.val_samples_per_topic is not None

        if is_training:
            return ConfigMode.TRAINING
        elif has_claude_eval:
            return ConfigMode.CLAUDE_EVAL
        else:
            return ConfigMode.CLAUDE_EVAL

    @field_validator('claude_frac')
    @classmethod
    def validate_fractions(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Fraction must be between 0 and 1')
        return v

    @model_validator(mode='after')
    def validate_mode_exclusivity(self):
        """Enforce that only one mode's parameters are set."""
        mode = self.get_mode()

        # Check training mode requirements
        if mode == ConfigMode.TRAINING:
            # Required training parameters
            required = {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'lr': self.lr,
                'lora_r': self.lora_r,
            }
            missing = [k for k, v in required.items() if v is None]
            if missing:
                raise ValueError(
                    f"Training mode requires these parameters: {missing}"
                )

            # Training should have train_topics
            if not self.train_topics:
                raise ValueError("Training mode requires non-empty train_topics")

            # Adam betas should be set together
            if (self.ab1 is None) != (self.ab2 is None):
                raise ValueError("ab1 and ab2 must both be set or both be None")

            # Logging should be configured
            if self.log_every is None:
                raise ValueError("Training mode requires log_every")

            # Heavy logging params should be together
            heavy_log_params = [self.first_heavy_log_step, self.heavy_log_increment_percent]
            if any(p is not None for p in heavy_log_params) and not all(p is not None for p in heavy_log_params):
                raise ValueError("first_heavy_log_step and heavy_log_increment_percent must both be set or both be None")

            # Checkpoint params should be together
            checkpoint_params = [self.first_checkpoint, self.checkpoint_increment_percent]
            if any(p is not None for p in checkpoint_params) and not all(p is not None for p in checkpoint_params):
                raise ValueError("first_checkpoint and checkpoint_increment_percent must both be set or both be None")

        # Check Claude eval mode requirements
        elif mode == ConfigMode.CLAUDE_EVAL:
            if not self.validation_topics:
                raise ValueError("Claude evaluation mode requires non-empty validation_topics")

            if self.val_samples_per_topic is None:
                raise ValueError("Claude evaluation mode requires val_samples_per_topic")

            # Should NOT have training params
            training_params = {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'lr': self.lr,
            }
            has_training = [k for k, v in training_params.items() if v is not None]
            if has_training:
                raise ValueError(
                    f"Claude evaluation mode should not have training parameters: {has_training}. "
                    f"Remove these or add all required training params to switch to training mode."
                )

        # Experiments are mutually exclusive
        if self.exclude_system_prompt and self.mismatch_prompts:
            raise ValueError("exclude_system_prompt and mismatch_prompts cannot both be True")

        return self

    def validate_formatting_params(self):
        """
        Validate that formatting parameters are set.

        Call explicitly before loading data for evaluation to ensure
        all required parameters have been loaded from checkpoint.
        """
        missing_params = []
        if self.system_tag is None:
            missing_params.append("system_tag")
        if self.intervention_prefix is None:
            missing_params.append("intervention_prefix")
        if self.review_prefix is None:
            missing_params.append("review_prefix")
        if self.flag_prefix is None:
            missing_params.append("flag_prefix")

        if missing_params:
            raise ValueError(
                f"Formatting parameters must be set: {', '.join(missing_params)}. "
                f"These should be loaded from checkpoint's args.json using load_checkpoint_args()."
            )


def load_config(data: dict) -> Config:
    """Load config from dictionary with backward compatibility."""
    # Handle legacy name: validation_claude_subsample_n is now eval_subsample_n
    # (val_samples_per_topic is the stratified split size, eval_subsample_n is evaluation subsample)
    if 'validation_claude_subsample_n' in data:
        if 'eval_subsample_n' not in data:
            data['eval_subsample_n'] = data['validation_claude_subsample_n']
        # Also set val_samples_per_topic if not already set (for backward compat with old configs)
        if 'val_samples_per_topic' not in data:
            data['val_samples_per_topic'] = data['validation_claude_subsample_n']

    return Config(**data)
