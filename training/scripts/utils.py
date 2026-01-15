import json, os, types
from scripts.config import load_config

def load_cfg(path=None):
    base_dir = os.path.dirname(os.path.dirname(__file__))

    if (path): cfg_path = os.path.join(base_dir, path)
    else: cfg_path = os.path.join(base_dir, 'cfg.json')
    with open(cfg_path, 'r') as f:
        data = json.load(f)
        return load_config(data)

def load_env():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(base_dir, 'env.json')
    with open(env_path, 'r') as f:
        env_vars = json.load(f)
        for key, value in env_vars.items():
            os.environ[key] = str(value)


def load_checkpoint_args(config, checkpoint_path):
    """
    Load checkpoint's essential training parameters for evaluation.

    Loads split-critical parameters (val_samples_per_topic, intervention_types,
    tags_to_filter) and formatting parameters. These must match training to ensure
    evaluation uses the same data split and filtering as training.

    Topics can differ from training to test generalization on held-out topics.

    Args:
        config: Config object to override
        checkpoint_path: Path to checkpoint directory (contains args.json)

    Returns:
        dict: Dictionary of checkpoint arguments
    """
    # Essential parameters that must match training
    split_critical_keys = [
        'val_samples_per_topic',
        'intervention_types',
        'tags_to_filter',
        'exclude_system_prompt',  # Affects sample loading (can change which samples exist)
        'mismatch_prompts'       # Affects sample loading (can skip samples without opposite prompt)
    ]

    formatting_keys = [
        'system_tag',
        'intervention_prefix',
        'review_prefix',
        'flag_prefix',
    ]
    
    essential_keys = split_critical_keys + formatting_keys

    # Handle both step directory and parent directory
    checkpoint_dir = checkpoint_path.rstrip('/')
    if 'step_' in os.path.basename(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)

    args_path = os.path.join(checkpoint_dir, "args.json")

    with open(args_path, 'r') as f:
        checkpoint_args = json.load(f)

    # Check for mismatches in split-critical parameters
    mismatches = []
    for key in split_critical_keys:
        if key in checkpoint_args and hasattr(config, key):
            config_value = getattr(config, key)
            checkpoint_value = checkpoint_args[key]
            if config_value != checkpoint_value:
                mismatches.append(f"  {key}: config={config_value}, checkpoint={checkpoint_value}")
    
    if mismatches:
        print("\n" + "="*80)
        print("WARNING: Split-critical parameters differ from checkpoint!")
        print("Using checkpoint values to ensure evaluation uses same split as training:")
        print("\n".join(mismatches))
        print("="*80 + "\n")

    # Override with checkpoint values
    for key in essential_keys:
        if key in checkpoint_args:
            original_value = getattr(config, key, None)
            checkpoint_value = checkpoint_args[key]

            # Warn if overriding a different value
            if original_value is not None and original_value != checkpoint_value:
                print(f"⚠️  Overriding {key} from checkpoint:")
                print(f"   Config:     {original_value}")
                print(f"   Checkpoint: {checkpoint_value}")

            setattr(config, key, checkpoint_value)

    return checkpoint_args