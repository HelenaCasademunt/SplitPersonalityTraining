#!/usr/bin/env python3
"""
Training orchestration script for external reviewer baseline experiments.

Runs all 16 possible combinations of 4 parameters (2^4 = 16 experiments):
- data_model: qwen/qwen3-14b or google/gemma-3-12b-it
- train_model: qwen/qwen3-14b or google/gemma-3-12b-it
- prob_exclude_system_prompt: 0.0 or 1.0
- lora_patch: True or False

Each experiment:
1. Checks if checkpoint already exists (skips if found)
2. Loads base config from external_reviewer_experiments/config_train.json
3. Modifies only the experimental parameters
4. Copies to cfg.json
5. Runs training
6. Saves checkpoint with descriptive name
"""

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from itertools import product

# Get project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Define parameter values for exhaustive sweep
DATA_MODELS = ["qwen/qwen3-14b", "google/gemma-3-12b-it"]
TRAIN_MODELS = ["qwen/qwen3-14b", "google/gemma-3-12b-it"]
PROB_EXCLUDE_VALUES = [0.0, 1.0]
LORA_PATCH_VALUES = [True, False]
CV_FOLDS = [0, 1, 2]  # 3-fold cross-validation


def generate_experiment_name(data_model, train_model, prob_exclude_system_prompt, lora_patch, cv_fold):
    """
    Generate a descriptive experiment name from parameters.

    Format: {train_model_short}{on/off}policy_{sys/nosys}_{patch/nopatch}_fold{N}
    """
    # Extract short model names
    if "qwen" in train_model.lower():
        train_short = "qwen14b"
    elif "gemma" in train_model.lower():
        train_short = "gemma12b"
    else:
        train_short = train_model.split("/")[-1].replace("-", "").lower()

    # Determine if on-policy or off-policy
    policy_type = "onpolicy" if data_model == train_model else "offpolicy"

    # System prompt suffix
    sys_suffix = "nosys" if prob_exclude_system_prompt == 1.0 else "withsys"

    # Build experiment name
    experiment_name = f"{train_short}_{policy_type}_{sys_suffix}"

    # Only add patch suffix when lora_patch is False
    if not lora_patch:
        experiment_name += "_nopatch"

    # Add CV fold
    experiment_name += f"_fold{cv_fold}"

    return experiment_name


def generate_all_experiments():
    """
    Generate all combinations with 3-fold CV.

    Returns:
        List of tuples: (name, data_model, train_model, prob_exclude_system_prompt, lora_patch, cv_fold)
    """
    experiments = []
    for data_model, train_model, prob_exclude, lora_patch, cv_fold in product(
        DATA_MODELS, TRAIN_MODELS, PROB_EXCLUDE_VALUES, LORA_PATCH_VALUES, CV_FOLDS
    ):
        name = generate_experiment_name(data_model, train_model, prob_exclude, lora_patch, cv_fold)
        experiments.append((name, data_model, train_model, prob_exclude, lora_patch, cv_fold))
    return experiments


def load_base_config():
    """Load the base training configuration."""
    config_path = SCRIPT_DIR / "config_train.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def check_checkpoint_exists(name):
    """Check if checkpoint directory already exists for this experiment."""
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / name
    if checkpoint_dir.exists():
        # Check if it has any step directories
        step_dirs = list(checkpoint_dir.glob("step_*"))
        if step_dirs:
            return True, checkpoint_dir
    return False, None


def setup_experiment(name, data_model, train_model, prob_exclude_system_prompt, lora_patch, cv_fold):
    """
    Setup configuration for a specific experiment.

    Args:
        name: Experiment name (used for checkpoint naming)
        data_model: Model whose inference data to load from JSON files
        train_model: Actual model to train (base model architecture)
        prob_exclude_system_prompt: 0.0 (include) or 1.0 (exclude)
        lora_patch: True or False
        cv_fold: Cross-validation fold number (0, 1, 2)

    Returns:
        env dict, or None if checkpoint already exists
    """
    # Check if checkpoint already exists
    exists, checkpoint_path = check_checkpoint_exists(name)
    if exists:
        print(f"\n{'='*80}")
        print(f"⏭️  SKIPPING experiment: {name}")
        print(f"  Checkpoint already exists at: {checkpoint_path}")
        print(f"{'='*80}\n")
        return None

    print(f"\n{'='*80}")
    print(f"Setting up experiment: {name}")
    print(f"  Data source: {data_model}")
    print(f"  Training model: {train_model}")
    print(f"  System prompt exclusion: {prob_exclude_system_prompt}")
    print(f"  LoRA patch: {lora_patch}")
    print(f"  CV fold: {cv_fold}")
    print(f"{'='*80}\n")

    # Load base config
    config = load_base_config()

    # Modify experimental parameters
    config['prob_exclude_system_prompt'] = prob_exclude_system_prompt
    config['lora_patch'] = lora_patch
    config['experiment_name'] = name
    config['model_name_for_data'] = data_model
    config['cv_fold'] = cv_fold

    # Write modified config to cfg.json
    cfg_path = PROJECT_ROOT / "cfg.json"
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Config written to {cfg_path}")

    # Set environment variables
    env = os.environ.copy()
    # base_model: the actual model architecture to train
    env['base_model'] = train_model

    return env


def run_training(env):
    """Run the training script with the given environment."""
    train_script = PROJECT_ROOT / "train_lora.py"

    print(f"Starting training...")
    print(f"Running: python {train_script}")

    # Run training script
    result = subprocess.run(
        [sys.executable, str(train_script)],
        env=env,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"❌ Training failed with return code {result.returncode}")
        return False

    print(f"✓ Training completed successfully")
    return True


def main():
    """Run all training experiments with 3-fold CV."""
    # Generate all combinations (2 * 2 * 2 * 2 * 3 = 48 experiments)
    EXPERIMENTS = generate_all_experiments()

    print(f"\n{'='*80}")
    print(f"EXTERNAL REVIEWER TRAINING EXPERIMENTS (3-FOLD CV)")
    print(f"Total experiments to run: {len(EXPERIMENTS)}")
    print(f"{'='*80}\n")

    # Verify we have exactly 48 experiments (2^4 * 3)
    assert len(EXPERIMENTS) == 48, f"Expected 48 experiments, got {len(EXPERIMENTS)}"

    # Check which experiments already exist
    existing_experiments = []
    for name, _, _, _, _, _ in EXPERIMENTS:
        exists, _ = check_checkpoint_exists(name)
        if exists:
            existing_experiments.append(name)

    if existing_experiments:
        print(f"Found {len(existing_experiments)} existing checkpoints (will be skipped):")
        for name in existing_experiments:
            print(f"  - {name}")
        print()

    print(f"Will run {len(EXPERIMENTS) - len(existing_experiments)} new experiments\n")

    results = []

    for i, (name, data_model, train_model, prob_exclude, lora_patch, cv_fold) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(EXPERIMENTS)}: {name}")
        print(f"{'='*80}")

        env = setup_experiment(name, data_model, train_model, prob_exclude, lora_patch, cv_fold)

        # If env is None, checkpoint exists - skip
        if env is None:
            results.append({
                'name': name,
                'success': True,
                'skipped': True,
                'data_model': data_model,
                'train_model': train_model,
                'prob_exclude_system_prompt': prob_exclude,
                'lora_patch': lora_patch,
                'cv_fold': cv_fold
            })
            continue

        success = run_training(env)

        results.append({
            'name': name,
            'success': success,
            'skipped': False,
            'data_model': data_model,
            'train_model': train_model,
            'prob_exclude_system_prompt': prob_exclude,
            'lora_patch': lora_patch,
            'cv_fold': cv_fold
        })

        if not success:
            print(f"\n⚠️  Experiment {name} failed. Continue with remaining experiments? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Stopping experiment runs.")
                break

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    for result in results:
        if result.get('skipped', False):
            status = "⏭️  SKIPPED"
        elif result['success']:
            status = "✓ SUCCESS"
        else:
            status = "❌ FAILED"
        print(f"{status}: {result['name']}")

    successful = sum(1 for r in results if r['success'] and not r.get('skipped', False))
    skipped = sum(1 for r in results if r.get('skipped', False))
    print(f"\nCompleted: {successful}/{len(results)} experiments successful, {skipped} skipped")


if __name__ == "__main__":
    main()
