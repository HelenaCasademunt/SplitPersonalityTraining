#!/usr/bin/env python3
"""Run all GPU-based evaluations for trained models.

Generates all 16 training experiment combinations and evaluates each one,
ensuring lora_patch matches the training configuration.
"""

import json
import os
import sys
from pathlib import Path
from itertools import product

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config
from scripts.utils import load_env
from evals import run_evaluation

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_DIR = Path(__file__).parent

# Define parameter values for exhaustive sweep (matching training script)
DATA_MODELS = ["google/gemma-3-12b-it", "qwen/qwen3-14b"]
TRAIN_MODELS = ["google/gemma-3-12b-it"]
PROB_EXCLUDE_VALUES = [0.0]
LORA_PATCH_VALUES = [True]
CV_FOLDS = [2]  # 3-fold cross-validation

# Eval on all data sources (matching training data models)
DATA_SOURCES = [
    ("google/gemma-3-12b-it", "gemma"),
    ("qwen/qwen3-14b", "qwen"),
]

# Sysprompt configurations for evaluation: (prob_exclude, prob_mismatch, name)
SYSPROMPT_CONFIGS = [
    (0.0, 1.0, "swap"),  
]


def generate_experiment_name(data_model, train_model, prob_exclude_system_prompt, lora_patch, cv_fold):
    """
    Generate a descriptive experiment name from parameters.
    Matches the naming convention from run_training_experiments.py
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


def generate_all_evaluations():
    """
    Generate all evaluation combinations directly.

    Yields:
        Tuples: (exp_name, train_model, data_source_model, data_name,
                 prob_excl_eval, prob_mismatch_eval, sysp_name, eval_name, lora_patch_train, cv_fold)
    """
    for data_model, train_model, prob_exclude_train, lora_patch_train, cv_fold in product(
        DATA_MODELS, TRAIN_MODELS, PROB_EXCLUDE_VALUES, LORA_PATCH_VALUES, CV_FOLDS
    ):
        exp_name = generate_experiment_name(data_model, train_model, prob_exclude_train, lora_patch_train, cv_fold)

        for data_source_model, data_name in DATA_SOURCES:
            for prob_excl_eval, prob_mismatch_eval, sysp_name in SYSPROMPT_CONFIGS:
                eval_name = f"{exp_name}_{data_name}_{sysp_name}sys"
                yield (
                    exp_name, train_model, data_source_model, data_name,
                    prob_excl_eval, prob_mismatch_eval, sysp_name, eval_name, lora_patch_train, cv_fold
                )


def load_lora_patch_from_checkpoint(checkpoint_path):
    """Load lora_patch value from checkpoint's args.json."""
    checkpoint_dir = Path(checkpoint_path).parent if 'step_' in str(checkpoint_path) else Path(checkpoint_path)
    args_path = checkpoint_dir / "args.json"
    
    if not args_path.exists():
        return None
    
    with open(args_path, 'r') as f:
        checkpoint_args = json.load(f)
    
    return checkpoint_args.get('lora_patch', None)


def find_checkpoint(exp_name):
    """Find latest checkpoint directory for experiment.
    
    Returns the checkpoint directory (parent of step_*), not the step directory itself.
    evals.py will find the steps within this directory.
    """
    ckpt_dir = PROJECT_ROOT / "checkpoints" / exp_name
    if not ckpt_dir.exists():
        return None
    steps = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return str(ckpt_dir) if steps else None


def main():
    # Change to project root directory so relative paths resolve correctly
    os.chdir(PROJECT_ROOT)
    
    # Load environment variables
    load_env()

    # Load base eval config
    with open(SCRIPT_DIR / "config_eval_base.json") as f:
        base_config = json.load(f)

    # Generate all evaluation combinations (as list so we can count and iterate twice)
    all_evals = list(generate_all_evaluations())

    total = len(all_evals)
    print(f"\n{'='*80}")
    print(f"EXTERNAL REVIEWER EVALUATION EXPERIMENTS")
    print(f"Total evaluations to run: {total}")
    print(f"{'='*80}\n")

    # Check which evaluations already exist
    existing_evals = []
    eval_results_dir = PROJECT_ROOT / "eval_results"
    for _, _, _, _, _, _, _, eval_name, _, _ in all_evals:
        # Check for evaluation results matching this experiment
        pattern = f"eval_results_*{eval_name}*.json"
        if list(eval_results_dir.glob(pattern)):
            existing_evals.append(eval_name)
    
    if existing_evals:
        print(f"Found {len(existing_evals)} existing evaluations (will be skipped):")
        for name in existing_evals[:10]:  # Show first 10
            print(f"  - {name}")
        if len(existing_evals) > 10:
            print(f"  ... and {len(existing_evals) - 10} more")
        print()
    
    print(f"Will run {total - len(existing_evals)} new evaluations\n")

    for i, (exp_name, train_model, data_source_model, data_name,
            prob_excl_eval, prob_mismatch_eval, sysp_name, eval_name, lora_patch_train, cv_fold) in enumerate(all_evals, 1):

        # Check if already done
        if eval_name in existing_evals:
            print(f"[{i}/{total}] ⏭️  {eval_name}")
            continue

        # Find checkpoint
        ckpt = find_checkpoint(exp_name)
        if not ckpt:
            print(f"[{i}/{total}] ❌ No checkpoint: {exp_name}")
            continue

        # Load lora_patch from checkpoint to verify it matches
        checkpoint_lora_patch = load_lora_patch_from_checkpoint(ckpt)
        if checkpoint_lora_patch is not None and checkpoint_lora_patch != lora_patch_train:
            print(f"[{i}/{total}] ⚠️  WARNING: {eval_name}")
            print(f"  lora_patch mismatch: expected {lora_patch_train}, checkpoint has {checkpoint_lora_patch}")
            print(f"  Using checkpoint value: {checkpoint_lora_patch}")
            lora_patch_to_use = checkpoint_lora_patch
        else:
            lora_patch_to_use = lora_patch_train

        print(f"[{i}/{total}] 🔄 {eval_name}")
        print(f"  Training exp: {exp_name}")
        print(f"  lora_patch: {lora_patch_to_use}")
        print(f"  CV fold: {cv_fold}")
        print(f"  prob_mismatch_prompts: {prob_mismatch_eval}")

        # Modify config
        config = base_config.copy()
        config.update({
            "checkpoint_path": ckpt,
            "prob_exclude_system_prompt": prob_excl_eval,
            "prob_mismatch_prompts": prob_mismatch_eval,
            "experiment_name": eval_name,
            "lora_patch": lora_patch_to_use,
            "model_name_for_data": data_source_model,
            "cv_fold": cv_fold,
        })

        # Set environment variable for base model (required for model loading)
        os.environ["base_model"] = train_model

        # Run eval using refactored run_evaluation function
        try:
            args = load_config(config)
            model, tokenizer = run_evaluation(args)

            # Cleanup
            del model, tokenizer
            import torch as t
            import gc
            if t.cuda.is_available():
                t.cuda.empty_cache()
                gc.collect()

            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            if input("Continue? (y/n): ").lower() != 'y':
                break


if __name__ == "__main__":
    main()
