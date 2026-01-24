#!/usr/bin/env python3
"""Run evaluations for all experiments and conditions."""

import csv
import gc
import json
import random
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.utils import load_env
from evals import run_evaluation, setup_distributed_environment


def read_experiments(csv_path):
    """Read experiment names from CSV."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0].strip():
                yield row[0].strip()


def get_conditions():
    """Return evaluation conditions with their augmentation parameters."""
    return {
        "eval_baseline": {"prob_exclude_system_prompt": 0.0, "prob_mismatch_prompts": 0.0},
        "eval_no_sysprompt": {"prob_exclude_system_prompt": 1.0, "prob_mismatch_prompts": 0.0},
        "eval_swapped": {"prob_exclude_system_prompt": 0.0, "prob_mismatch_prompts": 1.0},
    }


def main():
    # Setup
    random.seed(42), t.manual_seed(42), t.cuda.manual_seed(42), t.cuda.manual_seed_all(42)
    t._dynamo.config.cache_size_limit = 128
    
    base_dir = Path(__file__).resolve().parents[1]
    
    # Change to train-project directory for correct relative paths
    import os
    os.chdir(base_dir)
    
    load_env()

    csv_path = base_dir / "augmentation_experiment" / "sysprompt_train_sweep.csv"
    template_path = base_dir / "augmentation_experiment" / "cfg_eval.json"
    checkpoint_root = base_dir / "checkpoints"
    output_root = base_dir / "augmentation_experiment" / "eval_run_5"

    template = json.loads(template_path.read_text())
    conditions = get_conditions()

    for experiment in read_experiments(csv_path):
        checkpoint_dir = checkpoint_root / experiment
        if not checkpoint_dir.is_dir():
            print(f"⚠️  Skipping {experiment} (checkpoint not found)")
            continue

        print(f"\n{'='*80}\nLoading model: {experiment}\n{'='*80}\n")

        # Load model once, reuse across conditions
        model, tokenizer = None, None
        
        for condition_name, overrides in conditions.items():
            result_dir = output_root / f"results_{experiment}_{condition_name}"
            
            if result_dir.exists() and any(result_dir.glob("eval_results_*.json")):
                print(f"⏭️  Skipping {condition_name} (results exist)")
                continue

            print(f"\n{'='*80}\nRunning: {experiment} | {condition_name}\n{'='*80}\n")

            # Create fresh config for this condition (don't modify base_cfg)
            config = {**template, **overrides, "checkpoint_path": str(checkpoint_dir)}
            cfg = load_config(config)
            cfg.experiment_name = f"{experiment}_{condition_name}"

            # Run evaluation (reuses model if provided)
            model, tokenizer = run_evaluation(cfg, model, tokenizer)


            eval_results = base_dir / "eval_results"
            if eval_results.exists():
                result_dir.mkdir(parents=True, exist_ok=True)
                for f in eval_results.glob("*"):
                    f.rename(result_dir / f.name)
                eval_results.rmdir()

            print(f"✓ Completed: {condition_name}")

        # Aggressive cleanup after all conditions for this experiment
        if model is not None:
            del model, tokenizer
            model, tokenizer = None, None
            
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
            t.cuda.synchronize()
        gc.collect()


if __name__ == "__main__":
    main()
