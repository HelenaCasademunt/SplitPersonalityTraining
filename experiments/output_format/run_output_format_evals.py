#!/usr/bin/env python3
"""
Run evaluation on trained output format experiments.

Usage:
    python run_output_format_evals.py
"""

import json
import subprocess
from pathlib import Path

MODELS = {
    "qwen": "Qwen/Qwen3-14B",
    "gemma": "google/gemma-3-12b-it",
}

OUTPUT_FORMATS = ["flag_then_review", "flag_only"]

CV_FOLDS = 3

# Reduced max tokens for flag-first formats (only need to generate the flag)
MAX_TOKENS_FLAG_FIRST = 20
MAX_TOKENS_DEFAULT = 2000


def main():
    script_dir = Path(__file__).parent
    base_config_path = script_dir / "cfg_eval_base.json"
    cfg_path = script_dir / "cfg.json"
    env_path = script_dir / "env.json"
    checkpoints_dir = script_dir / "checkpoints"

    with open(base_config_path) as f:
        base_config = json.load(f)

    with open(env_path) as f:
        env = json.load(f)

    eval_results_dir = script_dir / "eval_results"
    eval_results_dir.mkdir(exist_ok=True)

    for model_name, model_path in MODELS.items():
        for output_format in OUTPUT_FORMATS:
            for fold in range(CV_FOLDS):
                experiment_name = f"{model_name}_{output_format}_fold{fold}"
                checkpoint_path = checkpoints_dir / experiment_name

                if not checkpoint_path.exists():
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {experiment_name} (checkpoint not found)")
                    print(f"{'='*60}")
                    continue

                # Check if results already exist
                result_file = eval_results_dir / f"eval_results_{experiment_name}.json"

                if result_file.exists():
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {experiment_name} (results already exist: {result_file.name})")
                    print(f"{'='*60}")
                    continue

                print(f"\n{'='*60}")
                print(f"Evaluating: {experiment_name}")
                print(f"{'='*60}")

                # Update env
                env["base_model"] = model_path
                with open(env_path, 'w') as f:
                    json.dump(env, f, indent=4)

                # Update config
                config = base_config.copy()
                config["checkpoint_path"] = str(checkpoint_path) + "/"
                config["output_format"] = output_format
                config["cv_fold"] = fold

                # Reduce max tokens for flag-first formats
                if output_format in ("flag_only", "flag_then_review"):
                    config["max_new_tokens_eval"] = MAX_TOKENS_FLAG_FIRST
                else:
                    config["max_new_tokens_eval"] = MAX_TOKENS_DEFAULT

                with open(cfg_path, 'w') as f:
                    json.dump(config, f, indent=4)

                subprocess.run(["python", "evals.py"], cwd=script_dir)


if __name__ == "__main__":
    main()
