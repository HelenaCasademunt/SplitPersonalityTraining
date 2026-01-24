#!/usr/bin/env python3
"""
Run training experiments with different output formats and models.

Usage:
    python run_output_format_experiments.py
"""

import json
import subprocess
from pathlib import Path

MODELS = {
    "qwen": "Qwen/Qwen3-14B",
    "gemma": "google/gemma-3-12b-it",
}

OUTPUT_FORMATS = ["flag_only", "flag_then_review"]

CV_FOLDS = 3


def main():
    script_dir = Path(__file__).parent
    base_config_path = script_dir / "cfg_train_base.json"
    cfg_path = script_dir / "cfg.json"
    env_path = script_dir / "env.json"
    checkpoints_dir = script_dir / "checkpoints"

    with open(base_config_path) as f:
        base_config = json.load(f)

    with open(env_path) as f:
        env = json.load(f)

    for model_name, model_path in MODELS.items():
        for output_format in OUTPUT_FORMATS:
            for fold in range(CV_FOLDS):
                experiment_name = f"{model_name}_{output_format}_fold{fold}"
                checkpoint_path = checkpoints_dir / experiment_name

                if checkpoint_path.exists():
                    print(f"SKIPPING: {experiment_name} (checkpoint already exists)")
                    continue

                print(f"\n{'='*60}")
                print(f"Running: {experiment_name}")
                print(f"{'='*60}")

                # Update env
                env["base_model"] = model_path
                with open(env_path, 'w') as f:
                    json.dump(env, f, indent=4)

                # Update config
                config = base_config.copy()
                config["output_format"] = output_format
                config["experiment_name"] = experiment_name
                config["cv_fold"] = fold
                with open(cfg_path, 'w') as f:
                    json.dump(config, f, indent=4)

                subprocess.run(["python", "train_lora.py"], cwd=script_dir)


if __name__ == "__main__":
    main()
