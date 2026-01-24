#!/usr/bin/env python3
"""
Run generalization experiments: train and evaluate models.

Experiment A: Train on 80% of data (val_samples_per_topic=80), evaluate on all topics.
Experiment B: Train on all topics except one, evaluate on the held-out topic.

Usage:
    python run_generalization_experiments.py

Set SLACK_WEBHOOK_URL environment variable to receive Slack notifications.
"""

import json
import os
import subprocess
from pathlib import Path
from knockknock import slack_sender

# Load env.json to get Slack config
with open(Path(__file__).parent / "env.json") as f:
    _env = json.load(f)
SLACK_WEBHOOK_URL = _env["SLACK_WEBHOOK_URL"]
SLACK_CHANNEL = _env.get("SLACK_CHANNEL", "@Experiment Notifier")  # Use @username for DMs

MODELS = {
    "gemma": "google/gemma-3-12b-it",
}

# Elicitation types to sweep; honesty_scare pairs with 'ur'
ELICITATION_TYPES = ["hp"]
# Map elicitation_type -> allowed intervention prefixes
ELICITATION_TO_PREFIXES = {"ur": ["honesty_scare"], "hp": ["split_personality"]}

TOPICS = [
    "confidence_assessment",
    "jailbreak_attempts",
    "oversimplification",
    "fabricated_statistics",
    "influence_seeking",
    "malicious_user_queries",
    "reward_hacks",
    "sycophancy",
    "unethical_instructions",
]

CV_FOLDS = 3
LORA_PATCH = True  # Enable lora patch for all experiments
EXP_SUFFIX = "_lp" if LORA_PATCH else ""  # Suffix for experiment names
ENABLE_CROSS_TOPIC = True  # Toggle leave-one-topic-out experiments


def filter_interventions_for_elicitation(intervention_types, elicitation_type):
    prefixes = ELICITATION_TO_PREFIXES.get(elicitation_type, [])
    filtered = [t for t in intervention_types if any(t.startswith(p) for p in prefixes)]
    # If nothing matched and we're in 'ur', try swapping split_personality -> honesty_scare
    if not filtered and elicitation_type == "ur":
        swapped = [t.replace("split_personality", "honesty_scare") for t in intervention_types]
        filtered = [t for t in swapped if any(t.startswith(p) for p in prefixes)]
    if not filtered:
        raise ValueError(f"No intervention_types remain after filtering for elicitation_type={elicitation_type}. Original: {intervention_types}")
    return filtered


@slack_sender(webhook_url=SLACK_WEBHOOK_URL, channel=SLACK_CHANNEL)
def run_training(script_dir, env_path, cfg_path, env, train_config):
    with open(env_path, 'w') as f:
        json.dump(env, f, indent=4)
    with open(cfg_path, 'w') as f:
        json.dump(train_config, f, indent=4)
    subprocess.run(["python", "train_lora.py"], cwd=script_dir)
    return train_config.get("experiment_name")


@slack_sender(webhook_url=SLACK_WEBHOOK_URL, channel=SLACK_CHANNEL)
def run_eval(script_dir, env_path, cfg_path, env, eval_config):
    with open(env_path, 'w') as f:
        json.dump(env, f, indent=4)
    with open(cfg_path, 'w') as f:
        json.dump(eval_config, f, indent=4)
    subprocess.run(["python", "evals.py"], cwd=script_dir)
    return Path(eval_config["checkpoint_path"].rstrip("/")).name


def main():
    script_dir = Path(__file__).parent
    train_base_path = script_dir / "cfg_generalization_train_base.json"
    eval_base_path = script_dir / "cfg_generalization_eval_base.json"
    cfg_path = script_dir / "cfg.json"
    env_path = script_dir / "env.json"
    checkpoints_dir = script_dir / "checkpoints"
    eval_results_dir = script_dir / "eval_results"
    eval_results_dir.mkdir(exist_ok=True)

    with open(train_base_path) as f:
        train_base = json.load(f)
    with open(eval_base_path) as f:
        eval_base = json.load(f)
    with open(env_path) as f:
        env = json.load(f)

    for elicitation in ELICITATION_TYPES:
        for model_name, model_path in MODELS.items():
            env["base_model"] = model_path

            # Experiment A: Train on 75% of all topics, evaluate on all topics
            for fold in range(CV_FOLDS):
                exp_name_a = f"{model_name}_all_topics_75pct_fold{fold}{EXP_SUFFIX}_{elicitation}"
                checkpoint_a = checkpoints_dir / exp_name_a
                result_a = eval_results_dir / f"eval_results_{exp_name_a}.json"

                if not checkpoint_a.exists():
                    print(f"\n{'='*60}")
                    print(f"Training: {exp_name_a}")
                    print(f"{'='*60}")
                    train_config = train_base.copy()
                    train_config["train_topics"] = TOPICS
                    train_config["validation_topics"] = TOPICS
                    train_config["val_samples_per_topic"] = 80
                    train_config["experiment_name"] = exp_name_a
                    train_config["cv_fold"] = fold
                    train_config["lora_patch"] = LORA_PATCH
                    train_config["elicitation_type"] = elicitation
                    train_config["intervention_types"] = filter_interventions_for_elicitation(
                        train_config.get("intervention_types", []), elicitation
                    )
                    run_training(script_dir, env_path, cfg_path, env, train_config)
                else:
                    print(f"SKIPPING training: {exp_name_a} (checkpoint exists)")

                if not result_a.exists() and checkpoint_a.exists():
                    print(f"\n{'='*60}")
                    print(f"Evaluating: {exp_name_a}")
                    print(f"{'='*60}")
                    eval_config = eval_base.copy()
                    eval_config["checkpoint_path"] = str(checkpoint_a) + "/"
                    eval_config["validation_topics"] = TOPICS
                    eval_config["val_samples_per_topic"] = 80
                    eval_config["cv_fold"] = fold
                    eval_config["elicitation_type"] = elicitation
                    eval_config["intervention_types"] = filter_interventions_for_elicitation(
                        eval_config.get("intervention_types", []), elicitation
                    )
                    run_eval(script_dir, env_path, cfg_path, env, eval_config)
                elif result_a.exists():
                    print(f"SKIPPING eval: {exp_name_a} (results exist)")

            if ENABLE_CROSS_TOPIC:
                # Experiment B: Leave-one-topic-out
                for held_out_topic in TOPICS:
                    train_topics = [t for t in TOPICS if t != held_out_topic]
                    exp_name_b = f"{model_name}_leave_out_{held_out_topic}{EXP_SUFFIX}_{elicitation}"
                    checkpoint_b = checkpoints_dir / exp_name_b
                    result_b = eval_results_dir / f"eval_results_{exp_name_b}.json"

                    if not checkpoint_b.exists():
                        print(f"\n{'='*60}")
                        print(f"Training: {exp_name_b}")
                        print(f"{'='*60}")
                        train_config = train_base.copy()
                        train_config["train_topics"] = train_topics
                        train_config["validation_topics"] = train_topics
                        train_config["val_samples_per_topic"] = 80  # Minimal validation during training
                        train_config["experiment_name"] = exp_name_b
                        train_config["cv_fold"] = 0
                        train_config["lora_patch"] = LORA_PATCH
                        train_config["elicitation_type"] = elicitation
                        train_config["intervention_types"] = filter_interventions_for_elicitation(
                            train_config.get("intervention_types", []), elicitation
                        )
                        run_training(script_dir, env_path, cfg_path, env, train_config)
                    else:
                        print(f"SKIPPING training: {exp_name_b} (checkpoint exists)")

                    if not result_b.exists() and checkpoint_b.exists():
                        print(f"\n{'='*60}")
                        print(f"Evaluating: {exp_name_b}")
                        print(f"{'='*60}")
                        eval_config = eval_base.copy()
                        eval_config["checkpoint_path"] = str(checkpoint_b) + "/"
                        eval_config["validation_topics"] = [held_out_topic]
                        # Use large value to get all samples for held-out topic evaluation
                        eval_config["val_samples_per_topic"] = 9999
                        eval_config["cv_fold"] = 0
                        eval_config["elicitation_type"] = elicitation
                        eval_config["intervention_types"] = filter_interventions_for_elicitation(
                            eval_config.get("intervention_types", []), elicitation
                        )
                        run_eval(script_dir, env_path, cfg_path, env, eval_config)
                    elif result_b.exists():
                        print(f"SKIPPING eval: {exp_name_b} (results exist)")

    return "All generalization experiments complete!"


if __name__ == "__main__":
    main()
