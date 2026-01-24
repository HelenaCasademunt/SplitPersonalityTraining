#!/usr/bin/env python3
"""Run evaluations using external API models (OpenRouter)."""

import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.external_api_model import ExternalAPIModel
from scripts.eval import run_model_on_claude
from scripts.config import load_config
from scripts.utils import load_env
from transformers import AutoTokenizer

# External models to evaluate
EXTERNAL_MODELS = [
    ("qwen/qwen3-14b", "qwen3-14b_external"),
    # Add frontier model here, e.g.:
    ("anthropic/claude-sonnet-4.5", "claude-sonnet-4.5_external"),
]

# Sysprompt conditions
SYSPROMPT = [(0.0, "with"), (1.0, "without")]

# Few-shot configuration
USE_FEW_SHOT = False  # Set to True to include a few-shot example from training data


def main():
    # Load environment variables (API keys, etc.)
    load_env()

    # Load external API eval config (separate from GPU eval config)
    with open(PROJECT_ROOT / "external_reviewer_experiments/config_eval_external_api.json") as f:
        base_config = json.load(f)

    # Set data source to Gemma
    os.environ['base_model'] = 'google/gemma-3-12b-it'

    total = len(EXTERNAL_MODELS) * len(SYSPROMPT)
    print(f"Running {total} external API evaluations\n")

    for i, ((model_name, short_name), (prob_excl, sysp_name)) in enumerate(
        [(m, s) for m in EXTERNAL_MODELS for s in SYSPROMPT], 1
    ):
        eval_name = f"{short_name}_gemma_{sysp_name}sys"

        # Check if already done
        if (PROJECT_ROOT / "eval_results" / f"eval_results_{eval_name}.json").exists():
            print(f"[{i}/{total}] ⏭️  {eval_name}")
            continue

        print(f"[{i}/{total}] 🔄 {eval_name}")
        print(f"  Model: {model_name}")
        print(f"  Sysprompt: prob_exclude={prob_excl}")

        # Create config
        config = base_config.copy()
        config.update({
            "prob_exclude_system_prompt": prob_excl,
            "experiment_name": eval_name,
            "model_name_for_data": "google/gemma-3-12b-it",  # Set data source for eval
        })


        # Write config (still needed for external API script compatibility)
        with open(PROJECT_ROOT / "cfg.json", "w") as f:
            json.dump(config, f, indent=2)

        # Load config and tokenizer
        cfg = load_config(config)
        # Use tokenizer matching the data source (Gemma), not the API model
        # This ensures prompt formatting matches what the data was generated with
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

        # Create external model wrapper
        # Few-shot example loading is handled internally if USE_FEW_SHOT is True
        if USE_FEW_SHOT:
            print(f"  📝 Using few-shot example from training data")
        model = ExternalAPIModel(model_name, tokenizer, cfg=cfg, use_few_shot=USE_FEW_SHOT)

        # Run evaluation using existing pipeline
        # Evaluate each topic separately to avoid subsample limit issues
        try:
            from tqdm import tqdm

            all_validation_topics = cfg.validation_topics.copy()
            all_samples_aggregated = []
            topic_results = {}

            # Check for existing checkpoint
            checkpoint_file = PROJECT_ROOT / "eval_results" / f"checkpoint_{eval_name}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)
                    topic_results = checkpoint.get("topic_results", {})
                    all_samples_aggregated = checkpoint.get("samples", [])
                    completed_topics = set(topic_results.keys())
                    print(f"  📂 Resuming from checkpoint: {len(completed_topics)}/{len(all_validation_topics)} topics completed")
            else:
                completed_topics = set()

            eval_quiet = getattr(cfg, 'eval_quiet', False)

            for topic in tqdm(all_validation_topics, desc="  Topics"):
                # Skip if already completed
                if topic in completed_topics:
                    continue

                # Evaluate one topic at a time
                cfg.validation_topics = [topic]
                cfg.train_topics = []

                cco, cnao, ctoto, _, split_samples = run_model_on_claude(
                    model=model,
                    tokenizer=tokenizer,
                    device="cpu",  # Doesn't matter for API
                    cfg=cfg,
                    quiet=eval_quiet,
                    distributed=False,
                    rank=0
                )

                topic_results[topic] = {
                    'accuracy%': round(cco/(1.0 - cnao + 0.000001), 3),
                    'na%': cnao,
                    'tot_n': ctoto
                }

                # Add topic label to each sample and aggregate
                for sample in split_samples:
                    sample['topic'] = topic
                    all_samples_aggregated.append(sample)

                # Save checkpoint after each topic
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "topic_results": topic_results,
                        "samples": all_samples_aggregated
                    }, f, indent=2)

            # Calculate overall stats
            total_correct = sum(int(r['accuracy%'] * r['tot_n'] * (1.0 - r['na%'])) for r in topic_results.values())
            total_na = sum(int(r['na%'] * r['tot_n']) for r in topic_results.values())
            total_samples = sum(int(r['tot_n']) for r in topic_results.values())
            overall_accuracy = total_correct / max(total_samples - total_na, 1)

            # Save results
            from scripts.eval import save_eval_results
            save_eval_results(
                all_samples=all_samples_aggregated,
                metadata={
                    "model": model_name,
                    "experiment_name": eval_name,
                    "prob_exclude_system_prompt": prob_excl,
                    "correct": total_correct,
                    "na": total_na,
                    "total": total_samples,
                    "accuracy": float(overall_accuracy),
                    "topic_results": topic_results
                },
                exp_name=eval_name
            )

            print(f"  ✓ Done - Accuracy: {overall_accuracy:.3f} ({total_samples} samples across {len(all_validation_topics)} topics)")

            # Clean up checkpoint file after successful completion
            if checkpoint_file.exists():
                checkpoint_file.unlink()

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print(f"  💾 Progress saved in checkpoint: {checkpoint_file}")
            if input("Continue? (y/n): ").lower() != 'y':
                break


if __name__ == "__main__":
    main()
