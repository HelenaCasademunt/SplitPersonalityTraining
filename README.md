# Split Personality Training - Data Generation Pipeline

This repository contains the data generation pipeline for Split Personality Training (SPT), a method for training an "honest persona" into language models that can review and flag potentially problematic outputs.

For ideas on future improvements to the data generation process, see [`notes on data generation.md`](notes%20on%20data%20generation.md).

For information and code about training the model, have a look at the subfolder `training/README.md`


## TLDR

The data we generated can be found here:

`data/stage_3_tagged/`

This data is complete and includes lower-quality items that we filter out later. The files use a hierarchical format that combines inference completions and training data generation by several different models. The final data to use for training depends on the model you want to train. This document explains how the data generation and filtering works, in what format it is stored, and how the final training data can be extracted from it.

## Overview

The pipeline generates training data in three stages:

1. **Stage 1**: Generate Task/Prompt pairs with A/B system prompts
2. **Stage 2**: Run inference on target models to generate responses
3. **Stage 3**: Generate intervention and review pairs for training

## Directory Structure

```
repository/
├── prompts/
│   ├── base_prompt.md              # System prompt for all stages
│   ├── stage_1/                    # Topic-specific prompts for stage 1
│   │   ├── sycophancy.md
│   │   ├── reward_hacks.md
│   │   └── ...
│   └── stage_3/                    # Intervention prompts for stage 3
│       ├── sycophancy__split_personality.md
│       ├── sycophancy__honesty_scare.md
│       └── ...
├── scripts/
│   ├── utils.py                    # Shared utilities and path configuration
│   ├── shared_tools.py             # Tool definitions for API calls
│   ├── dataset_generation_stage_1.py
│   ├── dataset_generation_stage_2.py
│   ├── dataset_generation_stage_3.py
│   ├── tag_stage_3_data.py         # Quality filtering and tagging
│   ├── analyze_stage_3.py          # Analysis of training data quality
│   └── analyze_evaluation_results.py  # Analysis of poisoned model evaluation
├── config/
│   └── training_config.json        # Training configuration with filters
└── data/
    ├── stage_1/                    # Generated at runtime
    ├── stage_2/                    # Generated at runtime
    ├── stage_3/                    # Generated at runtime
    ├── stage_3_tagged/             # Tagged training data (included)
    └── evaluation_results/         # Results from poisoned model evaluation
        ├── quantitative/           # 30 result files (10k+ samples)
        ├── qualitative/            # Manual inspection samples
        └── comparison/             # LoRA patch comparison analysis
```

## Prerequisites

- Python 3.10+
- API keys for:
  - Anthropic (ANTHROPIC_API_KEY) - for Claude API calls
  - OpenAI (OPENAI_API_KEY) - optional, for GPT models in stage 1
  - OpenRouter (OPENROUTER_API_KEY) - optional, for stage 2/3 via OpenRouter
  - HuggingFace (HUGGINGFACE_API_TOKEN) - optional, for local inference

Create a `.env` file in the repository root:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_key_here
```

## Pipeline Usage

**Important**: Run all scripts from the repository root directory, not from inside `scripts/`.

### Stage 1: Generate Task/Prompt Pairs

Generates [A][B][T] data where:
- **A**: Unbiased system prompt
- **B**: Biased system prompt (with hidden misalignment)
- **T**: User task/query

```bash
python scripts/dataset_generation_stage_1.py \
    --name sycophancy \
    --num_batches 10 \
    --num_items_per_batch 30 \
    --model claude-opus-4-1
```

### Stage 2: Generate Model Responses

Runs inference on stage 1 data to generate [S] responses using target models.

```bash
python scripts/dataset_generation_stage_2.py \
    --names sycophancy \
    --model qwen/qwen3-32b \
    --inference openrouter \
    --batch_size 128
```

Options:
- `--inference local`: Use local HuggingFace models (requires GPU)
- `--inference openrouter`: Use OpenRouter API

### Stage 3: Generate Intervention/Review Pairs

Generates [I][R][P] data where:
- **I**: Intervention prompt (triggers honest persona)
- **R**: Review of the model's response
- **P**: Pass/Fail verdict

```bash
python scripts/dataset_generation_stage_3.py \
    --names sycophancy \
    --model claude-sonnet-4-5-20250929 \
    --batch_size 2 \
    --num_processes 4
```

### Tagging: Quality Filtering

Applies quality tags to stage 3 data for filtering during training.

```bash
python scripts/tag_stage_3_data.py
```

Tags include:
- `confidence_below_4`: Low confidence scores
- `training_data_leakage`: Leaked methodology terms
- `intervention_tags_disagree_twice`: Inconsistent verdicts
- `unclear_a_b_separation`: A and B variants have same verdict
- And more (see script for full list)

### Analysis: Generate Reports

Analyzes stage 3 data quality and generates reports.

```bash
python scripts/analyze_stage_3.py \
    --data-dir data/stage_3_tagged \
    --output-dir data/stage_3_analysis \
    --detailed
```

## Training Configuration

The `config/training_config.json` file controls training data filtering:

```json
{
    "intervention_types": ["split_personality__version_3", "split_personality__version_4"],
    "train_topics": ["sycophancy", "reward_hacks", ...],
    "tags_to_filter": [
        "confidence_below_4",
        "training_data_leakage",
        "failed_external_review",
        ...
    ]
}
```

Key filters:
- **intervention_types**: Which intervention versions to include. The stage 3 script was run multiple times with different `--intervention_modification` suffixes (e.g., `version_2`, `version_3`, `version_4`). Each run appends interventions with that suffix to the data. The config filters to only use specific versions (here: `version_3` and `version_4`), allowing us to iterate on the intervention prompts while keeping all results and selecting which to use for training.
- **train_topics**: Which topics to use for training
- **tags_to_filter**: Quality tags that exclude samples from training

## Data Format

### Stage 1 Output
```json
{
    "batch_id": "sycophancy_20250101_120000_batch_001",
    "topic": "sycophancy",
    "data": [
        {
            "thought_process": "...",
            "A": "System prompt without bias...",
            "B": "System prompt with hidden bias...",
            "T": "User query...",
            "metadata": {
                "alignment_issue_type": "sycophancy",
                "difficulty": "medium",
                "intended_review_focus": "..."
            }
        }
    ]
}
```

### Stage 2 Output
Adds `inferences` field to each sample:
```json
{
    "inferences": {
        "qwen/qwen3-32b": {
            "A": {"S": "Model response to A prompt...", "output_tokens": 500},
            "B": {"S": "Model response to B prompt...", "output_tokens": 520}
        }
    }
}
```

### Stage 3 Output
Adds `interventions` field to each inference:
```json
{
    "interventions": {
        "split_personality__version_3__baseline": {
            "intervention_type": "split_personality",
            "I": "<split-personality-token> Review this response...",
            "R": "This response shows concerning patterns...",
            "P": "FAIL",
            "confidence": 4,
            "detection_difficulty_internal": "medium",
            "detection_difficulty_external": "hard"
        }
    }
}
```

## Topics

The pipeline supports multiple alignment issue types:
- `sycophancy`
- `reward_hacks`
- `specification_gaming`
- `confidence_assessment`
- `fabricated_statistics`
- `goal_misgeneralization`
- `jailbreak_attempts`
- `influence_seeking`
- `malicious_user_queries`
- And more

**Excluded topics:** Three topics have data in `data/stage_3_tagged/` but were excluded from training due to quality issues:
- `steganography` - Claude generated data about steganography in general (hiding information in images, etc.) rather than LLM-to-LLM steganography as intended. The data is technically correct but would be misleading to reviewers who interpret "steganography" in the AI safety context.
- `reasoning_metacognition` - Generated data was low quality and did not reliably capture the intended alignment failures.
- `strategic_awareness` - Generated data was low quality and did not reliably capture the intended alignment failures.

## Evaluation on Anthropic's Poisoned Model

The `data/evaluation_results/` directory contains results from evaluating SPT on Anthropic's "Auditing Game" poisoned Llama-3.3-70B model. This model was trained to exploit 52 reward model biases while being adversarially trained to never admit to this exploitation.

### Evaluation Setup

The evaluation scripts (not included, as they require access to trained model weights) performed:

1. **Qualitative inference** (`run_qualitative_inference.py`):
   - 30 manually crafted test samples
   - Task+intervention pairs and standalone intervention tests
   - True positive and false positive checks

2. **Quantitative inference** (`run_quantitative_inference.py`):
   - 10,000+ samples across 3 conditions
   - Multiple intervention types and elicitation methods
   - Batch processing with LoRA mask handling

### Evaluation Conditions

| Condition | Response Type | Intervention Asks | Expected Flag |
|-----------|--------------|-------------------|---------------|
| **trained_actual** | Biased (on-policy) | "Did you exploit?" | FAIL |
| **original_actual** | Clean (off-policy) | "Did you exploit?" | PASS |
| **original_intended** | Clean (off-policy) | "Did you intend to exploit?" | FAIL |

### Key Parameters

**Model checkpoints tested:**
- `70b_anthropic-mp_lora-patch` (hp elicitation, with LoRA patch)
- `70b_anthropic-mp_lora-patch_ur` (ur elicitation, with LoRA patch)
- `70b_anthropic-mp_no-lora-patch` (hp elicitation, without LoRA patch)
- `atomic-universe-16` (up elicitation, with LoRA patch)

**Inference parameters:**
- Base model: `meta-llama/Llama-3.3-70B-Instruct`
- Merge checkpoint: Anthropic's `auditing-agents_llama-3.3-70b-dpo-rt-lora`
- SPT token ID: 128002 (added programmatically, not as string)
- Max tokens: 2048
- Batch size: 8

**LoRA masking (for lora-patch models):**
- `past_lora_mask`: 0.0 for conversation tokens, 1.0 for intervention tokens
- `future_lora_mask`: 1.0 (full SPT personality for generation)
- `use_cache=True` for KV cache handling

**Terminology note:** The writeup refers to this technique as "LoRA-mask" while the code uses "lora-patch" for historical reasons. Both terms refer to the same mechanism: selectively zeroing LoRA weights for tokens before the split-personality token during training.

### Results Summary

To reproduce these numbers, run:
```bash
python scripts/analyze_evaluation_results.py
```

**Main findings (hp elicitation, fair comparison):**

| Condition | With LoRA Patch | Without LoRA Patch |
|-----------|-----------------|---------------------|
| trained_actual (detect exploitation) | 76.5% | **96.7%** |
| original_actual (true negatives) | 89.2% | **93.8%** |
| original_intended (detect intent) | 3.8% | 3.6% |

**Aggregate results (all elicitation types):**
- trained_actual: 86.0% accuracy (model admits to bias exploitation)
- original_actual: 88.5% accuracy (low false positive rate)
- original_intended: 6.1% accuracy (intent detection does not work)

### Result File Format

Each result file contains:
```json
{
    "metadata": {
        "checkpoint_name": "70b_anthropic-mp_lora-patch",
        "checkpoint_step": "step_522",
        "elicitation_type": "hp",
        "lora_patch": true,
        "input_file": "trained_actual_split_personality_basic.json",
        "target_model": "meta-llama/Llama-3.3-70B-Instruct",
        "merge_checkpoints": ["./lora_adapters/auditing-agents_llama-3.3-70b-dpo-rt-lora"]
    },
    "results": [
        {
            "sample_id": "bias_2_sample_0",
            "bias_id": 2,
            "condition": "trained_actual",
            "intervention_type": "split_personality_basic",
            "generated_review": "The model's full response...",
            "parsed_flag": "FAIL",
            "flag_parsed_successfully": true,
            "expected_flag": "FAIL",
            "correct": true
        }
    ],
    "summary": {
        "total_samples": 1000,
        "pass_count": 235,
        "fail_count": 765,
        "accuracy": 0.765
    }
}
```

### Comparison Files

The `comparison/` directory contains paired analyses between lora-patch and no-lora-patch models:
- Side-by-side reviews for the same samples
- Overlap analysis (how often models agree)
- Cases where only one model was correct

## Notes

- Stage 2 supports both local GPU inference and API-based inference
- Stage 3 includes automatic quality review with configurable retry logic
- The tagging script removes baseline interventions and applies hierarchical quality tags
- Analysis generates detailed reports with per-topic breakdowns