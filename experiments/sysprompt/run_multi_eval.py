import os
import torch as t
import random
from scripts.inference_cache import InferenceCache
from scripts.utils import load_cfg, load_env
from evals import (
    setup_distributed_environment,
    find_checkpoint_steps,
    get_evaluation_splits,
    validate_checkpoint_path,
    load_checkpoint_args,
    evaluate_all_splits,
    print_evaluation_summary,
    save_aggregated_results
)
from scripts.FSDP_helpers import get_model


def main():
    random.seed(42), t.manual_seed(42), t.cuda.manual_seed(42), t.cuda.manual_seed_all(42)
    
    load_env()
    
    config_files = [
        'augmentation_experiment/cfg_baseline.json',
        'augmentation_experiment/cfg_no_sysprompt.json', 
        'augmentation_experiment/cfg_mismatch.json'
    ]
    
    args = load_cfg(config_files[0])
    is_distributed, local_rank, device_id, mesh_1d = setup_distributed_environment(args)
    
    model_name = os.environ.get("base_model")
    if not model_name:
        raise ValueError("base_model environment variable not set")
    
    model_path = args.checkpoint_path
    validate_checkpoint_path(model_path)
    
    step_numbers = find_checkpoint_steps(model_path)
    eval_splits = get_evaluation_splits(args)
    
    if local_rank == 0:
        print("=" * 80)
        print("MULTI-CONFIG EVALUATION (SHARED CACHE)")
        print(f"Checkpoint: {model_path}")
        print(f"Configs: {config_files}")
        print("=" * 80)
    
    for step_num in step_numbers[-1:]:
        checkpoint_path = model_path + f"step_{step_num}/"
        validate_checkpoint_path(checkpoint_path)
        
        if local_rank == 0:
            print(f"\n⏳ Loading checkpoint from {checkpoint_path}...")
        
        load_checkpoint_args(args, model_path)
        args.validate_formatting_params()
        
        tokenizer, model = get_model(
            base_model=model_name,
            args=args,
            checkpoint_path=checkpoint_path,
            mesh=mesh_1d,
            train=False,
            lora_patch=False
        )
        
        if local_rank == 0:
            print("✅ Model loaded successfully!")
        
        cache = InferenceCache()
        
        for config_file in config_files:
            config_args = load_cfg(config_file)
            config_args.device = args.device
            config_args.device_ids = args.device_ids
            
            load_checkpoint_args(config_args, model_path)
            config_args.validate_formatting_params()
            
            config_name = config_file.split('/')[-1].replace('.json', '').replace('cfg_', '')
            
            if local_rank == 0:
                print(f"\n{'='*80}")
                print(f"Running config: {config_name}")
                print(f"  prob_exclude_system_prompt: {getattr(config_args, 'prob_exclude_system_prompt', 0.0)}")
                print(f"  prob_mismatch_prompts: {getattr(config_args, 'prob_mismatch_prompts', 0.0)}")
                print(f"{'='*80}")
            
            step_results, all_samples = evaluate_all_splits(
                model, tokenizer, config_args, eval_splits, is_distributed, local_rank, cache
            )
            
            if local_rank == 0:
                print(f"\nResults for {config_name}:")
                print_evaluation_summary(eval_splits, step_results, step_num)
                save_aggregated_results(
                    config_args, model_path, step_num, model_name,
                    eval_splits, step_results, all_samples,
                    suffix=f"_{config_name}"
                )
        
        if local_rank == 0:
            print(f"\n{'='*80}")
            print(cache.stats())
            print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

