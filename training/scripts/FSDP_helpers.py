import torch as t, gc
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy, register_fsdp_forward_method
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from functools import partial
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from scripts.lora_workaround import install_lora_token_masking, patch_huggingface_generation



# IMPORTANT DETAILS:
# 1. Make sure huggingface model is loaded to cpu and stays on cpu during LoRA wrapping.Else we get OOM errors when trying to load a large model. FSDP2 handles moving to GPU.
# 2. Keep everything in 16bf. Except maybe reducetype, this can either be bf16 or float32 for stability.
# 3. In FSDP2 the way sharding works is we shard each relevant submodule, then shard the root. For our purposes "relevant submodule" means we shard decoder layers in the LLM we're looking at, and also embedding/unembedding. Sometimes sharding tied embeddings causes errors, but we try to make it work.
# 4. Reshard after forwards should be TRUE on the submodules and FALSE on the root module. 
# 5. Use gradient checkpoint BEFORE fsdp2. This is very important. FSDP2 needs to be aware of the gradient checkpointing, else we will get OOM errors.
# 6. Make sure we're operating in an environment with latest version of torch, cuda, transformers, peft.
# 7. Putting up manual barriers and synchronization sometimes helps and sometimes hurts. Use it to debug, but when running finished code, aim to have no manual barriers/sync steps.


def synchronize_processes(s=""):
    if (not dist.is_initialized()): print(s)
    else:
        if (s!=""): print(f"@ RANK {dist.get_rank()} || {s}")
        t.cuda.synchronize()
        dist.barrier()
        if ((dist.get_rank() == 0) and s!=""): print("="*80 + f"\nBARRIER PASSED || {s}")



def print_mem(label: str):
    if (dist.get_rank() != 0): return
    if not t.cuda.is_available():
        print(f"[rank {dist.get_rank()}] {label} alloc=0MB reserved=0MB")
        return
    dev = t.cuda.current_device()
    t.cuda.synchronize()
    alloc = t.cuda.memory_allocated(dev) // (1024 * 1024)
    reserved = t.cuda.memory_reserved(dev) // (1024 * 1024)
    print(f"[rank {dist.get_rank()}] {label} alloc={alloc}MB reserved={reserved}MB")


def clear_cache(sync_distributed: bool = True) -> None:
    gc.collect()
    if t.cuda.is_available(): t.cuda.empty_cache()

def custom_wrap_policy(module, module_name=""):
    if isinstance(module, (Gemma3DecoderLayer, Qwen3DecoderLayer, LlamaDecoderLayer)): return True
    if isinstance(module, Gemma3TextScaledWordEmbedding): return True
    if "lm_head" in module_name: return True
    return False

def get_mp_policy():
    return MixedPrecisionPolicy(
        param_dtype=t.bfloat16,
        reduce_dtype=t.bfloat16,
    )

def apply_activation_checkpointing(model):
    non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    for module_path, module in list(model.named_modules()):
        if isinstance(module, (Gemma3DecoderLayer, Qwen3DecoderLayer, LlamaDecoderLayer)):
            parent_path, child_name = module_path.rsplit(".", 1) if "." in module_path else ("", module_path)
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, child_name, non_reentrant_wrapper(module))

def make_lora_config(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        use_rslora=True
    )


def get_model(base_model, args, mesh=None, checkpoint_path=None, use_keymapping=False, train=True, lora_patch=True, merge_checkpoints=None):
    print(f"[get_model] Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(f"[get_model] Loading model to CPU (this may take a few minutes for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=t.bfloat16, attn_implementation="sdpa", device_map="cpu"
    )
    print(f"[get_model] Model loaded to CPU")

    if merge_checkpoints:
        print(f"[get_model] Merging {len(merge_checkpoints)} checkpoints...")
        for path in merge_checkpoints:
            print(f"[get_model]   Merging checkpoint: {path}")
            if not path:
                continue
            peft_model = PeftModel.from_pretrained(
                model,
                path,
                is_trainable=False
            )
            model = peft_model.merge_and_unload()
        model = model.to(t.bfloat16)
        print(f"[get_model] Checkpoint merging complete")

    if checkpoint_path:
        print(f"[get_model] Loading LoRA checkpoint from {checkpoint_path}...")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=False
        )
        model = model.to(t.bfloat16)
        model.set_adapter("default")
        print(f"[get_model] LoRA checkpoint loaded")
    else:
        print(f"[get_model] Applying fresh LoRA config (r={args.lora_r})...")
        model = get_peft_model(model, make_lora_config(args)).to(t.bfloat16)
        print(f"[get_model] LoRA applied")

    if (lora_patch):
        install_lora_token_masking(model)
        patch_huggingface_generation(model)
        synchronize_processes("LORA PATCH")

    if (train):
        print(f"[get_model] Applying activation checkpointing...")
        apply_activation_checkpointing(model)
        model.config.use_cache = False
        model.train()
        print(f"[get_model] Model set to train mode")

        print(f"[get_model] Starting torch.compile (this can take 15-30+ mins for large models)...")
        model = t.compile(model)
        print(f"[get_model] torch.compile complete")
    else:
        model.config.use_cache = True
        model.eval()
        print(f"[get_model] Model set to eval mode")
    

    
    if (mesh is not None) and dist.is_initialized():
        print(f"[get_model] Applying FSDP sharding across devices...")
        for n, m in reversed(list(model.named_modules())):
            if custom_wrap_policy(m, n):
                fully_shard(m, mesh=mesh, mp_policy=get_mp_policy(), reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=get_mp_policy(), reshard_after_forward=False)
        register_fsdp_forward_method(model, "generate")
        print(f"[get_model] FSDP sharding complete")
    else:
        print(f"[get_model] Moving model to {args.device}...")
        model = model.to(args.device)
        print(f"[get_model] Model moved to device")

    print(f"[get_model] Model ready!")
    return tokenizer, model


