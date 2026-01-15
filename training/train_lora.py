import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

from scripts.utils import *
load_env()


import os, torch as t, random, datetime
import torch.distributed as dist
from scripts.trainer_LoRA import Trainer_LoRA
from scripts.data.full_pipeline import get_batched_dist_data
from scripts.FSDP_helpers import get_model, synchronize_processes
from transformers.utils import logging
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

logging.disable_progress_bar()
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

t.backends.cuda.matmul.allow_tf32 = True
t.set_float32_matmul_precision('high')

def main():
    random.seed(42), t.manual_seed(42), t.cuda.manual_seed(42), t.cuda.manual_seed_all(42) 
    
    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        t.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=t.device("cuda", rank))
    else:
        rank = 0
        world_size = 1
        t.cuda.set_device(0)

    args = load_cfg()
    args.device_ids, args.device = list(range(world_size)) if is_distributed else [0], f"cuda:{rank}"

    os.environ["base_model"] = args.target_model
    os.environ["data_model_source"] = args.data_model_source

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    distributed_data = get_batched_dist_data(tokenizer, world_size)
    
    if is_distributed:
        synchronize_processes("SYNC BEFORE CREATING MESH")
        mesh_1d = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    else:
        mesh_1d = None

    tokenizer, model = get_model(
        base_model = os.environ.get("base_model"),
        args = args,
        mesh = mesh_1d,
        checkpoint_path=None,
        train=True,
        lora_patch=False,
        merge_checkpoints=args.merge_checkpoints
    )

    if (rank==0):
        from torch.distributed.tensor import DTensor
        total_params = sum(p.numel() for p in model.parameters())
        sharded_params = sum(p.numel() for p in model.parameters() if isinstance(p, DTensor))
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Sharded params: {sharded_params:,} ({sharded_params/total_params:.1%})")
        print(f"Unsharded params: {total_params-sharded_params:,} ({(total_params-sharded_params)/total_params:.1%})")
    
    trainer = Trainer_LoRA(args, model, tokenizer, distributed_data[rank])
    trainer.train()
    
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
