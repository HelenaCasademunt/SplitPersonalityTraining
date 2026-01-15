from scripts.utils import load_cfg
from scripts.data.claude_data import get_claude_tokenized
from scripts.data.data_utils import pad, filter_by_length
import random, torch as t, os
import torch.distributed as dist



def full_data_pipeline(tokenizer):
    cfg = load_cfg()
    model_name = os.environ.get("base_model")

    if (cfg.use_dummy_data == 1):
        print("USING DUMMY DATA!!!!")
        dummy_samples = [
            (
                t.randint(0, tokenizer.vocab_size, (cfg.seq_len,), dtype=t.long),
                t.ones(cfg.seq_len, dtype=t.long)
            ) for _ in range(1024)]
        return dummy_samples

    claude_data = ([] if (cfg.claude_frac == 0.0) else get_claude_tokenized(tokenizer, model_name, cfg))

    claude_filtered = filter_by_length(claude_data, cfg.seq_len)

    claude_tensor = [(t.tensor(x, dtype=t.long), t.tensor(y, dtype=t.long)) for x, y in claude_filtered]

    claude_padded = [pad(x, cfg.seq_len, tokenizer) for x in claude_tensor]

    claude = [claude_padded[i % len(claude_padded)] for i in range(0, int(len(claude_padded) * cfg.claude_frac))]

    total_data = claude
    if not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0):
        print("=" * 80)
        print(f"total samples   || claude: {len(claude_data)}")
        print(f"after filtering || claude: {len(claude_filtered)}")
        print(f"after rationing || claude: {len(claude)}")
        print(f"TOTAL DATA : {len(total_data)}")
        print("=" * 80)

    random.shuffle(total_data)
    return total_data if cfg.num_samples == -1 else total_data[:cfg.num_samples]



def full_data_pipeline_distributed(tokenizer, world_size):
    batch_size = load_cfg().batch_size

    data = full_data_pipeline(tokenizer)
    data = data[:(len(data) - len(data)%(batch_size * world_size))]

    return [data[rank :: world_size] for rank in list(range(0, world_size))]



def get_batched_dist_data(tokenizer, world_size):
    data = full_data_pipeline_distributed(tokenizer, world_size)
    batch_size = load_cfg().batch_size
    batched = []
    for rank_data in data:
        rank_batches = []
        for i in range(0, len(rank_data), batch_size):
            batch_slice = rank_data[i:i + batch_size]
            if len(batch_slice) == batch_size:
                input_ids = t.stack([tokens for tokens, _ in batch_slice])
                grad_masks = t.stack([mask for _, mask in batch_slice])
                rank_batches.append((input_ids, grad_masks))
        batched.append(rank_batches)
    return batched
















