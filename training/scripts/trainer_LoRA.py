import wandb, torch as t, os, random, torch.distributed as dist
import json
from torch.optim import AdamW
from scripts.FSDP_helpers import clear_cache, synchronize_processes
from tqdm import tqdm
from safetensors.torch import save_file

#IMPORTANT DETAILS
# 1. We assume this is always running FSDP, so we can assume t.distribut.is_initialized() returns True etc


class Trainer_LoRA():
    def __init__(self, args, model, tokenizer, batches):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.batches = batches
        self.opt = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.global_step = 0
        self.losses = []
        self.grad_norms = []
        self.last_logged_loss = 0.0
        self.is_rank_0 = not t.distributed.is_initialized() or t.distributed.get_rank() == 0
        self.is_distributed = t.distributed.is_initialized()
        self.rank = t.distributed.get_rank() if self.is_distributed else 0

        if self.is_rank_0: 
            print("\n" + "="*80 + "\n ||| STARTING TRAINING!!!!!!!!!! ||| \n" + "="*80)
            print("\n\n" + "="*140)
            wandb.init(project="anthropic_model_poison_test", config=args)
            print("="*140 + "\n\n")
            total_steps = self.args.epochs * len(self.batches)
            self.pbar = tqdm(total=total_steps, desc="Training", unit="step", 
                           postfix={'epoch': 1, 'loss': 0.0, 'lr': args.lr})
        else: self.pbar = None

    def get_optimizer(self):
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        return AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
            fused=True,
            foreach=False
        )

    def get_scheduler(self):
        import math
        total_steps = self.args.epochs * len(self.batches)
        warmup_steps = int(total_steps * self.args.lr_warmup_frac)
        cooldown_steps = int(total_steps * self.args.lr_cooldown_frac)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            elif step < total_steps - cooldown_steps:
                return 1.0
            else:
                progress = (step - (total_steps - cooldown_steps)) / cooldown_steps
                return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

        return t.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def compute_loss_with_mask(self, batch):
        tokens, mask = batch

        pad_id           = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
        attention_mask   = (tokens != pad_id).long()

        logits           = self.model(tokens, attention_mask=attention_mask, adapter_mask=mask).logits

        predicted_logits = logits[:,:-1].reshape(-1, logits.size(-1))
        target_tokens    = tokens[:,1:].reshape(-1)
        mask             = mask[:,:-1].reshape(-1)

        loss = t.nn.functional.cross_entropy(predicted_logits, target_tokens, reduction="none") * mask

        return loss.sum() / mask.sum()

    def log(self):
        if self.global_step % self.args.log_every == 0:
            mean_loss = sum(self.losses)/self.args.log_every
            loss_value = t.tensor(mean_loss, device=self.args.device)
            if self.is_distributed:
                t.distributed.all_reduce(loss_value, op=t.distributed.ReduceOp.SUM)
                loss_value /= t.distributed.get_world_size()
            self.losses = []

            mean_grad_norm = sum(self.grad_norms)/self.args.log_every
            grad_norm_value = t.tensor(mean_grad_norm, device=self.args.device)
            if self.is_distributed:
                t.distributed.all_reduce(grad_norm_value, op=t.distributed.ReduceOp.SUM)
                grad_norm_value /= t.distributed.get_world_size()
            self.grad_norms = []

            if self.is_rank_0:
                self.last_logged_loss = loss_value.item()
                wandb.log({
                    "train_loss": self.last_logged_loss,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm_value.item()
                }, step=self.global_step)

                if self.pbar:
                    self.pbar.set_postfix({
                        'epoch': f"{(self.global_step // len(self.batches)) + 1}/{self.args.epochs}",
                        'loss': f"{self.last_logged_loss:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })


    def train_step(self, batch):
        self.opt.zero_grad(set_to_none=True)
        self.loss = self.compute_loss_with_mask(batch)
        self.loss.backward()

        self.grad_norm = t.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.opt.step()
        self.scheduler.step()
        self.global_step += 1

        self.losses.append(self.loss.item())
        self.grad_norms.append(self.grad_norm.item())
        self.log()
        
        if self.pbar:
            self.pbar.update(1)

    def train_epoch(self):
        random.shuffle(self.batches)

        clear_cache()
        synchronize_processes(f" EPOCH SYNC ")

        for batch in self.batches:
            tokens, mask = batch
            batch = (tokens.to(self.args.device), mask.to(self.args.device))
            self.train_step(batch)
        self.save_checkpoint()

    def train(self):
        self.model.train()
        for _ in range(self.args.epochs):
            self.train_epoch()
        
        if self.pbar: self.pbar.close()
        if self.is_rank_0: wandb.finish()


    def save_checkpoint(self):
        adapter_sd = {}
        for name, param in self.model.named_parameters():
            if "lora_" not in name or not param.requires_grad: continue
            full_param = param.full_tensor() if hasattr(param, "full_tensor") else param
            if (self.is_rank_0):
                cleaned_up_name = name.replace("._checkpoint_wrapped_module", "")
                cleaned_up_name = cleaned_up_name.replace("_orig_mod.", "")
                cleaned_up_name = cleaned_up_name.replace(".default.", ".").removesuffix(".default")
                if ("base_model" not in cleaned_up_name): cleaned_up_name = "base_model.model." + cleaned_up_name
                adapter_sd[cleaned_up_name] = full_param.cpu()
            else:
                del full_param

        if (self.is_rank_0):
            ckpt_dir  = f"./checkpoints/{wandb.run.name}"
            step_dir  = os.path.join(ckpt_dir, f"step_{self.global_step}")

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(step_dir, exist_ok=True)
        
            with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
                json.dump(vars(self.args), f)

            save_file(adapter_sd, os.path.join(step_dir, "adapter_model.safetensors"))
            self.model.peft_config["default"].save_pretrained(step_dir)
            self.pbar.write(f"Checkpoint saved at {step_dir} ({len(adapter_sd)} tensors)")
