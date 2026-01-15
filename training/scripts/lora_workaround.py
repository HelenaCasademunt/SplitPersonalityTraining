import torch
from torch import nn
from typing import Optional
import types
from peft.tuners.lora.layer import LoraLayer

# ==========================================================================================================================
# ======================================== FORWARD PASS PATCH ==============================================================
# ==========================================================================================================================

CURRENT_LORA_MASK:   Optional[torch.Tensor] = None
CURRENT_LORA_OFFSET: Optional[int]          = None
FORWARD_PASS_SHIFT:  Optional[int]          = None

def _broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < target.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    return expanded_mask.to(target.device, target.dtype)


def _slice_global_mask(target: torch.Tensor) -> Optional[torch.Tensor]:
    global FORWARD_PASS_SHIFT
    if CURRENT_LORA_MASK is None:
        return None
    
    if CURRENT_LORA_OFFSET is None:
        mask = CURRENT_LORA_MASK[:, :target.size(1)]
    else:
        FORWARD_PASS_SHIFT = target.size(1)
        mask = CURRENT_LORA_MASK[:, CURRENT_LORA_OFFSET: (CURRENT_LORA_OFFSET + target.size(1))]

    return _broadcast_mask(mask, target)


def _mask_dropout_hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
    mask = _slice_global_mask(output)
    if mask is None: return output
    return output * mask


def _patch_forward(model: nn.Module):
    original_forward = model.forward

    # DO NOT RESET MASK AFTER FORWARD!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THIS BREAKS ACTIVATION CHECKPOINTING!!!!!!!!!!!!!!!!!!!!!!
    def forward_with_mask(*args, **kwargs):
        global CURRENT_LORA_MASK, CURRENT_LORA_OFFSET, FORWARD_PASS_SHIFT
        mask = kwargs.pop("adapter_mask", None)
        if mask is not None:
            CURRENT_LORA_MASK = mask
        
        if (CURRENT_LORA_OFFSET is not None):
            out = original_forward(*args, **kwargs)
            CURRENT_LORA_OFFSET += FORWARD_PASS_SHIFT
            FORWARD_PASS_SHIFT  = 0
            return out
        else:
            return original_forward(*args, **kwargs)

    model.forward = forward_with_mask


def install_lora_token_masking(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for dropout in module.lora_dropout.values():
                dropout.register_forward_hook(_mask_dropout_hook)

    # Find the correct model to patch - different model architectures have different nesting
    # Gemma3: model.base_model.model.model
    # GPT2: model.base_model.model
    if hasattr(model.base_model.model, 'model'):
        # Gemma3 and similar
        _patch_forward(model.base_model.model.model)
    else:
        # GPT2 and similar
        _patch_forward(model.base_model.model)

# ==========================================================================================================================
# ======================================== HUGGINGFACE- GENERATION PATCH ===================================================
# ==========================================================================================================================


def _patch_generate(model):
    original_generate = model.generate

    def generate_with_mask(self, *args, **kwargs):
        global CURRENT_LORA_MASK, CURRENT_LORA_OFFSET, FORWARD_PASS_SHIFT
        past_mask = kwargs.pop("past_lora_mask", None)
        future_mask = kwargs.pop("future_lora_mask", None)

        if ((past_mask is None) or (future_mask is None)):
            raise ValueError("You either did not pass past_lora_mask or did not pass future_lora_mask, but those are required arguments.")
        
        if (not ('max_new_tokens' in kwargs)):
            raise ValueError("You need to pass max_new_tokens.")

        if (not ('use_cache' in kwargs)):
            raise ValueError("You need to pass use_cache=True/False")
        
        if (not isinstance(kwargs["max_new_tokens"], int)):
            raise ValueError("max_new_tokens has to be an int")
        
        if (not isinstance(future_mask, float)):
            raise ValueError("future_lora_mask must be a float")

        if past_mask.ndim != 2:
            raise ValueError("Your past_lora_mask has wrong dimensionality")
        
        future_tensor = torch.full(
            (past_mask.shape[0], kwargs["max_new_tokens"]),
            future_mask,
            device=past_mask.device,
            dtype=past_mask.dtype,
        )

        CURRENT_LORA_MASK   = torch.cat([past_mask, future_tensor], dim=1)

        if (kwargs['use_cache']):
            CURRENT_LORA_OFFSET = 0
            FORWARD_PASS_SHIFT  = 0
        else:
            CURRENT_LORA_OFFSET = None
            FORWARD_PASS_SHIFT  = None


        result = original_generate(*args, **kwargs)
        
        CURRENT_LORA_MASK   = None
        CURRENT_LORA_OFFSET = None
        FORWARD_PASS_SHIFT  = None
        
        return result

    model.generate = types.MethodType(generate_with_mask, model)

def patch_huggingface_generation(model) -> None:
    _patch_generate(model)