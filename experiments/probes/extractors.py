"""Activation extraction from transformer models.

Provides extractors that take a sample and return activation vectors
from specified layers. Designed to be reusable across different pipelines.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional, Protocol

import torch
from peft import PeftModel
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from scripts.data.data_utils import apply_chat_template_open, tokenize
from scripts.data.dataset import Sample

logger = logging.getLogger(__name__)

RESID_POST_HOOK = "blocks.{layer}.hook_resid_post"


@dataclass(frozen=True, slots=True)
class ActivationResult:
    """Activation vectors extracted from a single sample.

    Each tensor has shape (n_layers, hidden_dim).
    """
    response_mean: torch.Tensor
    response_last: torch.Tensor
    intervention_mean: torch.Tensor
    intervention_last: torch.Tensor

    @classmethod
    def stack(cls, results: list["ActivationResult"]) -> "ActivationResult":
        return cls(
            response_mean=torch.stack([r.response_mean for r in results]),
            response_last=torch.stack([r.response_last for r in results]),
            intervention_mean=torch.stack([r.intervention_mean for r in results]),
            intervention_last=torch.stack([r.intervention_last for r in results]),
        )


class ActivationExtractor(Protocol):
    """Protocol for activation extraction backends."""

    def extract(self, sample: Sample, layers: list[int]) -> ActivationResult:
        """Extract activations for response and intervention tokens."""
        ...

    @property
    def hidden_dim(self) -> int:
        ...

    @property
    def n_layers(self) -> int:
        ...


class StubActivationExtractor:
    """Stub extractor that returns random activations for testing."""

    def __init__(self, hidden_dim: int = 5120, n_layers: int = 48):
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers

    def extract(self, sample: Sample, layers: list[int]) -> ActivationResult:
        """Return random activations for testing."""
        n = len(layers)
        return ActivationResult(
            response_mean=torch.randn(n, self._hidden_dim),
            response_last=torch.randn(n, self._hidden_dim),
            intervention_mean=torch.randn(n, self._hidden_dim),
            intervention_last=torch.randn(n, self._hidden_dim),
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def n_layers(self) -> int:
        return self._n_layers


@contextmanager
def _cached_forward(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layers: list[int],
) -> Generator[dict, None, None]:
    """Run forward pass with cache, ensuring cleanup."""
    layer_names = {RESID_POST_HOOK.format(layer=layer) for layer in layers}

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in layer_names,
        )
    try:
        yield cache
    finally:
        del cache


def _load_model_with_lora(
    model_name: str,
    lora_path: Optional[str],
    torch_dtype: torch.dtype,
) -> Optional[AutoModelForCausalLM]:
    """Load HuggingFace model with optional LoRA weights merged."""
    if lora_path is None:
        return None

    logger.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    logger.info(f"Loading LoRA weights from: {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Merging LoRA weights into base model...")
    return lora_model.merge_and_unload()


class TransformerLensActivationExtractor:
    """Extract activations using TransformerLens."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        lora_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name

        hf_model = _load_model_with_lora(model_name, lora_path, torch_dtype)

        if hf_model is not None:
            self.model = HookedTransformer.from_pretrained(
                model_name,
                hf_model=hf_model,
                tokenizer=tokenizer,
                dtype=torch_dtype,
            )
        else:
            self.model = HookedTransformer.from_pretrained(
                model_name,
                tokenizer=tokenizer,
                dtype=torch_dtype,
            )

        self._hidden_dim = self.model.cfg.d_model
        self._n_layers = self.model.cfg.n_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def n_layers(self) -> int:
        return self._n_layers

    def extract(self, sample: Sample, layers: list[int]) -> ActivationResult:
        """Extract activations for response and intervention tokens."""
        response_tokens, intervention_tokens = self._tokenize(sample)
        response_len = len(response_tokens)
        total_len = response_len + len(intervention_tokens)

        tokens = torch.tensor(
            response_tokens + intervention_tokens,
            device=self.model.cfg.device,
        ).unsqueeze(0)

        with _cached_forward(self.model, tokens, layers) as cache:
            layer_results = [
                self._extract_layer(cache, layer, response_len, total_len)
                for layer in layers
            ]

        return ActivationResult.stack(layer_results)

    def _extract_layer(
        self,
        cache: dict,
        layer: int,
        response_len: int,
        total_len: int,
    ) -> ActivationResult:
        """Extract activations for a single layer."""
        resid = cache[RESID_POST_HOOK.format(layer=layer)]
        return ActivationResult(
            response_mean=resid[0, :response_len, :].mean(dim=0).cpu(),
            response_last=resid[0, response_len - 1, :].cpu(),
            intervention_mean=resid[0, response_len:, :].mean(dim=0).cpu(),
            intervention_last=resid[0, total_len - 1, :].cpu(),
        )

    def _tokenize(self, sample: Sample) -> tuple[list[int], list[int]]:
        """Tokenize sample using same format as training."""
        interaction = sample.to_interaction()
        s_block_tokens = apply_chat_template_open(interaction, self.tokenizer, self.model_name)
        intervention_tokens = tokenize(sample.intervention, self.tokenizer)
        return s_block_tokens, intervention_tokens


def create_extractor(
    model_name: str,
    lora_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> ActivationExtractor:
    """Factory function to create an activation extractor.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-14B")
        lora_path: Path to LoRA checkpoint directory (optional)
        torch_dtype: Data type for model weights

    Returns:
        Configured TransformerLensActivationExtractor
    """
    logger.info(f"Loading model: {model_name}")
    if lora_path:
        logger.info(f"With LoRA weights from: {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TransformerLensActivationExtractor(
        model_name,
        tokenizer,
        lora_path=lora_path,
        torch_dtype=torch_dtype,
    )
