#!/usr/bin/env python3
"""
Stage 2 Data Generation Script for HonestPersona Project

This script runs inference on stage 1 data using HuggingFace models to generate [S] responses.
It processes [A/B][T] data and adds [S] responses to the inferences field.

The script:
- Uses HuggingFace Hub for model inference
- Supports batch processing with configurable batch size
- Handles model-specific tokenization and padding
- Aggregates results from multiple models in the same output file
- Requires GPU and proper HF authentication
"""

import argparse
import concurrent.futures
import json
import time
from abc import ABC, abstractmethod
from os import environ
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict

import requests
from requests.exceptions import ChunkedEncodingError, RequestException
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib3.exceptions import ProtocolError

from utils import STAGE_1_DATA_DIR, STAGE_2_DATA_DIR, setup_server_environment


class InferenceResult(TypedDict):
    """Typed result structure for inference outputs."""

    S: str  # Generated response text
    input_tokens: int  # Number of input tokens
    output_tokens: int  # Number of output tokens
    string_length: int  # Character length of response
    generation_time: float  # Time taken to generate response
    temperature: float | None  # Sampling temperature used
    top_p: float | None  # Top-p parameter used
    top_k: int | None  # Top-k parameter used
    thinking_enabled: bool | None  # Whether thinking mode was enabled


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def format_conversation(
        self, system_prompt: str, user_message: str
    ) -> List[Dict[str, str]]:
        """
        Format system prompt and user message as a conversation.

        Args:
            system_prompt: The system prompt (A or B variant)
            user_message: The user message (T)

        Returns:
            List of conversation messages
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    @abstractmethod
    def run_batch_inference(
        self, conversations: List[List[Dict[str, str]]], max_new_tokens: int = 2048
    ) -> List[InferenceResult]:
        """Run inference on a batch of conversations."""
        pass


class HuggingFaceInferenceEngine(InferenceEngine):
    """Handles HuggingFace model loading and inference with proper tokenization."""

    def __init__(
        self,
        model_name: str,
        use_flash_attention: bool = False,
        use_quantization: bool = True,
        quantization_bits: int = 8,
    ):
        """
        Initialize the inference engine with a specific model.

        Args:
            model_name: HuggingFace model identifier (e.g., "google/gemma-3-12b-it")
            use_flash_attention: Whether to use flash_attention_2 if available
            use_quantization: Whether to use quantization for memory efficiency
            quantization_bits: Number of bits for quantization (4 or 8)
        """
        self.model_name = model_name

        # Set up server environment and authentication
        setup_server_environment()

        # Ensure GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required but not available. Please ensure CUDA is installed and a GPU is available."
            )

        self.device = torch.device("cuda")
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

        # Load tokenizer with proper configuration
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )

        # Configure padding - check model-specific requirements
        if "gemma" in model_name.lower():
            # Gemma models typically need left padding for batch inference
            self.tokenizer.padding_side = "left"
        elif "llama" in model_name.lower():
            # Llama models typically need left padding for batch inference
            self.tokenizer.padding_side = "left"
        else:
            # Default to left padding for batch generation
            self.tokenizer.padding_side = "left"

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Set pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}")

        # Load model with appropriate configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16,  # Use bfloat16 for better numerical stability
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # Reduce CPU memory usage
        }

        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Attempting to use flash_attention_2")

        # Add quantization for memory efficiency
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig

                if quantization_bits == 4:
                    # Configure 4-bit quantization (extreme memory savings)
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    print("Using 4-bit quantization for extreme memory efficiency")
                else:
                    # Configure 8-bit quantization (reasonable memory savings)
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    print("Using 8-bit quantization for memory efficiency")

                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("BitsAndBytes not available, using full precision model")
        else:
            print("Quantization disabled, using full precision model")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Disable gradient computation to save memory
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(
            f"Model loaded successfully. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB"
        )

    def run_batch_inference(
        self, conversations: List[List[Dict[str, str]]], max_new_tokens: int = 2048
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of conversations.

        Args:
            conversations: List of formatted conversation histories
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            List of inference results with response text and metadata
        """
        if not conversations:
            return []

        # Apply chat template to each conversation
        formatted_inputs = []
        for conversation in conversations:
            try:
                # Use tokenizer's chat template for proper formatting
                formatted = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                formatted_inputs.append(formatted)
            except Exception as e:
                raise RuntimeError(f"Failed to apply chat template: {e}")

        # Tokenize all inputs
        tokenized = self.tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            max_length=2048,  # Reasonable context limit
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        # Record input token counts for each sample
        input_token_counts = attention_mask.sum(dim=1).tolist()

        # Generate responses
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.8,  # Reasonable temperature for diversity
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                # Memory optimizations
                return_dict_in_generate=False,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        generation_time = time.time() - start_time

        # Clear GPU memory to prevent accumulation
        torch.cuda.empty_cache()

        # Force garbage collection to free up memory
        import gc

        gc.collect()

        # Reset the model's cache to free memory
        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()

        # Force CUDA memory cleanup
        torch.cuda.synchronize()

        # Clear CUDA graphs and private pools
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

        # Force memory defragmentation
        torch.cuda.empty_cache()

        # Extract and decode generated text
        results = []
        for i, output_ids in enumerate(outputs):
            # Get only the newly generated tokens
            input_length = input_ids[i].shape[0]
            generated_ids = output_ids[input_length:]

            # Decode the generated response
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Clean up the response (remove any trailing whitespace)
            response = response.strip()

            # Re-encode the cleaned response to get accurate token count (without padding)
            actual_output_tokens = len(
                self.tokenizer.encode(response, add_special_tokens=False)
            )
            result: InferenceResult = {
                "S": response,
                "input_tokens": input_token_counts[i],
                "output_tokens": actual_output_tokens,
                "string_length": len(response),
                "generation_time": generation_time
                / len(outputs),  # Average time per sample
                "temperature": 0.8,  # Hardcoded in generate() call
                "top_p": None,  # Not used in HuggingFace inference
                "top_k": None,  # Not used in HuggingFace inference
                "thinking_enabled": None,  # Not applicable for HuggingFace models
            }
            results.append(result)

        return results


class OpenRouterInferenceEngine(InferenceEngine):
    """Handles OpenRouter API inference with multiple model support."""

    def __init__(
        self,
        model_name: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        disable_thinking: bool = True,
    ):
        """
        Initialize the OpenRouter inference engine.

        Args:
            model_name: OpenRouter model identifier (e.g., "anthropic/claude-3-sonnet")
            temperature: Sampling temperature for generation
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            disable_thinking: Whether to disable thinking mode for models that support it
        """
        super().__init__(model_name)
        self._url = "https://openrouter.ai/api/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.disable_thinking = disable_thinking

        print(f"Initialized OpenRouter for model: {model_name}")
        if disable_thinking:
            print("  Thinking mode: DISABLED")

    def format_conversation(
        self, system_prompt: str, user_message: str
    ) -> List[Dict[str, str]]:
        """
        Format system prompt and user message as a conversation.
        For qwen models with thinking disabled, appends /no_think to system prompt.

        Args:
            system_prompt: The system prompt (A or B variant)
            user_message: The user message (T)

        Returns:
            List of conversation messages
        """
        # Add /no_think to system prompt for qwen models when thinking is disabled
        if self.disable_thinking and "qwen" in self.model_name.lower():
            system_prompt = system_prompt + "\n\n/no_think"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _create_request(self, messages: list, max_new_tokens: int):
        payload: dict[str, list | str | int | float | dict] = dict(
            model=self.model_name, messages=messages, max_tokens=max_new_tokens
        )
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.disable_thinking:
            payload["reasoning"] = {"enabled": False}
        return {"url": self._url, "headers": self._headers, "json": payload}

    def run_single_inference(
        self, conversation: List[Dict[str, str]], max_new_tokens: int = 2048
    ) -> InferenceResult:
        """Inference on a single conversation using the openrouter API."""
        assert isinstance(conversation, list), (
            f"`conversation` is of type {type(conversation)}, but expected `list`."
        )
        if not conversation:
            return {
                "S": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "string_length": 0,
                "generation_time": 0,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "thinking_enabled": not self.disable_thinking,
            }
        assert any([isinstance(turn, dict) for turn in conversation]), (
            f"`conversation` contains elements that are not `dict`: {[type(turn) for turn in conversation]}"
        )

        # Retry logic for network errors
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                http_response = requests.post(**self._create_request(conversation, max_new_tokens))
                generation_time = time.time() - start_time

                # Check HTTP status code
                if http_response.status_code != 200:
                    error_text = http_response.text[:500]
                    raise RuntimeError(
                        f"API request failed with status {http_response.status_code}. "
                        f"Response: {error_text}"
                    )

                # Extract response content
                response = http_response.json()

                # Check if response contains expected fields
                if "choices" not in response:
                    # Response doesn't have expected structure - likely an error
                    error_msg = response.get("error", {})
                    if isinstance(error_msg, dict):
                        error_code = error_msg.get("code", "unknown")
                        error_message = error_msg.get("message", "unknown error")
                        raise RuntimeError(
                            f"API error: {error_code} - {error_message}\n"
                            f"Full response: {json.dumps(response, indent=2)}"
                        )
                    else:
                        raise RuntimeError(
                            f"API returned unexpected response structure (no 'choices' field).\n"
                            f"Full response: {json.dumps(response, indent=2)}"
                        )

                response_message = response["choices"][0]["message"]
                response_text = response_message["content"]
                response_text = response_text.strip()

                # Get token usage from response
                usage = response["usage"]
                input_tokens = usage["prompt_tokens"] if usage else 0
                output_tokens = usage["completion_tokens"] if usage else 0

                # Check for thinking/reasoning content when thinking should be disabled
                # Based on empirical testing: OpenRouter returns reasoning in message.reasoning and message.reasoning_details
                if self.disable_thinking and "qwen" in self.model_name.lower():
                    reasoning_text = response_message.get("reasoning", "")
                    reasoning_details = response_message.get("reasoning_details", [])

                    # Check if reasoning field has substantial content (more than whitespace)
                    if reasoning_text and reasoning_text.strip() and len(reasoning_text.strip()) > 10:
                        raise RuntimeError(
                            f"Thinking mode is disabled but API returned reasoning content! "
                            f"Model: {self.model_name}, Reasoning length: {len(reasoning_text)}, "
                            f"Preview: {reasoning_text[:200]}..."
                        )

                    # Also check reasoning_details array for substantial content
                    if reasoning_details:
                        for detail in reasoning_details:
                            if isinstance(detail, dict):
                                detail_text = detail.get("text", "")
                                if detail_text and detail_text.strip() and len(detail_text.strip()) > 10:
                                    raise RuntimeError(
                                        f"Thinking mode is disabled but API returned reasoning_details content! "
                                        f"Model: {self.model_name}, Detail length: {len(detail_text)}, "
                                        f"Preview: {detail_text[:200]}..."
                                    )

                # If token counts not provided, estimate them
                if not usage:
                    print(
                        "Warning: token counts are only estimates, no exact usage stats were provided."
                    )
                    # Rough estimation: ~4 chars per token
                    input_text = " ".join([msg["content"] for msg in conversation])
                    input_tokens = len(input_text) // 4
                    output_tokens = len(response_text) // 4

                result: InferenceResult = {
                    "S": response_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "string_length": len(response_text),
                    "generation_time": generation_time,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "thinking_enabled": not self.disable_thinking,
                }
                return result
                
            except (ChunkedEncodingError, ProtocolError, RequestException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # exponential backoff
                    print(f"Network error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts. Raising error.")
                    raise

    def run_batch_inference(
        self, conversations: List[List[Dict[str, str]]], max_new_tokens: int = 2048
    ) -> List[InferenceResult]:
        """
        Run inference on a batch of conversations using OpenRouter API.

        Args:
            conversations: List of formatted conversation histories
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            List of inference results with response text and metadata
        """
        if not conversations:
            return []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # the next statement runs an inference for every conversation with the set token length in parallel
            results = list(
                executor.map(
                    lambda conversation: self.run_single_inference(
                        conversation, max_new_tokens=max_new_tokens
                    ),
                    conversations,
                )
            )

        print(results)
        return results


class Stage2DataGenerator:
    """Handles stage 2 data generation by running inference on stage 1 data."""

    def __init__(
        self,
        names: list,
        model: str,
        inference: Literal["local", "openrouter"],
        max_batch_size: int = 8,
        use_flash_attention: bool | None = None,
        use_quantization: bool | None = None,
        quantization_bits: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        disable_thinking: bool = True,
    ):
        """
        Initialize the data generator.

        Args:
            names: List of dataset names to process (or single name as string)
            model: HuggingFace model to use for inference
            max_batch_size: Maximum number of samples to process at once
            use_flash_attention: Whether to use flash_attention_2 if available
            use_quantization: Whether to use quantization for memory efficiency
            quantization_bits: Number of bits for quantization (4 or 8)
            temperature: Sampling temperature for generation
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            disable_thinking: Whether to disable thinking mode for models that support it
        """
        self.names = names if isinstance(names, list) else [names]
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.disable_thinking = disable_thinking

        self.model_name = model
        self.max_batch_size = max_batch_size

        # Default values for parameters are set within _initialize_inference.
        self.inference_engine = self._initialize_inference(
            model,
            inference,
            use_flash_attention,
            use_quantization,
            quantization_bits,
        )

        # Ensure stage 2 directories exist
        for name in self.names:
            stage2_dir = STAGE_2_DATA_DIR / name
            stage2_dir.mkdir(parents=True, exist_ok=True)

        print("Initialized Stage2DataGenerator:")
        print(f"  Dataset names: {', '.join(self.names)}")
        print(f"  Model: {model}")
        print(f"  Max batch size: {max_batch_size}")

    def _initialize_inference(
        self,
        model: str,
        inference: Literal["local", "openrouter"],
        use_flash_attention: bool | None = None,
        use_quantization: bool | None = None,
        quantization_bits: int | None = None,
    ) -> InferenceEngine:
        """Initializes supported InferenceEngine sub classes based on the inference type.

        Args:
            model (str): Model name.
            inference (str): Inference mode, can be local or openrouter.
        """
        if inference not in ["local", "openrouter"]:
            raise NotImplementedError(
                "Currently, only `local` and `openrouter` are supported inference modes."
            )

        self.inference = inference
        if inference == "local":
            # set default values and instantiate HuggingFace model
            self.use_flash_attention = (
                use_flash_attention if use_flash_attention is not None else False
            )
            self.use_quantization = (
                use_quantization if use_quantization is not None else True
            )
            self.quantization_bits = (
                quantization_bits if quantization_bits is not None else 8
            )
            return HuggingFaceInferenceEngine(
                model,
                self.use_flash_attention,
                self.use_quantization,
                self.quantization_bits,
            )

        if inference == "openrouter":
            # set all irrelevant variables to None and initialize Openrouter model
            self.use_flash_attention = None
            self.use_quantization = None
            self.quantization_bits = None
            return OpenRouterInferenceEngine(
                model,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                disable_thinking=self.disable_thinking,
            )

    def _reload_model(self):
        """Reload the model to clear accumulated memory."""
        print("Reloading model to clear memory...")
        assert isinstance(self.inference_engine, HuggingFaceInferenceEngine), "Can only reload model if inference is run locally."

        # Clear memory
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        torch.cuda.synchronize()

        # Delete old model
        del self.inference_engine
        torch.cuda.empty_cache()

        # Reload model
        self.inference_engine = HuggingFaceInferenceEngine(
            self.model_name,
            self.use_flash_attention,
            self.use_quantization,
            self.quantization_bits,
        )
        print("Model reloaded successfully")

    def _gather_inference_tasks(
        self, data: Dict[str, Any]
    ) -> List[Tuple[int, str, List[Dict[str, str]]]]:
        """
        Gather all inference tasks from the data that need to be processed.

        Args:
            data: The loaded JSON data from stage 1

        Returns:
            List of (sample_index, variant_type, conversation) tuples
        """
        tasks = []

        for sample_idx, sample in enumerate(data["data"]):
            # Initialize inferences if not present
            if "inferences" not in sample:
                sample["inferences"] = {}

            # Check if this model already has results
            if self.model_name not in sample["inferences"]:
                sample["inferences"][self.model_name] = {}

            # Check for A variant
            if "A" in sample and "A" not in sample["inferences"][self.model_name]:
                conversation = self.inference_engine.format_conversation(
                    sample["A"], sample["T"]
                )
                tasks.append((sample_idx, "A", conversation))

            # Check for B variant
            if "B" in sample and "B" not in sample["inferences"][self.model_name]:
                conversation = self.inference_engine.format_conversation(
                    sample["B"], sample["T"]
                )
                tasks.append((sample_idx, "B", conversation))

        return tasks

    def _process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Process a single input file and save the output.

        Args:
            input_path: Path to input stage 1 file
            output_path: Path to output stage 2 file
        """
        # Load the input data
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If output file exists, use it as the base (for aggregating multiple models)
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loading existing output file: {output_path.name}")
        else:
            print(f"Creating new output file: {output_path.name}")

        # Gather all inference tasks that need to be processed
        tasks = self._gather_inference_tasks(data)

        if not tasks:
            print(
                f"No inference tasks needed for model {self.model_name} in {input_path.name}"
            )
            return

        print(f"Found {len(tasks)} inference tasks to process")

        # Process tasks in batches
        total_tasks = len(tasks)
        for batch_start in range(0, total_tasks, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total_tasks)
            batch_tasks = tasks[batch_start:batch_end]

            print(
                f"Processing batch {batch_start // self.max_batch_size + 1}: tasks {batch_start + 1}-{batch_end}/{total_tasks}"
            )

            # Extract conversations from batch tasks
            conversations = [task[2] for task in batch_tasks]

            # Run batch inference
            try:
                results = self.inference_engine.run_batch_inference(conversations)
            except Exception as e:
                raise RuntimeError(f"Batch inference failed: {e}") from e

            # Store results back in the data structure
            for task, result in zip(batch_tasks, results):
                sample_idx, variant_type, _ = task
                data["data"][sample_idx]["inferences"][self.model_name][
                    variant_type
                ] = result

            print(f"Completed batch {batch_start // self.max_batch_size + 1}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

            # Save progress after each batch (enables resuming from last completed batch)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved results to {output_path.name}")

    def generate_inferences(self) -> None:
        """
        Generate inferences for all stage 1 data files across all datasets.
        """
        total_files_processed = 0

        for dataset_idx, name in enumerate(self.names, 1):
            input_dir = STAGE_1_DATA_DIR / name
            output_dir = STAGE_2_DATA_DIR / name

            if not input_dir.exists():
                print(f"Warning: Input directory not found: {input_dir}")
                continue

            # Get all input files
            input_files = sorted(list(input_dir.glob("batch_*.json")))
            if not input_files:
                print(f"Warning: No batch files found in {input_dir}")
                continue

            print(f"\n=== Dataset {dataset_idx}/{len(self.names)}: {name} ===")
            print(f"Input directory: {input_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Found {len(input_files)} files to process")
            print(f"Model: {self.model_name}")
            print(f"Max batch size: {self.max_batch_size}")

            # Process each file
            for file_idx, input_file in enumerate(input_files):
                output_file = output_dir / input_file.name

                print(
                    f"\n--- File {file_idx + 1}/{len(input_files)}: {input_file.name} ---"
                )

                try:
                    if self.inference == "local":
                        print(f"\n--- Memory before processing {input_file.name} ---")
                        print(
                            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB"
                        )
                        print(
                            f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB"
                        )
                        print(
                            f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB"
                        )

                    start_time = time.time()
                    self._process_file(input_file, output_file)
                    elapsed = time.time() - start_time
                    print(f"Completed {input_file.name} in {elapsed:.1f} seconds")
                    total_files_processed += 1

                    if self.inference == "local":
                        print(f"\n--- Memory after processing {input_file.name} ---")
                        print(
                            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB"
                        )
                        print(
                            f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB"
                        )
                        print(
                            f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB"
                        )

                    # Clear memory between files to prevent accumulation
                    print("Clearing memory...")
                    if self.inference == "local":
                        torch.cuda.empty_cache()

                    import gc

                    gc.collect()

                    if self.inference == "local":
                        # Force CUDA memory cleanup
                        torch.cuda.synchronize()

                        # Clear CUDA graphs and private pools
                        if hasattr(torch.cuda, "reset_peak_memory_stats"):
                            torch.cuda.reset_peak_memory_stats()

                        # Force memory defragmentation
                        torch.cuda.empty_cache()

                        print("--- Memory after clearing ---")
                        print(
                            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB"
                        )
                        print(
                            f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB"
                        )
                        print(
                            f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB"
                        )

                        # If memory is still high after clearing, reload the model
                        if (
                            torch.cuda.memory_allocated() > 30e9
                        ):  # If more than 30GB allocated
                            print(
                                "Memory still high after clearing - reloading model..."
                            )
                            self._reload_model()

                except Exception as e:
                    print(f"ERROR processing {input_file.name}: {e}")
                    raise

            print(f"\n=== Dataset {name} Complete ===")
            print(f"Processed {len(input_files)} files successfully")
            print(f"Output saved to: {output_dir}")

            # Clear memory between datasets
            print(f"Clearing memory after dataset {name}...")
            import gc

            gc.collect()
            if self.inference == "local":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        print("\n=== All Datasets Complete ===")
        print(f"Total files processed: {total_files_processed}")
        print(f"Datasets processed: {len(self.names)}")


def main():
    """Main entry point for the script."""
    # Set environment variables for better memory management
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    parser = argparse.ArgumentParser(
        description="Generate stage 2 inferences for HonestPersona project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/dataset_generation_stage_2.py --name TEST
  python src/scripts/dataset_generation_stage_2.py --names TEST sycophancy confidence_assessment
  python src/scripts/dataset_generation_stage_2.py --names all --model google/gemma-3-27b-it --batch_size 4
        """,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="TEST",
        help="Name of a single dataset (default: TEST). Use --names for multiple datasets.",
    )

    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Names of datasets to process. Use 'all' to process all available datasets except TEST.",
    )

    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Dataset names to exclude from processing (e.g., TEST influence_seeking malicious_user_queries)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model identifier to use for inference (default: google/gemma-3-27b-it)",
    )

    parser.add_argument(
        "--inference",
        type=str,
        default="openrouter",
        help="Inference provider to use, openrouter or local (default: openrouter)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Maximum batch size for inference (default: 128)",
    )

    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Use flash_attention_2 if available (requires flash-attn package)",
    )

    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable quantization (uses more memory but may be faster)",
    )

    parser.add_argument(
        "--quantization_bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Number of bits for quantization: 8 (default, good balance) or 4 (extreme memory savings)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation (default: 0.7 for qwen3-32b with thinking disabled, 0.8 otherwise)",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter (default: 0.8 for qwen3-32b with thinking disabled)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter (default: 20 for qwen3-32b)",
    )

    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for models that support it (default: disabled for qwen3-32b)",
    )

    args = parser.parse_args()

    # Determine which datasets to process
    if args.names:
        if len(args.names) == 1 and args.names[0] == "all":
            # Get all available datasets except TEST and excluded ones
            from utils import STAGE_1_DATA_DIR

            all_datasets = []
            for item in STAGE_1_DATA_DIR.iterdir():
                if item.is_dir() and item.name != "TEST" and item.name not in args.exclude:
                    all_datasets.append(item.name)
            dataset_names = sorted(all_datasets)
        else:
            # Filter out excluded datasets
            dataset_names = [name for name in args.names if name not in args.exclude]
    else:
        # Single dataset case
        if args.name not in args.exclude:
            dataset_names = [args.name]
        else:
            dataset_names = []
            print(f"Warning: Dataset {args.name} is excluded, nothing to process")

    if not dataset_names:
        print("No datasets to process after applying exclusions")
        return

    # Set default parameters for qwen3-32b if not specified
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    disable_thinking = not args.enable_thinking

    if "qwen3-32b" in args.model.lower():
        if temperature is None:
            temperature = 0.7 if disable_thinking else 0.6
        if top_p is None:
            top_p = 0.8 if disable_thinking else 0.95
        if top_k is None:
            top_k = 20
    else:
        # Default for other models
        if temperature is None:
            temperature = 0.8

    print("Starting Stage 2 data generation with:")
    print(f"  Datasets: {', '.join(dataset_names)}")
    if args.exclude:
        print(f"  Excluded: {', '.join(args.exclude)}")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {temperature}")
    if top_p is not None:
        print(f"  Top-p: {top_p}")
    if top_k is not None:
        print(f"  Top-k: {top_k}")
    if args.inference == "openrouter":
        print(f"  Thinking mode: {'ENABLED' if not disable_thinking else 'DISABLED'}")
    print(f"  Flash attention: {args.flash_attention}")
    print(f"  Quantization: {not args.no_quantization}")
    if not args.no_quantization:
        print(f"  Quantization bits: {args.quantization_bits}")

    try:
        generator = Stage2DataGenerator(
            names=dataset_names,
            model=args.model,
            inference=args.inference,
            max_batch_size=args.batch_size,
            use_flash_attention=args.flash_attention,
            use_quantization=not args.no_quantization,
            quantization_bits=args.quantization_bits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            disable_thinking=disable_thinking,
        )

        generator.generate_inferences()

        print("\n✓ Stage 2 data generation completed successfully!")

    except Exception as e:
        print(f"\n✗ Stage 2 data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
