#!/usr/bin/env python3
"""
Stage 3 Data Generation Script for HonestPersona Project

This script generates [I][R][P] intervention data by making API calls to Claude.
It processes [A/B][T][S] data from stage 2 and adds intervention assessments.

The script:
- Uses Claude API with base_prompt.md for intervention generation
- Supports multiple intervention types and review focuses
- Uses a triple-loop structure: intervention_types -> files -> batched_samples
- Handles batch processing with configurable batch size
- Stores results in the same structure as stage 2 with added interventions field
"""

import argparse
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import traceback

import anthropic
from anthropic import Anthropic, RateLimitError
import requests
from os import environ

from utils import (
    DEBUG_DIR,
    FEEDBACK_DATA_DIR,
    GENERATION_PROMPTS_STAGE_3_DIR,
    PROGRESS_DIR,
    PROMPTS_DIR,
    STAGE_2_DATA_DIR,
    STAGE_3_DATA_DIR,
    STAGE_3_REVIEW_DIR,
    get_env_var,
)
from shared_tools import STAGE_3_TOOLS, STAGE_3_REVIEW_TOOLS, TOPIC_REVIEW_FOCUS_MAPPING


class ContextLengthError(Exception):
    """Raised when the context length exceeds the model's limit."""
    pass


class RateLimitExceededTimeout(Exception):
    """Raised when rate limit retries exceed the maximum timeout (10 minutes)."""
    pass


def make_anthropic_call_with_rate_limit_retry(
    client: Anthropic,
    api_params: Dict,
    max_tokens: int,
    progress_log_path: Path,
    context_description: str = "API call"
) -> anthropic.types.Message:
    """
    Make an Anthropic API call with exponential backoff for rate limit errors.

    Args:
        client: Anthropic client instance
        api_params: Parameters for the API call
        max_tokens: Maximum tokens for response generation
        progress_log_path: Path to progress log file
        context_description: Description of what this call is for (for logging)

    Returns:
        The API response message

    Raises:
        RateLimitExceededTimeout: If retry backoff exceeds 10 minutes
        Other exceptions: Propagated as-is
    """
    max_wait_time = 600  # 10 minutes in seconds
    total_wait_time = 0
    retry_count = 0

    while True:
        try:
            # Use streaming for large max_tokens to avoid timeout
            if max_tokens > 10000:
                # Use stream=True and collect the full response
                with client.messages.stream(**api_params) as stream:
                    response = stream.get_final_message()
            else:
                response = client.messages.create(**api_params)

            # Success - return the response
            return response

        except RateLimitError as e:
            retry_count += 1

            # Calculate exponential backoff with jitter
            # Base: 2^retry_count, max 60 seconds, plus random jitter
            base_wait = min(60, (2 ** retry_count))
            jitter = random.uniform(0, 1)
            wait_time = base_wait + jitter

            # Check if we would exceed max wait time
            if total_wait_time + wait_time > max_wait_time:
                error_msg = (
                    f"Rate limit retry timeout exceeded for {context_description}. "
                    f"Total wait time would be {total_wait_time + wait_time:.1f}s (max: {max_wait_time}s). "
                    f"Made {retry_count} retry attempts. Worker shutting down to free resources."
                )

                # Log to progress file
                with open(progress_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()} | RATE_LIMIT_TIMEOUT: {error_msg}\n")

                print(f"⚠️ {error_msg}")
                raise RateLimitExceededTimeout(error_msg)

            # Log the rate limit and retry
            log_msg = (
                f"Rate limit hit for {context_description}. "
                f"Retry {retry_count}, waiting {wait_time:.1f}s "
                f"(total wait so far: {total_wait_time:.1f}s/{max_wait_time}s). "
                f"Error: {str(e)}"
            )

            with open(progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | RATE_LIMIT_RETRY: {log_msg}\n")

            print(f"⚠️ {log_msg}")

            # Wait before retrying
            time.sleep(wait_time)
            total_wait_time += wait_time


def make_openrouter_call_with_retry(
    api_params: Dict,
    api_key: str,
    max_tokens: int,
    progress_log_path: Path,
    context_description: str = "API call"
) -> Dict:
    """
    Make an OpenRouter API call using raw HTTP requests with exponential backoff for rate limit errors.

    OpenRouter uses OpenAI-compatible endpoints, not Anthropic-compatible ones.

    Args:
        api_params: Parameters for the API call (Anthropic format)
        api_key: OpenRouter API key
        max_tokens: Maximum tokens for response generation
        progress_log_path: Path to progress log file
        context_description: Description of what this call is for (for logging)

    Returns:
        Dict with parsed response in Anthropic-like format

    Raises:
        RateLimitExceededTimeout: If retry backoff exceeds 10 minutes
        Other exceptions: Propagated as-is
    """
    max_wait_time = 600  # 10 minutes in seconds
    total_wait_time = 0
    retry_count = 0

    # Convert Anthropic format to OpenAI format for OpenRouter
    openai_params = convert_anthropic_to_openai_format(api_params)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anthropics/anthropic-cookbook",
        "X-Title": "HonestPersona Research"
    }

    while True:
        try:
            response = requests.post(url, headers=headers, json=openai_params, timeout=300)

            # Check for rate limit
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded", response=response, body=response.text)

            # Check for other errors
            if response.status_code != 200:
                error_text = response.text[:500]
                raise RuntimeError(
                    f"OpenRouter API error {response.status_code}: {error_text}"
                )

            # Parse response and convert back to Anthropic-like format
            openai_response = response.json()
            anthropic_like_response = convert_openai_to_anthropic_format(openai_response)

            return anthropic_like_response

        except RateLimitError as e:
            retry_count += 1

            # Calculate exponential backoff with jitter
            base_wait = min(60, (2 ** retry_count))
            jitter = random.uniform(0, 1)
            wait_time = base_wait + jitter

            # Check if we would exceed max wait time
            if total_wait_time + wait_time > max_wait_time:
                error_msg = (
                    f"Rate limit retry timeout exceeded for {context_description}. "
                    f"Total wait time would be {total_wait_time + wait_time:.1f}s (max: {max_wait_time}s). "
                    f"Made {retry_count} retry attempts."
                )

                with open(progress_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()} | RATE_LIMIT_TIMEOUT: {error_msg}\n")

                raise RateLimitExceededTimeout(error_msg)

            # Log the retry
            log_msg = (
                f"Rate limit hit for {context_description}. "
                f"Retry {retry_count}, waiting {wait_time:.1f}s "
                f"(total wait: {total_wait_time:.1f}s)"
            )
            with open(progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | RATE_LIMIT_RETRY: {log_msg}\n")

            print(f"⚠️ {log_msg}")

            # Wait before retrying
            time.sleep(wait_time)
            total_wait_time += wait_time


def convert_anthropic_to_openai_format(anthropic_params: Dict) -> Dict:
    """
    Convert Anthropic API parameters to OpenAI-compatible format for OpenRouter.

    Args:
        anthropic_params: Dict with Anthropic format (model, max_tokens, system, messages, tools, etc.)

    Returns:
        Dict with OpenAI format
    """
    # Extract system message from Anthropic format
    system_content = ""
    if "system" in anthropic_params:
        system_param = anthropic_params["system"]
        if isinstance(system_param, str):
            system_content = system_param
        elif isinstance(system_param, list):
            # Handle array format with cache_control
            system_content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system_param
            )

    # Convert messages (prepend system as first message if present)
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})

    # Add user/assistant messages
    for msg in anthropic_params.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        # Handle content array format
        if isinstance(content, list):
            text_content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
            messages.append({"role": role, "content": text_content})
        else:
            messages.append({"role": role, "content": content})

    # Build OpenAI-compatible parameters
    # Convert model name to OpenRouter format
    model_name = anthropic_params['model']
    # OpenRouter uses simpler names without date suffixes
    # claude-sonnet-4-5-20250929 -> anthropic/claude-sonnet-4.5
    # claude-sonnet-4-20250514 -> anthropic/claude-sonnet-4
    if 'claude-sonnet-4.5' in model_name or 'claude-sonnet-4-5' in model_name:
        openrouter_model = "anthropic/claude-sonnet-4.5"
    elif 'claude-sonnet-4' in model_name:
        openrouter_model = "anthropic/claude-sonnet-4"
    elif 'claude-opus-4' in model_name:
        openrouter_model = "anthropic/claude-opus-4"
    elif 'claude-haiku-4' in model_name:
        openrouter_model = "anthropic/claude-haiku-4"
    else:
        # Fallback: just add anthropic/ prefix
        openrouter_model = f"anthropic/{model_name}"

    openai_params = {
        "model": openrouter_model,
        "messages": messages,
        "max_tokens": anthropic_params.get("max_tokens", 4096),
    }

    # Add thinking mode if present
    if "thinking" in anthropic_params:
        thinking = anthropic_params["thinking"]
        if thinking.get("type") == "enabled":
            # OpenRouter uses extended_thinking parameter
            openai_params["reasoning_effort"] = "high"  # or use budget_tokens if supported

    # Handle tools - convert Anthropic tools to OpenAI format
    if "tools" in anthropic_params:
        openai_tools = []
        for tool in anthropic_params["tools"]:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        openai_params["tools"] = openai_tools

    # Handle tool_choice
    if "tool_choice" in anthropic_params:
        tool_choice = anthropic_params["tool_choice"]
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
            openai_params["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name")}
            }

    return openai_params


def convert_openai_to_anthropic_format(openai_response: Dict) -> Dict:
    """
    Convert OpenAI API response to Anthropic-like format.

    Args:
        openai_response: Dict with OpenAI format response

    Returns:
        Dict mimicking Anthropic response structure
    """
    choice = openai_response["choices"][0]
    message = choice["message"]

    # Build content array
    content = []

    # Add text content if present
    if message.get("content"):
        content.append({
            "type": "text",
            "text": message["content"]
        })

    # Add tool calls if present
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            function = tool_call["function"]
            content.append({
                "type": "tool_use",
                "name": function["name"],
                "input": json.loads(function["arguments"])
            })

    # Build Anthropic-like response
    return {
        "content": content,
        "role": "assistant",
        "model": openai_response.get("model", ""),
        "stop_reason": choice.get("finish_reason"),
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0)
        }
    }


class Stage3DataGenerator:
    """Handles stage 3 data generation for the HonestPersona project."""
    
    def __init__(self, name: str = "TEST", model: str = "claude-sonnet-4-20250514",
                 thinking_budget: int = 1000,
                 max_tokens: int = 40_000,  # Large token limit supported with streaming
                 num_tasks_per_batch: int = 4,
                 intervention_modification: str = "baseline",
                 review_model: Optional[str] = None,
                 dry_run: bool = False,
                 progress_log_path: Optional[Path] = None,
                 stage2_model: Optional[str] = None,
                 use_openrouter: bool = False):
        """
        Initialize the data generator.

        Args:
            name: Name of the dataset (used for file naming)
            model: Claude model to use for generation
            thinking_budget: Token budget for thinking mode, 0 to disable
            max_tokens: Maximum tokens for response generation
            num_tasks_per_batch: Number of sample groups to process per batch
            intervention_modification: Suffix for intervention storage key (default: "baseline")
            review_model: Model to use for quality review (default: None, uses same as generation model)
            dry_run: If True, skip API calls and just log what would be done (default: False)
            progress_log_path: Path to progress log file for rate limit logging (default: None)
            stage2_model: Filter to only process this stage 2 inference model (default: None, processes all models)
            use_openrouter: If True, use OpenRouter API instead of direct Anthropic API (default: False)
        """
        self.name = name
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.num_tasks_per_batch = num_tasks_per_batch
        self.intervention_modification = intervention_modification
        self.review_model = review_model or model  # Default to generation model if not specified
        self.dry_run = dry_run
        self.progress_log_path = progress_log_path or (PROGRESS_DIR / "stage_3_progress.log")
        self.stage2_model = stage2_model
        self.use_openrouter = use_openrouter

        # Initialize API client or key
        if not dry_run:
            if use_openrouter:
                # Store OpenRouter API key for raw HTTP requests
                # Try both OPENROUTER_API_KEY and OPEN_ROUTER_API_KEY
                try:
                    self.openrouter_api_key = get_env_var("OPENROUTER_API_KEY")
                except ValueError:
                    self.openrouter_api_key = get_env_var("OPEN_ROUTER_API_KEY")
                self.client = None  # Don't use Anthropic SDK with OpenRouter
            else:
                # Use direct Anthropic API
                self.client = Anthropic(api_key=get_env_var("ANTHROPIC_API_KEY"))
                self.openrouter_api_key = None
        else:
            self.client = None
            self.openrouter_api_key = None
        
        # Ensure directories exist
        STAGE_3_DATA_DIR.mkdir(parents=True, exist_ok=True)
        FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize prompt cache
        self._prompt_cache = {}
        
        # Discover intervention types from prompt files
        self.intervention_configs = self._discover_intervention_configs()
        
        print(f"Initialized Stage3DataGenerator:")
        print(f"  Dataset name: {name}")
        print(f"  Model: {model}")
        print(f"  API: {'OpenRouter' if use_openrouter else 'Anthropic Direct'}")
        print(f"  Max batch size: {num_tasks_per_batch}")
        print(f"  Found intervention configs: {len(self.intervention_configs)}")
        for intervention_type in self.intervention_configs:
            print(f"    - {intervention_type}")
    
    def _discover_intervention_configs(self) -> List[str]:
        """
        Discover intervention types from prompt files in the stage 3 directory.
        Now using hybrid approach: NAME__intervention.md files only.
        
        Returns:
            List of intervention types
        """
        interventions = []
        # Look for new hybrid format: NAME__intervention.md
        prompt_files = list(GENERATION_PROMPTS_STAGE_3_DIR.glob(f"{self.name}__*.md"))
        
        for prompt_file in prompt_files:
            # Parse filename: NAME__intervention.md
            parts = prompt_file.stem.split('__')
            if len(parts) == 2 and parts[0] == self.name:
                intervention_type = parts[1]
                interventions.append(intervention_type)
        
        if not interventions:
            raise FileNotFoundError(
                f"No intervention config files found in {GENERATION_PROMPTS_STAGE_3_DIR} "
                f"matching pattern {self.name}__*.md"
            )
        
        return sorted(interventions)
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from base_prompt.md."""
        cache_key = "system_prompt"
        if cache_key not in self._prompt_cache:
            base_prompt_path = PROMPTS_DIR / "base_prompt.md"
            self._prompt_cache[cache_key] = base_prompt_path.read_text(encoding='utf-8')
        return self._prompt_cache[cache_key]
    
    def _load_user_prompt(self, intervention_type: str, num_samples: int) -> str:
        """Load the user prompt for the specific intervention (hybrid approach)."""
        cache_key = f"user_prompt_{intervention_type}_{num_samples}"
        if cache_key not in self._prompt_cache:
            prompt_path = GENERATION_PROMPTS_STAGE_3_DIR / f"{self.name}__{intervention_type}.md"
            prompt_template = prompt_path.read_text(encoding='utf-8')
            
            # Add legal review focuses for this topic
            legal_focuses = TOPIC_REVIEW_FOCUS_MAPPING.get(self.name, [])
            if legal_focuses:
                legal_focuses_str = "\n".join([f"- {focus}" for focus in legal_focuses])
                prompt_template = prompt_template.replace("{legal_review_focuses}", legal_focuses_str)
            
            self._prompt_cache[cache_key] = prompt_template.format(num_samples=num_samples)
        return self._prompt_cache[cache_key]
    
    def _group_tasks_for_batching(self, tasks: List[Tuple[int, str, str]]) -> List[List[Tuple[int, str, str]]]:
        """
        Group tasks for batching. Groups by sample_index first, then adds all variants.
        
        Args:
            tasks: List of (sample_index, variant_type, model_name) tuples
            
        Returns:
            List of batches, where each batch is a list of tasks
        """
        # Group tasks by sample_index
        sample_groups = {}
        for sample_idx, variant_type, model_name in tasks:
            if sample_idx not in sample_groups:
                sample_groups[sample_idx] = []
            sample_groups[sample_idx].append((sample_idx, variant_type, model_name))
        
        # Create batches by selecting num_tasks_per_batch sample groups
        batches = []
        sample_indices = sorted(sample_groups.keys())
        
        for i in range(0, len(sample_indices), self.num_tasks_per_batch):
            batch_sample_indices = sample_indices[i:i + self.num_tasks_per_batch]
            batch_tasks = []
            
            # Add all variants for the selected sample indices
            for sample_idx in batch_sample_indices:
                batch_tasks.extend(sample_groups[sample_idx])
            
            batches.append(batch_tasks)
        
        return batches
    
    def _make_api_call(self, intervention_type: str, 
                       batch_samples: List[Dict], max_retries: int = 5) -> Optional[Dict]:
        """
        Make a single API call to Claude with retry logic and prompt caching.
        
        Args:
            intervention_type: Type of intervention to apply
            batch_samples: List of sample dictionaries with A/B, T, S data
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed response data or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                print(f"Making API call (attempt {attempt + 1}/{max_retries})...")
                
                # Load prompts with caching
                system_prompt = self._load_system_prompt()
                user_prompt = self._load_user_prompt(intervention_type, len(batch_samples))
                
                # Add sample data to user prompt
                sample_data_text = "\n\n## Samples to Analyze:\n\n"
                for i, sample in enumerate(batch_samples, 1):
                    sample_data_text += f"### Sample {i}:\n"
                    # Only show the system prompt that was actually used
                    if 'used_variant' in sample and sample['used_variant'] == 'B':
                        sample_data_text += f"**System Prompt (B variant used):** {sample['B']}\n"
                        sample_data_text += f"**Original unbiased prompt (A) for reference:** {sample['A']}\n"
                    else:
                        sample_data_text += f"**System Prompt (A variant used):** {sample['A']}\n"
                        if 'B' in sample and sample['B']:
                            sample_data_text += f"**Biased variant (B) exists but was not used for this response**\n"
                    sample_data_text += f"**User Message (T):** {sample['T']}\n"
                    sample_data_text += f"**Model Response (S):** {sample['S']}\n"
                    if 'thought_process' in sample:
                        sample_data_text += f"**Original Thought Process:** {sample['thought_process']}\n"
                    # Extract and highlight intended_review_focus from metadata
                    if 'metadata' in sample and 'intended_review_focus' in sample['metadata']:
                        sample_data_text += f"**Intended Review Focus:** {sample['metadata']['intended_review_focus']}\n"
                    if 'metadata' in sample:
                        sample_data_text += f"**Metadata:** {json.dumps(sample['metadata'])}\n"
                    sample_data_text += "\n"
                
                full_user_prompt = user_prompt + sample_data_text
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": full_user_prompt,
                                    # Do not cache this: The full user prompt is different each time because of sample_data_text.
                                    # "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        }
                    ],
                    "tools": STAGE_3_TOOLS,
                    "tool_choice": {"type": "tool", "name": "stage_3_generation"}
                }
                
                # Add thinking mode if budget > 0
                if self.thinking_budget > 0:
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    }
                
                # Log API call parameters for debugging
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_3__in.json", 'w') as f:
                    json.dump(api_params, f, indent=2, ensure_ascii=False, default=str)
                
                # Also save just the plain text user message for easier review
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_3__user_message.txt", 'w', encoding='utf-8') as f:
                    f.write(full_user_prompt)

                # Make API call with rate limit retry handling
                if self.use_openrouter:
                    # Use OpenRouter with raw HTTP requests
                    response = make_openrouter_call_with_retry(
                        api_params=api_params,
                        api_key=self.openrouter_api_key,
                        max_tokens=self.max_tokens,
                        progress_log_path=self.progress_log_path,
                        context_description=f"Stage 3 generation for {self.name}"
                    )
                else:
                    # Use Anthropic SDK
                    response = make_anthropic_call_with_rate_limit_retry(
                        client=self.client,
                        api_params=api_params,
                        max_tokens=self.max_tokens,
                        progress_log_path=self.progress_log_path,
                        context_description=f"Stage 3 generation for {self.name}"
                    )

                # Log API response for debugging
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_3__out.txt", 'w') as f:
                    f.write(f"Response type: {type(response)}\n")

                    # Handle both Anthropic SDK response and raw dict response
                    if isinstance(response, dict):
                        # OpenRouter response (dict format)
                        f.write(f"Response (dict): {json.dumps(response, indent=2, ensure_ascii=False, default=str)}\n")
                        response_content = response.get("content", [])
                    else:
                        # Anthropic SDK response
                        f.write(f"Response content length: {len(response.content) if response.content else 0}\n")
                        response_content = response.content

                    if response_content:
                        for i, content in enumerate(response_content):
                            f.write(f"\nContent {i}:\n")
                            # Handle both dict and object formats
                            if isinstance(content, dict):
                                f.write(f"  Type: {content.get('type')}\n")
                                if 'name' in content:
                                    f.write(f"  Name: {content['name']}\n")
                                if 'input' in content:
                                    f.write(f"  Input: {json.dumps(content['input'], indent=2, ensure_ascii=False, default=str)}\n")
                                if 'text' in content:
                                    f.write(f"  Text: {content['text']}\n")
                            else:
                                f.write(f"  Type: {content.type}\n")
                                if hasattr(content, 'name'):
                                    f.write(f"  Name: {content.name}\n")
                                if hasattr(content, 'input'):
                                    f.write(f"  Input: {json.dumps(content.input, indent=2, ensure_ascii=False, default=str)}\n")
                                if hasattr(content, 'text'):
                                    f.write(f"  Text: {content.text}\n")

                # Extract tool call content
                if isinstance(response, dict):
                    # OpenRouter dict response
                    response_content = response.get("content", [])
                else:
                    # Anthropic SDK response
                    response_content = response.content

                if response_content and len(response_content) > 0:
                    tool_call = response_content[0]

                    # Handle both dict and object formats
                    if isinstance(tool_call, dict):
                        if tool_call.get("type") == "tool_use" and tool_call.get("name") == "stage_3_generation":
                            return tool_call.get("input")
                    else:
                        if tool_call.type == "tool_use" and tool_call.name == "stage_3_generation":
                            return tool_call.input
                
                print(f"Unexpected response format on attempt {attempt + 1}")
                
            except RateLimitExceededTimeout:
                # Rate limit timeout - propagate immediately, don't retry
                raise
            except Exception as e:
                # Check if this is a context length error
                error_str = str(e)
                if "context limit" in error_str.lower() or "context window" in error_str.lower() or \
                   ("max_tokens" in error_str and "exceed" in error_str):
                    print(f"Context length error detected: {e}")
                    raise ContextLengthError(f"Input and max_tokens exceed context limit: {e}")

                print(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"All {max_retries} attempts failed")
                    raise
    
    def _gather_intervention_tasks(self, data: Dict, intervention_type: str) -> List[Tuple[int, str, str]]:
        """
        Gather all intervention tasks that need to be processed for a specific intervention config.

        Args:
            data: The loaded JSON data from stage 2
            intervention_type: Type of intervention

        Returns:
            List of (sample_index, variant_type, model_name) tuples
        """
        tasks = []
        intervention_key = f"{intervention_type}__{self.intervention_modification}"

        for sample_idx, sample in enumerate(data["data"]):
            if "inferences" not in sample:
                continue

            for model_name, model_data in sample["inferences"].items():
                # Filter by stage2_model if specified
                if self.stage2_model and model_name != self.stage2_model:
                    continue

                for variant_type in ["A", "B"]:
                    if variant_type not in model_data:
                        continue

                    # Initialize interventions field if not present
                    if "interventions" not in model_data[variant_type]:
                        model_data[variant_type]["interventions"] = {}

                    # Check if this intervention configuration already has results
                    if intervention_key not in model_data[variant_type]["interventions"]:
                        tasks.append((sample_idx, variant_type, model_name))

        return tasks

    def _check_file_needs_processing(self, input_path: Path, output_path: Path) -> bool:
        """
        Check if a file needs processing for any intervention type.

        Args:
            input_path: Path to input stage 2 file
            output_path: Path to output stage 3 file

        Returns:
            True if file needs processing for at least one intervention type, False otherwise
        """
        # If output doesn't exist, definitely needs processing
        if not output_path.exists():
            return True

        # Load output file to check intervention status
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return True

        # Check if any intervention type is missing
        for intervention_type in self.intervention_configs:
            tasks = self._gather_intervention_tasks(data, intervention_type)
            if tasks:  # If any tasks need processing, file needs work
                return True

        return False
    
    def _process_file(self, input_path: Path, output_path: Path, intervention_type: str) -> None:
        """
        Process a single input file and save the output with interventions.
        
        Args:
            input_path: Path to input stage 2 file
            output_path: Path to output stage 3 file
            intervention_type: Type of intervention to apply
        """
        # Load the input data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If output file exists, use it as the base
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loading existing output file: {output_path.name}")
        else:
            print(f"Creating new output file: {output_path.name}")
        
        # Gather all intervention tasks that need to be processed
        tasks = self._gather_intervention_tasks(data, intervention_type)

        if not tasks:
            print(f"No intervention tasks needed for {intervention_type}__{self.intervention_modification} in {input_path.name}")
            return

        print(f"Found {len(tasks)} intervention tasks to process")

        # DRY RUN: Log what would be done and return
        if self.dry_run:
            print(f"[DRY RUN] Would process {len(tasks)} tasks for {intervention_type}__{self.intervention_modification}")
            print(f"[DRY RUN] Tasks: {tasks[:5]}{'...' if len(tasks) > 5 else ''}")  # Show first 5 tasks
            return
        
        # Group tasks into batches
        batches = self._group_tasks_for_batching(tasks)
        
        # Process tasks in batches
        total_batches = len(batches)
        intervention_key = f"{intervention_type}__{self.intervention_modification}"
        
        for batch_idx, batch_tasks in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_tasks)} tasks)")
            
            # Prepare sample data for API call
            batch_samples = []
            for sample_idx, variant_type, model_name in batch_tasks:
                sample_data = data["data"][sample_idx].copy()
                # Add the model response from the specific variant
                sample_data["S"] = data["data"][sample_idx]["inferences"][model_name][variant_type]["S"]
                # Mark which variant was actually used to generate this response
                sample_data["used_variant"] = variant_type
                batch_samples.append(sample_data)
            
            # Make API call with retry logic for quality control
            max_review_attempts = 3
            batch_result = None
            review_passed = False
            num_reviews_made = 0

            # Flag to track if we hit a context length error
            context_length_error_info = None

            # Track context length errors during review
            review_context_length_errors = []

            # Track generation/validation errors
            generation_error_info = None

            for review_attempt in range(max_review_attempts):
                num_reviews_made = review_attempt + 1
                print(f"  Review attempt {num_reviews_made}/{max_review_attempts}")

                try:
                    batch_result = self._make_api_call(intervention_type, batch_samples)
                except ContextLengthError as e:
                    # Context length exceeded - save minimal error records and continue
                    print(f"  ⚠️  Context length error for batch {batch_idx + 1}: {e}")
                    print(f"  Saving error records for {len(batch_tasks)} tasks and skipping batch...")
                    context_length_error_info = str(e)
                    break  # Don't retry - this won't succeed with same inputs
                except Exception as e:
                    raise RuntimeError(f"Batch API call failed: {e}")

                # Validate that we got a response
                if batch_result is None:
                    print(f"  ✗ API call returned None on attempt {num_reviews_made}")
                    if review_attempt < max_review_attempts - 1:
                        print(f"  Retrying...")
                        continue
                    else:
                        # After all retries, mark as generation error
                        generation_error_info = f"API returned None after {max_review_attempts} attempts"
                        print(f"  ⚠️  Generation failed: {generation_error_info}")
                        print(f"  Saving error records for {len(batch_tasks)} tasks and skipping batch...")
                        break

                # Validate response structure - treat as retryable error
                if "batch_interventions" not in batch_result:
                    print(f"  ✗ Invalid response structure on attempt {num_reviews_made}: missing 'batch_interventions' key")
                    print(f"     Response keys: {list(batch_result.keys())}")
                    if review_attempt < max_review_attempts - 1:
                        print(f"  Retrying...")
                        continue
                    else:
                        # After all retries, mark as generation error
                        generation_error_info = f"Invalid response structure after {max_review_attempts} attempts: missing 'batch_interventions' key. Response keys: {list(batch_result.keys())}"
                        print(f"  ⚠️  Generation failed: {generation_error_info}")
                        print(f"  Saving error records for {len(batch_tasks)} tasks and skipping batch...")
                        break

                # Review the generated data
                review_passed, review_context_error = self.review_stage_3_data(
                    batch_result=batch_result,
                    batch_samples=batch_samples,
                    dataset_name=self.name,
                    intervention_type=intervention_type,
                    batch_num=batch_idx + 1,
                    review_model=self.review_model,
                    base_prompt=self._load_system_prompt()
                )

                # Track if this review attempt had a context length error
                review_context_length_errors.append(review_context_error)

                if review_passed:
                    print(f"  ✓ Review passed on attempt {num_reviews_made}")
                    break
                else:
                    if review_context_error:
                        print(f"  ✗ Review failed on attempt {num_reviews_made} (context length error)")
                        print(f"  Context length errors cannot be fixed by retrying - skipping remaining attempts")
                        break  # Don't retry context length errors - they won't succeed
                    else:
                        print(f"  ✗ Review failed on attempt {num_reviews_made}")
                    if review_attempt < max_review_attempts - 1:
                        print(f"  Retrying...")

            # Handle context length error by creating minimal error records
            if context_length_error_info is not None:
                # Create minimal error records for all tasks in this batch
                for task_idx, (sample_idx, variant_type, model_name) in enumerate(batch_tasks):
                    stage_3_analysis = f"item_{task_idx + 1}_of_{len(batch_tasks)}_in_batch_{batch_idx + 1}_of_{len(batches)}"

                    error_record = {
                        "error": "context_too_large",
                        "error_message": context_length_error_info,
                        "intervention_type": intervention_type,
                        "generation_time": time.time(),
                        "model_used": self.model,
                        "review_model_used": self.review_model,
                        "num_stage_3_reviews_made": 0,
                        "stage_3_reviews_passed": False,
                        "stage_3_analysis": stage_3_analysis,
                        # Minimal placeholders to maintain schema compatibility
                        "thought_process": "[ERROR: Context length exceeded - batch too large for model context window]",
                        "selected_review_focus": "N/A",
                        "I": "[ERROR: Context length exceeded]",
                        "R": "[ERROR: Context length exceeded]",
                        "P": "ERROR",
                        "confidence": 0,
                        "detection_difficulty_internal": "NA",
                        "detection_difficulty_external": "NA",
                        "faulty_data": True,
                        "batch_thinking": f"[ERROR: Context length exceeded for batch {batch_idx + 1}]"
                    }

                    data["data"][sample_idx]["inferences"][model_name][variant_type]["interventions"][intervention_key] = error_record

                print(f"Saved {len(batch_tasks)} error records for batch {batch_idx + 1}")
                continue  # Skip to next batch

            # Handle generation/validation error by creating minimal error records
            if generation_error_info is not None:
                # Create minimal error records for all tasks in this batch
                for task_idx, (sample_idx, variant_type, model_name) in enumerate(batch_tasks):
                    stage_3_analysis = f"item_{task_idx + 1}_of_{len(batch_tasks)}_in_batch_{batch_idx + 1}_of_{len(batches)}"

                    error_record = {
                        "error": "generation_failed",
                        "error_during_generation_or_review": generation_error_info,
                        "intervention_type": intervention_type,
                        "generation_time": time.time(),
                        "model_used": self.model,
                        "review_model_used": self.review_model,
                        "num_stage_3_reviews_made": num_reviews_made,
                        "stage_3_reviews_passed": False,
                        "stage_3_analysis": stage_3_analysis,
                        # Minimal placeholders to maintain schema compatibility
                        "thought_process": f"[ERROR: Generation failed - {generation_error_info}]",
                        "selected_review_focus": "N/A",
                        "I": "[ERROR: Generation failed]",
                        "R": "[ERROR: Generation failed]",
                        "P": "ERROR",
                        "confidence": 0,
                        "detection_difficulty_internal": "NA",
                        "detection_difficulty_external": "NA",
                        "faulty_data": True,
                        "batch_thinking": f"[ERROR: Generation failed for batch {batch_idx + 1}]"
                    }

                    data["data"][sample_idx]["inferences"][model_name][variant_type]["interventions"][intervention_key] = error_record

                print(f"Saved {len(batch_tasks)} error records for batch {batch_idx + 1} (generation failed)")
                continue  # Skip to next batch

            # Check if we should skip this batch due to any error
            # (this should have been handled above, but double-check here)
            if context_length_error_info is not None or generation_error_info is not None:
                # Error records already saved above
                continue

            # If all reviews failed, use the last result anyway
            if not review_passed:
                print(f"  ⚠ All {max_review_attempts} reviews failed, using last result")

            # Validate response structure
            # This should not fail at this point since we validated in the loop above,
            # but keep as a safeguard
            if batch_result is None or "batch_interventions" not in batch_result:
                error_msg = "batch_result is None" if batch_result is None else f"missing 'batch_interventions' key (keys: {list(batch_result.keys())})"
                raise ValueError(f"Invalid response structure for batch {batch_idx + 1}: {error_msg}. This should have been caught earlier in retry loop.")

            interventions = batch_result["batch_interventions"]
            if isinstance(interventions, list):
                pass
            elif isinstance(interventions, str):
                # This should not happen if the API works, but sometimes it seems to happen anyway. This requires debugging.
                print("  !!! STRING FORMAT !!!")
                print(interventions)
                print("  !!! STRING FORMAT !!!")
                interventions = interventions.strip()
                # Clean up the JSON string - remove XML tags and other artifacts
                if '</invoke>' in interventions:
                    interventions = interventions[:interventions.rfind('</invoke>')].strip()
                
                # Try to extract just the JSON array part
                try:
                    interventions = json.loads(interventions)
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    # Look for the main JSON structure
                    if interventions.startswith('[') and '}' in interventions:
                        # Try to find the end of the JSON array
                        bracket_count = 0
                        end_pos = 0
                        for i, char in enumerate(interventions):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_pos = i + 1
                                    break
                        
                        if end_pos > 0:
                            cleaned_json = interventions[:end_pos]
                            try:
                                interventions = json.loads(cleaned_json)
                                print(f"Successfully parsed cleaned JSON with {len(interventions)} interventions")
                            except json.JSONDecodeError:
                                raise ValueError(f"Could not parse JSON even after cleaning: {e}")
                        else:
                            raise ValueError(f"Could not find valid JSON array structure: {e}")
                    else:
                        raise ValueError(f"Unexpected JSON format: {e}")
            else:
                assert False
            if len(interventions) != len(batch_tasks):
                raise ValueError(f"Mismatch between tasks ({len(batch_tasks)}) and interventions ({len(interventions)}) for batch {batch_idx + 1}:\n{interventions}\n{batch_tasks}")
            
            # Extract batch_thinking for individual intervention records
            batch_thinking = batch_result.get("batch_thinking", "")
            
            # Check if all review attempts failed due to context length
            all_reviews_failed_due_to_context = (
                not review_passed and
                len(review_context_length_errors) > 0 and
                all(review_context_length_errors)
            )

            # Store results back in the data structure
            for task_idx, ((sample_idx, variant_type, model_name), intervention_data) in enumerate(zip(batch_tasks, interventions)):
                # Create stage_3_analysis field with position info
                stage_3_analysis = f"item_{task_idx + 1}_of_{len(batch_tasks)}_in_batch_{batch_idx + 1}_of_{len(batches)}"

                # Add metadata to intervention
                intervention_record = {
                    "thought_process": intervention_data["thought_process"],
                    "intervention_type": intervention_type,
                    "selected_review_focus": intervention_data["selected_review_focus"],
                    "confidence": intervention_data["confidence"],
                    "I": intervention_data["I"],
                    "R": intervention_data["R"],
                    "P": intervention_data["P"],
                    "detection_difficulty_internal": intervention_data["detection_difficulty_internal"],
                    "detection_difficulty_external": intervention_data["detection_difficulty_external"],
                    "faulty_data": intervention_data["faulty_data"],
                    "batch_thinking": batch_thinking,
                    "stage_3_analysis": stage_3_analysis,
                    "generation_time": time.time(),
                    "model_used": self.model,
                    "review_model_used": self.review_model,
                    "num_stage_3_reviews_made": num_reviews_made,
                    "stage_3_reviews_passed": review_passed,
                    "stage_3_reviews_all_failed_due_to_overlong_context": all_reviews_failed_due_to_context
                }

                data["data"][sample_idx]["inferences"][model_name][variant_type]["interventions"][intervention_key] = intervention_record
            
            # Update feedback log
            if "project_feedback" in batch_result and batch_result["project_feedback"].strip():
                self._update_feedback_log(batch_result, intervention_type, batch_idx + 1)
            
            print(f"Completed batch {batch_idx + 1}")
        
        # Save the updated data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to {output_path.name}")

    def _process_file_all_interventions(self, input_path: Path, output_path: Path) -> None:
        """
        Process a single file with ALL intervention types.

        Args:
            input_path: Path to input stage 2 file
            output_path: Path to output stage 3 file
        """
        print(f"\n{'='*60}")
        print(f"Processing file: {input_path.name}")
        print(f"Topic: {self.name}")
        print(f"Intervention types: {len(self.intervention_configs)}")
        print(f"{'='*60}")

        # Process each intervention type for this file
        for intervention_idx, intervention_type in enumerate(self.intervention_configs, 1):
            print(f"\n--- Intervention {intervention_idx}/{len(self.intervention_configs)}: {intervention_type} ---")
            try:
                self._process_file(input_path, output_path, intervention_type)
                print(f"✓ Completed {intervention_type} for {input_path.name}")
            except Exception as e:
                error_msg = f"Error processing {input_path.name} for intervention {intervention_type}: {e}"
                print(f"✗ {error_msg}")
                # Re-raise to mark the entire file as failed
                raise RuntimeError(error_msg)

        print(f"\n{'='*60}")
        print(f"✓ Completed all interventions for {input_path.name}")
        print(f"{'='*60}\n")

    def _update_feedback_log(self, batch_result: Dict, intervention_type: str, 
                           batch_num: int) -> None:
        """
        Update the cumulative feedback log.
        
        Args:
            batch_result: The data returned from the API call
            intervention_type: Type of intervention
            batch_num: Batch number for identification
        """
        feedback = batch_result.get("project_feedback", "")
        if not feedback.strip():
            return
        
        timestamp = datetime.now().isoformat()
        feedback_entry = {
            "timestamp": timestamp,
            "intervention_config": f"{intervention_type}__{self.intervention_modification}",
            "batch_number": batch_num,
            "project_feedback": feedback,
            "topic": self.name,
            "stage": 3
        }
        
        # Append to cumulative feedback file
        feedback_filename = f"cumulative_{self.name}.jsonl"
        feedback_path = FEEDBACK_DATA_DIR / feedback_filename
        
        with open(feedback_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        print(f"Updated feedback log: {feedback_path}")
    
    def generate_interventions(self, continue_on_error: bool = False) -> None:
        """
        Generate interventions for all stage 2 data files and intervention configurations.
        """
        input_dir = STAGE_2_DATA_DIR / self.name
        output_dir = STAGE_3_DATA_DIR / self.name
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Get all input files
        input_files = sorted(list(input_dir.glob("batch_*.json")))
        if not input_files:
            raise FileNotFoundError(f"No batch files found in {input_dir}")
        
        print(f"\n=== Stage 3 Data Generation ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(input_files)} files to process")
        print(f"Intervention configurations: {len(self.intervention_configs)}")
        print(f"Model: {self.model}")
        print(f"Batch size: {self.num_tasks_per_batch}")
        
        # Double loop: interventions -> files (review focus now determined by model)
        total_interventions = len(self.intervention_configs)
        for config_idx, intervention_type in enumerate(self.intervention_configs):
            print(f"\n--- Intervention {config_idx + 1}/{total_interventions}: {intervention_type} ---")
            
            for file_idx, input_file in enumerate(input_files):
                output_file = output_dir / input_file.name
                
                print(f"\nFile {file_idx + 1}/{len(input_files)}: {input_file.name}")
                
                try:
                    start_time = time.time()
                    self._process_file(input_file, output_file, intervention_type)
                    elapsed = time.time() - start_time
                    print(f"Completed {input_file.name} for {intervention_type} in {elapsed:.1f} seconds")
                
                except Exception as e:
                    print(f"ERROR processing {input_file.name} for {intervention_type}: {e}")
                    if not continue_on_error:
                        raise
        
        print(f"\n=== Stage 3 Generation Complete ===")
        print(f"Processed {len(input_files)} files with {total_interventions} intervention types")
        print(f"Output saved to: {output_dir}")
    
    def review_stage_3_data(
        self,
        batch_result: Dict,
        batch_samples: List[Dict],
        dataset_name: str,
        intervention_type: str,
        batch_num: int,
        review_model: str,
        base_prompt: str
    ) -> tuple[bool, bool]:
        """
        Review Stage 3 data quality by making an API call to evaluate the generated interventions.

        Args:
            batch_result: The parsed response from the API containing interventions
            batch_samples: The original samples that were sent to the API
            dataset_name: Name of the dataset (e.g., "TEST", "sycophancy")
            intervention_type: Type of intervention (e.g., "split_personality__version_3")
            batch_num: Batch number for logging
            review_model: Model to use for review
            anthropic_client: Anthropic client instance
            base_prompt: System prompt containing review guidelines

        Returns:
            Tuple of (review_passed, context_length_error)
            - review_passed: True if the data passes quality controls, False otherwise
            - context_length_error: True if review failed due to context length, False otherwise
        """
        # Prepare review prompt with full context
        review_user_message = """Please review the following Stage 3 generated data for quality issues.

    IMPORTANT: You are reviewing a BATCH of interventions. If ANY intervention in the batch has quality issues, you must reject the ENTIRE batch (return false). The batch will be regenerated as a whole.

    ## Original Input Samples:

    """
        review_user_message += json.dumps(batch_samples, indent=2, ensure_ascii=False)
        review_user_message += "\n\n## Generated Stage 3 Output:\n\n"
        review_user_message += json.dumps(batch_result, indent=2, ensure_ascii=False)
        review_user_message += "\n\nPlease use the stage_3_review tool to provide your evaluation. You have access to the stage_3_generation tool for reference (to see what instructions were given), but you MUST use the stage_3_review tool for your response. Using the wrong tool is a critical error."
        # Debugging code:
        # # print the keys in batch_samples and batch_result
        # print(f"FIELD LENGTHS, {len(batch_samples)} samples:")
        # for bs in batch_samples:
        #     print(f"Batch sample keys: {bs.keys()}")
        #     for k in bs.keys():
        #         print(f"{k}: {len(bs[k])}")
        # print(f"Batch result keys: {batch_result.keys()}")
        # for k in batch_result.keys():
        #     print(f"{k}: {len(batch_result[k])}")

        # Make API call to review model - provide both tools but require use of review tool
        api_params = {
            "model": review_model,
            "max_tokens": self.max_tokens,
            "system": base_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": review_user_message
                }
            ],
            "tools": STAGE_3_TOOLS + STAGE_3_REVIEW_TOOLS,  # Provide both for reference
            "tool_choice": {"type": "tool", "name": "stage_3_review"}  # But require review tool
        }

        # Log API call parameters for debugging
        with open(DEBUG_DIR / f"LAST_CALL__{dataset_name}__stage_3_review__in.json", 'w') as f:
            json.dump(api_params, f, indent=2, ensure_ascii=False, default=str)

        # Also save just the plain text user message for easier review
        with open(DEBUG_DIR / f"LAST_CALL__{dataset_name}__stage_3_review__user_message.txt", 'w', encoding='utf-8') as f:
            f.write(review_user_message)

        # Make API call with rate limit retry handling
        try:
            if self.use_openrouter:
                # Use OpenRouter with raw HTTP requests
                response = make_openrouter_call_with_retry(
                    api_params=api_params,
                    api_key=self.openrouter_api_key,
                    max_tokens=self.max_tokens,
                    progress_log_path=self.progress_log_path,
                    context_description=f"Stage 3 review for {dataset_name} batch {batch_num}"
                )
            else:
                # Use Anthropic SDK
                response = make_anthropic_call_with_rate_limit_retry(
                    client=self.client,
                    api_params=api_params,
                    max_tokens=self.max_tokens,
                    progress_log_path=self.progress_log_path,
                    context_description=f"Stage 3 review for {dataset_name} batch {batch_num}"
                )
        except RateLimitExceededTimeout:
            # Rate limit timeout - propagate immediately
            raise
        except Exception as e:
            # Check if this is a context length error
            error_str = str(e)
            if "context limit" in error_str.lower() or "context window" in error_str.lower() or \
               "prompt is too long" in error_str.lower() or \
               ("max_tokens" in error_str and "exceed" in error_str):
                print(f"  ⚠️  Context length error during review: {e}")
                print(f"  Review prompt too large - automatically rejecting batch (cannot verify quality)")
                # Return (False, True) - failed review due to context length
                return (False, True)
            else:
                # Re-raise other errors
                raise

        # Log API response for debugging
        with open(DEBUG_DIR / f"LAST_CALL__{dataset_name}__stage_3_review__out.txt", 'w') as f:
            f.write(f"Response type: {type(response)}\n")

            # Handle both dict and Anthropic SDK response
            if isinstance(response, dict):
                f.write(f"Response (dict): {json.dumps(response, indent=2, ensure_ascii=False, default=str)}\n")
                response_content = response.get("content", [])
            else:
                f.write(f"Response content length: {len(response.content) if response.content else 0}\n")
                response_content = response.content

            if response_content:
                for i, content in enumerate(response_content):
                    f.write(f"\nContent {i}:\n")
                    if isinstance(content, dict):
                        f.write(f"  Type: {content.get('type')}\n")
                        if 'name' in content:
                            f.write(f"  Name: {content['name']}\n")
                        if 'input' in content:
                            f.write(f"  Input: {json.dumps(content['input'], indent=2, ensure_ascii=False, default=str)}\n")
                        if 'text' in content:
                            f.write(f"  Text: {content['text']}\n")
                    else:
                        f.write(f"  Type: {content.type}\n")
                        if hasattr(content, 'name'):
                            f.write(f"  Name: {content.name}\n")
                        if hasattr(content, 'input'):
                            f.write(f"  Input: {json.dumps(content.input, indent=2, ensure_ascii=False, default=str)}\n")
                        if hasattr(content, 'text'):
                            f.write(f"  Text: {content.text}\n")

        # Extract review verdict - handle both dict and object formats
        if isinstance(response, dict):
            response_content = response.get("content", [])
        else:
            response_content = response.content

        if not response_content or len(response_content) == 0:
            raise ValueError("Review API returned empty response")

        tool_call = response_content[0]

        # Extract data handling both dict and object formats
        if isinstance(tool_call, dict):
            tool_type = tool_call.get("type")
            tool_name = tool_call.get("name")
            review_data = tool_call.get("input")
        else:
            tool_type = tool_call.type
            tool_name = tool_call.name
            review_data = tool_call.input

        if tool_type != "tool_use":
            raise ValueError(f"Review API did not return tool use, got type: {tool_type}")

        if tool_name != "stage_3_review":
            raise ValueError(f"Review API used wrong tool: {tool_name} (expected stage_3_review)")
        verdict = review_data.get("final_verdict")

        if verdict is None:
            raise ValueError("Review API did not return final_verdict field")

        if not isinstance(verdict, bool):
            raise ValueError(f"Review API returned non-boolean verdict: {verdict} (type: {type(verdict)})")

        # Log the review
        self._log_review(
            dataset_name=dataset_name,
            intervention_type=intervention_type,
            batch_num=batch_num,
            review_data=review_data,
            verdict=verdict,
            review_model=review_model
        )

        # Return (verdict, context_length_error=False)
        return (verdict, False)


    def _log_review(
        self,
        dataset_name: str,
        intervention_type: str,
        batch_num: int,
        review_data: Dict,
        verdict: bool,
        review_model: str
    ) -> None:
        """
        Log review results to JSONL file.

        Args:
            dataset_name: Name of the dataset
            intervention_type: Type of intervention
            batch_num: Batch number
            review_data: The review tool output
            verdict: True if passed, False if failed
            review_model: Model used for review
        """
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "intervention_type": intervention_type,
            "batch_number": batch_num,
            "review_model": review_model,
            "verdict": verdict,
            "thought_process": review_data.get("thought_process", ""),
            "written_evaluation": review_data.get("written_evaluation", "")
        }

        # Append to log file
        log_file = STAGE_3_REVIEW_DIR / f"{dataset_name}_reviews.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        print(f"Logged review to {log_file}")


def process_single_file(args_tuple):
    """
    Process a single file with all intervention types. This function is designed to be called by multiprocessing workers.

    Args:
        args_tuple: Tuple of (topic_name, input_path, output_path, model, thinking_budget, max_tokens, batch_size, intervention_modification, review_model, progress_log_path, dry_run, stage2_model, use_openrouter, file_index, total_files)

    Returns:
        Tuple of (topic_name, filename, success, error_message)
    """
    topic_name, input_path, output_path, model, thinking_budget, max_tokens, batch_size, intervention_modification, review_model, progress_log_path, dry_run, stage2_model, use_openrouter, file_index, total_files = args_tuple

    filename = input_path.name

    try:
        # Create generator for this file
        generator = Stage3DataGenerator(
            name=topic_name,
            model=model,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            num_tasks_per_batch=batch_size,
            intervention_modification=intervention_modification,
            review_model=review_model,
            dry_run=dry_run,
            progress_log_path=progress_log_path,
            stage2_model=stage2_model,
            use_openrouter=use_openrouter
        )

        # Process all intervention types for this file
        generator._process_file_all_interventions(input_path, output_path)

        # Log success
        dry_run_prefix = "[DRY RUN] " if dry_run else ""
        success_msg = f"{dry_run_prefix}SUCCESS: {topic_name}/{filename} ({file_index}/{total_files})"
        with open(progress_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {success_msg}\n")

        print(f"✓ [Worker] {success_msg}")
        return (topic_name, filename, True, None)

    except Exception as e:
        # Get full error with traceback for debugging
        import traceback
        error_details = traceback.format_exc()

        # For the log, include first line of error and type
        error_type = type(e).__name__
        error_msg = str(e).split('\n')[0] if str(e) else error_type

        fail_msg = f"FAIL: {topic_name}/{filename} - {error_type}: {error_msg}"

        # Log failure with more details
        with open(progress_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {fail_msg}\n")
            # Also write full traceback for debugging
            f.write(f"  Full error:\n")
            for line in error_details.split('\n'):
                if line.strip():
                    f.write(f"    {line}\n")
            f.write("\n")

        print(f"✗ [Worker] {fail_msg}")
        return (topic_name, filename, False, f"{error_type}: {error_msg}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate stage 3 interventions for HonestPersona project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/dataset_generation_stage_3.py --name TEST
  python src/scripts/dataset_generation_stage_3.py --names sycophancy confidence_assessment
  python src/scripts/dataset_generation_stage_3.py --names all --batch_size 4
        """
    )

    parser.add_argument(
        "--name",
        type=str,
        default="TEST",
        help="Name of a single dataset (default: TEST). Use --names for multiple datasets."
    )

    parser.add_argument(
        "--names",
        type=str,
        nargs='+',
        help="Names of datasets to process. Use 'all' to process all available datasets except TEST."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use for intervention generation"
    )

    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Token budget for thinking mode, 0 to disable (default: 0)"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=40_000,
        help="Maximum tokens for response generation (default: 40000)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of sample groups to process per batch (default: 4)"
    )

    parser.add_argument(
        "--intervention_modification",
        type=str,
        default="baseline",
        help="Suffix for intervention storage key (default: baseline)"
    )

    parser.add_argument(
        "--review_model",
        type=str,
        default=None,
        help="Model to use for Stage 3 quality review (default: None, uses same model as generation)"
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue on error (default: False). If set, failed files will be logged but processing continues."
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes to use (default: 4). Set to 1 to disable multiprocessing."
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode: scan files and log what would be done, but skip API calls and don't modify files."
    )

    parser.add_argument(
        "--stage2_model",
        type=str,
        default=None,
        help="Filter to only process this stage 2 inference model (e.g., 'qwen/qwen3-32b'). If not specified, processes all models."
    )

    parser.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Use OpenRouter API instead of direct Anthropic API. Useful to bypass Anthropic spend limits. Requires OPENROUTER_API_KEY in environment."
    )

    args = parser.parse_args()

    # Determine which datasets to process
    if args.names:
        if len(args.names) == 1 and args.names[0] == "all":
            # Get all available datasets except TEST
            all_datasets = []
            for item in STAGE_2_DATA_DIR.iterdir():
                if item.is_dir() and item.name != "TEST":
                    all_datasets.append(item.name)
            dataset_names = sorted(all_datasets)
        else:
            dataset_names = args.names
    else:
        dataset_names = [args.name]

    dry_run_label = " [DRY RUN MODE]" if args.dry_run else ""
    print(f"Starting Stage 3 data generation{dry_run_label} with:")
    print(f"  Datasets: {', '.join(dataset_names)}")
    print(f"  Model: {args.model}")
    print(f"  Thinking budget: {args.thinking_budget}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Intervention modification: {args.intervention_modification}")
    print(f"  Number of processes: {args.num_processes}")
    if args.stage2_model:
        print(f"  Stage 2 model filter: {args.stage2_model}")
    if args.dry_run:
        print(f"  DRY RUN: Will scan and log only, no API calls or file modifications")

    # Setup progress log
    progress_log_path = PROGRESS_DIR / "stage_3_progress.log"
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    progress_log_path.touch(exist_ok=True)

    # Initialize/overwrite progress log
    with open(progress_log_path, 'w', encoding='utf-8') as f:
        dry_run_label = " [DRY RUN]" if args.dry_run else ""
        f.write(f"Stage 3 Progress Log{dry_run_label} - Started at {datetime.now().isoformat()}\n")
        f.write(f"Datasets: {', '.join(dataset_names)}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Intervention modification: {args.intervention_modification}\n")
        if args.stage2_model:
            f.write(f"Stage 2 model filter: {args.stage2_model}\n")
        if args.dry_run:
            f.write(f"DRY RUN MODE: No API calls or file modifications\n")
        f.write(f"{'='*80}\n\n")

    print(f"\nProgress log: {progress_log_path}")

    try:
        # Step 1: Scan all files to build work queue
        print(f"\n{'='*80}")
        print("Scanning files to determine work queue...")
        print(f"{'='*80}")

        work_queue = []

        for dataset_name in dataset_names:
            # Create a temporary generator just to discover intervention configs
            temp_generator = Stage3DataGenerator(
                name=dataset_name,
                model=args.model,
                thinking_budget=args.thinking_budget,
                max_tokens=args.max_tokens,
                num_tasks_per_batch=args.batch_size,
                intervention_modification=args.intervention_modification,
                review_model=args.review_model,
                dry_run=args.dry_run,
                progress_log_path=progress_log_path,
                use_openrouter=args.use_openrouter
            )

            input_dir = STAGE_2_DATA_DIR / dataset_name
            output_dir = STAGE_3_DATA_DIR / dataset_name

            if not input_dir.exists():
                print(f"⚠ Warning: Input directory not found for {dataset_name}: {input_dir}")
                continue

            # Get all input files
            input_files = sorted(list(input_dir.glob("batch_*.json")))
            if not input_files:
                print(f"⚠ Warning: No batch files found in {input_dir}")
                continue

            # Check each file
            for input_file in input_files:
                output_file = output_dir / input_file.name
                if temp_generator._check_file_needs_processing(input_file, output_file):
                    work_queue.append((dataset_name, input_file, output_file))

        total_files = len(work_queue)

        if total_files == 0:
            print("\n✓ No files need processing! All interventions are up to date.")
            return

        print(f"\nFound {total_files} files needing processing:")
        # Group by dataset for display
        by_dataset = {}
        for dataset_name, input_file, output_file in work_queue:
            if dataset_name not in by_dataset:
                by_dataset[dataset_name] = []
            by_dataset[dataset_name].append(input_file.name)

        for dataset_name in sorted(by_dataset.keys()):
            print(f"  {dataset_name}: {len(by_dataset[dataset_name])} files")

        # Step 2: Process files in parallel
        print(f"\n{'='*80}")
        print(f"Processing {total_files} files with {args.num_processes} processes")
        print(f"{'='*80}\n")

        # Prepare worker arguments
        worker_args = [
            (
                dataset_name,
                input_file,
                output_file,
                args.model,
                args.thinking_budget,
                args.max_tokens,
                args.batch_size,
                args.intervention_modification,
                args.review_model,
                progress_log_path,
                args.dry_run,
                args.stage2_model,
                args.use_openrouter,
                i,
                total_files
            )
            for i, (dataset_name, input_file, output_file) in enumerate(work_queue, 1)
        ]

        if args.num_processes == 1:
            # Sequential processing
            print("Running in sequential mode (no multiprocessing)\n")
            results = [process_single_file(arg) for arg in worker_args]
        else:
            # Parallel processing
            print(f"Running in parallel mode with {args.num_processes} processes\n")
            with Pool(processes=args.num_processes) as pool:
                results = pool.map(process_single_file, worker_args)

        # Step 3: Report results
        successful = []
        failed = []

        for topic_name, filename, success, error_msg in results:
            if success:
                successful.append((topic_name, filename))
            else:
                failed.append((topic_name, filename, error_msg))

        print(f"\n{'='*80}")
        print("Final Results:")
        print(f"  Total files: {total_files}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print(f"\nFailed files:")
            for topic_name, filename, error_msg in failed:
                print(f"  - {topic_name}/{filename}")
                # print(f"    Error: {error_msg[:100]}...")  # First 100 chars of error
                print(f"    Error:\n\n--------------------------------\n\n{error_msg}\n\n--------------------------------\n\n")

        print(f"\nProgress log: {progress_log_path}")
        print(f"{'='*80}")

        # Write final summary to progress log
        with open(progress_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Completed at {datetime.now().isoformat()}\n")
            f.write(f"Total files: {total_files}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n")

        if failed and not args.continue_on_error:
            raise Exception(f"Failed to process {len(failed)} file(s)")

        print(f"\n✓ Stage 3 data generation completed!")

    except Exception as e:
        print(f"\n✗ Stage 3 data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()