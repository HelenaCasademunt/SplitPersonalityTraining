#!/usr/bin/env python3
"""
Stage 1 Data Generation Script for HonestPersona Project

This script generates [A/B][T] training data by making API calls to Claude.
It handles retries, saves data in batches, and maintains a feedback log.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from anthropic import Anthropic
import openai

from utils import (
    DEBUG_DIR,
    FEEDBACK_DATA_DIR,
    GENERATION_PROMPTS_STAGE_1_DIR,
    STAGE_1_DATA_DIR,
    PROMPTS_DIR,
    get_env_var,
)
from shared_tools import AVAILABLE_TOOLS


class Stage1DataGenerator:
    """Handles stage 1 data generation for the HonestPersona project."""
    
    def __init__(self, name: str = "TEST", model: str = "claude-opus-4-1", thinking_budget: int = 1000, max_tokens: int = 8000, trim_excess_samples: bool = False):
        """
        Initialize the data generator.
        
        Args:
            name: Name of the dataset (used for file naming)
            model: Model to use for generation (Claude or GPT models)
            thinking_budget: Token budget for thinking mode, 0 to disable (default: 1000, only for Claude models)
            max_tokens: Maximum tokens for response generation (default: 8000)
            trim_excess_samples: If True, trim excess samples instead of raising error (default: False)
        """
        self.name = name
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.trim_excess_samples = trim_excess_samples
        
        # Determine if this is a Claude or OpenAI model
        self.is_openai_model = model.startswith("gpt-") or model.startswith("o1-")
        
        # Initialize appropriate client
        if self.is_openai_model:
            self.client = openai.OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
            # OpenAI models don't support thinking mode
            if self.thinking_budget > 0:
                print(f"Warning: Thinking mode not supported for OpenAI model {model}, disabling")
                self.thinking_budget = 0
        else:
            self.client = Anthropic(api_key=get_env_var("ANTHROPIC_API_KEY"))
        
        # Ensure directories exist
        STAGE_1_DATA_DIR.mkdir(parents=True, exist_ok=True)
        FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize prompt cache
        self._prompt_cache = {}
        
    def _load_system_prompt(self) -> str:
        """Load the system prompt from base_prompt.md."""
        cache_key = "system_prompt"
        if cache_key not in self._prompt_cache:
            base_prompt_path = PROMPTS_DIR / "base_prompt.md"
            self._prompt_cache[cache_key] = base_prompt_path.read_text(encoding='utf-8')
        return self._prompt_cache[cache_key]
    
    def _load_user_prompt(self, num_samples: int = 5) -> str:
        """Load the user prompt for the specific dataset with num_samples formatting."""
        cache_key = f"user_prompt_{self.name}_{num_samples}"
        if cache_key not in self._prompt_cache:
            prompt_path = GENERATION_PROMPTS_STAGE_1_DIR / f"{self.name}.md"
            prompt_template = prompt_path.read_text(encoding='utf-8')
            self._prompt_cache[cache_key] = prompt_template.format(num_samples=num_samples)
        return self._prompt_cache[cache_key]
    
    def _make_api_call(self, num_samples: int = 5, max_retries: int = 5) -> Optional[Dict]:
        """
        Make a single API call with retry logic.
        
        Args:
            num_samples: Number of samples to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed response data or None if all retries failed
        """
        if self.is_openai_model:
            return self._make_openai_api_call(num_samples, max_retries)
        else:
            return self._make_claude_api_call(num_samples, max_retries)
    
    def _make_openai_api_call(self, num_samples: int = 5, max_retries: int = 5) -> Optional[Dict]:
        """
        Make a single API call to OpenAI with retry logic.
        
        Args:
            num_samples: Number of samples to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed response data or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                print(f"Making OpenAI API call (attempt {attempt + 1}/{max_retries})...")
                
                # Load prompts with caching
                system_prompt = self._load_system_prompt()
                user_prompt = self._load_user_prompt(num_samples)
                
                # Convert tools to OpenAI format
                tools = []
                for tool in AVAILABLE_TOOLS:
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["input_schema"]
                        }
                    }
                    tools.append(openai_tool)
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "stage_1_generation"}}
                }
                
                # Use max_completion_tokens for newer models like gpt-5, max_tokens for others
                if self.model.startswith("gpt-5") or self.model.startswith("o1-"):
                    api_params["max_completion_tokens"] = self.max_tokens
                else:
                    api_params["max_tokens"] = self.max_tokens
                
                # Log API call parameters for debugging
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_1__in.json", 'w') as f:
                    json.dump(api_params, f, indent=2, ensure_ascii=False, default=str)

                # Make API call
                response = self.client.chat.completions.create(**api_params)

                # Log API response for debugging
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_1__out.txt", 'w') as f:
                    f.write(f"Response type: {type(response)}\n")
                    f.write(f"Response: {response}\n")
                
                # Extract tool call content
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if choice.message.tool_calls:
                        tool_call = choice.message.tool_calls[0]
                        if tool_call.function.name == "stage_1_generation":
                            return json.loads(tool_call.function.arguments)
                
                print(f"Unexpected response format on attempt {attempt + 1}")
                
            except Exception as e:
                print(f"OpenAI API call failed on attempt {attempt + 1}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"All {max_retries} attempts failed")
                    return None
        
        return None
    
    def _make_claude_api_call(self, num_samples: int = 5, max_retries: int = 5) -> Optional[Dict]:
        """
        Make a single API call to Claude with retry logic and prompt caching.
        
        Args:
            num_samples: Number of samples to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed response data or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                print(f"Making API call (attempt {attempt + 1}/{max_retries})...")
                
                # Load prompts with caching
                system_prompt = self._load_system_prompt()
                user_prompt = self._load_user_prompt(num_samples)
                
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
                                    "text": user_prompt,
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        }
                    ],
                    "tools": AVAILABLE_TOOLS,
                    "tool_choice": {"type": "tool", "name": "stage_1_generation"}
                }
                
                # Add thinking mode if budget > 0
                if self.thinking_budget > 0:
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    }
                
                # Log API call parameters for debugging
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_1__in.json", 'w') as f:
                    json.dump(api_params, f, indent=2, ensure_ascii=False, default=str)

                # Use streaming for long requests to avoid 10-minute timeout
                with self.client.messages.stream(**api_params) as stream:
                    # Consume the stream
                    for _ in stream:
                        pass
                    response = stream.get_final_message()

                # Log API response for debugging
                with open(DEBUG_DIR / f"LAST_CALL__{self.name}__stage_1__out.txt", 'w') as f:
                    f.write(f"Response type: {type(response)}\n")
                    f.write(f"Response content length: {len(response.content) if response.content else 0}\n")
                    if response.content:
                        for i, content in enumerate(response.content):
                            f.write(f"\nContent {i}:\n")
                            f.write(f"  Type: {content.type}\n")
                            if hasattr(content, 'name'):
                                f.write(f"  Name: {content.name}\n")
                            if hasattr(content, 'input'):
                                f.write(f"  Input: {json.dumps(content.input, indent=2, ensure_ascii=False, default=str)}\n")
                            if hasattr(content, 'text'):
                                f.write(f"  Text: {content.text}\n")
                
                # Extract tool call content
                if response.content and len(response.content) > 0:
                    tool_call = response.content[0]
                    if tool_call.type == "tool_use" and tool_call.name == "stage_1_generation":
                        return tool_call.input
                
                print(f"Unexpected response format on attempt {attempt + 1}")
                
            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"All {max_retries} attempts failed")
                    return None
        
        return None
    
    def _save_batch_data(self, batch_data: Dict, batch_id: str) -> None:
        """
        Save batch data to a JSON file.
        
        Args:
            batch_data: The data returned from the API call
            batch_id: Unique identifier for this batch
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare the batch file data
        batch_file_data = {
            "batch_id": batch_id,
            "generation_timestamp": timestamp,
            "model_used": self.model,
            "stage": 1,
            "topic": self.name,
            "data": batch_data.get("batch_data", []),
            "project_feedback": batch_data.get("project_feedback", "")
        }
        
        # Save batch file
        batch_filename = f"batch_{batch_id}.json"
        batch_path = STAGE_1_DATA_DIR / self.name / batch_filename
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(batch_file_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved batch data to {batch_path}")
    
    def _update_feedback_log(self, batch_data: Dict, batch_id: str) -> None:
        """
        Update the cumulative feedback log.
        
        Args:
            batch_data: The data returned from the API call
            batch_id: Unique identifier for this batch
        """
        feedback = batch_data.get("project_feedback", "")
        if not feedback.strip():
            return
        
        timestamp = datetime.now().isoformat()
        feedback_entry = {
            "timestamp": timestamp,
            "batch_id": batch_id,
            "project_feedback": feedback,
            "topic": self.name,
            "stage": 1
        }
        
        # Append to cumulative feedback file
        feedback_filename = f"cumulative_{self.name}.jsonl"
        feedback_path = FEEDBACK_DATA_DIR / feedback_filename
        
        with open(feedback_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        print(f"Updated feedback log: {feedback_path}")
    
    def _validate_batch_data(self, batch_data: Dict, batch_num: int, expected_count: int) -> None:
        """
        Strictly validate batch data structure and content.
        Raises detailed errors if validation fails.
        
        Args:
            batch_data: The batch data to validate
            batch_num: Batch number for error messages
            expected_count: Expected number of samples
        """
        # Check top-level structure
        if not isinstance(batch_data, dict):
            error_msg = f"Batch {batch_num}: batch_data must be a dictionary, got {type(batch_data)}"
            print(f"VALIDATION ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        if "batch_data" not in batch_data:
            error_msg = f"Batch {batch_num}: missing 'batch_data' key"
            print(f"VALIDATION ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        samples = batch_data["batch_data"]
        if not isinstance(samples, list):
            error_msg = f"Batch {batch_num}: 'batch_data' must be a list, got {type(samples)}"
            print(f"VALIDATION ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check sample count
        actual_count = len(samples)
        if actual_count != expected_count:
            if actual_count > expected_count and self.trim_excess_samples:
                print(f"⚠️  Batch {batch_num}: got {actual_count} samples, expected {expected_count}. Trimming to {expected_count} samples.")
                # Trim excess samples
                samples = samples[:expected_count]
                actual_count = len(samples)
                # Update the batch_data to reflect the trimmed samples
                batch_data["batch_data"] = samples
            else:
                error_msg = f"Batch {batch_num}: expected {expected_count} samples, got {actual_count}"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
        
        # Validate each sample
        required_fields = ["thought_process", "A", "T", "metadata"]
        metadata_required_fields = ["alignment_issue_type", "difficulty", "intended_review_focus", "notes"]
        valid_difficulties = ["easy", "medium", "hard", "FPT"]
        
        for i, sample in enumerate(samples):
            sample_num = i + 1
            
            if not isinstance(sample, dict):
                error_msg = f"Batch {batch_num}, Sample {sample_num}: must be a dictionary, got {type(sample)}"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Check required fields
            for field in required_fields:
                if field not in sample:
                    error_msg = f"Batch {batch_num}, Sample {sample_num}: missing required field '{field}'"
                    print(f"VALIDATION ERROR: {error_msg}")
                    raise ValueError(error_msg)
                
                if not isinstance(sample[field], str) and field != "metadata":
                    error_msg = f"Batch {batch_num}, Sample {sample_num}: field '{field}' must be a string, got {type(sample[field])}"
                    print(f"VALIDATION ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Validate metadata structure
            metadata = sample["metadata"]
            if not isinstance(metadata, dict):
                error_msg = f"Batch {batch_num}, Sample {sample_num}: metadata must be a dictionary, got {type(metadata)}"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            for meta_field in metadata_required_fields:
                if meta_field not in metadata:
                    error_msg = f"Batch {batch_num}, Sample {sample_num}: metadata missing required field '{meta_field}'"
                    print(f"VALIDATION ERROR: {error_msg}")
                    raise ValueError(error_msg)
                
                if not isinstance(metadata[meta_field], str):
                    error_msg = f"Batch {batch_num}, Sample {sample_num}: metadata['{meta_field}'] must be a string, got {type(metadata[meta_field])}"
                    print(f"VALIDATION ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Validate difficulty enum
            if metadata["difficulty"] not in valid_difficulties:
                error_msg = f"Batch {batch_num}, Sample {sample_num}: invalid difficulty '{metadata['difficulty']}', must be one of {valid_difficulties}"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Validate non-empty strings
            if not sample["thought_process"].strip():
                error_msg = f"Batch {batch_num}, Sample {sample_num}: thought_process cannot be empty"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            if not sample["A"].strip():
                error_msg = f"Batch {batch_num}, Sample {sample_num}: system prompt A cannot be empty"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            if not sample["T"].strip():
                error_msg = f"Batch {batch_num}, Sample {sample_num}: user task T cannot be empty"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
        
        print(f"✓ Batch {batch_num} validation passed: {actual_count} samples with correct structure")
    
    def generate_batches(self, num_batches: int, num_items_per_batch: int = 5) -> None:
        """
        Generate multiple batches of data.
        
        Args:
            num_batches: Number of batches to generate
            num_items_per_batch: Number of items per batch (for logging)
        """
        print(f"Starting generation of {num_batches} batches with {num_items_per_batch} items each")
        print(f"Using model: {self.model}")
        print(f"Dataset name: {self.name}")
        
        successful_batches = 0
        
        for batch_num in range(num_batches):
            print(f"\n--- Generating batch {batch_num + 1}/{num_batches} ---")
            
            # Generate unique batch ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"{self.name}_{timestamp}_batch_{batch_num + 1:03d}"
            
            # Make API call
            batch_data = self._make_api_call(num_items_per_batch)
            
            if batch_data is None:
                raise RuntimeError(f"Failed to generate batch {batch_num + 1}")
            
            # Validate and fix response structure
            if "batch_data" not in batch_data:
                raise ValueError(f"Invalid response structure for batch {batch_num + 1}: missing 'batch_data' key")
            
            # Handle string responses (same fix as Stage 3)
            batch_samples = batch_data["batch_data"]
            if isinstance(batch_samples, str):
                # This should not happen if the API works, but sometimes it does
                print("  !!! STRING FORMAT !!!")
                print(f"Received string response with length: {len(batch_samples)}")
                print("  !!! STRING FORMAT !!!")
                batch_samples = batch_samples.strip()
                
                # Clean up the JSON string - remove XML tags and other artifacts
                if '</invoke>' in batch_samples:
                    batch_samples = batch_samples[:batch_samples.rfind('</invoke>')].strip()
                
                # Try to extract just the JSON array part
                try:
                    batch_samples = json.loads(batch_samples)
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    # Look for the main JSON structure
                    if batch_samples.startswith('[') and '}' in batch_samples:
                        # Try to find the end of the JSON array
                        bracket_count = 0
                        end_pos = 0
                        for i, char in enumerate(batch_samples):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_pos = i + 1
                                    break
                        
                        if end_pos > 0:
                            cleaned_json = batch_samples[:end_pos]
                            try:
                                batch_samples = json.loads(cleaned_json)
                                print(f"Successfully parsed cleaned JSON with {len(batch_samples)} samples")
                            except json.JSONDecodeError:
                                error_msg = f"Could not parse JSON even after cleaning for batch {batch_num + 1}: {e}"
                                print(f"ERROR: {error_msg}")
                                raise ValueError(error_msg)
                        else:
                            error_msg = f"Could not find valid JSON array structure for batch {batch_num + 1}: {e}"
                            print(f"ERROR: {error_msg}")
                            raise ValueError(error_msg)
                    else:
                        error_msg = f"Unexpected JSON format for batch {batch_num + 1}: {e}"
                        print(f"ERROR: {error_msg}")
                        raise ValueError(error_msg)
                
                # Update the batch_data with parsed samples
                batch_data["batch_data"] = batch_samples
            
            # Strict validation of batch structure
            self._validate_batch_data(batch_data, batch_num + 1, num_items_per_batch)
            
            # Save data
            self._save_batch_data(batch_data, batch_id)
            self._update_feedback_log(batch_data, batch_id)
            
            successful_batches += 1
            print(f"Successfully generated batch {batch_num + 1}")
            
            # Small delay between batches
            if batch_num < num_batches - 1:
                time.sleep(2)
        
        print(f"\n--- Generation Complete ---")
        print(f"Successfully generated {successful_batches}/{num_batches} batches")
        print(f"Data saved to: {STAGE_1_DATA_DIR / self.name}")
        print(f"Feedback saved to: {FEEDBACK_DATA_DIR}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate stage 1 training data for HonestPersona")
    parser.add_argument(
        "--name", 
        type=str, 
        default="TEST",
        help="Name of the dataset (default: TEST)"
    )
    parser.add_argument(
        "--num_batches", 
        type=int, 
        default=1,
        help="Number of batches to generate (default: 1)"
    )
    parser.add_argument(
        "--num_items_per_batch",
        type=int,
        default=30,
        help="Number of items per batch (default: 30)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="claude-opus-4-1",
        help="Claude model to use (default: claude-opus-4-1)"
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        # Note: API error: "Thinking may not be enabled when tool_choice forces tool use."
        # I'm keeping this argument in but disabled for now.
        # Let's check the results before we enable this again, because it would require refactoring:
        # We would need to ask for one message with thinking enabled and no tools, followed by a message with thinking disabled and tools.
        default=0,
        help="Token budget for thinking mode, 0 to disable"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8000,
        help="Maximum tokens for response generation (default: 8000, max 32000 for claude-opus-4-1)"
    )
    parser.add_argument(
        "--trim_excess_samples",
        action="store_true",
        help="Trim excess samples instead of raising error when model produces too many samples (useful for OpenAI models)"
    )
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = Stage1DataGenerator(
        name=args.name,
        model=args.model,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        trim_excess_samples=args.trim_excess_samples
    )
    generator.generate_batches(args.num_batches, args.num_items_per_batch)


if __name__ == "__main__":
    main() 