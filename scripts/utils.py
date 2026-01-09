import copy
import json
import os
import re
import time
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# Load environment variables from .env file
load_dotenv()

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
STAGE_1_DATA_DIR = DATA_DIR / "stage_1"
STAGE_2_DATA_DIR = DATA_DIR / "stage_2"
STAGE_3_DATA_DIR = DATA_DIR / "stage_3"
STAGE_3_TAGGED_DATA_DIR = DATA_DIR / "stage_3_tagged"
STAGE_3_REVIEW_DIR = DATA_DIR / "stage_3_reviews"
FEEDBACK_DATA_DIR = DATA_DIR / "feedback"
DEBUG_DIR = DATA_DIR / "debug"
PROGRESS_DIR = DATA_DIR / "progress"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Prompt directories
PROMPTS_DIR = PROJECT_ROOT / "prompts"
GENERATION_PROMPTS_STAGE_1_DIR = PROMPTS_DIR / "stage_1"
GENERATION_PROMPTS_STAGE_3_DIR = PROMPTS_DIR / "stage_3"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    STAGE_1_DATA_DIR,
    STAGE_2_DATA_DIR,
    STAGE_3_DATA_DIR,
    STAGE_3_TAGGED_DATA_DIR,
    STAGE_3_REVIEW_DIR,
    DEBUG_DIR,
    FEEDBACK_DATA_DIR,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    PROGRESS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def get_env_var(name: str, required: bool = True) -> str:
    """
    Get an environment variable.

    Args:
        name (str): Name of the environment variable
        required (bool): Whether the variable is required

    Returns:
        str: Value of the environment variable

    Raises:
        ValueError: If the variable is required but not set
    """
    value = os.getenv(name)
    if required and value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


# Set environment variables if they exist, but don't require them
anthropic_key = get_env_var("ANTHROPIC_API_KEY", required=False)
if anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

openai_key = get_env_var("OPENAI_API_KEY", required=False)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

open_router_key = get_env_var("OPENROUTER_API_KEY", required=False)
if open_router_key:
    os.environ["OPENROUTER_API_KEY"] = open_router_key

huggingface_user = get_env_var("HUGGINGFACE_USER", required=False)
if huggingface_user:
    os.environ["HUGGINGFACE_USER"] = huggingface_user

huggingface_token = get_env_var("HUGGINGFACE_API_TOKEN", required=False)
if huggingface_token:
    os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_token


def setup_server_environment():
    """
    Set up the server environment for local model inference.

    This function:
    1. Logs into HuggingFace using credentials from environment variables
    2. Sets the HuggingFace cache directory

    Raises:
        RuntimeError: If HuggingFace credentials are missing
    """
    # Check for required HuggingFace credentials
    hf_user = get_env_var("HUGGINGFACE_USER", required=True)
    hf_token = get_env_var("HUGGINGFACE_API_TOKEN", required=True)

    # Set HuggingFace cache directory (use local cache by default)
    hf_cache_dir = Path.home() / ".cache" / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")

    # Log into HuggingFace
    try:
        from huggingface_hub import login

        login(token=hf_token)
        print(f"Successfully logged into HuggingFace as user: {hf_user}")
        print(f"HuggingFace cache directory: {hf_cache_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to log into HuggingFace: {e}")

    return hf_cache_dir


def get_data_file_path(filename: str, data_type: str = "processed") -> Path:
    """
    Get the full path to a data file.

    Args:
        filename (str): Name of the data file
        data_type (str): Type of data directory ('processed')

    Returns:
        Path: Full path to the data file
    """
    data_dirs = {
        "processed": PROCESSED_DATA_DIR,
    }

    if data_type not in data_dirs:
        raise ValueError(f"data_type must be one of {list(data_dirs.keys())}")

    return data_dirs[data_type] / filename


def get_output_file_path(filename: str, output_type: str = "reports") -> Path:
    """
    Get the full path to an output file.

    Args:
        filename (str): Name of the output file
        output_type (str): Type of output directory ('logs' or 'reports')

    Returns:
        Path: Full path to the output file
    """
    output_dirs = {"logs": LOGS_DIR, "reports": REPORTS_DIR}

    if output_type not in output_dirs:
        raise ValueError(f"output_type must be one of {list(output_dirs.keys())}")

    return output_dirs[output_type] / filename


def cache_results(name: str, caching: bool = True):
    """
    Decorator that caches function results in PROCESSED_DATA_DIR as JSONL files.
    The cache filename is created by combining the decorator name and function arguments.

    Args:
        name (str): Base name for the cache file

    Returns:
        Function that loads cached results if they exist, otherwise runs the function and caches the results.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            assert len(args) == 0, "Please use kwargs only"
            # Create cache filename from name and arguments
            file_name = f"{name}"
            if args:
                file_name += "__" + "__".join(str(arg) for arg in args)
            if kwargs:
                file_name += "__" + "__".join(f"{k}_{v}" for k, v in kwargs.items())
            cache_path = PROCESSED_DATA_DIR / f"{file_name}.jsonl"
            results = None
            if not cache_path.exists() or not caching:
                if not caching:
                    print(f"Skipping {cache_path} because caching is disabled")
                print(f"Generating {cache_path}")
                results = func(*args, **kwargs)
                with open(cache_path, "w") as f:
                    for item in results:
                        f.write(json.dumps(item) + "\n")
                first_item = results[0]
                if not isinstance(first_item, dict):
                    raise ValueError("Cached results must contain only dictionaries")
                reference_keys = set(first_item.keys())
                reference_types = {k: type(v) for k, v in first_item.items()}
                for item in results[1:]:
                    if not isinstance(item, dict):
                        raise ValueError(
                            "Cached results must contain only dictionaries"
                        )
                    if set(item.keys()) != reference_keys:
                        raise ValueError(
                            f"All dictionaries must have the same keys:\n{reference_keys}\n{set(item.keys())}"
                        )
                    for key, value in item.items():
                        if type(value) is not reference_types[key]:
                            raise ValueError(
                                f"Type mismatch for key '{key}': expected {reference_types[key]}, got {type(value)}"
                            )
                print(f"Cached {cache_path}")
            with open(cache_path) as f:
                cached_results = [json.loads(line) for line in f]
                if results is not None:
                    assert cached_results == results, (
                        "Cached results don't match generated results."
                    )
                print(f"Content of {cache_path}:")
                print(f"    {len(cached_results)} items")
                for k, v in cached_results[0].items():
                    print(f"    {k}: {type(v)}")
                return cached_results

        return wrapper

    return decorator


@dataclass()
class InteractionWithModel:
    model: str
    initial_system_message: str
    print_thoughts_of_thinking_models: bool = False
    key: Optional[str] = None
    client: Optional[OpenAI | Anthropic] = None
    messages: Optional[List[Dict[str, str]]] = None

    def to_json_dict(self):
        return {
            "model": self.model,
            "initial_system_message": self.get_system_message_content(),
            "messages": self.get_non_system_messages(),
        }

    def __post_init__(self):
        assert self.model.count("/") == 1, self.model
        org_name, model_name = tuple(self.model.split("/"))
        self.org_name = org_name
        self.model_name = model_name
        if self.org_name in ["openai"]:
            self.client = OpenAI(
                api_key=self.key if self.key else os.environ["OPENAI_API_KEY"]
            )
        elif self.org_name in ["deepseek"]:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.key if self.key else os.environ["OPENROUTER_API_KEY"],
            )
        elif self.org_name in ["anthropic"]:
            self.client = Anthropic(
                api_key=self.key if self.key else os.environ["ANTHROPIC_API_KEY"]
            )
        else:
            raise ValueError(self.model)
        self._initialize_messages_list()

    def _initialize_messages_list(self):
        if self.org_name in ["openai"]:
            self.messages = []
            if self.initial_system_message is not None:
                self.messages.append(
                    {"role": "system", "content": self.initial_system_message}
                )
        elif self.org_name in ["deepseek"]:
            self.messages = []
            if self.initial_system_message is not None:
                self.messages.append(
                    {"role": "system", "content": self.initial_system_message}
                )
        elif self.org_name in ["anthropic"]:
            self.messages = []
        else:
            raise ValueError(self.model)

    def add_user_message(self, content):
        assert not self.messages or self.messages[-1]["role"] != "user", (
            "Only one user message at a time."
        )
        assert isinstance(content, str), content
        user_message = {"role": "user", "content": content}
        self.messages.append(user_message)

    def add_assistant_message(self, content):
        assert self.messages[-1]["role"] in ["user", "system"], (
            "Last message must be user or system message."
        )
        assert isinstance(content, str), content
        assistant_message = {"role": "assistant", "content": content}
        self.messages.append(assistant_message)

    def add_system_message(self, content):
        assert isinstance(content, str), content
        system_message = {"role": "system", "content": content}
        self.messages.append(system_message)

    def generate(self, n_times=None):
        """
        n_times means that we generate n_times responses and return them all, but only save the first one.
        """
        all_variants = []
        for _ in range(n_times or 1):
            if self.org_name in ["openai"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                )
                assert len(response.choices) == 1
                response_message = response.choices[0].message
                response_role = response_message.role
                response_content = response_message.content
            elif self.org_name in ["deepseek"]:
                num_consecutive_errors = 0
                while True:
                    try:
                        response = self.client.chat.completions.create(
                            model=f"deepseek/{self.model_name}",
                            messages=self.messages,
                            temperature=0.6,
                        )
                        assert response.choices is not None, response
                        break
                    except (JSONDecodeError, AssertionError):
                        num_consecutive_errors += 1
                        print("Error in DeepSeek response")
                        time.sleep(10)
                        if num_consecutive_errors >= 10:
                            raise Exception("Too many consecutive errors")
                assert len(response.choices) == 1
                response_message = response.choices[0].message
                response_role = response_message.role
                response_content = response_message.content
            elif self.org_name in ["anthropic"]:
                if self.model_name == "claude-3-7-sonnet-20250219__thinking":
                    with self.client.messages.stream(
                        max_tokens=20000,
                        thinking={"type": "enabled", "budget_tokens": 16000},
                        messages=self.messages,
                        model="claude-3-7-sonnet-20250219",
                    ) as stream:
                        for text in stream.text_stream:
                            pass
                        response = stream.get_final_message()
                    assert len(response.content) == 2, response.content
                    response_role = response.role
                    if self.print_thoughts_of_thinking_models:
                        print(f"----\n{response.content[0].thinking}\n----")
                    response_content = response.content[1].text
                else:
                    token_count = self.client.messages.count_tokens(
                        model=self.model_name,
                        system=self.initial_system_message,
                        messages=self.messages,
                    )
                    if token_count.input_tokens > 20_000:
                        print("Note: token_count", token_count.input_tokens)
                    num_consecutive_errors = 0
                    while True:
                        try:
                            response = self.client.messages.create(
                                model=self.model_name,
                                max_tokens=4096,
                                system=self.initial_system_message,
                                messages=self.messages,
                            )
                            break
                        except RateLimitError:
                            num_consecutive_errors += 1
                            print("Rate limit error in Anthropic")
                            time.sleep(10)
                            if num_consecutive_errors >= 10:
                                raise Exception("Too many consecutive errors")
                    assert len(response.content) == 1, response.content
                    response_role = response.role
                    response_content = response.content[0].text
            else:
                raise ValueError(self.model)
            assert isinstance(response_content, str), response_content
            assert response_role in ["assistant"], response_role
            all_variants.append(response_content)
        canonical_response = all_variants[0]
        if not self.messages or self.messages[-1]["role"] != "assistant":
            self.messages.append(
                {
                    "role": "assistant",
                    "content": canonical_response,
                }
            )
        else:
            existing_prefix = self.messages[-1]["content"]
            canonical_response = existing_prefix + canonical_response
            self.messages[-1]["content"] = canonical_response
            all_variants = [existing_prefix + a for a in all_variants]
        if n_times is None:
            return canonical_response
        assert canonical_response in all_variants, (canonical_response, all_variants)
        return all_variants

    def get_messages(self):
        return copy.deepcopy(self.messages)

    def get_system_message_content(self):
        return self.initial_system_message

    def get_non_system_messages(self):
        if self.org_name in ["openai", "deepseek"]:
            if self.messages and self.messages[0]["role"] == "system":
                messages = self.messages[1:]
            else:
                messages = self.messages
        elif self.org_name in ["anthropic"]:
            messages = self.messages
        else:
            raise ValueError(self.model)
        assert all(m["role"] in ["user", "assistant"] for m in messages), messages
        return copy.deepcopy(messages)


def extract_markup_tags_from_message(
    message,
    required_tags,
    optional_tags,
    tags_with_multiple_instances,
    convert_to_integer,
    ignore_extra_text_outside_of_tags=False,
):
    assert len(required_tags) + len(optional_tags) + len(
        tags_with_multiple_instances
    ) == len(
        set(required_tags) | set(optional_tags) | set(tags_with_multiple_instances)
    ), "The lists must not overlap"
    res = {}
    for block_name in required_tags + optional_tags + tags_with_multiple_instances:
        pattern = f"<{block_name}>(.*?)</{block_name}>"
        matches = re.findall(pattern, message, re.DOTALL)
        assert (block_name not in required_tags) or matches, (block_name, message)
        if block_name in required_tags:
            assert len(matches) == 1, matches
            val = matches[0]
        elif block_name in optional_tags:
            assert len(matches) <= 1, matches
            val = matches[0] if matches else None
        else:
            assert block_name in tags_with_multiple_instances, block_name
            val = matches
        if block_name in convert_to_integer:
            if isinstance(val, list):
                val = [int(a) for a in val]
            else:
                val = int(val)
        res[block_name] = val
    tmp = message
    for block_name, val in res.items():
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            a = f"<{block_name}>{v}</{block_name}>"
            tmp = tmp.replace(a, "")
    tmp = tmp.strip()
    if tmp != "" and not ignore_extra_text_outside_of_tags:
        raise ValueError(
            f"After removing all tags, the message contained meaningful characters:\n{message.strip()}\n-----\n{tmp}"
        )
    for key, value in res.items():
        if isinstance(value, list):
            res[key] = [v.strip() if isinstance(v, str) else v for v in value]
        elif isinstance(value, str):
            res[key] = value.strip()
    return res
