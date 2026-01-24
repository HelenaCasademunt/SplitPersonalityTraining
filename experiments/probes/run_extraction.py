"""Run activation extraction for multiple models."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils import load_cfg, load_env
from scripts.probes.extract_activations import extract_activations

MODELS = {
    "qwen": "Qwen/Qwen3-14B",
}

load_env()
cfg = load_cfg("cfg_extraction.json")

for name, model_path in MODELS.items():
    print(f"\n{'#'*80}")
    print(f"# Extracting activations for: {name} ({model_path})")
    print(f"{'#'*80}")
    extract_activations(cfg, model_name=model_path, experiment_name=name)
