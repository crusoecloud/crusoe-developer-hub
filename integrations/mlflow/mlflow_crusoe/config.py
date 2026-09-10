"""Configuration constants and helpers for the Crusoe deployment plugin."""

import os
from typing import Any, Dict, Optional


DEFAULT_API_BASE = "https://api.inference.crusoecloud.com/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1

# Known Crusoe models and their context windows
CRUSOE_MODELS: Dict[str, int] = {
    "meta-llama/Llama-3.3-70B-Instruct": 131072,
    "deepseek-ai/DeepSeek-V3-0324": 163840,
    "deepseek-ai/DeepSeek-R1-0528": 163840,
    "google/gemma-3-12b-it": 131072,
    "openai/gpt-oss-120b": 131072,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 262144
}


def get_api_key(config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the Crusoe API key from config or environment.

    Priority:
        1. config["api_key"]
        2. CRUSOE_API_KEY environment variable

    Raises:
        ValueError: If no API key is found.
    """
    if config and config.get("api_key"):
        return config["api_key"]

    api_key = os.environ.get("CRUSOE_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "Crusoe API key is required. Provide it via config={'api_key': '...'} "
        "or set the CRUSOE_API_KEY environment variable."
    )
