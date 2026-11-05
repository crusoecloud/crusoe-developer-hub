"""Shared Crusoe model factory for the Agno recipes.

Uses Agno's OpenAILike class against Crusoe's OpenAI-compatible endpoint. When the native
provider from agno-agi/agno#8920 ships, replace the body with
``from agno.models.crusoe import Crusoe; return Crusoe(id=model_id)``.
"""

import os

from agno.models.openai.like import OpenAILike

BASE_URL = "https://api.inference.crusoecloud.com/v1"


def crusoe(model_id: str = "zai/GLM-5.3", **kwargs) -> OpenAILike:
    """Return an Agno model bound to a Crusoe-hosted model id."""
    return OpenAILike(
        id=model_id,
        api_key=os.environ["CRUSOE_API_KEY"],
        base_url=BASE_URL,
        **kwargs,
    )
