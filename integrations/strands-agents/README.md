# Strands Agents with Crusoe

Run [Strands Agents](https://strandsagents.com/) on [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference). Crusoe is a documented community model provider for Strands ([strands-agents/harness-sdk#3225](https://github.com/strands-agents/harness-sdk/pull/3225)): the SDK's `OpenAIModel` talks to Crusoe's OpenAI-compatible endpoint, so tools, streaming, and multi-agent patterns need no adapter code.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)

## Setup

```bash
cd integrations/strands-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```python
import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={
        "api_key": os.environ["CRUSOE_API_KEY"],
        "base_url": "https://api.inference.crusoecloud.com/v1/",
    },
    model_id="zai/GLM-5.3",
    params={"max_tokens": 2048, "temperature": 0.1},
)


@tool
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


agent = Agent(model=model, tools=[word_count])
agent("How many words are in 'serverless inference on renewable power'?")
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`agent.py`](agent.py) | An agent that chains two custom `@tool` functions to size rack power and backup generators |

```bash
python agent.py
```

The script makes live requests that incur inference charges.

## Notes

- `client_args` are passed straight to the OpenAI Python client, so timeouts, retries, and proxies configure the same way.
- Tools are plain Python functions under `@tool`; Strands builds the schema from the signature and docstring. The `strands-agents-tools` package adds ready-made tools such as `http_request` and `file_read`.
- `params` accepts any OpenAI-compatible chat completion parameter. For `openai/gpt-oss-120b`, add `"reasoning_effort": "medium"`.
- Pick any model from `https://api.inference.crusoecloud.com/v1/models`. `zai/GLM-5.3`, `moonshotai/Kimi-K2.6`, and `nvidia/Nemotron-3-Super-120B-A12B` are strong tool callers.

[Back to integrations](../README.md)
