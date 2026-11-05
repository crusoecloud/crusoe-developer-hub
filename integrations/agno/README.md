# Agno with Crusoe

Run [Agno](https://docs.agno.com/) agents on [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference). Agno's `OpenAILike` model class speaks to any OpenAI-compatible endpoint, so Crusoe works today with three arguments. A native `agno.models.crusoe.Crusoe` class is in review upstream ([agno-agi/agno#8920](https://github.com/agno-agi/agno/pull/8920)); once it merges, `Crusoe(id="zai/GLM-5.3")` replaces the `OpenAILike` block below and the rest of the code stays the same.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Security > Inference API Key**)

## Setup

```bash
cd integrations/agno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```python
import os

from agno.agent import Agent
from agno.models.openai.like import OpenAILike

crusoe = OpenAILike(
    id="zai/GLM-5.3",
    api_key=os.environ["CRUSOE_API_KEY"],
    base_url="https://api.inference.crusoecloud.com/v1",
)

agent = Agent(model=crusoe, markdown=True)
agent.print_response("Write a two sentence horror story about a cold aisle.", stream=True)
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`basic.py`](basic.py) | Sync, streaming, and async runs from one agent |
| [`tool_use.py`](tool_use.py) | A custom tool function the agent calls to look up a GPU spec |
| [`structured_output.py`](structured_output.py) | A Pydantic `output_schema`, enforced by Crusoe's guided decoding |

```bash
python basic.py
python tool_use.py
python structured_output.py
```

Each script makes live requests that incur inference charges.

## Notes

- `OpenAILike` accepts the same generation parameters as Agno's OpenAI model: `temperature`, `max_tokens`, `top_p`, and `reasoning_effort` for `openai/gpt-oss-120b`.
- Pick any model from `https://api.inference.crusoecloud.com/v1/models`. GLM, Kimi, and Nemotron Super handle multi-step tool use well; `google/gemma-4-31b-it` is a good fit for structured output.
- Add Agno's bundled tools (`agno.tools.*`) to the `tools` list alongside your own functions.

[Back to integrations](../README.md)
