# Pydantic AI with Crusoe

Build type-safe agents on [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference) with [Pydantic AI](https://ai.pydantic.dev/). Crusoe is a native Pydantic AI provider ([pydantic/pydantic-ai#6457](https://github.com/pydantic/pydantic-ai/pull/6457)): name a model as `crusoe:<model id>` and the SDK picks the right model profile from the vendor prefix, so tool calling, streaming, and structured output work across the catalog.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)

## Setup

From the developer hub repository root:

```bash
cd integrations/pydantic-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```python
from pydantic_ai import Agent

agent = Agent("crusoe:zai/GLM-5.3", instructions="Answer in one sentence.")
print(agent.run_sync("Why do data centers care about power usage effectiveness?").output)
```

Or construct the model explicitly to pass a custom provider, key, or HTTP client:

```python
from pydantic_ai import Agent
from pydantic_ai.models.crusoe import CrusoeModel
from pydantic_ai.providers.crusoe import CrusoeProvider

model = CrusoeModel("zai/GLM-5.3", provider=CrusoeProvider(api_key="your-api-key"))
agent = Agent(model)
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`basic.py`](basic.py) | Sync and streamed runs, plus token usage from the response |
| [`tool_use.py`](tool_use.py) | A tool the model calls to look up live inference rates, with typed dependencies |
| [`structured_output.py`](structured_output.py) | A Pydantic model as the agent output, validated by guided decoding on Crusoe |

Run any of them with the virtual environment active:

```bash
python basic.py
python tool_use.py
python structured_output.py
```

Each script makes live requests that incur inference charges.

## Notes

- **Model names carry the lab prefix.** `zai/GLM-5.3`, `deepseek-ai/DeepSeek-V4-Pro`, `meta-llama/Llama-3.3-70B-Instruct`, and `openai/gpt-oss-120b` all work; keep the prefix, since it selects the model profile. The current catalog is at `https://api.inference.crusoecloud.com/v1/models`.
- **Structured output.** Crusoe serves every model with guided decoding, so `NativeOutput` works even for model families that lack native JSON schema support through their own vendor.
- **Reasoning models.** `openai/gpt-oss-120b` accepts `reasoning_effort` through `OpenAIChatModelSettings`; GLM and Kimi models return reasoning interleaved with tool calls.
- **Tracing.** Pair this with the [Managed MLflow cookbook](../../examples/cookbooks/managed-mlflow/) or [Logfire](https://ai.pydantic.dev/logfire/) to see every tool call.

[Back to integrations](../README.md)
