# Google ADK with Crusoe

Use [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference) models in agents built with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/).

Crusoe exposes an OpenAI-compatible API and is a supported [LiteLLM](https://docs.litellm.ai/) provider, so ADK agents can use Crusoe models through ADK's built-in `LiteLlm` wrapper with no extra adapter code.

## Setup

Use Python 3.10 or later with an activated virtual environment, as described in the [ADK Python quickstart](https://google.github.io/adk-docs/get-started/python/).

1. Create an account at the [Crusoe Console](https://console.crusoe.ai/) and generate an Inference API key under **Security > Inference API Key**.
2. From the developer hub repository root, enter this integration and install its dependencies:

```bash
cd integrations/google-adk
python -m pip install -r requirements.txt
```

3. Set your API key:

```bash
export CRUSOE_API_KEY="your-api-key"
```

Alternatively, copy [`crusoe/.env.example`](crusoe/.env.example) to `crusoe/.env` and replace the placeholder locally. ADK loads the agent's `.env` file when it starts. Keep that file untracked.

> **Note:** If your installed LiteLLM version predates the endpoint update ([BerriAI/litellm#33121](https://github.com/BerriAI/litellm/pull/33121)), also set
> `export CRUSOE_API_BASE="https://api.inference.crusoecloud.com/v1"`.

## Quick start

Point ADK's `LiteLlm` model wrapper at a Crusoe model using the `crusoe/` prefix:

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name="crusoe_agent",
    model=LiteLlm(model="crusoe/zai/GLM-5.2"),
    instruction="You are a helpful assistant.",
)
```

A complete tool-calling agent lives in [`crusoe/agent.py`](crusoe/agent.py). The `crusoe` package name is a valid Python identifier for ADK's agent loader; `google-adk` remains the integration directory. From `integrations/google-adk`, run either interface:

```bash
adk run crusoe   # interactive terminal session
adk web          # browser-based dev UI; select crusoe
```

Ask for the current time in London, Tokyo, New York, or San Francisco. The agent calls the local time tool and uses a Crusoe model to answer. Model requests can incur inference charges. Exit the terminal session or stop the web server with `Ctrl+C` when finished.

## Available models

| Model | Context window |
|---|---|
| `crusoe/zai/GLM-5.2` | 256,000 |
| `crusoe/zai/GLM-5.1` | 202,000 |
| `crusoe/nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` | 262,144 |
| `crusoe/google/gemma-4-31b-it` | 262,144 |
| `crusoe/deepseek-ai/DeepSeek-V3-0324` | 163,840 |
| `crusoe/meta-llama/Llama-3.3-70B-Instruct` | 131,072 |
| `crusoe/openai/gpt-oss-120b` | 131,072 |
| `crusoe/moonshotai/Kimi-K2.6` | 262,144 |

The current catalog is always available from `https://api.inference.crusoecloud.com/v1/models` and the [Crusoe Managed Inference docs](https://docs.crusoecloud.com/managed-inference/overview).

## Model parameters

`LiteLlm` forwards generation parameters to the Crusoe API:

```python
model = LiteLlm(
    model="crusoe/zai/GLM-5.2",
    temperature=0.2,
    max_tokens=1024,
    top_p=0.95,
)
```
