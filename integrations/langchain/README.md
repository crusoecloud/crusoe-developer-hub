# LangChain × Crusoe AI

[![PyPI version](https://badge.fury.io/py/langchain-crusoe.svg)](https://badge.fury.io/py/langchain-crusoe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package provides LangChain integrations for [Crusoe AI's Managed Inference](https://www.crusoe.ai/cloud/managed-inference) service, giving you access to leading open-source models powered by Crusoe's proprietary MemoryAlloy™ inference engine.

## Features

- **OpenAI-compatible API**: Drop-in replacement via `BaseChatOpenAI`
- **Leading open-source models**: Llama 3.3, DeepSeek V3/R1, Qwen3, Gemma 3, Kimi-K2, GPT-OSS-120B
- **Full LangChain support**: Streaming, async, tool calling, structured output
- **Ultra-low latency**: Powered by MemoryAlloy cluster-wide KV cache technology
- **LangSmith integration**: Built-in tracing and observability

## Installation

```bash
pip install -U langchain-crusoe
```

## Setup

1. Create an account at [Crusoe Cloud](https://console.crusoecloud.com/)
2. Generate an Inference API key in the **Security** tab
3. Set your API key:

```bash
export CRUSOE_API_KEY="your-api-key"
```

## Quick Start

```python
from langchain_crusoe import ChatCrusoe

llm = ChatCrusoe(
    model="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0,
    max_tokens=1024,
)

# Simple invocation
response = llm.invoke("Explain quantum computing in one paragraph.")
print(response.content)

# Streaming
for chunk in llm.stream("Write a haiku about open source."):
    print(chunk.content, end="", flush=True)
```

## Available Models

| Model | Provider | Context Length |
|-------|----------|---------------|
| `meta-llama/Llama-3.3-70B-Instruct` | Meta | 128k |
| `openai/gpt-oss-120b` | OpenAI | 128k |
| `deepseek-ai/DeepSeek-V3-0324` | DeepSeek | 160k |
| `deepseek-ai/DeepSeek-V4-Flash` | DeepSeek | 1M |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek | 1M |
| `google/gemma-4-31b-it` | Google | 262k |
| `moonshotai/Kimi-K2.6` | Moonshot AI | 256k |
| `nvidia/Nemotron-3-Nano-30B-A3B` | NVIDIA | 262k |
| `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` | NVIDIA | 262k |
| `nvidia/Nemotron-3-Super-120B-A12B` | NVIDIA | 262k |
| `nvidia/Nemotron-3-Ultra-550B` | NVIDIA | 262k |
| `qwen/Qwen3-235B-A22B` | Qwen | 131k |
| `zai/GLM-5.1` | Z.ai | 202k |
| `zai/GLM-5.2` | Z.ai | 256k |

See the latest model list at the [Crusoe Intelligence Foundry](https://console.crusoecloud.com/foundry/models).

## Advanced Usage

### Tool Calling

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get current weather for a location."""
    location: str = Field(description="City and state, e.g. San Francisco, CA")

llm_with_tools = llm.bind_tools([GetWeather])
response = llm_with_tools.invoke("What's the weather in Seattle?")
print(response.tool_calls)
```

### Structured Output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

structured_llm = llm.with_structured_output(Summary)
result = structured_llm.invoke("Summarize the benefits of open-source AI models.")
```

### Async Support

```python
import asyncio

async def main():
    response = await llm.ainvoke("Hello, async world!")
    print(response.content)

asyncio.run(main())
```

### Project ID

Optionally set a Crusoe Project ID for usage attribution:

```python
llm = ChatCrusoe(
    model="meta-llama/Llama-3.3-70B-Instruct",
    crusoe_project_id="my-project-id",
)
```

Or via environment variable:

```bash
export CRUSOE_PROJECT_ID="my-project-id"
```

## Configuration

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `api_key` | `CRUSOE_API_KEY` | Required | Your Crusoe inference API key (required) |
| `model` | Not applicable | `meta-llama/Llama-3.3-70B-Instruct` | Model to use |
| `crusoe_api_base` | `CRUSOE_API_BASE` | `https://api.inference.crusoecloud.com/v1` | API endpoint |
| `crusoe_project_id` | `CRUSOE_PROJECT_ID` | `None` | Project ID for attribution |
| `temperature` | Not applicable | `0.7` (inherited) | Sampling temperature |
| `max_tokens` | Not applicable | `None` | Max tokens to generate |
| `timeout` | Not applicable | `None` | Request timeout |
| `max_retries` | Not applicable | `2` | Max retry attempts |

## Development

```bash
# Clone the repo
git clone https://github.com/crusoecloud/crusoe-developer-hub.git
cd crusoe-developer-hub/integrations/langchain

# Install dependencies
poetry install --with lint,typing,test,test_integration

# Run unit tests
make tests

# Run integration tests (requires CRUSOE_API_KEY)
make integration_tests

# Run linting
make lint
```

## Links

- [Crusoe Cloud Console](https://console.crusoecloud.com/)
- [Crusoe Serverless Inference Docs](https://docs.crusoecloud.com/serverless-inference/overview)
- [LangChain Documentation](https://docs.langchain.com/)
- [API Reference](https://python.langchain.com/api_reference/crusoe/) *(available after package is published)*
