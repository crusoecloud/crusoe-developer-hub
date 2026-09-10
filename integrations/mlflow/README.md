# mlflow-crusoe

[![PyPI](https://img.shields.io/pypi/v/mlflow-crusoe.svg)](https://pypi.org/project/mlflow-crusoe/)

An [MLflow](https://mlflow.org/) deployment plugin for [Crusoe AI](https://www.crusoe.ai/)'s managed inference API.

This plugin lets you manage named model endpoint configurations and run inference on Crusoe's renewable-powered GPU infrastructure directly from MLflow's deployment interface.

## Installation

```bash
pip install mlflow-crusoe
```

## Setup

Set your Crusoe API key:

```bash
export CRUSOE_API_KEY="your-api-key"
```

You can generate one from the [Crusoe Console](https://console.crusoecloud.com/) under **Security > Inference API Key**.

## Usage

```python
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("crusoe")

# Create a deployment
client.create_deployment(
    name="my-llm",
    model_uri="meta-llama/Llama-3.3-70B-Instruct",
    config={"temperature": 0.7, "max_tokens": 2048},
)

# Run inference
result = client.predict("my-llm", inputs={"prompt": "Hello!"})
print(result["choices"][0]["message"]["content"])

# List deployments
client.list_deployments()

# Update
client.update_deployment("my-llm", model_uri="deepseek-ai/DeepSeek-V3")

# Delete
client.delete_deployment("my-llm")
```

### Input formats

The `predict` method accepts three input formats:

```python
# Chat messages (recommended)
client.predict("my-llm", inputs={
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
})

# Simple prompt
client.predict("my-llm", inputs={"prompt": "Hello!"})

# Plain string
client.predict("my-llm", inputs="Hello!")
```

### CLI

```bash
# Help
mlflow deployments help -t crusoe

# Create
mlflow deployments create -t crusoe --name my-llm -m meta-llama/Llama-3.3-70B-Instruct

# List
mlflow deployments list -t crusoe

# Predict
mlflow deployments predict -t crusoe --name my-llm --input '{"prompt": "Hi"}'

# Delete
mlflow deployments delete -t crusoe --name my-llm
```

## Available Models

| Model | Identifier |
|-------|-----------|
| Meta Llama 3.3 70B Instruct | `meta-llama/Llama-3.3-70B-Instruct` |
| DeepSeek V3 (0324) | `deepseek-ai/DeepSeek-V3-0324` |
| DeepSeek R1 (0528) | `deepseek-ai/DeepSeek-R1-0528` |
| Google Gemma 3 12B | `google/gemma-3-12b-it` |
| OpenAI GPT-OSS 120B | `openai/gpt-oss-120b` |
| Qwen3 235B A22B Instruct | `Qwen/Qwen3-235B-A22B-Instruct-2507` |

See the [Crusoe docs](https://docs.crusoecloud.com/managed-inference/overview/index.html) for latest availability.

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | `CRUSOE_API_KEY` env var | API key |
| `api_base` | `https://api.crusoe.ai/v1` | API base URL |
| `temperature` | `0.1` | Sampling temperature (0-2) |
| `max_tokens` | `1024` | Max tokens to generate |
| `top_p` | None | Nucleus sampling |
| `frequency_penalty` | None | Frequency repetition penalty |
| `presence_penalty` | None | Presence repetition penalty |
| `stop` | None | Comma-separated stop sequences |

## References

- [Crusoe Managed Inference Docs](https://docs.crusoecloud.com/managed-inference/overview/index.html)
- [MLflow Deployment Plugins](https://mlflow.org/docs/latest/ml/plugins/)
- [Crusoe Console](https://console.crusoecloud.com/)
