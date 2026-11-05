# vLLM with Crusoe

[Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference) is powered by [vLLM](https://docs.vllm.ai/), and Crusoe is listed in vLLM's deployment frameworks ([vllm-project/vllm#49353](https://github.com/vllm-project/vllm/pull/49353)). Two things follow for developers:

- **Client code is portable.** Anything written against a self-hosted vLLM OpenAI server runs against Crusoe endpoints by changing the base URL and key, and the reverse holds when you move a workload from Serverless Inference onto your own GPUs.
- **vLLM features are exposed.** Guided decoding (`response_format` with a JSON schema), tool calling, and streaming behave as they do on a vLLM server.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)

## Setup

```bash
cd integrations/vllm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
)

response = client.chat.completions.create(
    model="zai/GLM-5.3",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
print(response.choices[0].message.content)
```

Verify with curl:

```bash
curl https://api.inference.crusoecloud.com/v1/chat/completions \
  -H "Authorization: Bearer $CRUSOE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "zai/GLM-5.3", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`client.py`](client.py) | One client, three vLLM-backed features: streaming, JSON-schema guided decoding, and the model catalog |
| [`portable.py`](portable.py) | The same code against Crusoe or a local `vllm serve`, switched by two environment variables |

```bash
python client.py
VLLM_BASE_URL=https://api.inference.crusoecloud.com/v1 python portable.py
```

To run `portable.py` against your own vLLM server on a Crusoe GPU VM:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000   # on the VM
VLLM_BASE_URL=http://<vm-ip>:8000/v1 VLLM_API_KEY=unused VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct python portable.py
```

## Self-hosting vLLM on Crusoe

For dedicated capacity, run vLLM yourself on Crusoe GPU VMs or Crusoe Managed Kubernetes. The [Solutions Library](https://github.com/crusoecloud/solutions-library) has deployment recipes, and [Self-Serve Deployments](https://docs.crusoecloud.com/self-serve-deployments/overview) gives you a managed vLLM replica set with the same API as Serverless Inference.

[Back to integrations](../README.md)
