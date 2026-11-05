# Portkey AI Gateway with Crusoe

Route Crusoe Managed Inference through the [Portkey AI Gateway](https://github.com/Portkey-AI/gateway) for retries, fallbacks, load balancing, and observability across providers. A native `crusoe` provider is open upstream ([Portkey-AI/gateway#1732](https://github.com/Portkey-AI/gateway/pull/1732)); until it merges, Portkey's custom-host path for OpenAI-compatible endpoints works unchanged.

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)
- Docker, or a Portkey account

## Run the gateway locally

```bash
docker run --rm -p 8787:8787 portkeyai/gateway:latest
```

## Call Crusoe through it

Send requests to the gateway with a provider config that names Crusoe's base URL as the custom host:

```bash
export CRUSOE_API_KEY="your-api-key"

curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-portkey-provider: openai" \
  -H "x-portkey-custom-host: https://api.inference.crusoecloud.com/v1" \
  -H "Authorization: Bearer $CRUSOE_API_KEY" \
  -d '{"model": "zai/GLM-5.3", "messages": [{"role": "user", "content": "Hello from Portkey"}]}'
```

The same works from the OpenAI SDK by pointing `base_url` at the gateway and adding the two `x-portkey-*` headers as `default_headers`:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8787/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
    default_headers={
        "x-portkey-provider": "openai",
        "x-portkey-custom-host": "https://api.inference.crusoecloud.com/v1",
    },
)
print(client.chat.completions.create(model="zai/GLM-5.3", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

## Fallback config

A gateway config can fall back from one Crusoe model to another, or from Crusoe to a second provider, with the same custom-host target:

```json
{
  "strategy": { "mode": "fallback" },
  "targets": [
    { "provider": "openai", "custom_host": "https://api.inference.crusoecloud.com/v1", "api_key": "<crusoe api key>", "override_params": { "model": "zai/GLM-5.3" } },
    { "provider": "openai", "custom_host": "https://api.inference.crusoecloud.com/v1", "api_key": "<crusoe api key>", "override_params": { "model": "meta-llama/Llama-3.3-70B-Instruct" } }
  ]
}
```

The config travels in the `x-portkey-config` header as JSON, and JSON does not expand environment variables, so the key has to be substituted before the request. Build the header in the shell with `jq`, which reads the key from the environment and never puts it in a file:

```bash
PORTKEY_CONFIG=$(jq -nc --arg k "$CRUSOE_API_KEY" '{
  strategy: {mode: "fallback"},
  targets: [
    {provider: "openai", custom_host: "https://api.inference.crusoecloud.com/v1", api_key: $k, override_params: {model: "zai/GLM-5.3"}},
    {provider: "openai", custom_host: "https://api.inference.crusoecloud.com/v1", api_key: $k, override_params: {model: "meta-llama/Llama-3.3-70B-Instruct"}}
  ]}')

curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-portkey-config: $PORTKEY_CONFIG" \
  -d '{"messages": [{"role": "user", "content": "Hello from Portkey"}]}'
```

After Portkey-AI/gateway#1732 merges, replace `"provider": "openai"` plus `custom_host` with `"provider": "crusoe"`.

[Back to integrations](../README.md)
