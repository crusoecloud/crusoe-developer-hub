# crusoe-litellm-gateway

Your own OpenAI-compatible AI gateway to
[Crusoe Managed Inference](https://docs.crusoecloud.com/managed-inference/overview),
in one `docker compose up`. Built on the [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/quick_start)
and its native [Crusoe provider](https://docs.litellm.ai/docs/providers/crusoe).

One Crusoe API key upstream. Virtual keys, budgets, fallbacks, and the whole
model catalog behind friendly names downstream.

## Quickstart (60 seconds, no database)

```bash
git clone https://github.com/crusoecloud/crusoe-developer-hub.git
cd crusoe-developer-hub/integrations/litellm

export CRUSOE_API_KEY="your-key"          # https://console.crusoecloud.com/
docker compose up -d

curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev-master" \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-5.2", "messages": [{"role": "user", "content": "hello"}]}'
```

Any OpenAI-compatible client works — point it at `http://localhost:4000/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-local-dev-master")
print(client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

Run `scripts/smoke.sh` to verify health, model list, and a live completion.

## Friendly model names

Crusoe's full model IDs are exact and verbose
(`crusoe/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`). The gateway maps them to
names people actually type:

| Gateway name | Crusoe model |
|---|---|
| `glm-5.2` | `zai/GLM-5.2` — 256k context, the featured default |
| `glm-5.1` | `zai/GLM-5.1` |
| `nemotron-ultra` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B` |
| `nemotron-super` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| `nemotron-nano` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` |
| `nemotron-nano-omni` | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` |
| `gemma-4` | `google/gemma-4-31b-it` |
| `deepseek-v3` | `deepseek-ai/DeepSeek-V3-0324` |
| `deepseek-v4-pro` | `deepseek-ai/DeepSeek-V4-Pro` |
| `deepseek-v4-flash` | `deepseek-ai/Deepseek-V4-Flash` |
| `llama-3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` |
| `kimi-k2.6` | `moonshotai/Kimi-K2.6` |
| `gpt-oss-120b` | `openai/gpt-oss-120b` |
| `qwen3-235b` | `Qwen/Qwen3-235B-A22B-Instruct-2507` |

The catalog evolves — `GET /v1/models` against the Crusoe API is the source of
truth, and adding a model is a four-line entry in
[config/litellm-config.yaml](config/litellm-config.yaml).

## Fallbacks

`glm-5.2` automatically fails over to `llama-3.3-70b` if the primary errors.
Chains are configured per model in `litellm_settings.fallbacks`.

## Production mode: virtual keys, budgets, spend tracking

The production overlay adds Postgres and switches to the database-backed
LiteLLM image:

```bash
export CRUSOE_API_KEY="your-key"
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 16)"   # required in production
export POSTGRES_PASSWORD="$(openssl rand -hex 16)"

docker compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

Now mint scoped virtual keys instead of sharing the master key:

```bash
curl http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"models": ["glm-5.2", "gemma-4"], "max_budget": 25, "duration": "30d"}'
```

Each key gets its own model allowlist, budget, expiry, and rate limits; usage
shows up in the spend logs and the admin UI at `http://localhost:4000/ui`.
The real `CRUSOE_API_KEY` never leaves the gateway host.

> Cost fields are deliberately left unset — spend is tracked in tokens.
> If your team has contract pricing, add `input_cost_per_token` /
> `output_cost_per_token` per model in the config to get dollar amounts.

## Clients

- **curl** — [examples/curl.sh](examples/curl.sh), including streaming
- **OpenAI SDK** — [examples/openai_sdk.py](examples/openai_sdk.py)
- **CrewAI** — [examples/crewai_demo.py](examples/crewai_demo.py) (CrewAI also
  supports Crusoe directly via LiteLLM's `crusoe/` prefix)
- **deep-repo-agent** — [examples/deep-repo-agent.md](examples/deep-repo-agent.md):
  a full tool-calling agent routed through the gateway with zero code changes

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CRUSOE_API_KEY` | — (required) | Upstream key from the [Crusoe console](https://console.crusoecloud.com/) |
| `LITELLM_MASTER_KEY` | `sk-local-dev-master` | Gateway admin key — override for anything beyond local dev |
| `CRUSOE_API_BASE` | `https://api.inference.crusoecloud.com/v1` | Crusoe endpoint; pinned explicitly in the config |
| `POSTGRES_PASSWORD` | `litellm` | Production overlay only |

The endpoint is pinned per model via `CRUSOE_API_BASE` so the gateway targets
Crusoe's canonical API regardless of the LiteLLM release you run.

## License

MIT
