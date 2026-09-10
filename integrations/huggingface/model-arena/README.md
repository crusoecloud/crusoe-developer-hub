---
title: Model Arena
emoji: ⚔️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: true
---

# Model Arena

**Compare, Cost, and Crash-Test Open-Source LLMs on Crusoe Managed AI**

No single model rules them all. This app lets you prove it.

---

## What This App Does

### Arena — Head-to-Head Model Comparison
Pit open-source LLMs against each other side by side. Same prompt, real latency and token metrics. Vote on the best response and contribute to the community leaderboard.

### Cost — Real Economics Calculator
See what it actually costs to run open-source models on Crusoe Managed AI vs. proprietary APIs (OpenAI, Anthropic). Scale from 100 to 1M queries/day and watch the math change.

### Resilience — Failover Simulator
Configure a 3-step agent pipeline, assign models, then simulate an endpoint outage. Watch the pipeline reroute to a fallback model in real time.

---

## Architecture

```
model-arena/
├── app.py                # Streamlit entry point — page config, tabs, API key check
├── core/
│   ├── __init__.py
│   ├── models.py         # Model registry with Crusoe + proprietary pricing
│   ├── inference.py      # OpenAI-compatible client — streaming + non-streaming + metrics
│   ├── cost_engine.py    # Cost calculations — per-query, daily, monthly, yearly projections
│   └── leaderboard.py    # JSON-based voting system — ELO-style win rates
├── tabs/
│   ├── __init__.py
│   ├── arena.py          # Side-by-side comparison with parallel ThreadPoolExecutor inference
│   ├── cost.py           # Interactive cost calculator with Plotly bar charts
│   └── resilience.py     # 3-step agent pipeline with simulated outages + failover
├── .streamlit/           # Streamlit theme config
└── requirements.txt
```

### How the Arena works

```
User enters prompt + selects 2-3 models
                │
                ▼
    ┌─── ThreadPoolExecutor ───┐
    │                          │
    ▼                          ▼
┌─────────┐            ┌─────────┐
│ Model A  │            │ Model B  │
│ Crusoe   │            │ Crusoe   │
│ Foundry  │            │ Foundry  │
└────┬─────┘            └────┬─────┘
     │                       │
     ▼                       ▼
┌─────────────────────────────────┐
│  Side-by-side results           │
│  + latency, tokens/sec, TTFT   │
│  + community voting             │
└─────────────────────────────────┘
```

## Models Available

All served through Crusoe Managed AI:

| Model | Provider | Parameters | Input $/1M | Output $/1M |
|-------|----------|------------|-----------|-------------|
| Llama 3.3 70B | Meta | 70B | $0.25 | $0.75 |
| DeepSeek R1 | DeepSeek | 671B MoE | $1.35 | $5.40 |
| DeepSeek V3 | DeepSeek | 671B MoE | $0.50 | $1.50 |
| Qwen3 235B | Alibaba | 235B MoE | $0.22 | $0.80 |
| Kimi K2 | Moonshot | 1T MoE | $0.60 | $2.50 |
| Gemma 3 12B | Google | 12B | $0.08 | $0.30 |
| Nemotron 3 Super 120B | NVIDIA | 120B MoE | $0.30 | $2.40 |
| Nemotron 3 Nano 30B | NVIDIA | 30B MoE | $0.05 | $0.20 |
| GPT-OSS 120B | OpenAI | 120B | $0.15 | $0.60 |

**Proprietary baselines** (for cost comparison only): GPT-4o, GPT-4o Mini, Claude Sonnet 4, Claude Haiku 3.5

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Crusoe Cloud](https://crusoe.ai) API key

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/model-arena
cd model-arena

pip install -r requirements.txt
export CRUSOE_API_KEY=your_key_here
streamlit run app.py
```

Open **http://localhost:8501**.

### Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK (version 1.41.0)
3. Add `CRUSOE_API_KEY` as a Space secret in Settings
4. Push:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/model-arena
git push hf main
```

---

## Use as a Framework

This project is designed to be forked. Here's how to extend it:

### Add a new Crusoe model

Edit `core/models.py` — add to the `CRUSOE_MODELS` dict:

```python
"your-model": ModelInfo(
    id="org/model-name",              # Exact model ID on Crusoe Foundry
    display_name="Model Display Name",
    provider="Provider",
    param_count="70B",
    input_price=0.25,                  # Per 1M tokens
    output_price=0.75,
    description="Short description",
    strengths=["fast", "code"],
)
```

It will automatically appear in Arena, Cost, and Resilience tabs.

### Add a proprietary model baseline

Same file, add to `PROPRIETARY_MODELS` dict for cost comparison.

### Customize inference

Edit `core/inference.py`:
- `run_inference()` — Single-shot with full metrics (TTFT, latency, tokens/sec)
- `run_inference_streaming()` — Generator for streaming UIs
- Returns `InferenceResult` dataclass with all timing data

### Add a new tab

1. Create `tabs/your_tab.py` with a `render()` function
2. Import in `app.py` and add to `st.tabs`

### Swap the endpoint

Works with any OpenAI-compatible API — change the `base_url` in `core/inference.py`:

```python
client = OpenAI(
    base_url="https://your-endpoint.com/v1/",
    api_key=os.environ.get("YOUR_API_KEY"),
)
```

---

## Key Implementation Details

| Component | How it works |
|-----------|-------------|
| **Parallel inference** | `ThreadPoolExecutor` runs 2-3 model calls concurrently |
| **Leaderboard** | JSON file with win/loss records, sorted by win rate |
| **Cost engine** | Calculates per-query, daily, monthly, yearly projections from token counts |
| **Resilience** | 3-step pipeline (Summarize > Classify > Extract) with killable primary models |

---

## API Reference (Crusoe Cloud Foundry)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1/",
    api_key="YOUR_CRUSOE_API_KEY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=1,
    top_p=0.95,
)
```

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://api.inference.crusoecloud.com/v1/',
  apiKey: 'YOUR_CRUSOE_API_KEY',
});

const response = await client.chat.completions.create({
  model: 'Qwen/Qwen3-235B-A22B-Instruct-2507',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CRUSOE_API_KEY` | Yes | Your Crusoe Cloud Foundry API key |

---

## License

MIT — fork it, remix it, ship it.

Built by Crusoe AI Developer Relations · [crusoe.ai](https://crusoe.ai)
