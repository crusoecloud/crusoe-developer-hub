---
title: Nemotron Duo - AI Pair Programming
emoji: 🤖
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: "5.9.0"
app_file: app.py
pinned: false
license: mit
short_description: AI Pair Programming with NVIDIA Nemotron on Crusoe Cloud
---

# Nemotron Duo: AI Pair Programming

**A live pair-programming demo where two NVIDIA Nemotron models collaborate: one codes fast, the other reviews like a senior engineer.**

---

## How It Works

```
User describes what they want to build
                │
                ▼
┌───────────────────────────────┐
│  ⚡ Nemotron Nano (30B, 3B   │
│     active parameters)        │
│  "The Fast Coder"             │
│                               │
│  → Rapid implementation       │
│  → Clean, functional code     │
│  → Brief inline comments      │
│  → Ships fast                 │
└──────────────┬────────────────┘
               │ Generated code
               ▼
┌───────────────────────────────┐
│  🧠 Nemotron Super (120B,    │
│     12B active parameters)    │
│  "The Senior Reviewer"        │
│                               │
│  → Bug detection              │
│  → Security analysis          │
│  → Architecture review        │
│  → Concrete improvements      │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Split-panel results          │
│  + Metrics: tokens, TTFT,     │
│    TPS, elapsed time          │
│  + Cost savings estimate      │
└───────────────────────────────┘
```

**Why two models?** Using Nano for fast generation + Super for review is faster *and* cheaper than using Super for everything. The demo tracks and displays the cost savings in real time.

---

## Features

- **Two-phase workflow:** Nano generates code, Super reviews and improves it
- **Adjustable review depth:** Light / Medium / Thorough
- **Live metrics:** Tokens, time-to-first-token (TTFT), tokens-per-second, elapsed time
- **Cost comparison:** Shows savings vs. using only the large model
- **Example prompts:** Pre-built coding scenarios to try
- **Custom CSS:** Green panel (Nano) and purple panel (Super) for visual clarity

---

## Architecture

```
crusoe-nvidia-nemotron-demo/
├── app.py                # Full Gradio app: 2 models, split-panel UI, metrics
├── requirements.txt
└── README.md
```

### Single-file design

The entire app lives in `app.py`:

- **Config:** OpenAI client pointed at Crusoe Foundry, model IDs, system prompts
- **Nano system prompt:** Focus on speed, working implementations, concise explanations
- **Super system prompt:** Critical review, bug/security analysis, concrete improvements with code
- **Inference:** Non-streaming with full metric capture (tokens, TTFT, TPS)
- **Cost engine:** Compares Duo approach cost vs. Super-only cost
- **UI:** Gradio Blocks with split panels, metrics accordion, example prompts

## Models

| Role | Model | Active Params | Full Params |
|------|-------|--------------|-------------|
| Fast Coder | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 3B | 30B MoE |
| Senior Reviewer | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` | 12B | 120B MoE |

Both served via Crusoe Cloud Foundry's OpenAI-compatible endpoint.

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Crusoe Cloud](https://crusoe.ai) API key

### Local Development

```bash
# From the developer hub repository root
cd integrations/huggingface/crusoe-nvidia-nemotron-demo

pip install -r requirements.txt
export CRUSOE_API_KEY=your_key_here
python app.py
```

Open **http://localhost:7860**.

### Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Add `CRUSOE_API_KEY` as a Space secret in Settings
4. Upload this demo's `app.py`, `requirements.txt`, and `README.md` to the root of the new Space using its Files tab.

---

## Use as a Framework

### Swap the model pair

In `app.py`, change the model constants:

```python
NANO_MODEL = "org/your-fast-model"
SUPER_MODEL = "org/your-reviewer-model"
```

Any OpenAI-compatible models work. Good pairings:
- Fast: Gemma 3 12B, Nemotron Nano → Review: Qwen3 235B, DeepSeek R1
- Fast: Llama 3.3 70B → Review: Kimi K2

### Change the workflow

The system prompts define each model's role. Edit them to create different workflows:

- **Writer + Editor:** One model drafts prose, the other edits for clarity
- **Planner + Implementer:** One model creates a plan, the other writes the code
- **Debater:** Both models argue opposing sides of a technical decision

### Customize the review depth

The review depth maps to the system prompt. Edit the depth options:

```python
if depth == "thorough":
    super_prompt += "Do a line-by-line review. Miss nothing."
```

### Add a third model

Extend the pipeline, for example, add a "QA Tester" that writes tests for the generated code:

```python
QA_MODEL = "org/your-qa-model"
QA_SYSTEM = "You write comprehensive unit tests for the provided code..."
```

### Swap the endpoint

Works with any OpenAI-compatible API:

```python
client = OpenAI(
    base_url="https://your-endpoint.com/v1/",
    api_key=os.environ.get("YOUR_API_KEY"),
)
```

---

## API Reference (Crusoe Cloud Foundry)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1/",
    api_key="YOUR_CRUSOE_API_KEY",
)

# Fast generation with Nano
nano_response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    messages=[{"role": "user", "content": "Build a REST API in FastAPI"}],
    temperature=1,
    top_p=0.95,
)

# Senior review with Super
super_response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    messages=[
        {"role": "user", "content": f"Review this code:\n{nano_response.choices[0].message.content}"}
    ],
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

// Fast generation
const nano = await client.chat.completions.create({
  model: 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B',
  messages: [{ role: 'user', content: 'Build a REST API in FastAPI' }],
});

// Senior review
const review = await client.chat.completions.create({
  model: 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B',
  messages: [{ role: 'user', content: `Review this code:\n${nano.choices[0].message.content}` }],
});
```

---

## Dependencies

```
gradio>=5.9.0
openai>=1.40.0
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CRUSOE_API_KEY` | Yes | Your Crusoe Cloud Foundry API key |
| `CRUSOE_API_BASE` | No | Override endpoint (default: Crusoe Foundry proxy) |

---

## License

MIT. Fork it, remix it, ship it.

Built by Crusoe AI Developer Relations · [crusoe.ai](https://crusoe.ai)
