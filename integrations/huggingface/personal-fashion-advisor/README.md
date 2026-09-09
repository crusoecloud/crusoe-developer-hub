---
title: "Style Savvy: Your Personal Fashion Advisor"
emoji: 👗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
short_description: AI fashion stylist powered by Crusoe Managed AI
---

# Style Savvy: Your Personal Fashion Advisor

**An AI-powered fashion chatbot that gives personalized outfit recommendations, seasonal trend analysis, and inclusive style advice — powered by Crusoe Managed Inference.**

---

## How It Works

```
User describes their style, occasion, or question
                    │
                    ▼
        ┌───────────────────────┐
        │  Streamlit Chat UI    │
        │  with streaming       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Crusoe Foundry API   │
        │  (OpenAI-compatible)  │
        │                       │
        │  DeepSeek R1 or       │
        │  Kimi K2 (selectable) │
        └───────────────────────┘
```

The app sends your full conversation history to the selected model with a fashion-expert system prompt. Responses stream in real-time. Thinking model output (`<think>` blocks) is automatically filtered so you only see the final advice.

---

## Features

- **Personalized Outfit Recommendations** — Describe your style, body type, or occasion and get tailored suggestions
- **Seasonal Trend Analysis** — Ask about current fashion trends and what's in style
- **Body Type & Style Inclusivity** — Advice for all body types, no gatekeeping
- **Sustainable Fashion Tips** — Eco-friendly and ethical fashion guidance
- **Model Selection** — Switch between DeepSeek R1 (deep reasoning) and Kimi K2 (fast, conversational) in the sidebar
- **Streaming Responses** — Answers appear in real-time as the model generates them
- **Thinking Model Support** — Automatically strips `<think>` reasoning blocks from output

---

## Models Available

| Model | Description |
|-------|-------------|
| **DeepSeek R1** (`deepseek-ai/DeepSeek-R1-0528`) | Thinking model — deeper, more considered advice |
| **Kimi K2** (`moonshotai/Kimi-K2-Thinking`) | Fast and conversational |

---

## Required Secrets

Set these in **Settings → Variables and secrets** in your HF Space:

| Name | Description |
|------|-------------|
| `CRUSOE_API_KEY` | Crusoe Managed Inference API key |

Optionally override the endpoint:

| Name | Default |
|------|---------|
| `CRUSOE_BASE_URL` | `https://managed-inference-api-proxy.crusoecloud.com/v1/` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (chat UI with streaming) |
| LLM Inference | Crusoe Cloud Foundry (OpenAI-compatible) |
| Container | Docker (Python 3.11-slim) |
| Dependencies | `streamlit`, `openai` |

---

## Local Development

```bash
git clone https://huggingface.co/spaces/crusoeai/personal-fashion-advisor
cd personal-fashion-advisor

# Set your API key
export CRUSOE_API_KEY="your-key-here"

# Install and run
pip install streamlit openai
streamlit run app.py
```

Open **http://localhost:8501** and start chatting.

---

## Deploy to Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Add `CRUSOE_API_KEY` in Settings → Variables and secrets
3. Push the code — the Dockerfile handles everything

---

## Built by

Crusoe DevRel · Powered by [Crusoe Managed Inference](https://crusoe.ai)
