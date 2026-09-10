---
title: Crusoe AI Market Intelligence
emoji: ⚡
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
short_description: Brand sentiment & competitive analysis dashboard
---

# Crusoe AI Market Intelligence

**Real-time brand sentiment, perception analysis, and competitive landscape dashboard — powered by Crusoe Managed AI.**

---

## How It Works

```
Click "Run Analysis"
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│  NewsAPI           │     │  NewsAPI              │
│  "Crusoe AI"      │     │  Competitor coverage   │
│  (20 articles)    │     │  (15 articles)         │
└────────┬──────────┘     └───────────┬────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
         ┌───────────────────────┐
         │  Crusoe Foundry LLM   │
         │  (6 models available) │
         │  Structured JSON out  │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Glass-morphism       │
         │  Dashboard UI         │
         │  (Streamlit)          │
         └───────────────────────┘
```

1. Fetches recent news articles about Crusoe AI and 6 competitors from NewsAPI
2. Sends all articles to a Crusoe-hosted LLM with a structured analysis prompt
3. Parses the JSON response and renders an executive-grade dashboard

---

## Features

- **Executive Summary** — C-suite-ready narrative generated from current coverage
- **Sentiment Scores** — Overall and developer sentiment on a -1.0 to 1.0 scale
- **Brand Perception Breakdown** — How much coverage frames Crusoe as a data center company vs. cloud provider vs. managed AI platform
- **Key Themes & Opportunities** — Extracted from article analysis
- **Risk Monitoring** — Watch areas identified from coverage patterns
- **Competitive Landscape** — Head-to-head analysis against 6 neocloud competitors:
  - CoreWeave, Lambda Labs, Together AI, Voltage Park, Fluidstack, Nebius
- **Competitor Cards** — Sentiment, coverage volume, threat level, and key narrative per competitor
- **Crusoe Advantages & Gaps** — Strategic positioning analysis
- **Source Articles** — Expandable list of all articles used in the analysis
- **Model Selector** — Choose from 6 models in the sidebar (including thinking models)

---

## Models Available

| Model | Provider | Notes |
|-------|----------|-------|
| Qwen3 235B | Qwen | Flagship — best overall quality |
| DeepSeek R1-0528 | DeepSeek | Thinking model — slower, deeper |
| Llama 3.3 70B | Meta | Fast & efficient |
| GPT-OSS 120B | OpenAI | Open weights model |
| Gemma 3 12B | Google | Lightweight & fast |
| Kimi-K2 Thinking | Moonshot AI | Thinking model — complex analysis |

Thinking models automatically have their `<think>` blocks stripped from the output.

---

## Required Secrets

Set these in **Settings → Variables and secrets** in your HF Space:

| Name | Description |
|------|-------------|
| `CRUSOE_API_KEY` | Crusoe Managed Inference API key |
| `NEWSAPI_KEY` | [NewsAPI](https://newsapi.org/) API key (free tier works) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit with custom glass-morphism CSS |
| LLM Inference | Crusoe Cloud Foundry (OpenAI-compatible) |
| News Data | NewsAPI (articles about Crusoe AI + competitors) |
| Dependencies | `streamlit`, `openai`, `newsapi-python`, `httpx` |

---

## Local Development

```bash
git clone https://huggingface.co/spaces/crusoeai/<space-name>
cd crusoe-test

# Set your API keys
export CRUSOE_API_KEY="your-crusoe-key"
export NEWSAPI_KEY="your-newsapi-key"

# Install and run
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** and click **Run Analysis**.

---

## Built by

Crusoe DevRel · Powered by [Crusoe Managed Inference](https://crusoe.ai)
