---
title: One-Shot Deploy
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
---

# One-Shot Deploy

Describe a demo in plain English → Crusoe's multi-model agent writes the code and ships it to Hugging Face Spaces in minutes.

## How it works

| Stage | Model | What happens |
|-------|-------|-------------|
| 1 · Intent | **DeepSeek R1** | Reasons about your prompt, selects a template, writes the AI system prompt |
| 2 · Codegen | **Kimi-K2** | Writes a complete single-file Streamlit app wired to Crusoe inference |
| 3 · Validate | *(ast parser)* | Checks syntax and structural patterns |
| 4 · Deploy | **Qwen3** (healer) | Pushes to HF Spaces, injects secrets, auto-fixes on build failure (up to 3×) |

## Required secrets

Set these in **Settings → Variables and secrets** in your Space:

| Name | Description |
|------|-------------|
| `CRUSOE_API_KEY` | Crusoe Managed Inference API key |
| `HF_TOKEN` | Hugging Face token with write scope |
| `HF_USERNAME` | Your Hugging Face username |

Optionally override the default models:

| Name | Default |
|------|---------|
| `CRUSOE_BASE_URL` | `https://api.crusoe.ai/v1` |
| `INTENT_MODEL` | `deepseek-ai/DeepSeek-R1` |
| `CODEGEN_MODEL` | `moonshotai/Kimi-K2-Instruct` |
| `HEALER_MODEL` | `Qwen/Qwen3-235B-A22B` |

## Built by

Crusoe DevRel · Powered by [Crusoe Managed Inference](https://crusoe.ai)
