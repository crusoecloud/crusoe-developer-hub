# Crusoe Hugging Face Demos

A collection of Hugging Face Spaces demo projects covering LLM game battles, model comparison, voice, document analysis, and agent-driven deployment. These are application demos with source code, rather than planned placeholders.

Each folder contains a Gradio, Streamlit, or Docker app and its Space metadata. Read the individual README for setup and limitations; deployment links below are project references, and availability can vary. These projects belong in [Integrations](../README.md) because they demonstrate the Hugging Face Spaces workflow. For standalone tasks organized by developer goal, browse [Examples](../../examples/README.md).

## Demos

| Demo | Description | SDK | Space link |
|------|-------------|-----|------------|
| [one-shot-deploy](one-shot-deploy/) | Describe a demo in plain English → a multi-model agent writes the code and ships it to HF Spaces | Docker | [demo_agent](https://huggingface.co/crusoeai/spaces) |
| [model-arena](model-arena/) | Compare, cost, and crash-test open-source LLMs on Crusoe Managed AI | Streamlit | [foundry-model-arena](https://huggingface.co/spaces/crusoeai/foundry-model-arena) |
| [arcade-race](arcade-race/) | Race four LLMs to generate playable 80s arcade games | Gradio | [pixel-prix](https://huggingface.co/spaces/crusoeai/pixel-prix) |
| [debate-arena](debate-arena/) | Two Gemma personas debate any topic with live-streamed speech bubbles | Docker | [debate-arena](https://huggingface.co/spaces/crusoeai/debate-arena) |
| [fake-podcast](fake-podcast/) | Generate AI podcast conversations between two LLM hosts with real TTS audio | Docker | [fake-podcast](https://huggingface.co/spaces/crusoeai/fake-podcast) |
| [gemma-canvas](gemma-canvas/) | Image-aware chat with Gemma on Crusoe Foundry | Gradio | [gemma-canvas](https://huggingface.co/spaces/crusoeai/gemma-canvas) |
| [crusoe-hf-space-legal](crusoe-hf-space-legal/) | Document analysis, codebase intelligence, and KV-cache demo | Gradio | [doc_analysis](https://huggingface.co/spaces/crusoeai/doc_analysis) |
| [crusoe-nvidia-nemotron-demo](crusoe-nvidia-nemotron-demo/) | Nemotron Duo: AI pair programming with NVIDIA Nemotron | Gradio | [Nemotron Duo](https://huggingface.co/spaces/crusoeai/crusoe_nvidia_nemotron_demo) |
| [personal-fashion-advisor](personal-fashion-advisor/) | AI fashion stylist with personalized outfit recommendations | Docker | [personal-fashion-advisor](https://huggingface.co/spaces/crusoeai/personal-fashion-advisor) |
| [crusoe-sentiment-streamlit](crusoe-sentiment-streamlit/) | Sentiment analysis across Reddit, Twitter/X, and news; the current implementation uses OpenRouter | Streamlit | Not listed |
| [crusoe-test](crusoe-test/) | Brand sentiment & competitive analysis dashboard | Streamlit | Not listed |
| [voice-translation-relay](voice-translation-relay/) | Speech translation demo with fallback behavior documented in its README | Gradio | Not listed |

### LLM Game Battles

Head-to-head games where two LLMs on Crusoe Foundry compete in real time:

| Demo | Space link |
|------|------------|
| [battleship-ai](battleship-ai/) | [battleship-ai](https://huggingface.co/spaces/crusoeai/battleship-ai) |
| [chess-ai](chess-ai/) | [chess-ai](https://huggingface.co/spaces/crusoeai/chess-ai) |
| [connect-four-ai](connect-four-ai/) | [connect-four-ai](https://huggingface.co/spaces/crusoeai/connect-four-ai) |
| [othello-ai](othello-ai/) | [othello-ai](https://huggingface.co/spaces/crusoeai/othello-ai) |
| [pong-ai](pong-ai/) | [pong-ai](https://huggingface.co/spaces/crusoeai/pong-ai) |
| [snake-battle-ai](snake-battle-ai/) | [snake-battle-ai](https://huggingface.co/spaces/crusoeai/snake-battle-ai) |
| [tictactoe-5x5-ai](tictactoe-5x5-ai/) | [tictactoe-5x5-ai](https://huggingface.co/spaces/crusoeai/tictactoe-5x5-ai) |
| [tron-lightcycles-ai](tron-lightcycles-ai/) | Not listed |

## Setup

Choose a demo and read its README before installing its dependencies. Many demos read a Crusoe Inference API key from the environment:

```bash
export CRUSOE_API_KEY="your-api-key"
```

Obtain a key through the [Crusoe Console](https://console.crusoecloud.com/). Configuration differs between apps:

| Demo | Configuration to check |
|---|---|
| [Nemotron Duo](crusoe-nvidia-nemotron-demo/README.md#quick-start) | `CRUSOE_API_KEY`, Python dependencies, and optional `CRUSOE_API_BASE` |
| [One-Shot Deploy](one-shot-deploy/README.md#required-secrets) | `CRUSOE_API_KEY`, `HF_TOKEN` with write scope, and `HF_USERNAME`; the app creates and publishes Spaces |
| [Sentiment intelligence](crusoe-sentiment-streamlit/README.md) | `OPENROUTER_API_KEY` and source-specific Reddit, Twitter/X, and news credentials, as defined in [config.py](crusoe-sentiment-streamlit/config.py) |
| [Market intelligence](crusoe-test/README.md#required-secrets) | `CRUSOE_API_KEY` and `NEWSAPI_KEY` |
| [Tron Light Cycles](tron-lightcycles-ai/README.md) | The current [app.py](tron-lightcycles-ai/app.py) reads `API_KEY` and optional `API_URL` from the environment |
| [Voice Translation Relay](voice-translation-relay/README.md#configuration) | Enter the key in the app's configuration panel and review its audio fallback limitations |

Some READMEs focus on Space deployment rather than local execution. Model IDs, endpoint defaults, and credential names differ across the preserved demos; check the selected app's configuration before sending requests. Inference and third-party services can incur usage charges, and deployed Spaces can incur hosting costs. Stop local processes when finished and remove hosted resources you no longer need.

[Back to Integrations](../README.md)
