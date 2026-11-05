# Pipecat with Crusoe

Build voice and multimodal agents on [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference) with [Pipecat](https://docs.pipecat.ai/). Pipecat ships `CrusoeLLMService` ([pipecat-ai/pipecat#5024](https://github.com/pipecat-ai/pipecat/pull/5024)), an OpenAI-compatible LLM service with Crusoe's endpoint and defaults built in, so it drops into any Pipecat pipeline alongside your speech-to-text and text-to-speech services.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)
- For the voice bot: a [Deepgram](https://deepgram.com/) key for speech-to-text and a [Cartesia](https://cartesia.ai/) key for text-to-speech. Swap either for another Pipecat service if you prefer.

## Setup

```bash
cd integrations/pipecat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

## Quick start

```python
import os

from pipecat.services.crusoe.llm import CrusoeLLMService

llm = CrusoeLLMService(
    api_key=os.environ["CRUSOE_API_KEY"],
    settings=CrusoeLLMService.Settings(
        model="zai/GLM-5.3",
        system_instruction="You are a helpful assistant in a voice conversation. Keep replies short.",
    ),
)
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`bot.py`](bot.py) | A voice bot with two function-calling tools, runnable on a WebRTC page, Daily, or Twilio through Pipecat's runner |

Run it with the local WebRTC transport and open the printed URL in a browser:

```bash
python bot.py -t webrtc
```

The bot introduces itself, then answers questions about the weather or restaurants by calling its tools. It makes live requests that incur inference charges plus the speech providers' own costs.

## Notes

- `CrusoeLLMService` accepts the same `Settings` as Pipecat's OpenAI service: `model`, `temperature`, `max_tokens`, `system_instruction`, and `tools`. Crusoe does not use the `developer` message role, so the service sends system instructions as a `system` message for you.
- Tools are plain async functions registered through `LLMContext(tools=[...])`; Pipecat builds the schema from the signature and docstring.
- Set `CRUSOE_MODEL` in `.env` to switch models without editing code. Models with fast time to first token, such as `zai/GLM-5.3-Flash`, `nvidia/Nemotron-3-Nano-30B-A3B`, and `meta-llama/Llama-3.3-70B-Instruct`, keep voice turns snappy.
- `pipecat init` offers Crusoe as an LLM choice when scaffolding a new project.

[Back to integrations](../README.md)
