# LiveKit Agents with Crusoe

Run [LiveKit Agents](https://docs.livekit.io/agents/) voice agents on [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference). LiveKit's OpenAI plugin accepts any OpenAI-compatible endpoint, so Crusoe works today with a `base_url` and key. A dedicated `LLM.with_crusoe(...)` constructor with Crusoe's model list is in review upstream ([livekit/agents#6408](https://github.com/livekit/agents/pull/6408)); when it merges, it replaces the explicit `base_url` and the rest of the agent is unchanged.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Security > Inference API Key**)
- A LiveKit Cloud project or self-hosted server (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`)
- Speech services for the voice path: the recipe uses [Deepgram](https://deepgram.com/) for speech-to-text and [Cartesia](https://cartesia.ai/) for text-to-speech

## Setup

```bash
cd integrations/livekit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
python agent.py download-files   # fetch the VAD and turn-detection models once
```

## Quick start

```python
import os

from livekit.plugins import openai

llm = openai.LLM(
    model="zai/GLM-5.3",
    base_url="https://api.inference.crusoecloud.com/v1/",
    api_key=os.environ["CRUSOE_API_KEY"],
)
```

After livekit/agents#6408 ships: `llm = openai.LLM.with_crusoe(model="zai/GLM-5.3")`.

## Recipes

| Script | What it shows |
| --- | --- |
| [`agent.py`](agent.py) | A voice agent with a function tool, Silero VAD, and the LiveKit turn detector, served through the Agents worker |

Test it in your terminal without a room, then connect it to a LiveKit room:

```bash
python agent.py console   # local mic and speaker
python agent.py dev       # register with LiveKit and join rooms
```

Both make live requests that incur inference charges plus the speech providers' costs.

## Notes

- `openai.LLM` forwards `temperature`, `parallel_tool_calls`, `tool_choice`, and `reasoning_effort` to Crusoe. Set `reasoning_effort="low"` for `openai/gpt-oss-120b` in voice use to keep latency down.
- Lower-latency models suit voice turns: `zai/GLM-5.3-Flash`, `nvidia/Nemotron-3-Nano-30B-A3B`, `meta-llama/Llama-3.3-70B-Instruct`.
- The model catalog is at `https://api.inference.crusoecloud.com/v1/models`.

[Back to integrations](../README.md)
