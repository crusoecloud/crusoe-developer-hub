# aisuite with Crusoe

Call [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference) through [aisuite](https://github.com/andrewyng/aisuite), the thin multi-provider client from Andrew Ng's team. Crusoe ships as a built-in aisuite provider ([andrewyng/aisuite#351](https://github.com/andrewyng/aisuite/pull/351)), so a model string of `crusoe:<model id>` is the whole integration, and swapping providers is a one-line change.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Security > Inference API Key**)

## Setup

```bash
cd integrations/aisuite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```python
import aisuite as ai

client = ai.Client()
response = client.chat.completions.create(
    model="crusoe:zai/GLM-5.3",
    messages=[{"role": "user", "content": "Explain liquid cooling in one sentence."}],
)
print(response.choices[0].message.content)
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`chat.py`](chat.py) | The same prompt sent to two Crusoe models by changing only the model string |
| [`tool_use.py`](tool_use.py) | aisuite's automatic tool loop (`max_turns`) with a plain Python function as the tool |

```bash
python chat.py
python tool_use.py
```

Both scripts make live requests that incur inference charges.

## Notes

- Configure the key in code instead of the environment with `ai.Client({"crusoe": {"api_key": "..."}})`.
- Any model from `https://api.inference.crusoecloud.com/v1/models` works after the `crusoe:` prefix.
- Extra keyword arguments such as `temperature`, `max_tokens`, or `reasoning_effort` pass through to the Crusoe API unchanged.

[Back to integrations](../README.md)
