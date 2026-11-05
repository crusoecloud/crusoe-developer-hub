# SGLang with Crusoe

Write [SGLang](https://docs.sglang.ai/) frontend programs that run on [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference). SGLang ships a `Crusoe` backend ([sgl-project/sglang#20475](https://github.com/sgl-project/sglang/pull/20475)): it is the OpenAI backend with Crusoe's endpoint and `CRUSOE_API_KEY` wired in, so multi-call programs, forks, and streaming work without running an SGLang server.

## Prerequisites

- Python 3.10 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Security > Inference API Key**)

## Setup

The frontend language needs only the base package, not the GPU runtime:

```bash
cd integrations/sglang
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```python
import sglang as sgl

sgl.set_default_backend(sgl.Crusoe("zai/GLM-5.3"))


@sgl.function
def qa(s, question):
    s += sgl.system("Answer in one sentence.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))


state = qa.run(question="What is a cold aisle?")
print(state["answer"])
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`program.py`](program.py) | A multi-turn program, a fork that runs branches in parallel, batch execution, and streaming |

```bash
python program.py
```

The script makes live requests that incur inference charges.

## Notes

- `sgl.Crusoe(model_name, api_key=None, base_url=None)` reads `CRUSOE_API_KEY` when no key is passed. Override `base_url` to target a Self-Serve Deployment.
- **Platform.** The `Crusoe` backend ships in sglang 0.5.11 and later, which publish Linux wheels only. On macOS, pip resolves to 0.5.10.post1, the last universal wheel; `program.py` detects that and uses `sgl.OpenAI(model, base_url=..., api_key=...)`, which is the same backend with the endpoint spelled out. Run on Linux, or in Docker, to use `sgl.Crusoe` directly.
- Pass generation parameters through `sgl.gen(...)`: `max_tokens`, `temperature`, `top_p`, `stop`.
- Regex-constrained and choice-constrained generation (`sgl.gen(regex=...)`, `sgl.select`) need a local SGLang runtime; on the hosted endpoint, ask for JSON in the prompt or use the OpenAI SDK's `response_format` as in the [vLLM recipe](../vllm/client.py).

[Back to integrations](../README.md)
