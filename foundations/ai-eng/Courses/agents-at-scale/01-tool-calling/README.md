# Chapter 1: Tool calling explained

A tool call is text. The model emits a string in the format its chat template taught it, the serving layer parses it into the `tool_calls` shape, and your code runs the function and sends the result back. The loop that does this is the agent. Everything else in the course wraps around it.

## Files

| file | what it shows |
|---|---|
| `call.sh` | One chat completion with a `calc` tool, with curl, so nothing sits between you and the wire. Look at `finish_reason`, `content: null` and the `arguments` string. |
| `loop.py` | The agent loop in about thirty lines: call, check for tool calls, execute, append a `tool` message with the matching `tool_call_id`, repeat, return when the model answers in prose. |
| `descriptions.py` | Descriptions are prompts. The same question against vague and precise tool descriptions, plus a `convert_units` schema generated from a Pydantic model. |
| `behaviours.py` | Parallel tool calls in one turn, the four `tool_choice` settings, `reasoning_content` on reasoning models, and assembling streamed tool-call deltas by index. |
| `failures.py` | Three failures on purpose: malformed arguments, a hallucinated tool, a missing required argument. Runs without a key; it exercises the try/except that keeps the loop alive. |
| `compare_models.sh` | The same question on two open models. Same code, different habits. |

## Run

From the repository root, with `.env` filled in:

```bash
uv run --env-file .env bash 01-tool-calling/call.sh
uv run --env-file .env python 01-tool-calling/loop.py
uv run --env-file .env python 01-tool-calling/descriptions.py
uv run --env-file .env python 01-tool-calling/behaviours.py
uv run python 01-tool-calling/failures.py
uv run --env-file .env bash 01-tool-calling/compare_models.sh
```

Set `VERBOSE=0` to hide the per-call trace that `loop.py` prints to stderr.

## Environment variables

| variable | required | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | yes (except `failures.py`) | bearer token for the endpoint |
| `LLM_BASE_URL` | no | defaults to `https://api.inference.crusoecloud.com/v1` |
| `MODEL` | no | defaults to `zai-org/GLM-5.3`; the model id is config, never a constant |

## Notes

- The `calc` tool uses `eval`. It is a teaching trap, named as such in the course. Chapter 6 replaces it with a sandbox. Do not ship it.
- `behaviours.py reasoning` only prints reasoning when the model returns a `reasoning_content` field. Whether to send that field back on the next turn depends on the model family; read the model card.
- `failures.py` reproduces the three failures by constructing the tool calls directly instead of replaying cached model responses, so it runs offline.

## Verified

All six files ran against the endpoint on 5 November 2026 with `zai-org/GLM-5.3`: `call.sh` returned `finish_reason: tool_calls` with the arguments string; `loop.py` made two sequential `calc` calls; `behaviours.py` showed two parallel calls in one turn, the four `tool_choice` settings, a `reasoning` field and assembled streaming deltas; `failures.py` produced the three error tool results offline; `compare_models.sh` showed Qwen3.8-27B making one combined call where GLM-5.3 made two. With the vague descriptions, GLM-5.3 still chose `convert_units`, so this model does not reproduce the wrong-tool moment from the course.
