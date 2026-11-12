# Chapter 3: Building a production agent on open models

The chapter 1 loop, the chapter 2 tools, a typed final answer and a trace. Spec first, then the output contract, then the agent, then the plumbing that makes it a service, then the same agent on Pydantic AI.

## Files

| file | what it shows |
|---|---|
| `SPEC.md` | The spec, written before the code. The done condition on the last four lines becomes the eval in chapter 4, the verifier's job in chapter 5, a hook in chapter 6 and the stop condition in chapter 7. |
| `prompts/system.md` | The system prompt, seven lines. Tool descriptions carry most of the instruction load. |
| `schemas.py` | The output contract: `Source` with a verbatim `quote`, `Brief` with at least three sources. A quote turns a judgment call into a substring search. |
| `agent.py` | `run_brief`, the loop with a `submit_brief` final-answer tool; `render` and `dispatch_write`, which write the brief through MCP; `run_brief_traced`, the one entry point the evals and the nightly loop also call; `RunResult`, what a run leaves behind. |
| `trace.py` | One JSONL line per step and per tool result, flushed immediately. Not logging: the eval dataset, the bill, and later the fine-tuning data. |
| `config.py` | Pydantic Settings: key, base URL, model, fallbacks, timeout, retries, step budget, concurrency. Every value comes from the environment. |
| `cost.py`, `prices.toml` | Dollars per run from the usage block: uncached input, cached input, output. Fill the prices in from the pricing page and record the date. |
| `streaming.py` | The streaming variant of the model call: content tokens printed as they arrive, tool-call deltas assembled by index. |
| `structured_output.py` | The other way to get typed output: `response_format` with a JSON schema, for a single extraction call with no tools. |
| `agent_pydantic.py` | The port: `Agent(model, output_type=Brief, toolsets=[MCPToolset(...)], retries=2)`. About twenty lines. |
| `agent_pydantic_deps.py` | Two additions: a native tool with typed dependencies through `RunContext`, and `Agent.instrument_all()` for OpenTelemetry spans. |
| `models.py` | Resolves a `crusoe:<model id>` string to the Crusoe provider shipped with Pydantic AI, and, for the Nemotron models, turns off strict tool definitions and forced tool choice: the endpoint answers a 500 to `strict: true` on a tool and to `tool_choice: required`. |
| `shared.py` | One process-wide semaphore, sized to the concurrency quota, imported by the service, the team and the evals. |
| `service.py`, `Dockerfile` | A stateless FastAPI wrapper: accept a topic, return a run id, do the work in the background, poll a status URL. One workspace per request, results copied out, workspace deleted. |
| `vllm.sh` | The no lock-in proof: the same agent against a vLLM server on your own GPU, by changing two environment variables. |

## Run

From the repository root:

```bash
uv run --env-file .env python 03-agent/agent.py "MemoryAlloy KV cache"
STREAM=1 uv run --env-file .env python 03-agent/agent.py "MemoryAlloy KV cache"
MODEL=deepseek-ai/DeepSeek-V4.1-Flash uv run --env-file .env python 03-agent/agent.py "MemoryAlloy KV cache"
MODEL=Qwen/Qwen3.8-27B uv run --env-file .env python 03-agent/agent.py "MemoryAlloy KV cache"
uv run --env-file .env python 03-agent/structured_output.py
uv run --env-file .env python 03-agent/agent_pydantic.py "MemoryAlloy KV cache"
uv run --env-file .env python 03-agent/agent_pydantic_deps.py "MemoryAlloy KV cache"
```

Each run of `agent.py` leaves `runs/single/<slug>/` with `trace.jsonl`, `brief.json`, `run.json` and the rendered brief under `briefs/`.

The service:

```bash
uv run --env-file .env uvicorn service:app --app-dir 03-agent --port 8080
```

```bash
curl -X POST localhost:8080/brief -H 'Content-Type: application/json' -d '{"topic": "MemoryAlloy KV cache"}'
```

Then poll the `status_url` from the response until it flips from `pending` to `done`. Results land in `results/runs/<run id>/`.

With Docker:

```bash
docker build -t brief-service .
docker run --env-file .env -p 8080:8080 brief-service
```

## Environment variables

| variable | required | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | yes | read by `config.py` |
| `LLM_BASE_URL` | no | defaults to the Crusoe endpoint |
| `MODEL` | no | the primary model id; defaults to `zai-org/GLM-5.3` |
| `FALLBACK_MODELS` | no | JSON list; used by chapter 6 |
| `MAX_STEPS`, `MAX_CONCURRENCY`, `REQUEST_TIMEOUT_S`, `MAX_RETRIES` | no | the rest of `Settings` |
| `STREAM` | no | `1` streams tokens to stderr while the agent works |
| `PYDANTIC_AI_MODEL` | no | the `provider:model` string for the Pydantic AI scripts; defaults to `crusoe:zai-org/GLM-5.3` |
| `TAVILY_API_KEY` | yes | the search API behind the MCP server |
| `RESULTS_DIR` | no | where the service copies finished runs; defaults to `results/runs` |

## Notes

- The course was recorded on Pydantic AI 1.x, where the MCP server is `MCPServerStdio(...)` and instrumentation is `instrument=True` on the agent. This repository pins Pydantic AI 2.x because FastMCP 4 and Pydantic AI 1.x need incompatible versions of the MCP SDK. In 2.x the same server is `MCPToolset(Client(transport))` and instrumentation is `Agent.instrument_all()`. Nothing else in the port changed.
- `cost.py` returns zero until `prices.toml` is filled in. Every run still records the token counts the cost is computed from.
- The trace records every tool result in full, so a run loaded back from disk grades exactly like the live run did. Fetched pages are capped at 20,000 characters by `fetch_url`, so a trace is a few hundred kilobytes at most.
- `vllm.sh serve` is meant to run on the GPU VM; `vllm.sh` with no argument runs the agent from your laptop against it. Look up the tool-call parser for the model family in the vLLM docs rather than guessing.

## Verified

Ran against the endpoint on 5 November 2026: `agent.py` wrote a grounded brief on MemoryAlloy in eight steps with 20k of 32k prompt tokens served from cache; `structured_output.py` returned a valid `Brief` through `response_format`; `agent_pydantic.py` and `agent_pydantic_deps.py` produced briefs through the Crusoe provider with the MCP toolset; `service.py` accepted a topic, returned a run id and reported `done` with the brief on the status URL. `vllm.sh` was not run, since it needs a GPU VM.
