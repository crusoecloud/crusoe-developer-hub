# Chapter 6: Why agents need harnesses

Agent = model + harness. Every harness component encodes an assumption about what the model cannot do on its own. The course breaks the chapter 3 agent five ways and fixes each one with a component. When the model gets better, the component comes out. Harnesses do not shrink; they move.

## Files

| file | failure | component |
|---|---|---|
| `context.py` | A 2,000-line page fetched four times; 26k tokens of prompt and the agent loses its plan. | `offload` keeps the head and tail of a big tool result and writes the rest to disk for `read_file`. `compact` summarises the middle of the conversation with a cheap model past 70 percent of the window. |
| `control.py` | The same `web_search` eight times. Twenty concurrent runs, 429s and timeouts. | `LoopGuard` nudges on the second identical call and raises `Stuck` on the third. `complete` retries with exponential backoff and jitter on rate limits, server errors and timeouts only. `complete_with_fallback` walks the fallback chain in config. `make_dispatch_once` makes a retried turn idempotent by tool call id. |
| `hooks.py` | A brief that validated, with a quote that is not on the page. | `after_tool` runs the chapter 4 grounding check after every `write_file`: silent on success, verbose on failure. `before_tool` blocks writes outside the workspace at the harness, not only in the server. |
| `sandbox.py` | A `run_python` tool that runs model-written code on the host, next to your keys. | A fresh Docker container per call: no network, no environment, read-only workspace, memory, CPU and PID limits, a timeout, truncated output. |
| `loop.py` | | The chapter 3 loop with every component wired around the one dispatch line. The model never sees an exception; it sees a tool result that says what happened. |
| `../AGENTS.md` | | The ratchet: every line traces to a failure shown here or in chapter 4. Under 60 lines. Delete a line when the trace shows it has not fired in a month. |

## Run

From the repository root:

```bash
uv run --env-file .env python 06-harness/loop.py "MemoryAlloy KV cache"
```

The trace in `runs/harness/<slug>/trace.jsonl` records `loop_guard` and `verify_failed` events, and the `model` field on each step shows when a fallback served the call.

The sandbox proofs, with Docker running:

```bash
uv run python 06-harness/sandbox.py
```

That runs four pieces of code inside the container: list the environment (empty), look for secrets (none), reach example.com (blocked), fork 500 times (killed by the PID limit). Pass your own code as the argument to run it in the same box.

## Environment variables

| variable | required | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | yes | model calls |
| `MODEL`, `FALLBACK_MODELS` | no | the fallback chain; defaults are `zai-org/GLM-5.3`, then `deepseek-ai/DeepSeek-V4.1-Flash`, then `openai/gpt-oss-120b` |
| `COMPACTION_MODEL` | no | the cheap summariser; defaults to `nvidia/Nemotron-3.5-Lightning-30B-A3B` |
| `CONTEXT_WINDOW` | no | tokens; compaction starts at 70 percent of it. Defaults to 128000 |
| `SANDBOX_IMAGE` | no | the container image; defaults to `python:3.12-slim`. Build your own with the packages the agent may use. |
| `TAVILY_API_KEY` | yes | the search API behind the MCP server |

## Notes

- The naive, host-side `run_python` from the course is deliberately not in this repository. Its entire body was a `subprocess.run(["python", "-c", code])` on the host. The sandbox is the tool.
- The course's compaction summariser was Nemotron 3 Nano. That model was deprecated on 2 November 2026 with Nemotron 3.5 Lightning as the replacement, which is why the default here is Lightning.
- `count_tokens` uses the four-characters-per-token estimate from the course. The threshold is 70 percent of the window on purpose, so being within ten percent is enough.

## Verified

`loop.py` ran against the endpoint on 5 November 2026 and wrote a brief whose every quote passed the grounding hook, with the loop guard firing once and 197k of 224k prompt tokens served from cache. `sandbox.py` ran against Docker: the container sees only the image's own environment variables, no secrets, name resolution fails for example.com, and the fork proof dies at the PID limit. `offload`, `LoopGuard` and the hooks were also exercised offline.
