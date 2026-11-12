# Agents at Scale (with Open Models)

Code for the course by Emmanuel Acheampong: production agents on open-weight models, from a hand-written tool-calling loop to a harnessed, evaluated, multi-agent system that runs itself on a schedule.

Every model call goes through one OpenAI-compatible endpoint, Crusoe Managed Inference at `https://api.inference.crusoecloud.com/v1`, and every model is open-weight. The base URL and the model id are config, never constants, so the same code runs on the managed endpoint or on your own vLLM server.

The spine of the course is one project, `brief`: give it a topic, it searches the web, reads pages, writes a markdown brief with at least three grounded citations, and saves it to disk. Each chapter wraps something new around the same loop.

## Course outline

| time | chapter | folder |
|---|---|---|
| 0:00:00 | Open: what you will build | |
| 0:05:00 | 1. Tool calling explained | [01-tool-calling](01-tool-calling/) |
| 0:25:00 | 2. MCP explained | [02-mcp](02-mcp/) |
| 0:45:00 | 3. Building a production agent on open models | [03-agent](03-agent/) |
| 1:20:00 | 4. Agent evals | [04-evals](04-evals/) |
| 1:45:00 | 5. Multi-agent systems | [05-multi-agent](05-multi-agent/) |
| 2:10:00 | 6. Why agents need harnesses | [06-harness](06-harness/) |
| 2:35:00 | 7. Loop engineering | [07-loops](07-loops/) |
| 2:55:00 | Close: what you can do now | |

Chapters 1 to 3 build. Chapter 4 proves. Chapters 5 to 7 scale.

## Setup

You need Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop for the chapter 6 sandbox, and an API key for the endpoint. Crusoe gives new accounts introductory credits; create the key in the Crusoe console under Intelligence API keys.

This folder lives inside the Crusoe Developer Hub repository under `foundations/ai-eng/Courses/agents-at-scale`. Clone the hub and work from this folder:

```bash
git clone https://github.com/crusoecloud/crusoe-developer-hub
```

```bash
cd crusoe-developer-hub/foundations/ai-eng/Courses/agents-at-scale
```

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your key in `.env` as `CRUSOE_API_KEY`, and a Tavily key as `TAVILY_API_KEY` for the search tool the chapter 2 server exposes. If your shell already exports `CRUSOE_API_KEY`, `uv run --env-file` keeps the shell's value and ignores the file, so unset the old export first. Then run any snippet from the repository root with `uv run --env-file .env python <path>`; every folder's README lists its commands.

Without uv:

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

## Lessons

| folder | what you build | run first |
|---|---|---|
| [01-tool-calling](01-tool-calling/) | One curl call, the thirty-line agent loop, descriptions as prompts, parallel calls, `tool_choice`, streaming deltas, three failures survived | `uv run --env-file .env python 01-tool-calling/loop.py` |
| [02-mcp](02-mcp/) | A FastMCP server with `web_search`, `fetch_url`, `write_file`, `read_file`; the ten-line adapter that connects the loop to it; stdio and Streamable HTTP | `uv run --env-file .env python 02-mcp/client.py` |
| [03-agent](03-agent/) | The spec, the typed output contract, `submit_brief`, the JSONL trace, config, cost, the service wrapper, the port to Pydantic AI, the vLLM proof | `uv run --env-file .env python 03-agent/agent.py "MemoryAlloy KV cache"` |
| [04-evals](04-evals/) | Twenty tasks, six deterministic graders, pass^k, a results table, a calibrated judge, trajectory checks, the PR gate | `uv run --env-file .env python 04-evals/run.py --tasks 04-evals/smoke.jsonl --k 1` |
| [05-multi-agent](05-multi-agent/) | Planner, parallel researchers, writer, verifier; typed handoffs; a different model per role | `uv run --env-file .env python 05-multi-agent/team.py "MemoryAlloy KV cache"` |
| [06-harness](06-harness/) | Context offloading and compaction, a loop guard, backoff and a fallback chain, verification hooks, a Docker sandbox | `uv run --env-file .env python 06-harness/loop.py "MemoryAlloy KV cache"` |
| [07-loops](07-loops/) | Triggered runs, durable state, budgets, machine-checkable stop conditions, escalation, a kill switch, the nightly workflow | `uv run --env-file .env python 07-loops/loop.py` |

Root files: [AGENTS.md](AGENTS.md) holds the agent conventions, every line traced to a failure from chapters 4 and 6. [config.toml](config.toml) holds the model per role, the fallbacks per role and the budgets. `.github/workflows/` holds the pull-request smoke eval and the nightly loop as GitHub Actions workflows; copy them to the root of your own repository to use them.

## Environment variables

| variable | used by | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | everything that calls a model | bearer token for the endpoint |
| `LLM_BASE_URL` | all chapters | defaults to `https://api.inference.crusoecloud.com/v1`; point it at vLLM to leave the provider |
| `MODEL` | all chapters | the primary model id; defaults to `zai-org/GLM-5.3` |
| `TAVILY_API_KEY` | chapters 2 to 7 | the search API behind `web_search` |
| `BRIEF_WORKSPACE` | chapters 2 to 7 | where the MCP server reads and writes; defaults to `workspace/` |
| `BRIEF_DRY_RUN` | chapter 7 | `1` by default; set `0` to publish and escalate for real |

Each folder README lists the variables specific to it.

## Models

The ids below are what `GET /v1/models` on the endpoint returned on 5 November 2026, spelled exactly as the endpoint spells them. The catalog changes; the course treats that as the point. Check the current list on the Crusoe docs available-models page, or with `curl $LLM_BASE_URL/models -H "Authorization: Bearer $CRUSOE_API_KEY"`, and change the id in config rather than in code.

| model id | role in the course |
|---|---|
| `zai-org/GLM-5.3` | primary agent model; planner and writer in chapter 5 |
| `zai-org/GLM-5.3-Flash` | judge in chapter 4 when the agent is not GLM |
| `deepseek-ai/DeepSeek-V4.1-Flash` | verifier in chapter 5; first fallback in chapter 6; comparison model in chapter 3 |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | researcher workers in chapter 5; compaction summariser in chapter 6 |
| `Qwen/Qwen3.8-27B` | the small dense model for the chapter 1 and 3 comparisons |
| `openai/gpt-oss-120b` | second fallback in chapter 6 |
| `google/gemma-4-31b-it` | optional extra row on the eval matrix |

Nemotron 3 Nano, the compaction model in the recorded course, was deprecated on 2 November 2026 with Nemotron 3.5 Lightning as its replacement, so Lightning is the default here.

## Layout

```
agents-at-scale/
├── README.md
├── AGENTS.md
├── config.toml
├── pyproject.toml  uv.lock  requirements.txt
├── .env.example
├── 01-tool-calling/
├── 02-mcp/
├── 03-agent/
├── 04-evals/
├── 05-multi-agent/
├── 06-harness/
├── 07-loops/
└── .github/workflows/   eval.yml (PR smoke eval), brief.yml (nightly loop)
```

Runs, workspaces, state and eval results are written next to the code and ignored by git.

## License

MIT. See [LICENSE](LICENSE).
