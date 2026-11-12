# Chapter 5: Multi-agent systems

Split an agent for one of four reasons: context isolation, parallelism, specialisation, or model tiering. Do not split if you cannot name which one applies. Handoffs are typed Pydantic models, never transcripts. The verifier is a separate agent on a different model family. The chapter 4 evals decide whether the split helped.

## Files

| file | what it shows |
|---|---|
| `team.py` | Four Pydantic AI agents: planner, researcher, writer, verifier. `Plan`, `Finding` and `Verdict` are the handoff contracts. `run_team` fans the researchers out under a semaphore, then loops writer and verifier up to three times before raising `NeedsHuman`. `run_team_traced` wraps a run with per-role usage and cost; chapter 7 calls it. |
| `prompts/` | One short prompt per role. The verifier's "You do not trust either" sentence does most of the work. |
| `../config.toml` | The model per role, and the fallbacks per role that chapter 6 uses. Change a role's model, rerun the evals. |

## Run

From the repository root:

```bash
uv run --env-file .env python 05-multi-agent/team.py "MemoryAlloy KV cache"
```

Each researcher's findings land in `runs/team/<slug>/findings/<subquestion>.json` before they reach the writer, so you can read exactly what the writer was given. The brief lands under `briefs/` in the same directory, and `team.json` holds per-role token usage and cost.

## Environment variables

| variable | required | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | yes | all four roles |
| `TAVILY_API_KEY` | yes | the researchers and the verifier search and fetch over MCP |
| `MAX_CONCURRENCY` | no | the process-wide semaphore from chapter 3; the team's own fan-out limit comes from `config.toml` |

## Notes

- The researchers and the verifier get the MCP server as `MCPToolset(Client(transport))`, the Pydantic AI 2.x spelling of the course's `MCPServerStdio`. See the chapter 3 README.
- Pydantic AI sends every tool definition with `strict: true` and forces a tool call with `tool_choice: required` when the output comes through a tool. The endpoint returns a 500 for both on Nemotron 3.5 Lightning, so `03-agent/models.py` clears both flags for `nvidia/` models; the researchers still get the argument repair loop, only without constrained decoding, and the output tool is requested by prompt instead of forced.

## Scale notes

- One process-wide semaphore, `03-agent/shared.py`, bounds the request rate across the service, the team and the evals. Per-team limits add up; a process-wide one does not.
- All researchers share a byte-identical system prompt and tool list, so the serving layer's prefix cache serves the prefill once. The subquestion goes in the user message, after the shared prefix.
- The telephone game variant from the course, where researchers return whole transcripts, is described rather than shipped: the writer's prompt grows by tens of thousands of tokens and it cites a URL nobody fetched.

## Verified

`team.py` ran against the endpoint on 5 November 2026 to a verified brief on MemoryAlloy in 519 seconds: the planner produced the subquestions, the researchers on Nemotron 3.5 Lightning fanned out and wrote their findings files, the writer and verifier converged, and `team.json` recorded the usage per role, with the verifier's re-fetches served mostly from cache. Two request shapes the endpoint rejects on Nemotron, strict tool definitions and `tool_choice: required`, are cleared by `03-agent/models.py`.
