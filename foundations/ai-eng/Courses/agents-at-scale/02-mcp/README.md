# Chapter 2: MCP explained

Tool calling is the contract between the model and your app. MCP is the contract between your app and whoever provides the tools: discovery, invocation and transport. One server can serve many agents written in any framework. This chapter builds the server the capstone needs and connects the chapter 1 loop to it with a ten-line adapter.

## Files

| file | what it shows |
|---|---|
| `server.py` | A FastMCP server with the four capstone tools: `web_search`, `fetch_url`, `write_file`, `read_file`. Type hints become the input schema, docstrings become the descriptions. `write_file` and `read_file` refuse paths outside the workspace. |
| `search.py` | The search adapter behind `web_search`: Tavily by default. A Brave Search adapter is also there; set `SEARCH_PROVIDER=brave` and `BRAVE_API_KEY` to use it. |
| `loop.py` | The chapter 1 loop made async, taking `tools` and a `dispatch` function instead of a registry. The loop body did not change. |
| `client.py` | The adapter: list the MCP tools, map `name`, `description` and `input_schema` into OpenAI tool definitions, dispatch `tool_calls` to `call_tool`. The stdio transport is built with the full environment, because the MCP SDK starts servers with a minimal one by default and the search key would not reach `server.py`. |
| `inspect_tools.py` | Lists the tools and their generated schemas, and calls `web_search` by hand. Tools are normal code and can be tested without a model. |

## Run

From the repository root:

```bash
uv run --env-file .env python 02-mcp/inspect_tools.py "MemoryAlloy KV cache"
uv run --env-file .env python 02-mcp/client.py "Find three sources on the MemoryAlloy KV cache and save a summary to memoryalloy.md"
```

The file lands in `workspace/`. Set `BRIEF_WORKSPACE` to put it somewhere else.

Remote transport, in two terminals:

```bash
uv run --env-file .env python 02-mcp/server.py --http
```

```bash
MCP_URL=http://127.0.0.1:8000/mcp uv run --env-file .env python 02-mcp/client.py
```

The server log shows `Mcp-Method` and `Mcp-Name` headers on each request, which is what a gateway or rate limiter sees. Start a second instance with `MCP_PORT=8001` and put any round-robin proxy in front: because the core protocol is stateless, requests can land on either instance.

The MCP Inspector works against the stdio server as well:

```bash
npx @modelcontextprotocol/inspector uv run --env-file .env python 02-mcp/server.py
```

## Environment variables

| variable | required | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | yes | model calls from `client.py` |
| `TAVILY_API_KEY` | yes | the search API behind `web_search` |
| `SEARCH_PROVIDER` | no | `tavily` (default) or `brave` |
| `BRIEF_WORKSPACE` | no | where `write_file` and `read_file` operate; defaults to `workspace/` |
| `MCP_URL` | no | point `client.py` at a running HTTP server instead of spawning the stdio one |
| `MCP_PORT` | no | port for `server.py --http`; defaults to 8000 |

## What changed on 28 July 2026

Six changes in the 2026-07-28 specification that matter at scale: a stateless core (no initialize handshake, no session header), `Mcp-Method` and `Mcp-Name` HTTP headers for routing, Multi Round-Trip Requests instead of server-initiated elicitation, cacheable `tools/list` results with `ttlMs` and a deterministic order, Tasks as a formal extension for long-running work, and deprecations of Roots, Sampling, Logging and the HTTP+SSE transport.

A Multi Round-Trip Request looks like this on the wire. The first response to `tools/call`:

```json
{"resultType": "input_required",
 "requests": [{"id": "q1", "message": "This will delete 3 files. Continue?", "schema": {"type": "boolean"}}]}
```

The client asks the user, then retries the same call with the answers attached:

```json
{"method": "tools/call", "params": {"name": "cleanup", "arguments": {},
 "inputResponses": {"q1": true}}}
```

The course names the schema field `inputSchema`; the MCP SDK v2 that FastMCP 4 builds on renamed it to `input_schema`, which is what the adapter reads.

## Two things to remember

Every tool description from every server you connect is text the model reads on every turn. The course demonstrates a poisoned `web_search` description that tells the model to include the contents of `.env` in its query, and the model obeys. That description is deliberately not in this repository. Pin servers to a version, read descriptions the way you would read a post-install script, allowlist tools per agent, and watch for description changes in updates.

Agents are a prefill problem. Fetch the tool list once per run, respect the TTL, keep the list small and the order fixed, so the prompt prefix stays byte-identical and the serving layer's prefix cache hits. Chapter 3 records `cached_tokens` from the usage block so you can see it working.

## Verified

Ran against the endpoint on 5 November 2026: `inspect_tools.py` lists the four tools with their generated schemas and returns Tavily results for a query; `client.py` ran the chapter 1 loop over MCP with two searches, three fetches and one `write_file`, and left `workspace/memoryalloy.md`. The HTTP transport and the two-instance round-robin were not exercised.
