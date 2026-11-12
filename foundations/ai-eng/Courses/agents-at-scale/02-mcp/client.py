import asyncio, json, os, sys
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from loop import run_agent

SERVER = os.environ.get("MCP_URL") or PythonStdioTransport(
    str(Path(__file__).with_name("server.py")), env=dict(os.environ), python_cmd=sys.executable)


def to_openai_tools(mcp_tools):
    return [{"type": "function", "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema}} for t in mcp_tools]


async def main(task: str):
    async with Client(SERVER) as mcp:
        tools = to_openai_tools(await mcp.list_tools())

        async def dispatch(call):
            res = await mcp.call_tool(call.function.name, json.loads(call.function.arguments or "{}"))
            return res.data if res.data is not None else "\n".join(c.text for c in res.content)

        print(await run_agent(task, tools, dispatch))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Find three sources on the MemoryAlloy KV cache and save a summary to memoryalloy.md"))
