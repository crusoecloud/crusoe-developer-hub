import asyncio, json, os, sys
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

SERVER = os.environ.get("MCP_URL") or PythonStdioTransport(
    str(Path(__file__).with_name("server.py")), env=dict(os.environ), python_cmd=sys.executable)


async def main(query: str | None):
    async with Client(SERVER) as mcp:
        for t in await mcp.list_tools():
            print(f"{t.name}: {t.description}")
            print("  ", json.dumps(t.input_schema))
        if query:
            res = await mcp.call_tool("web_search", {"query": query, "max_results": 3})
            print(json.dumps(res.data, indent=2))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or None))
