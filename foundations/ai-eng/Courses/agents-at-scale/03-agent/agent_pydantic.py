import asyncio, os, sys
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from fastmcp import Client
from schemas import Brief
from models import model_for
from agent import SYSTEM, mcp_transport, render, brief_relpath

WORKSPACE = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()
MODEL_SPEC = os.environ.get("PYDANTIC_AI_MODEL", "crusoe:zai-org/GLM-5.3")

server = MCPToolset(Client(mcp_transport(WORKSPACE)))

brief_agent = Agent(
    model_for(MODEL_SPEC),
    output_type=Brief,
    system_prompt=SYSTEM,
    toolsets=[server],
    retries=2,
)


async def run(topic: str) -> Brief:
    async with brief_agent:
        result = await brief_agent.run(f"Write a brief on: {topic}")
        print(result.usage, file=sys.stderr)
        return result.output


async def main(topic: str) -> None:
    brief = await run(topic)
    path = WORKSPACE / brief_relpath(brief)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(brief))
    print(path.read_text())
    print(f"written: {path}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "MemoryAlloy KV cache"))
