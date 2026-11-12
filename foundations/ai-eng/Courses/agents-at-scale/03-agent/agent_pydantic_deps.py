import asyncio, os, sys
from dataclasses import dataclass
from pathlib import Path
import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPToolset
from fastmcp import Client
from schemas import Brief
from models import model_for
from agent import SYSTEM, mcp_transport, render, brief_relpath

WORKSPACE = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()
MODEL_SPEC = os.environ.get("PYDANTIC_AI_MODEL", "crusoe:zai-org/GLM-5.3")


@dataclass
class Deps:
    workdir: Path
    http: httpx.AsyncClient


server = MCPToolset(Client(mcp_transport(WORKSPACE)))

Agent.instrument_all()
brief_agent = Agent(model_for(MODEL_SPEC), output_type=Brief, system_prompt=SYSTEM,
                    toolsets=[server], deps_type=Deps, retries=2)


@brief_agent.tool
async def read_notes(ctx: RunContext[Deps], name: str) -> str:
    """Read a note the user left in the workspace notes folder."""
    return (ctx.deps.workdir / "notes" / f"{name}.md").read_text()


async def main(topic: str) -> None:
    async with httpx.AsyncClient() as http, brief_agent:
        result = await brief_agent.run(f"Write a brief on: {topic}", deps=Deps(WORKSPACE, http))
    brief = result.output
    path = WORKSPACE / brief_relpath(brief)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(brief))
    print(result.usage, file=sys.stderr)
    print(f"written: {path}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "MemoryAlloy KV cache"))
