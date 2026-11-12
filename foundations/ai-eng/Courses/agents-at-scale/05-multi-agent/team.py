import asyncio, json, os, sys, time, tomllib
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from fastmcp import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
from schemas import Brief
from models import model_for
from agent import mcp_transport, render, slugify, brief_relpath
from config import settings
from cost import cost_usd
from trace import Usage
from shared import SLOTS

cfg = tomllib.loads((ROOT / "config.toml").read_text())
PROMPTS = Path(__file__).with_name("prompts")


class Plan(BaseModel):
    subquestions: list[str] = Field(min_length=3, max_length=6)


class Finding(BaseModel):
    claim: str
    url: str
    quote: str


class Verdict(BaseModel):
    passed: bool
    problems: list[str]


class NeedsHuman(Exception):
    def __init__(self, topic: str, problems: list[str]):
        super().__init__(f"{topic}: {problems}")
        self.topic, self.problems = topic, problems


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def dump(x) -> str:
    return json.dumps([i.model_dump(mode="json") for i in x] if isinstance(x, list) else x.model_dump(mode="json"), indent=1)


def build_team(workspace: Path) -> dict[str, Agent]:
    server = MCPToolset(Client(mcp_transport(workspace)))
    roles = cfg["role_model"]
    return {
        "planner": Agent(model_for(roles["planner"]), output_type=Plan, system_prompt=prompt("planner"), retries=3),
        "researcher": Agent(model_for(roles["researcher"]), output_type=list[Finding], system_prompt=prompt("researcher"), toolsets=[server], retries=3),
        "writer": Agent(model_for(roles["writer"]), output_type=Brief, system_prompt=prompt("writer"), retries=3),
        "verifier": Agent(model_for(roles["verifier"]), output_type=Verdict, system_prompt=prompt("verifier"), toolsets=[server], retries=3),
    }


class RoleUsage:
    def __init__(self):
        self.by_role: dict[str, Usage] = {}

    def add(self, role: str, run_usage) -> None:
        u = self.by_role.setdefault(role, Usage())
        u.prompt_tokens += getattr(run_usage, "input_tokens", None) or getattr(run_usage, "request_tokens", 0) or 0
        u.completion_tokens += getattr(run_usage, "output_tokens", None) or getattr(run_usage, "response_tokens", 0) or 0
        u.cached_tokens += getattr(run_usage, "cache_read_tokens", 0) or 0

    def cost(self) -> float:
        return sum(cost_usd(u, cfg["role_model"][role]) for role, u in self.by_role.items())

    def total(self) -> Usage:
        t = Usage()
        for u in self.by_role.values():
            t.prompt_tokens += u.prompt_tokens
            t.completion_tokens += u.completion_tokens
            t.cached_tokens += u.cached_tokens
        return t


async def run_team(topic: str, workspace: Path = Path("workspace"), usage: RoleUsage | None = None) -> Brief:
    usage = usage or RoleUsage()
    workspace = Path(workspace).resolve()
    team = build_team(workspace)
    planner, researcher, writer, verifier = (team[r] for r in ("planner", "researcher", "writer", "verifier"))
    async with researcher, verifier:
        plan_run = await planner.run(topic)
        usage.add("planner", plan_run.usage)
        plan = plan_run.output
        sem = asyncio.Semaphore(cfg["budget"]["max_concurrency"])

        async def research(q: str) -> list[Finding]:
            async with sem, SLOTS:
                r = await researcher.run(q)
                usage.add("researcher", r.usage)
                out = (workspace / "findings" / f"{slugify(q)}.json")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(dump(r.output))
                return r.output

        findings = [f for batch in await asyncio.gather(*(research(q) for q in plan.subquestions)) for f in batch]

        problems: list[str] = []
        for attempt in range(3):
            w = await writer.run(f"Topic: {topic}\nFindings:\n{dump(findings)}\nFix these problems: {problems}")
            usage.add("writer", w.usage)
            brief = w.output
            v = await verifier.run(f"Brief:\n{dump(brief)}\nFindings:\n{dump(findings)}")
            usage.add("verifier", v.usage)
            verdict = v.output
            if verdict.passed:
                return brief
            problems = verdict.problems
    raise NeedsHuman(topic, problems)


@dataclass
class TeamResult:
    run_id: str
    topic: str
    workdir: Path
    brief: Brief | None
    usage: Usage
    cost_usd: float
    seconds: float
    problems: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def brief_path(self) -> Path | None:
        return self.workdir / brief_relpath(self.brief) if self.brief else None

    @property
    def trace_path(self) -> Path:
        return self.workdir / "team.json"


async def run_team_traced(topic: str, workdir: Path) -> TeamResult:
    run_id = f"team_{uuid4().hex[:8]}"
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    usage = RoleUsage()
    t0 = time.time()
    brief, problems, error = None, [], None
    try:
        brief = await run_team(topic, workspace=workdir, usage=usage)
    except NeedsHuman as e:
        problems = e.problems
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    result = TeamResult(run_id, topic, workdir, brief, usage.total(), usage.cost(), time.time() - t0, problems, error)
    if brief:
        result.brief_path.parent.mkdir(parents=True, exist_ok=True)
        result.brief_path.write_text(render(brief))
        (workdir / "brief.json").write_text(brief.model_dump_json(indent=2))
    result.trace_path.write_text(json.dumps({
        "run_id": run_id, "topic": topic, "seconds": round(result.seconds, 1), "cost_usd": result.cost_usd,
        "usage_by_role": {r: u.as_dict() for r, u in usage.by_role.items()},
        "problems": problems, "error": error}, indent=2))
    return result


async def main(topic: str) -> None:
    result = await run_team_traced(topic, Path("runs") / "team" / slugify(topic))
    if result.error:
        print(f"failed: {result.error}")
        sys.exit(1)
    if result.brief is None:
        print("needs a human:")
        for p in result.problems:
            print(f"- {p}")
        sys.exit(2)
    print(result.brief_path.read_text())
    print(f"seconds={result.seconds:.1f} usd={result.cost_usd:.4f} usage={result.usage.as_dict()}")
    print(f"per-role usage: {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "MemoryAlloy KV cache"))
