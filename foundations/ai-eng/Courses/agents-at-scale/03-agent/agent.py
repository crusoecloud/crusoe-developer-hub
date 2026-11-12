import asyncio, json, os, re, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import structlog
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from pydantic import ValidationError
from config import settings, client
from schemas import Brief
from trace import Trace, Usage, ToolRecord
from cost import cost_usd
from mcp_tools import to_openai_tools, make_dispatch
from streaming import complete_streaming

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "02-mcp" / "server.py"
SYSTEM = (Path(__file__).with_name("prompts") / "system.md").read_text()
STREAM = os.environ.get("STREAM", "0") == "1"
log = structlog.get_logger()

SUBMIT = {"type": "function", "function": {
    "name": "submit_brief",
    "description": "Call exactly once, when the brief is complete and every quote is verbatim from a fetched page.",
    "parameters": Brief.model_json_schema()}}


class StepBudgetExceeded(Exception):
    pass


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "brief"


def render(brief: Brief) -> str:
    body = "\n\n".join(brief.sections)
    srcs = "\n".join(f"- {s.url}\n  > {s.quote}" for s in brief.sources)
    return f"# {brief.title}\n\n{brief.summary}\n\n{body}\n\n## Sources\n{srcs}\n"


def brief_relpath(brief: Brief) -> str:
    return f"briefs/{slugify(brief.title)}.md"


async def dispatch_write(brief: Brief, trace: Trace, mcp) -> str:
    path = brief_relpath(brief)
    res = await mcp.call_tool("write_file", {"path": path, "content": render(brief)})
    result = res.data if res.data is not None else "\n".join(c.text for c in res.content)
    trace.tool_result(_FakeCall("write_file", json.dumps({"path": path})), {"path": path}, str(result))
    trace.note("wrote", path=path)
    return path


class _FakeCall:
    def __init__(self, name: str, arguments: str):
        self.id = f"call_{uuid4().hex[:6]}"
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


async def complete(model: str, messages: list[dict], tools: list[dict]):
    if STREAM:
        return await complete_streaming(model, messages, tools, echo=sys.stderr)
    resp = await client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
    return resp.choices[0].message, resp.usage


async def run_brief(topic: str, model: str, tools, dispatch, trace: Trace, mcp) -> Brief:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Write a brief on: {topic}"}]
    for step in range(settings.max_steps):
        t0 = time.time()
        msg, usage = await complete(model, messages, tools + [SUBMIT])
        trace.record(step, msg, usage, ms=int((time.time() - t0) * 1000), model=model)
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Finish by calling submit_brief."})
            continue
        for call in msg.tool_calls:
            if call.function.name == "submit_brief":
                try:
                    brief = Brief.model_validate_json(call.function.arguments)
                    await dispatch_write(brief, trace, mcp)
                    return brief
                except ValidationError as e:
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": f"invalid brief: {e}"})
                    continue
            result = await dispatch(call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    raise StepBudgetExceeded(topic)


@dataclass
class RunResult:
    run_id: str
    model: str
    workdir: Path
    brief: Brief | None
    usage: Usage
    seconds: float
    tool_calls: list[ToolRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def brief_path(self) -> Path | None:
        return self.workdir / brief_relpath(self.brief) if self.brief else None

    @property
    def trace_path(self) -> Path:
        return self.workdir / "trace.jsonl"

    @property
    def tool_results(self) -> list[ToolRecord]:
        return self.tool_calls

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.usage, self.model)

    def fetched(self) -> dict[str, str]:
        return {r.args.get("url", ""): r.result for r in self.tool_calls if r.name == "fetch_url"}

    def save(self) -> None:
        if self.brief:
            (self.workdir / "brief.json").write_text(self.brief.model_dump_json(indent=2))
        (self.workdir / "run.json").write_text(json.dumps({
            "run_id": self.run_id, "model": self.model, "seconds": round(self.seconds, 2),
            "usage": self.usage.as_dict(), "cost_usd": self.cost_usd, "error": self.error,
            "brief_path": str(self.brief_path) if self.brief_path else None}, indent=2))

    @classmethod
    def load(cls, workdir: Path) -> "RunResult":
        workdir = Path(workdir)
        meta = json.loads((workdir / "run.json").read_text())
        brief = Brief.model_validate_json((workdir / "brief.json").read_text()) if (workdir / "brief.json").exists() else None
        calls = []
        for line in (workdir / "trace.jsonl").read_text().splitlines():
            obj = json.loads(line)
            if isinstance(obj.get("tool"), dict):
                t = obj["tool"]
                try:
                    args = json.loads(t["arguments"])
                except ValueError:
                    args = {}
                calls.append(ToolRecord(t["id"], t["name"], t["arguments"], args, t["result"]))
        return cls(meta["run_id"], meta["model"], workdir, brief, Usage(**meta["usage"]), meta["seconds"], calls, meta.get("error"))


def mcp_transport(workdir: Path) -> PythonStdioTransport:
    return PythonStdioTransport(str(SERVER), env={**os.environ, "BRIEF_WORKSPACE": str(workdir)}, python_cmd=sys.executable)


async def run_brief_traced(topic: str, model: str, workdir: Path, run_fn=run_brief) -> RunResult:
    run_id = f"r_{uuid4().hex[:8]}"
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    trace = Trace(run_id, workdir / "trace.jsonl")
    brief, error = None, None
    with structlog.contextvars.bound_contextvars(run_id=run_id, model=model):
        log.info("run.start", topic=topic)
        async with Client(mcp_transport(workdir)) as mcp:
            tools = to_openai_tools(await mcp.list_tools())
            dispatch = make_dispatch(mcp, trace)
            try:
                brief = await run_fn(topic, model, tools, dispatch, trace, mcp)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                trace.note("failed", error=error)
        log.info("run.done", steps=trace.steps, usd=round(cost_usd(trace.usage, model), 4), s=round(trace.elapsed, 1), error=error)
    result = RunResult(run_id, model, workdir, brief, trace.usage, trace.elapsed, trace.tool_calls, error)
    trace.close()
    result.save()
    return result


async def main(topic: str) -> None:
    result = await run_brief_traced(topic, model=settings.model, workdir=Path("runs") / "single" / slugify(topic))
    if result.error:
        print(f"failed: {result.error}")
        print(f"trace: {result.trace_path}")
        sys.exit(1)
    print(result.brief_path.read_text())
    print(f"run_id={result.run_id} model={result.model} steps={len(result.tool_calls)} tools "
          f"usage={result.usage.as_dict()} seconds={result.seconds:.1f} usd={result.cost_usd:.4f}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "MemoryAlloy KV cache"))
