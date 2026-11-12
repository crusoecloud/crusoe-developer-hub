import asyncio, json, os, sys, time
from pathlib import Path
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
sys.path.insert(0, str(ROOT / "04-evals"))
from config import settings
from schemas import Brief
from agent import SYSTEM, SUBMIT, StepBudgetExceeded, dispatch_write, run_brief_traced, slugify
from graders import grounding_problems as brief_grounding_problems
from context import offload, compact
from control import LoopGuard, complete_with_fallback, make_dispatch_once
from hooks import Blocked, before_tool, after_tool


async def run_brief_harnessed(topic: str, model: str, tools, dispatch, trace, mcp) -> Brief:
    work = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()
    guard = LoopGuard()
    dispatch_once = make_dispatch_once(dispatch)
    fetched: dict[str, str] = {}
    models = [model, *settings.fallback_models]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Write a brief on: {topic}"}]
    for step in range(settings.max_steps):
        messages = await compact(messages, window=settings.context_window)
        t0 = time.time()
        resp = await complete_with_fallback(messages, tools + [SUBMIT], models)
        msg = resp.choices[0].message
        trace.record(step, msg, resp.usage, ms=int((time.time() - t0) * 1000), model=resp.model)
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Finish by calling submit_brief."})
            continue
        for call in msg.tool_calls:
            name = call.function.name
            if name == "submit_brief":
                try:
                    brief = Brief.model_validate_json(call.function.arguments)
                except ValidationError as e:
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": f"invalid brief: {e}"})
                    continue
                problems = brief_grounding_problems(brief, fetched)
                if problems:
                    trace.note("verify_failed", problems=problems)
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": "VERIFY FAILED, fix before finishing:\n- " + "\n- ".join(problems)})
                    continue
                await dispatch_write(brief, trace, mcp)
                return brief
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = {}
            nudge = guard.check(call)
            if nudge:
                trace.note("loop_guard", name=name)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": nudge})
                continue
            try:
                before_tool(name, args, work)
                result = await dispatch_once(call)
                if name == "fetch_url" and "url" in args:
                    fetched[args["url"]] = result
                result = offload(result, name, work=work)
                result = after_tool(name, args, result, fetched, work) or result
            except Blocked as e:
                result = f"blocked by policy: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    raise StepBudgetExceeded(topic)


async def main(topic: str) -> None:
    workdir = Path("runs") / "harness" / slugify(topic)
    os.environ["BRIEF_WORKSPACE"] = str(workdir.resolve())
    result = await run_brief_traced(topic, model=settings.model, workdir=workdir, run_fn=run_brief_harnessed)
    if result.error:
        print(f"failed: {result.error}")
        print(f"trace: {result.trace_path}")
        sys.exit(1)
    print(result.brief_path.read_text())
    print(f"usage={result.usage.as_dict()} seconds={result.seconds:.1f} usd={result.cost_usd:.4f}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "MemoryAlloy KV cache"))
