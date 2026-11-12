import json, os, sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
from config import client

WORK = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()
COMPACTION_MODEL = os.environ.get("COMPACTION_MODEL", "nvidia/Nemotron-3.5-Lightning-30B-A3B")


def offload(result: str, tool: str, keep: int = 1500, work: Path = WORK) -> str:
    if len(result) <= 2 * keep:
        return result
    path = work / "tool_out" / f"{tool}-{uuid4().hex[:8]}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result)
    rel = path.relative_to(work)
    return (f"{result[:keep]}\n\n[... {len(result):,} chars total; full output at {rel}; "
            f"call read_file(path, start_line, end_line) for more ...]\n\n{result[-keep:]}")


def count_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


async def summarize(messages: list[dict], model: str = COMPACTION_MODEL) -> str:
    resp = await client.chat.completions.create(model=model, temperature=0, messages=[
        {"role": "system", "content": "Summarise an agent's work so far for the agent itself. Report: the plan, decisions already made, "
                                      "every URL fetched with one line on what it contained, and open questions. No narrative."},
        {"role": "user", "content": json.dumps(messages)[:120000]}])
    return resp.choices[0].message.content or ""


async def compact(messages: list[dict], window: int) -> list[dict]:
    if count_tokens(messages) < 0.7 * window or len(messages) <= 8:
        return messages
    head, middle, tail = messages[:2], messages[2:-6], messages[-6:]
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    summary = await summarize(middle + messages[-6:len(messages) - len(tail)])
    return head + [{"role": "user", "content": f"[Summary of earlier work]\n{summary}"}] + tail
