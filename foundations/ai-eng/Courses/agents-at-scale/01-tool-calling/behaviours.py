import asyncio, json, os, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import AsyncOpenAI
from loop import run, TOOLS, REGISTRY

client = AsyncOpenAI(
    api_key=os.environ["CRUSOE_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
)
MODEL = os.environ.get("MODEL", "zai-org/GLM-5.3")

CITIES = {"Accra": "Africa/Accra", "Seattle": "America/Los_Angeles", "London": "Europe/London", "Tokyo": "Asia/Tokyo"}


def get_time(city: str) -> str:
    return datetime.now(ZoneInfo(CITIES[city])).strftime("%H:%M on %A")


TIME_TOOL = {"type": "function", "function": {
    "name": "get_time",
    "description": "Current local time in a city. Call it once per city.",
    "parameters": {"type": "object",
        "properties": {"city": {"type": "string", "enum": list(CITIES)}},
        "required": ["city"]}}}


async def parallel_calls():
    messages = [{"role": "user", "content": "What time is it right now in Accra and in Seattle?"}]
    resp = await client.chat.completions.create(model=MODEL, messages=messages, tools=[TIME_TOOL], tool_choice="auto")
    msg = resp.choices[0].message
    print(f"tool calls in one assistant turn: {len(msg.tool_calls or [])}")
    for call in msg.tool_calls or []:
        print(f"  {call.function.name}({call.function.arguments})")


async def tool_choice_variants():
    messages = [{"role": "user", "content": "What is 12 * 12?"}]
    for choice in ("auto", "none", "required", {"type": "function", "function": {"name": "calc"}}):
        resp = await client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, tool_choice=choice)
        msg = resp.choices[0].message
        label = choice if isinstance(choice, str) else "named: calc"
        print(f"tool_choice={label!s:12} finish_reason={resp.choices[0].finish_reason:10} tool_calls={len(msg.tool_calls or [])}")


async def reasoning_content():
    resp = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "What is 17 * 23? Use the tool."}], tools=TOOLS)
    msg = resp.choices[0].message
    extra = msg.model_extra or {}
    field = next((k for k in ("reasoning_content", "reasoning") if extra.get(k)), None)
    print("reasoning field:", field or "none returned by this model")
    if field:
        print(str(extra[field])[:400])


async def streaming():
    calls = {}
    stream = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "What is 17 * 23? Use the tool."}], tools=TOOLS, stream=True)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
        for d in delta.tool_calls or []:
            c = calls.setdefault(d.index, {"id": d.id, "name": "", "arguments": ""})
            if d.function.name:
                c["name"] = d.function.name
            if d.function.arguments:
                c["arguments"] += d.function.arguments
    print()
    for c in calls.values():
        print(f"assembled call: {c['name']}({json.loads(c['arguments'])})")


DEMOS = {"parallel": parallel_calls, "tool_choice": tool_choice_variants, "reasoning": reasoning_content, "streaming": streaming}

async def main(names: list[str]) -> None:
    for name in names:
        print(f"== {name} ==")
        await DEMOS[name]()
        print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(DEMOS)))
