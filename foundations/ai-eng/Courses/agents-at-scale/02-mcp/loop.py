import os, sys
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["CRUSOE_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
)
MODEL = os.environ.get("MODEL", "zai-org/GLM-5.3")
VERBOSE = os.environ.get("VERBOSE", "1") != "0"


async def run_agent(user_msg: str, tools: list[dict], dispatch, max_steps: int = 15) -> str:
    messages = [{"role": "system", "content": "You are a precise assistant. Use tools when they apply."},
                {"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        resp = await client.chat.completions.create(model=MODEL, messages=messages, tools=tools, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            try:
                result = await dispatch(call)
            except Exception as e:
                result = f"error: {type(e).__name__}: {e}"
            if VERBOSE:
                print(f"[tool] {call.function.name}({call.function.arguments[:120]}) -> {str(result)[:120]!r}", file=sys.stderr)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "stopped: step budget exhausted"
