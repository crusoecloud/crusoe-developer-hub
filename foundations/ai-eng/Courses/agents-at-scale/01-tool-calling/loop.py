import json, os, sys
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["CRUSOE_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
)
MODEL = os.environ.get("MODEL", "zai-org/GLM-5.3")
VERBOSE = os.environ.get("VERBOSE", "1") != "0"


def calc(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))


TOOLS = [{"type": "function", "function": {
    "name": "calc",
    "description": "Evaluate an arithmetic expression. Use it for all math; never do math in your head.",
    "parameters": {"type": "object",
        "properties": {"expression": {"type": "string", "description": "Python arithmetic, e.g. '17 * 23'"}},
        "required": ["expression"]}}}]
REGISTRY = {"calc": calc}


def run(user_msg: str, max_steps: int = 10, tools=None, registry=None) -> str:
    tools = TOOLS if tools is None else tools
    registry = REGISTRY if registry is None else registry
    messages = [{"role": "system", "content": "You are a precise assistant. Use tools when they apply."},
                {"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            fn = registry.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
                result = fn(**args) if fn else f"error: no tool named {call.function.name}"
            except Exception as e:
                result = f"error: {type(e).__name__}: {e}"
            if VERBOSE:
                print(f"[tool] {call.function.name}({call.function.arguments}) -> {result}", file=sys.stderr)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "stopped: step budget exhausted"


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:]) or "What is 17 * 23, and then that times 3?"))
