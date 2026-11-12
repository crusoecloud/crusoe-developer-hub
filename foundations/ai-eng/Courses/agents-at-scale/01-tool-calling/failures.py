import json
from types import SimpleNamespace


def calc(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))


REGISTRY = {"calc": calc}


def fake_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def dispatch(call) -> dict:
    fn = REGISTRY.get(call.function.name)
    try:
        args = json.loads(call.function.arguments or "{}")
        result = fn(**args) if fn else f"error: no tool named {call.function.name}"
    except Exception as e:
        result = f"error: {type(e).__name__}: {e}"
    return {"role": "tool", "tool_call_id": call.id, "content": str(result)}


FAILURES = [
    ("failure 1 of 3: malformed arguments", fake_call("call_1", "calc", '{"expression": "17 * 23",}')),
    ("failure 2 of 3: hallucinated tool", fake_call("call_2", "get_weather", '{"city": "Accra"}')),
    ("failure 3 of 3: missing required argument", fake_call("call_3", "calc", "{}")),
    ("a good call, for contrast", fake_call("call_4", "calc", '{"expression": "17 * 23"}')),
]

if __name__ == "__main__":
    for label, call in FAILURES:
        print(label)
        print("  ", dispatch(call))
