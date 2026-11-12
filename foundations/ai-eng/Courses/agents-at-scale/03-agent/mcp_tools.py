import json


def to_openai_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema}} for t in mcp_tools]


def make_dispatch(mcp, trace):
    async def dispatch(call) -> str:
        args = {}
        try:
            args = json.loads(call.function.arguments or "{}")
            res = await mcp.call_tool(call.function.name, args)
            result = res.data if res.data is not None else "\n".join(c.text for c in res.content)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        result = result if isinstance(result, str) else json.dumps(result)
        trace.tool_result(call, args, result)
        return result
    return dispatch
