import sys
from pydantic import BaseModel
from config import client


class Fn(BaseModel):
    name: str
    arguments: str


class Call(BaseModel):
    id: str
    type: str = "function"
    function: Fn


class Msg(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[Call] | None = None


class TokenDetails(BaseModel):
    cached_tokens: int = 0


class StreamUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: TokenDetails = TokenDetails()


async def complete_streaming(model: str, messages: list[dict], tools: list[dict], echo=sys.stdout):
    calls: dict[int, dict] = {}
    content = ""
    usage = None
    stream = await client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto", stream=True,
        stream_options={"include_usage": True})
    async for chunk in stream:
        if chunk.usage is not None:
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            usage = StreamUsage(prompt_tokens=chunk.usage.prompt_tokens or 0, completion_tokens=chunk.usage.completion_tokens or 0,
                                prompt_tokens_details=TokenDetails(cached_tokens=(getattr(details, "cached_tokens", None) or 0)))
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            if echo:
                print(delta.content, end="", flush=True, file=echo)
        for d in delta.tool_calls or []:
            c = calls.setdefault(d.index, {"id": d.id, "name": "", "arguments": ""})
            if d.id:
                c["id"] = d.id
            if d.function.name:
                c["name"] = d.function.name
            if d.function.arguments:
                c["arguments"] += d.function.arguments
    if echo and content:
        print(file=echo)
    tool_calls = [Call(id=c["id"] or f"call_{i}", function=Fn(name=c["name"], arguments=c["arguments"])) for i, c in sorted(calls.items())] or None
    return Msg(content=content or None, tool_calls=tool_calls), usage
