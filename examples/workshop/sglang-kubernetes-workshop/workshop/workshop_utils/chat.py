"""Talk to the served model.

Two kinds of endpoint matter here:

- the OpenAI-compatible API (``/v1/models``, ``/v1/chat/completions``), which
  any OpenAI client library can use
- SGLang's native endpoints (``/get_server_info``, ``/metrics``), which tell
  you how the engine is configured and how busy it is

References: https://docs.sglang.io/basic_usage/openai_api_completions.html and
https://docs.sglang.io/basic_usage/native_api.html
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests


def list_models(base_url: str) -> list[str]:
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def server_info(base_url: str) -> dict:
    """Server arguments and memory pool sizes. The path moved between releases, so try both."""
    for path in ("/get_server_info", "/server_info"):
        r = requests.get(f"{base_url}{path}", timeout=10)
        if r.ok:
            return r.json()
    r.raise_for_status()
    return {}


def metrics(base_url: str, prefix: str = "sglang:") -> dict[str, float]:
    """Parse the Prometheus text exposition into ``{metric_name: value}`` for simple gauges and counters."""
    r = requests.get(f"{base_url}/metrics", timeout=10)
    r.raise_for_status()
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        if not line.startswith(prefix) or " " not in line:
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


@dataclass
class ChatResult:
    text: str = ""
    reasoning: str = ""
    ttft_s: float | None = None
    total_s: float = 0.0
    completion_tokens: int | None = None
    chunks: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float | None:
        if not self.completion_tokens or self.ttft_s is None:
            return None
        decode = self.total_s - self.ttft_s
        return self.completion_tokens / decode if decode > 0.01 else None


def chat(
    base_url: str,
    model: str,
    prompt: str,
    system: str | None = None,
    thinking: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    stream: bool = True,
    echo: bool = True,
) -> ChatResult:
    """Send one chat turn and measure time to first token and decode speed.

    ``thinking`` toggles Qwen3's reasoning mode through the chat template
    (``enable_thinking``). With the ``qwen3`` reasoning parser the thinking
    text arrives in ``reasoning_content`` and the answer in ``content``.
    """
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    res = ChatResult()
    t0 = time.monotonic()
    if not stream:
        resp = client.chat.completions.create(stream=False, **kwargs)
        res.total_s = time.monotonic() - t0
        msg = resp.choices[0].message
        res.text = msg.content or ""
        res.reasoning = getattr(msg, "reasoning_content", "") or ""
        res.completion_tokens = resp.usage.completion_tokens if resp.usage else None
        if echo:
            print(res.text)
        return res

    for chunk in client.chat.completions.create(
        stream=True, stream_options={"include_usage": True}, **kwargs
    ):
        now = time.monotonic()
        if chunk.usage and chunk.usage.completion_tokens:
            res.completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = delta.content
        if (reasoning or content) and res.ttft_s is None:
            res.ttft_s = now - t0
        if reasoning:
            res.reasoning += reasoning
        if content:
            res.text += content
            if echo:
                print(content, end="", flush=True)
        res.chunks += 1
    res.total_s = time.monotonic() - t0
    if echo:
        tps = f"{res.tokens_per_s:.1f} tok/s" if res.tokens_per_s else "n/a"
        ttft = f"{res.ttft_s * 1000:.0f} ms" if res.ttft_s is not None else "n/a"
        print(f"\n\n[ttft {ttft} | {res.completion_tokens or '?'} tokens | {tps}]")
    return res
