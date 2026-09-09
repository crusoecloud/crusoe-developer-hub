"""Debate Arena backend — two Gemma 4 instances argue via streaming SSE."""
from __future__ import annotations

import asyncio
import json
import pathlib
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from config import (
    CHARS_PER_SECOND,
    CRUSOE_API_BASE,
    CRUSOE_API_KEY,
    MAX_CONTEXT_MESSAGES,
    MAX_ROUNDS,
    MIN_ROUNDS,
    MODEL_NAME,
)
from prompts import (
    PERSONA_PAIRS,
    build_system_prompt,
    build_user_prompt,
    persona_pairs_public,
)

app = FastAPI(title="Debate Arena API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=CRUSOE_API_BASE, api_key=CRUSOE_API_KEY)
    return _client


class DebateRequest(BaseModel):
    topic: str
    rounds: int = 3
    persona_pair_id: str = "optimist-skeptic"
    # When the client is speaking the text aloud (TTS), it can override the
    # server throttle — TTS sets the perceived pace, so we want turns delivered
    # fast. Pass 0 to disable throttling; omit to use the server default.
    chars_per_second: float | None = None


def trim_context(messages: list[dict]) -> list[dict]:
    if len(messages) > MAX_CONTEXT_MESSAGES:
        return messages[-MAX_CONTEXT_MESSAGES:]
    return messages


# Sentinel values pushed onto the queue to signal stream events.
_STREAM_END = object()


def _run_stream_blocking(
    system: str,
    messages: list[dict],
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Blocking: stream Crusoe chat completions, push each text delta into queue."""
    try:
        stream = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=1,
            top_p=0.95,
            max_tokens=400,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                asyncio.run_coroutine_threadsafe(queue.put(content), loop)
    except Exception as exc:  # noqa: BLE001
        asyncio.run_coroutine_threadsafe(queue.put({"__error__": str(exc)}), loop)
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(_STREAM_END), loop)


async def stream_turn_chunks(
    system: str, messages: list[dict]
) -> AsyncGenerator[str | dict, None]:
    """Async generator yielding each text delta as it arrives, or a dict on error."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    worker = asyncio.create_task(
        asyncio.to_thread(_run_stream_blocking, system, messages, queue, loop)
    )

    while True:
        item = await queue.get()
        if item is _STREAM_END:
            break
        yield item

    await worker


async def generate_debate_stream(
    topic: str,
    rounds: int,
    pair: dict,
    disconnect_event: asyncio.Event,
    chars_per_second: float,
) -> AsyncGenerator[str, None]:
    persona_a = pair["a"]
    persona_b = pair["b"]
    system_a = build_system_prompt(persona_a, persona_b, topic)
    system_b = build_system_prompt(persona_b, persona_a, topic)

    history_a: list[dict] = []
    history_b: list[dict] = []
    last_text: dict[str, str] = {"a": "", "b": ""}

    start_event = {
        "event": "start",
        "topic": topic,
        "total_rounds": rounds,
        "persona_a": {
            "name": persona_a["name"],
            "color": persona_a["color"],
            "emoji": persona_a["emoji"],
        },
        "persona_b": {
            "name": persona_b["name"],
            "color": persona_b["color"],
            "emoji": persona_b["emoji"],
        },
    }
    yield f"data: {json.dumps(start_event)}\n\n"

    for round_num in range(rounds):
        for turn_in_round, (key, persona, system, own_hist) in enumerate(
            [
                ("a", persona_a, system_a, history_a),
                ("b", persona_b, system_b, history_b),
            ]
        ):
            if disconnect_event.is_set():
                yield "data: [DONE]\n\n"
                return

            opp_key = "b" if key == "a" else "a"
            user_msg = build_user_prompt(
                topic, round_num, rounds, turn_in_round, last_text[opp_key]
            )
            messages = trim_context(own_hist) + [{"role": "user", "content": user_msg}]

            accumulated: list[str] = []
            had_error = False

            async for item in stream_turn_chunks(system, messages):
                if isinstance(item, dict) and "__error__" in item:
                    yield f"data: {json.dumps({'error': item['__error__']})}\n\n"
                    had_error = True
                    break
                # item is a text delta string
                accumulated.append(item)
                event = {
                    "speaker": key,
                    "name": persona["name"],
                    "color": persona["color"],
                    "chunk": item,
                    "round": round_num + 1,
                    "total_rounds": rounds,
                    "done": False,
                }
                yield f"data: {json.dumps(event)}\n\n"
                if chars_per_second > 0:
                    # Throttle to reading pace. Producer thread keeps draining the
                    # upstream stream into the queue while we sleep.
                    await asyncio.sleep(len(item) / chars_per_second)

            if had_error:
                yield "data: [DONE]\n\n"
                return

            full_text = "".join(accumulated).strip()
            done_event = {
                "speaker": key,
                "name": persona["name"],
                "color": persona["color"],
                "text": full_text,
                "round": round_num + 1,
                "total_rounds": rounds,
                "done": True,
            }
            yield f"data: {json.dumps(done_event)}\n\n"

            own_hist.append({"role": "user", "content": user_msg})
            own_hist.append({"role": "assistant", "content": full_text})
            last_text[key] = full_text

    yield "data: [DONE]\n\n"


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/api/personas")
def list_personas():
    return {"pairs": persona_pairs_public()}


@app.post("/api/debate")
async def debate(req: DebateRequest, request: Request):
    if not req.topic.strip():
        raise HTTPException(400, "Topic is required")
    if req.persona_pair_id not in PERSONA_PAIRS:
        raise HTTPException(400, f"Unknown persona pair: {req.persona_pair_id}")

    rounds = max(MIN_ROUNDS, min(req.rounds, MAX_ROUNDS))
    pair = PERSONA_PAIRS[req.persona_pair_id]
    cps = req.chars_per_second if req.chars_per_second is not None else CHARS_PER_SECOND

    disconnect_event = asyncio.Event()

    async def event_generator():
        async for event in generate_debate_stream(
            req.topic, rounds, pair, disconnect_event, cps
        ):
            if await request.is_disconnected():
                disconnect_event.set()
                break
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Serve built React frontend ---
if STATIC_DIR.exists():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
