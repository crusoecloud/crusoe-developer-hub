"""FakePod Backend — Dual-model podcast generator with voice synthesis."""
from __future__ import annotations

import json
import asyncio
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

import io
import edge_tts

from config import (
    CRUSOE_API_KEY,
    CRUSOE_API_BASE,
    VOICECHAT_API_BASE,
    HOST_A_MODEL,
    HOST_B_MODEL,
    VOICECHAT_MODEL,
    DEFAULT_HOST_A,
    DEFAULT_HOST_B,
    INSPIRE_CATEGORIES,
    TTS_VOICE_A,
    TTS_VOICE_B,
)
from prompts import (
    host_system_prompt,
    opening_prompt,
    response_prompt,
    variety_prompt,
    deeper_prompt,
    wrapup_prompt,
    inspire_prompt,
)

import pathlib

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="FakePod API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static frontend (built React app) ---
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

# --- Clients (lazy-initialized to avoid import-time network issues) ---
_text_client: OpenAI | None = None
_voice_client: OpenAI | None = None


def get_text_client() -> OpenAI:
    global _text_client
    if _text_client is None:
        _text_client = OpenAI(base_url=CRUSOE_API_BASE, api_key=CRUSOE_API_KEY)
    return _text_client


def get_voice_client() -> OpenAI:
    global _voice_client
    if _voice_client is None:
        _voice_client = OpenAI(base_url=VOICECHAT_API_BASE, api_key=CRUSOE_API_KEY)
    return _voice_client


# --- Request / Response models ---
class HostConfig(BaseModel):
    name: str = ""
    personality: str = ""


class GenerateRequest(BaseModel):
    topic: str
    angle: Optional[str] = None
    rounds: int = 0  # 0 = continuous until client disconnects
    host_a: Optional[HostConfig] = None
    host_b: Optional[HostConfig] = None


class ContinueRequest(BaseModel):
    topic: str
    transcript: List[dict]
    action: str = "deeper"
    host_a: Optional[HostConfig] = None
    host_b: Optional[HostConfig] = None


class InspireRequest(BaseModel):
    category: Optional[str] = "all"


class VoiceRequest(BaseModel):
    text: str
    host: str = "a"  # "a" or "b"


# --- Helpers ---
# Max conversation messages to keep in context to avoid token overflow.
# Each "turn" is 2 messages (user prompt + assistant response).
# 20 turns = 40 messages ≈ ~12k-15k tokens of context.
MAX_CONTEXT_TURNS = 20

# Safety limit: max rounds before auto-stopping (~80 rounds ≈ 20min at ~15s/round)
MAX_ROUNDS = 120


def build_hosts(host_a_cfg: HostConfig | None, host_b_cfg: HostConfig | None):
    ha = dict(DEFAULT_HOST_A)
    hb = dict(DEFAULT_HOST_B)
    if host_a_cfg:
        if host_a_cfg.name:
            ha["name"] = host_a_cfg.name
        if host_a_cfg.personality:
            ha["personality"] = host_a_cfg.personality
    if host_b_cfg:
        if host_b_cfg.name:
            hb["name"] = host_b_cfg.name
        if host_b_cfg.personality:
            hb["personality"] = host_b_cfg.personality
    return ha, hb


def trim_conversation(conversation: list[dict]) -> list[dict]:
    """Keep only the last MAX_CONTEXT_TURNS turns (pairs of user+assistant)."""
    max_messages = MAX_CONTEXT_TURNS * 2
    if len(conversation) > max_messages:
        return conversation[-max_messages:]
    return conversation


def generate_line(model: str, system: str, messages: list[dict]) -> str:
    response = get_text_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=1,
        top_p=0.95,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def pick_user_prompt(topic: str, angle: str | None, round_num: int) -> str:
    """Choose the right prompt for this round to keep the convo dynamic."""
    if round_num == 0:
        return opening_prompt(topic, angle)
    # Every 3-5 rounds, inject a variety prompt to shift the conversation
    if round_num >= 4 and round_num % 3 == 0:
        return variety_prompt(topic, round_num)
    return response_prompt(topic)


async def generate_dialogue_stream(
    topic: str,
    angle: str | None,
    max_rounds: int,
    ha: dict,
    hb: dict,
    disconnect_event: asyncio.Event,
) -> AsyncGenerator[str, None]:
    """Yield SSE events for each dialogue line. Runs until max_rounds or client disconnects."""
    system_a = host_system_prompt(ha, hb)
    system_b = host_system_prompt(hb, ha)
    conversation: list[dict] = []

    target = max_rounds if max_rounds > 0 else MAX_ROUNDS
    round_num = 0

    while round_num < target:
        if disconnect_event.is_set():
            break

        # Alternate hosts
        if round_num % 2 == 0:
            model, system, host, host_key = HOST_A_MODEL, system_a, ha, "a"
        else:
            model, system, host, host_key = HOST_B_MODEL, system_b, hb, "b"

        user_msg = pick_user_prompt(topic, angle, round_num)
        messages = trim_conversation(conversation) + [{"role": "user", "content": user_msg}]

        try:
            line = await asyncio.to_thread(generate_line, model, system, messages)
        except Exception as e:
            # Send error event and stop
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            break

        # Add to conversation history
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": line})

        event = {
            "host": host_key,
            "name": host["name"],
            "color": host["color"],
            "text": line,
            "round": round_num + 1,
            "total_rounds": target,
        }
        yield f"data: {json.dumps(event)}\n\n"
        round_num += 1

    yield "data: [DONE]\n\n"


# --- Routes ---
@app.get("/api/health")
def health():
    return {"status": "ok", "models": [HOST_A_MODEL, HOST_B_MODEL, VOICECHAT_MODEL]}


@app.get("/api/hosts")
def get_default_hosts():
    return {"host_a": DEFAULT_HOST_A, "host_b": DEFAULT_HOST_B}


@app.get("/api/categories")
def get_categories():
    return {"categories": INSPIRE_CATEGORIES}


@app.post("/api/inspire")
def inspire(req: InspireRequest):
    prompt = inspire_prompt(req.category)
    try:
        response = get_text_client().chat.completions.create(
            model=HOST_A_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.2,
            top_p=0.95,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        cleaned = raw
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        ideas = json.loads(cleaned)
        if not isinstance(ideas, list):
            raise ValueError("Not a list")
        return {"ideas": ideas[:4]}
    except Exception:
        return {
            "ideas": [
                "What if trees could file lawsuits?",
                "The secret economics of vending machines",
                "Why do we dream about falling?",
                "History's most overhyped inventions",
            ]
        }


@app.post("/api/generate")
async def generate_episode(req: GenerateRequest, request: Request):
    if not req.topic.strip():
        raise HTTPException(400, "Topic is required")

    ha, hb = build_hosts(req.host_a, req.host_b)

    # rounds=0 means continuous, otherwise clamp to safety max
    if req.rounds > 0:
        rounds = max(2, min(req.rounds, MAX_ROUNDS))
    else:
        rounds = 0  # continuous mode

    disconnect_event = asyncio.Event()

    async def event_generator():
        async for event in generate_dialogue_stream(
            req.topic, req.angle, rounds, ha, hb, disconnect_event
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


@app.post("/api/continue")
async def continue_episode(req: ContinueRequest):
    ha, hb = build_hosts(req.host_a, req.host_b)

    system_a = host_system_prompt(ha, hb)
    system_b = host_system_prompt(hb, ha)

    last_host = req.transcript[-1].get("host", "a") if req.transcript else "a"
    next_host = "b" if last_host == "a" else "a"
    model = HOST_B_MODEL if next_host == "b" else HOST_A_MODEL
    system = system_b if next_host == "b" else system_a
    host = hb if next_host == "b" else ha

    # Build message history from transcript (trimmed)
    messages = []
    for entry in req.transcript:
        messages.append({"role": "assistant", "content": entry["text"]})

    # Trim to last N messages
    messages = trim_conversation(messages)

    if req.action == "wrapup":
        messages.append({"role": "user", "content": wrapup_prompt(req.topic)})
    else:
        messages.append({"role": "user", "content": deeper_prompt(req.topic)})

    line = await asyncio.to_thread(generate_line, model, system, messages)

    return {
        "host": next_host,
        "name": host["name"],
        "color": host["color"],
        "text": line,
    }


@app.get("/api/tts/test")
async def test_tts():
    """Quick check if edge-tts can reach Microsoft's servers."""
    try:
        communicate = edge_tts.Communicate(text="test", voice=TTS_VOICE_A)
        got_audio = False
        async with asyncio.timeout(10):
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and len(chunk["data"]) > 0:
                    got_audio = True
                    break
        return {"available": got_audio}
    except Exception:
        return {"available": False}


@app.post("/api/tts")
async def text_to_speech(req: VoiceRequest):
    """Generate MP3 audio from text using edge-tts with distinct voices per host."""
    voice = TTS_VOICE_A if req.host == "a" else TTS_VOICE_B
    try:
        communicate = edge_tts.Communicate(text=req.text, voice=voice)
        audio_buffer = io.BytesIO()
        async with asyncio.timeout(15):
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
        if audio_buffer.tell() == 0:
            raise HTTPException(500, "TTS returned empty audio")
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"},
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "TTS timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")


# --- Serve built React frontend ---
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
