"""Arcade Race — 4 LLMs race to generate a playable 80s arcade game.

Powered by Crusoe Managed AI. Each model gets the same prompt; the first
self-contained HTML5 game out wins. Users play all four side-by-side.
"""
from __future__ import annotations

import html as html_lib
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field

import gradio as gr
from openai import OpenAI

CRUSOE_API_BASE = os.getenv(
    "CRUSOE_API_BASE",
    "https://api.inference.crusoecloud.com/v1/",
)

AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/Deepseek-V4-Flash",
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
    "zai/GLM-5.1",
    "openai/gpt-oss-120b",
    "moonshotai/Kimi-K2-Thinking",
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "meta-llama/Llama-3.3-70B-Instruct",
]

DEFAULT_LINEUP = [
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/Deepseek-V4-Flash",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "zai/GLM-5.1",
]

SYSTEM_PROMPT = """You generate self-contained HTML5 arcade games in 80s/16-bit pixel-art style.

Hard requirements:
- Output ONE complete HTML document: <!DOCTYPE html>, <html>, inline <style>, inline <script>.
- Use <canvas> for rendering. NO external libraries, NO CDN imports, NO image/audio files.
- Pixel-art aesthetic: limited palette, chunky pixels, scanline or CRT feel, neon-on-black.
- Working game loop with score, lives/health if applicable, and game-over with auto-restart after 2 seconds.
- Canvas should be ~480x640 (or similar arcade ratio) and centered on the page.
- Black or themed background filling the viewport.

AUTOPLAY (REQUIRED):
- The game MUST start in autoplay mode automatically on page load — no human input needed.
- Implement a simple heuristic bot in JS that drives the player every frame:
  * Racer/dodger: move toward gaps in obstacle rows, avoid the closest threat.
  * Shooter: fire continuously; strafe away from incoming projectiles/enemies.
  * Snake/Pac-Man-like: head toward nearest pellet/food while avoiding walls and tail.
  * Breakout/Pong: track ball x-position with the paddle; lead slightly.
  * Platformer: jump when an obstacle is within ~80px ahead, walk forward otherwise.
- The bot should be good enough to play for at least ~30 seconds of action without dying instantly.
- Show "AUTOPLAY" text overlay in a corner of the canvas.
- Optional: pressing 'H' or clicking a button can toggle to human keyboard control (arrow keys / WASD).
- On game-over, auto-restart so the loop continues indefinitely.

Output ONLY the HTML inside ```html code fences. No commentary before or after.
"""

REQUEST_TIMEOUT = 120.0   # hard ceiling on the entire request
NO_FIRST_CHUNK_TIMEOUT = 15.0  # if no byte arrives by then, mark errored
MODEL_LIST_TTL = 300.0  # cache /v1/models for 5 min

EXAMPLE_PROMPTS = [
    "A simple pixel-art racing mini game with retro 8-bit/16-bit arcade style. Add other cars or obstacles to avoid, and a score system based on distance or survival time.",
    "A vertical-scrolling space shooter with neon enemies, three lives, and increasing wave difficulty.",
    "A Frogger-style game where the player crosses a busy 8-bit highway and a river of logs.",
    "A Breakout clone with rainbow bricks and a paddle that gets shorter every level.",
    "A Snake clone with a CRT-glow aesthetic and wraparound walls.",
]


# ─── Crusoe client ─────────────────────────────────────────────────────────────
_clients: dict[str, OpenAI] = {}
_model_catalog: dict[str, tuple[float, set[str] | None]] = {}  # api_key -> (ts, ids|None)


def get_available_models(client: OpenAI) -> set[str] | None:
    """Return the set of model IDs Crusoe will accept, or None if the
    /v1/models endpoint is unavailable. Cached for MODEL_LIST_TTL."""
    key = client.api_key or ""
    ts, cached = _model_catalog.get(key, (0.0, None))
    if time.time() - ts < MODEL_LIST_TTL and cached is not None:
        return cached
    try:
        resp = client.models.list()
        ids = {m.id for m in resp.data}
        _model_catalog[key] = (time.time(), ids)
        return ids
    except Exception:
        # Endpoint may not implement /v1/models or it timed out.
        # Cache None briefly so we don't hammer it.
        _model_catalog[key] = (time.time(), None)
        return None


def get_client() -> OpenAI:
    key = os.getenv("CRUSOE_API_KEY", "").strip()
    if not key:
        raise gr.Error(
            "CRUSOE_API_KEY is not set. Add it under Settings → Variables and "
            "secrets (Secret tab), then Factory rebuild the Space."
        )
    client = _clients.get(key)
    if client is None:
        client = OpenAI(base_url=CRUSOE_API_BASE, api_key=key)
        _clients[key] = client
    return client


# ─── Per-slot state ────────────────────────────────────────────────────────────
@dataclass
class Slot:
    model: str
    code: str = ""           # main `delta.content` stream
    reasoning: str = ""      # `delta.reasoning_content` (R1/GLM/etc.)
    chunks: int = 0
    ttft: float | None = None  # seconds to first byte (content OR reasoning)
    elapsed: float = 0.0
    started_at: float | None = None
    done: bool = False
    error: str | None = None
    final_html: str | None = None
    first_chunk_shape: str | None = None  # debug: keys present in first delta


# ─── HTML extraction ───────────────────────────────────────────────────────────
# Reasoning-model traces. Strip these first so they don't poison extraction
# or eat the byte/line counts in the analysis panel.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
ANALYSIS_BLOCK_RE = re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL)

# Fence regexes — accept ANY language tag (html / javascript / js / ts / etc.)
# so the tag word doesn't bleed into the captured content.
HTML_FENCE_RE = re.compile(r"```(?:[A-Za-z][\w+#.-]*)?[ \t]*\n?(.*?)```", re.DOTALL)
HTML_OPEN_FENCE_RE = re.compile(r"```(?:[A-Za-z][\w+#.-]*)?[ \t]*\n?(.*)$", re.DOTALL)
GETELEM_RE = re.compile(r"""getElementById\(\s*['"]([\w-]+)['"]\s*\)""")
JS_LIKELY_RE = re.compile(
    r"\b(?:function|const|let|var|requestAnimationFrame|addEventListener)\b"
)
HTML_DOC_RE = re.compile(
    r"(<!DOCTYPE\s+html.*?</html>|<html[^>]*>.*?</html>)",
    re.DOTALL | re.IGNORECASE,
)
FRAGMENT_START_RE = re.compile(
    r"<(?:!DOCTYPE|html|head|body|style|script|canvas)\b",
    re.IGNORECASE,
)


def _strip_reasoning(text: str) -> str:
    """Remove <think> blocks and similar reasoning traces from any model."""
    text = THINK_BLOCK_RE.sub("", text)
    text = ANALYSIS_BLOCK_RE.sub("", text)
    # Truncated open <think> with no close — drop it
    text = THINK_OPEN_RE.sub("", text)
    return text


def _wrap_if_fragment(html_str: str) -> str:
    if re.search(r"<html\b", html_str, re.IGNORECASE):
        return html_str
    return (
        "<!DOCTYPE html><html><body style='margin:0;background:#000;"
        "color:#fff;font-family:monospace;display:flex;align-items:center;"
        f"justify-content:center;min-height:100vh'>{html_str}</body></html>"
    )


def _wrap_js(js: str) -> str:
    """Wrap raw JS in a minimal HTML shell with <canvas> elements matching
    every id the JS references via getElementById."""
    ids = list(dict.fromkeys(GETELEM_RE.findall(js)))  # dedupe, preserve order
    if not ids:
        ids = ["c", "canvas", "game"]
    canvas_tags = "\n  ".join(
        f'<canvas id="{cid}" width="480" height="640"></canvas>' for cid in ids
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#000;height:100%}}
body{{display:flex;align-items:center;justify-content:center;flex-direction:column}}
canvas{{image-rendering:pixelated;image-rendering:crisp-edges;max-width:100%;max-height:100vh}}
canvas:not(:first-of-type){{display:none}}
</style></head><body>
  {canvas_tags}
<script>
{js}
</script></body></html>"""


def _looks_like_js(s: str) -> bool:
    return "<" not in s and bool(JS_LIKELY_RE.search(s))


def extract_html(text: str) -> str | None:
    """Pull a self-contained HTML doc out of model output. Tolerant of
    missing fences, prose around the code, reasoning traces, and bare
    fragments."""
    if not text:
        return None

    # 0. Strip <think> reasoning blocks first — otherwise they leak into the
    #    iframe and may swallow the actual HTML in regex matches.
    text = _strip_reasoning(text)

    # 1. Closed fence (any language tag). Take the LARGEST fence so we don't
    #    grab a tiny inline snippet from prose-like commentary.
    fences = list(HTML_FENCE_RE.finditer(text))
    if fences:
        best = max(fences, key=lambda m: len(m.group(1).strip()))
        candidate = best.group(1).strip()
        if candidate:
            if "<" in candidate:
                return _wrap_if_fragment(candidate)
            if _looks_like_js(candidate):
                return _wrap_js(candidate)

    # 2. Full <!DOCTYPE/<html> document anywhere in the text
    m = HTML_DOC_RE.search(text)
    if m:
        return m.group(1).strip()

    # 3. Open fence with no close (truncated stream): take everything after
    m = HTML_OPEN_FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r"`{1,3}\s*$", "", candidate).strip()
        if candidate:
            if "<" in candidate:
                return _wrap_if_fragment(candidate)
            if _looks_like_js(candidate):
                return _wrap_js(candidate)

    # 4. Raw HTML/JS fragment — accept anything with <canvas> OR (<style> AND <script>)
    lower = text.lower()
    has_canvas = "<canvas" in lower
    has_script = "<script" in lower
    has_style = "<style" in lower
    if has_canvas or (has_script and has_style):
        start_m = FRAGMENT_START_RE.search(text)
        if start_m:
            start = start_m.start()
            # End at the latest closing tag we can find
            end = len(text)
            for tag in ("</html>", "</body>", "</script>", "</style>"):
                idx = lower.rfind(tag)
                if idx != -1:
                    end = max(end, idx + len(tag))
            fragment = text[start:end].strip()
            if fragment:
                return _wrap_if_fragment(fragment)

    # 5. Whole text starts with a tag — try as-is
    stripped = text.strip()
    if stripped.startswith("<"):
        return _wrap_if_fragment(stripped)

    # 6. Whole text looks like JS (no HTML at all): wrap it.
    if _looks_like_js(stripped):
        return _wrap_js(stripped)

    return None


# ─── Worker ────────────────────────────────────────────────────────────────────
def stream_one(slot: Slot, prompt: str, q: queue.Queue, available: set[str] | None) -> None:
    """Run one model stream. Push ('tick', idx) onto the queue on each chunk."""
    client = get_client()
    slot.started_at = time.perf_counter()
    # Note: /v1/models returns a partial catalog on Crusoe — models can be
    # callable even when not listed. We don't pre-reject based on `available`.
    try:
        stream = client.chat.completions.create(
            model=slot.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=12000,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            # Disable reasoning/thinking mode so the model emits the game
            # directly to `content` instead of burning tokens in CoT. This is
            # the vLLM/sglang convention used by Qwen3, GLM, etc.; servers
            # ignore unknown chat_template_kwargs gracefully.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for chunk in stream:
            if slot.done:  # watchdog gave up
                return
            if not chunk.choices:
                continue
            delta_obj = chunk.choices[0].delta
            # Use model_dump() to get every field including non-standard ones
            # (Pydantic's model_extra mechanism varies by SDK version).
            try:
                delta_dict = delta_obj.model_dump()
            except Exception:
                delta_dict = {}

            content = delta_dict.get("content") or ""
            # Various reasoning-mode field names across providers.
            reasoning = (
                delta_dict.get("reasoning_content")
                or delta_dict.get("reasoning")
                or delta_dict.get("thinking")
                or delta_dict.get("thought")
                or ""
            )

            # On the first non-empty chunk, snapshot its keys for debugging.
            if slot.first_chunk_shape is None and any(
                v for k, v in delta_dict.items() if isinstance(v, str)
            ):
                non_empty = [k for k, v in delta_dict.items() if v not in (None, "", [], {})]
                slot.first_chunk_shape = ", ".join(non_empty) or "(no keys)"

            if not content and not reasoning:
                continue
            now = time.perf_counter()
            if slot.ttft is None:
                slot.ttft = now - slot.started_at
            if content:
                slot.code += content
            if reasoning:
                slot.reasoning += reasoning
            slot.chunks += 1
            slot.elapsed = now - slot.started_at
            q.put(("tick",))

        if not slot.done:
            # Fallback: if no content arrived but reasoning has the game in it,
            # extract from reasoning. Treat it as the source.
            primary = slot.code if slot.code.strip() else slot.reasoning
            slot.final_html = extract_html(primary)
            slot.done = True
            q.put(("done",))
    except Exception as exc:  # noqa: BLE001
        parts = [f"{type(exc).__name__}: {exc}"]
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                parts.append(f"status={resp.status_code}")
                body = resp.text
                if body:
                    parts.append(f"body={body[:300]}")
            except Exception:
                pass
        slot.error = " | ".join(parts)
        slot.done = True
        q.put(("done",))


# ─── Rendering ─────────────────────────────────────────────────────────────────
def fmt_mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def render_card(idx: int, slot: Slot) -> str:
    if slot.done and not slot.error:
        timer_color = "#00ff88"
    elif slot.done and slot.error:
        timer_color = "#ff4d6d"
    elif slot.started_at is None:
        timer_color = "#666"
    else:
        timer_color = "#ffd700"

    timer = fmt_mmss(slot.elapsed)
    model_label = slot.model.split("/")[-1]

    if slot.error:
        body = (
            f'<div class="ar-error">⚠️ {html_lib.escape(slot.error[:400])}</div>'
        )
    elif slot.done and slot.final_html:
        encoded = html_lib.escape(slot.final_html, quote=True)
        body = (
            f'<iframe srcdoc="{encoded}" sandbox="allow-scripts" '
            f'class="ar-iframe" title="game-{idx}"></iframe>'
        )
    elif slot.done:
        # Show previews of both content and reasoning streams so the failure
        # mode is obvious — empty content + populated reasoning means the model
        # wrote everything in its CoT scratchpad.
        def _preview(s: str, limit: int = 500) -> str:
            if not s:
                return "(empty)"
            head = s[:limit]
            if len(s) > limit:
                head += f"\n\n… [{len(s) - limit:,} more chars]"
            return html_lib.escape(head)

        content_block = (
            f"<div style=\"color:#94a3b8;font-size:0.7rem;margin-top:8px\">"
            f"<b>delta.content</b> ({len(slot.code):,} chars)</div>"
            f"<pre style=\"max-height:200px;overflow:auto;color:#cbd5e1;"
            f"background:#0a0a0d;padding:8px;border-radius:6px;"
            f"font-size:0.72rem;margin:4px 0;white-space:pre-wrap\">"
            f"{_preview(slot.code)}</pre>"
        )
        reasoning_block = ""
        if slot.reasoning:
            reasoning_block = (
                f"<div style=\"color:#fbbf24;font-size:0.7rem;margin-top:8px\">"
                f"<b>delta.reasoning_content</b> ({len(slot.reasoning):,} chars)</div>"
                f"<pre style=\"max-height:200px;overflow:auto;color:#fde68a;"
                f"background:#0a0a0d;padding:8px;border-radius:6px;"
                f"font-size:0.72rem;margin:4px 0;white-space:pre-wrap\">"
                f"{_preview(slot.reasoning)}</pre>"
            )
        body = (
            '<div class="ar-error">'
            "<strong>No HTML found in output.</strong><br>"
            "<span style=\"color:#999;font-size:0.78rem\">"
            "Model finished but extractor couldn't locate runnable HTML."
            "</span>"
            f"{content_block}{reasoning_block}"
            "</div>"
        )
    elif slot.started_at is None:
        body = (
            '<div class="ar-placeholder">'
            '<div style="color:#666">Queued</div>'
            "</div>"
        )
    elif slot.ttft is None:
        # Connected to API but no first byte yet — possible bad model ID
        # or slow cold-start. Make this visible.
        body = (
            '<div class="ar-placeholder">'
            '<div class="ar-pulse" style="background:#fbbf24;'
            'box-shadow:0 0 12px #fbbf24"></div>'
            f'<div style="color:#fbbf24">⏳ Connecting…</div>'
            f'<div style="color:#666;font-size:0.75rem;margin-top:4px">'
            f'waiting for first byte · {slot.elapsed:.1f}s</div>'
            f'<div style="color:#555;font-size:0.7rem;margin-top:2px">'
            f'(auto-fail at {NO_FIRST_CHUNK_TIMEOUT:.0f}s)</div>'
            "</div>"
        )
    else:
        kb = len(slot.code) / 1024
        rkb = len(slot.reasoning) / 1024
        rate = slot.chunks / slot.elapsed if slot.elapsed > 0.1 else 0
        reasoning_note = (
            f' · reasoning: {rkb:.1f} KB' if slot.reasoning else ""
        )
        body = (
            '<div class="ar-placeholder">'
            '<div class="ar-pulse"></div>'
            f'<div style="color:#00ff88">▶ Streaming…</div>'
            f'<div style="color:#94a3b8;font-size:0.78rem;margin-top:4px">'
            f'{kb:.1f} KB · {slot.chunks} chunks · {rate:.0f}/s{reasoning_note}'
            f'</div>'
            "</div>"
        )

    return f"""
<div class="ar-card">
  <div class="ar-card-head">
    <span class="ar-model">{html_lib.escape(model_label)}</span>
    <span class="ar-timer" style="color:{timer_color}">{timer}</span>
  </div>
  <div class="ar-card-body">{body}</div>
</div>
"""


# ─── Comparative analysis (for marketing) ─────────────────────────────────────
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
FN_RE = re.compile(r"\bfunction\s+\w+\s*\(|=>\s*\{")
STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def analyze_code(code: str) -> dict | None:
    if not code:
        return None
    lower = code.lower()
    return {
        "lines": code.count("\n") + 1,
        "bytes": len(code),
        "has_canvas": "<canvas" in lower,
        "has_autoplay_text": "autoplay" in lower,
        "has_restart": bool(re.search(r"restart|game.?over", code, re.IGNORECASE)),
        "has_score": "score" in lower,
        "has_keyboard": bool(re.search(r"keydown|keyup", code, re.IGNORECASE)),
        "has_animation": "requestanimationframe" in lower,
        "colors": len(set(HEX_RE.findall(code))),
        "functions": len(FN_RE.findall(code)),
        "css_size": sum(len(m.group(1)) for m in STYLE_RE.finditer(code)),
        "js_size": sum(len(m.group(1)) for m in SCRIPT_RE.finditer(code)),
    }


def _slot_status(s: Slot) -> str:
    if s.error:
        return "❌ Error"
    if s.done and s.final_html:
        return "✅ Done"
    if s.done:
        return "🟡 No HTML"
    if s.started_at is not None and (s.code or s.reasoning):
        return "▶ Streaming"
    if s.started_at is not None:
        return "⏳ Connecting"
    return "⏳ Queued"


def make_analysis(slots: list[Slot]) -> str:
    if not any(s.code or s.reasoning or s.started_at for s in slots):
        return ""

    # Analyze whichever stream has bytes — reasoning is the source of truth
    # for reasoning-mode models that never emit a `content` channel.
    metrics = [analyze_code(s.code or s.reasoning) for s in slots]

    # Comparison table
    md = "## 📊 Comparative Analysis\n\n"
    md += "| Model | Status | TTFT | Total | Chunks | Content | Reasoning | Lines | CSS | JS | Colors | Funcs |\n"
    md += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for s, m in zip(slots, metrics):
        name = s.model.split("/")[-1]
        status = _slot_status(s)
        ttft = f"{s.ttft:.2f}s" if s.ttft else "—"
        total = f"{s.elapsed:.1f}s" if s.elapsed > 0 else "—"
        rate = f"{s.chunks/s.elapsed:.0f}/s" if s.elapsed > 0.1 else "—"
        chunks_cell = f"{s.chunks} ({rate})" if s.chunks else "—"
        content_cell = f"{len(s.code):,}" if s.code else "—"
        reasoning_cell = f"{len(s.reasoning):,}" if s.reasoning else "—"
        if m:
            md += (
                f"| {name} | {status} | {ttft} | {total} | {chunks_cell} | "
                f"{content_cell} | {reasoning_cell} | "
                f"{m['lines']:,} | {m['css_size']:,} | {m['js_size']:,} | "
                f"{m['colors']} | {m['functions']} |\n"
            )
        else:
            md += (
                f"| {name} | {status} | {ttft} | {total} | {chunks_cell} | "
                f"{content_cell} | {reasoning_cell} | — | — | — | — | — |\n"
            )

    # Winners (only when at least one slot has finished)
    valid = [(s, m) for s, m in zip(slots, metrics) if m and s.elapsed > 0 and not s.error]
    if valid:
        fastest = min(valid, key=lambda x: x[0].elapsed)
        slowest = max(valid, key=lambda x: x[0].elapsed)
        biggest = max(valid, key=lambda x: x[1]["bytes"])
        smallest = min(valid, key=lambda x: x[1]["bytes"])
        most_colorful = max(valid, key=lambda x: x[1]["colors"])
        best_ttft = min(valid, key=lambda x: x[0].ttft or 999)

        def n(s):
            return s.model.split("/")[-1]

        md += "\n### 🏆 Marketing-ready verdicts\n\n"
        md += f"- **⚡ Fastest end-to-end:** `{n(fastest[0])}` — {fastest[0].elapsed:.1f}s total\n"
        md += f"- **🚀 Snappiest first token:** `{n(best_ttft[0])}` — TTFT {best_ttft[0].ttft:.2f}s\n"
        md += f"- **📦 Most comprehensive output:** `{n(biggest[0])}` — {biggest[1]['bytes']:,} bytes / {biggest[1]['lines']:,} lines\n"
        md += f"- **🎨 Most visually rich:** `{n(most_colorful[0])}` — {most_colorful[1]['colors']} unique colors in palette\n"
        md += f"- **🪶 Most concise:** `{n(smallest[0])}` — {smallest[1]['bytes']:,} bytes\n"
        md += f"- **🐢 Slowest:** `{n(slowest[0])}` — {slowest[0].elapsed:.1f}s\n"

    # Feature checklist
    md += "\n### ✅ Feature checklist\n\n"
    md += "| Feature | " + " | ".join(s.model.split("/")[-1] for s in slots) + " |\n"
    md += "|---" * (len(slots) + 1) + "|\n"
    feature_keys = [
        ("Canvas rendering", "has_canvas"),
        ("Autoplay overlay", "has_autoplay_text"),
        ("Game-over / restart", "has_restart"),
        ("Score system", "has_score"),
        ("Keyboard fallback", "has_keyboard"),
        ("Animation loop", "has_animation"),
    ]
    for label, key in feature_keys:
        cells = []
        for m in metrics:
            cells.append("✅" if (m and m.get(key)) else "—")
        md += f"| {label} | " + " | ".join(cells) + " |\n"

    # Per-model commentary
    md += "\n### 🔍 Per-model breakdown\n\n"
    for s, m in zip(slots, metrics):
        name = s.model.split("/")[-1]
        if s.error:
            md += f"**{name}** — ⚠️ Error: `{html_lib.escape(s.error[:160])}`\n\n"
            continue
        if not m:
            md += f"**{name}** — No analyzable output yet\n\n"
            continue
        ttft_str = f"TTFT `{s.ttft:.2f}s`" if s.ttft else "TTFT —"
        total_str = f"total `{s.elapsed:.1f}s`" if s.elapsed else "total —"
        chunks_str = f"`{s.chunks}` chunks"
        rate_str = f"`{s.chunks/s.elapsed:.0f}` chunks/s" if s.elapsed > 0.1 else "—"
        md += (
            f"**{name}** — {ttft_str} · {total_str} · {chunks_str} ({rate_str})  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;Output `{m['bytes']:,}` bytes / `{m['lines']:,}` lines · "
            f"CSS `{m['css_size']:,}` / JS `{m['js_size']:,}` · "
            f"~`{m['functions']}` functions · `{m['colors']}` colors\n\n"
        )

    return md


def render_metrics(slot: Slot) -> str:
    if slot.error:
        return f"❌ {html_lib.escape(slot.error[:200])}"
    if not slot.done and slot.started_at is None:
        return "_Idle_"
    ttft = f"{slot.ttft:.2f}s" if slot.ttft else "—"
    total = f"{slot.elapsed:.2f}s"
    bytes_out = f"{len(slot.code):,}"
    status = "✅ Done" if slot.done else "⏳ Streaming"
    return (
        f"**{status}** &nbsp;·&nbsp; "
        f"TTFT: `{ttft}` &nbsp;·&nbsp; "
        f"Total: `{total}` &nbsp;·&nbsp; "
        f"Bytes: `{bytes_out}`"
    )


# ─── Race orchestrator ─────────────────────────────────────────────────────────
def race(prompt: str, m1: str, m2: str, m3: str, m4: str):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a game prompt first.")

    models = [m1, m2, m3, m4]
    slots = [Slot(model=m) for m in models]
    q: queue.Queue = queue.Queue()

    # One-time catalog probe — lets us fail unknown model IDs immediately.
    try:
        available = get_available_models(get_client())
    except Exception:
        available = None

    threads = [
        threading.Thread(target=stream_one, args=(s, prompt, q, available), daemon=True)
        for s in slots
    ]
    for t in threads:
        t.start()

    EMIT_INTERVAL = 0.2
    last_emit = 0.0

    def snapshot():
        per_slot = tuple(
            v
            for s, idx in zip(slots, range(4))
            for v in (render_card(idx, s), render_metrics(s), s.code)
        )
        return per_slot + (make_analysis(slots),)

    yield snapshot()

    while not all(s.done for s in slots):
        try:
            q.get(timeout=0.15)
        except queue.Empty:
            pass
        now = time.perf_counter()
        for s in slots:
            if s.done or s.started_at is None:
                continue
            s.elapsed = now - s.started_at
            # Watchdog: if no first chunk arrived in NO_FIRST_CHUNK_TIMEOUT,
            # mark the slot errored so the UI doesn't pulse forever.
            if s.ttft is None and s.elapsed > NO_FIRST_CHUNK_TIMEOUT:
                s.error = (
                    f"No response from {s.model} after {NO_FIRST_CHUNK_TIMEOUT:.0f}s. "
                    "Model ID may not exist on Crusoe Managed AI, or the endpoint is "
                    "stalled. Check the available model catalog."
                )
                s.done = True
                q.put(("done",))
        if now - last_emit >= EMIT_INTERVAL:
            last_emit = now
            yield snapshot()

    yield snapshot()


# ─── UI ────────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container {
  max-width: 1280px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

#ar-frame {
  background: #f4f06b;
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

#ar-prompt-shell {
  background: #0e0e10;
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 14px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color: #e6e6e6;
  font-size: 0.9rem;
  border: 1px solid #2a2a2e;
}

.ar-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.ar-card {
  background: #15151a;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #2a2a2e;
  display: flex;
  flex-direction: column;
}

.ar-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: #1c1c22;
  border-bottom: 1px solid #2a2a2e;
}

.ar-model {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.95rem;
  font-weight: 600;
  color: #e6e6e6;
  letter-spacing: 0.02em;
}

.ar-timer {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 1.4rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-shadow: 0 0 8px currentColor;
}

.ar-card-body {
  background: #000;
  height: 420px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.ar-iframe {
  width: 100%;
  height: 100%;
  border: 0;
  background: #000;
  display: block;
}

.ar-placeholder {
  color: #888;
  text-align: center;
  font-family: ui-monospace, monospace;
  font-size: 0.85rem;
}

.ar-pulse {
  width: 16px; height: 16px; border-radius: 50%;
  background: #ffd700;
  margin: 0 auto 10px;
  animation: ar-pulse 1.2s ease-in-out infinite;
  box-shadow: 0 0 12px #ffd700;
}

@keyframes ar-pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.85); }
  50%      { opacity: 1.0; transform: scale(1.1); }
}

.ar-error {
  color: #ff8a9b;
  padding: 16px;
  font-family: ui-monospace, monospace;
  font-size: 0.85rem;
  text-align: left;
  white-space: pre-wrap;
}

#ar-header h1 {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  letter-spacing: 0.04em;
  margin: 0 0 0.2rem;
  font-size: 1.9rem;
  background: linear-gradient(90deg,#f59e0b,#ec4899,#8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
#ar-header p { color: #94a3b8; margin: 0; font-size: 0.9rem; }
"""


def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber"), css=CSS, title="Arcade Race") as demo:
        gr.HTML(
            """
            <div id="ar-header" style="text-align:center; padding: 1rem 0 0.6rem;">
              <h1>🕹️ ARCADE RACE</h1>
              <p>Four LLMs · one prompt · race to generate a playable 80s arcade game · powered by Crusoe Managed AI</p>
            </div>
            """
        )

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="Describe the game you want…",
                lines=2,
                scale=4,
                value=EXAMPLE_PROMPTS[0],
            )
            race_btn = gr.Button("▶ RACE", variant="primary", scale=1)

        with gr.Accordion("Models (pick 4)", open=False):
            with gr.Row():
                model_dropdowns = [
                    gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_LINEUP[i],
                        label=f"Slot {i+1}",
                    )
                    for i in range(4)
                ]

        gr.Examples(
            examples=[[p] for p in EXAMPLE_PROMPTS],
            inputs=[prompt_box],
            label="Try a prompt",
        )

        with gr.Column(elem_id="ar-frame"):
            prompt_echo = gr.HTML(
                elem_id="ar-prompt-shell",
                value='<span style="color:#9ca3af">// type a prompt above and hit RACE</span>',
            )
            card_html: list[gr.HTML] = []
            with gr.Row():
                card_html.append(gr.HTML(value=render_card(0, Slot(model=DEFAULT_LINEUP[0]))))
                card_html.append(gr.HTML(value=render_card(1, Slot(model=DEFAULT_LINEUP[1]))))
            with gr.Row():
                card_html.append(gr.HTML(value=render_card(2, Slot(model=DEFAULT_LINEUP[2]))))
                card_html.append(gr.HTML(value=render_card(3, Slot(model=DEFAULT_LINEUP[3]))))

        with gr.Accordion("📊 Comparative analysis", open=True):
            analysis_md = gr.Markdown(
                "_Run a race to see per-model speed, output size, palette, "
                "feature coverage, and marketing-ready verdicts._"
            )

        with gr.Accordion("Per-model details", open=False):
            metric_md: list[gr.Markdown] = []
            code_box: list[gr.Code] = []
            for i in range(4):
                with gr.Tab(f"Slot {i+1}"):
                    metric_md.append(gr.Markdown("_Idle_"))
                    code_box.append(
                        gr.Code(
                            label="Raw stream",
                            language="html",
                            lines=18,
                            interactive=False,
                        )
                    )

        # Output order matches snapshot(): for each slot (0..3): card, metrics, code; then analysis
        outputs: list[gr.components.Component] = []
        for i in range(4):
            outputs.extend([card_html[i], metric_md[i], code_box[i]])
        outputs.append(analysis_md)

        def echo_prompt(p: str) -> str:
            safe = html_lib.escape((p or "").strip())
            if not safe:
                return '<span style="color:#9ca3af">// type a prompt above and hit RACE</span>'
            return (
                '<span style="color:#7dd3fc">$</span> '
                f'<span style="color:#e6e6e6">{safe}</span>'
            )

        race_btn.click(
            echo_prompt, inputs=prompt_box, outputs=prompt_echo
        ).then(
            race, inputs=[prompt_box, *model_dropdowns], outputs=outputs
        )
        prompt_box.submit(
            echo_prompt, inputs=prompt_box, outputs=prompt_echo
        ).then(
            race, inputs=[prompt_box, *model_dropdowns], outputs=outputs
        )

        gr.HTML(
            """
            <div style="text-align:center; padding: 1rem 0; color:#888; font-size:0.85rem;">
              Built by <strong>Crusoe AI</strong> Developer Relations ·
              <a href="https://crusoe.ai" style="color:#f59e0b">crusoe.ai</a>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
    )
