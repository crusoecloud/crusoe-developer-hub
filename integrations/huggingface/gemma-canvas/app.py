"""Gemma Canvas — image understanding chat with google/gemma-4-31b-it on Crusoe Foundry."""
from __future__ import annotations

import base64
import io
import os
from typing import List

import gradio as gr
from openai import OpenAI
from PIL import Image

# GEMMA_MODEL is the canonical name; DEBATE_MODEL is accepted as a legacy fallback
# so the env var can be reused across sibling demos without a rename.
MODEL = os.getenv("GEMMA_MODEL") or os.getenv("DEBATE_MODEL", "google/gemma-4-31b-it")
CRUSOE_API_BASE = os.getenv(
    "CRUSOE_API_BASE",
    "https://managed-inference-api-proxy.crusoecloud.com/v1/",
)

SYSTEM_PROMPT = (
    "You are Gemma Canvas, a friendly visual analyst. The user uploads an image "
    "and asks questions about it. Describe what you see, answer follow-ups, and "
    "infer context when helpful. Be concise, concrete, and honest about uncertainty."
)

QUICK_PROMPTS = [
    "Describe this image in detail.",
    "What's the mood or atmosphere?",
    "Identify any text or logos.",
    "If this were a product photo, what's being sold?",
    "List 3 things a human might miss.",
    "Guess where and when this was taken.",
]

_clients: dict[str, OpenAI] = {}


def get_client() -> OpenAI:
    """Return a cached OpenAI client using the CRUSOE_API_KEY env/Space secret."""
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


def image_to_data_url(img: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL (JPEG for size)."""
    # Convert to RGB so JPEG encoding works for any input mode
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Cap long side at 1024px to keep payload small
    max_side = 1024
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(
    image: Image.Image,
    history: List[dict],
    new_user_text: str,
) -> list[dict]:
    """Build OpenAI-format messages list. The image is attached to the *first* user turn.

    `history` is Gradio's messages-format list: `[{"role": "user"|"assistant", "content": str}, ...]`.
    """
    data_url = image_to_data_url(image)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    first_user_seen = False
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            if not first_user_seen:
                first_user_seen = True
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": content},
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    # New user turn. If no prior user turn exists, attach the image inline.
    if not first_user_seen:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": new_user_text},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": new_user_text})
    return messages


def respond(
    user_message: str,
    image: Image.Image | None,
    chat_history: List[dict],
):
    """Stream response back. Yields updated (chatbot, textbox) state."""
    if image is None:
        raise gr.Error("Upload an image first.")
    if not user_message.strip():
        raise gr.Error("Type a question about the image.")

    client = get_client()
    messages = build_messages(image, chat_history, user_message.strip())

    # Append the user turn + empty assistant placeholder in Gradio messages format.
    chat_history = chat_history + [
        {"role": "user", "content": user_message.strip()},
        {"role": "assistant", "content": ""},
    ]
    yield chat_history, ""

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=600,
            stream=True,
        )
        assembled = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0].delta, "content", None)
            if not delta:
                continue
            assembled += delta
            chat_history[-1]["content"] = assembled
            yield chat_history, ""
    except Exception as exc:  # noqa: BLE001
        # Surface enough context to diagnose Crusoe proxy errors (404 "No API found"
        # when a model endpoint doesn't exist for this request shape, 401/403 auth, etc).
        parts = [f"{type(exc).__name__}: {exc}"]
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                parts.append(f"status={resp.status_code}")
                body = resp.text
                if body:
                    parts.append(f"body={body[:400]}")
            except Exception:
                pass
        err = "⚠️ " + " | ".join(parts)
        chat_history[-1]["content"] = err
        yield chat_history, ""


def clear_chat():
    return [], ""


def apply_quick_prompt(prompt: str):
    return prompt


CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
#gemma-header { text-align: center; padding: 1.5rem 0 0.5rem; }
#gemma-header h1 { font-size: 2rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg,#818cf8,#f472b6); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
#gemma-header p { color: #94a3b8; margin-top: 0.25rem; font-size: 0.9rem; }
"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS, title="Gemma Canvas") as demo:
    gr.HTML(
        """
        <div id="gemma-header">
          <h1>🎨 Gemma Canvas</h1>
          <p>Image-aware chat · google/gemma-4-31b-it · Crusoe Foundry</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload image", type="pil", height=340)
            gr.Markdown("**Quick prompts** (click to fill the box)")
            with gr.Row():
                quick_btns = [gr.Button(p, size="sm") for p in QUICK_PROMPTS[:3]]
            with gr.Row():
                quick_btns += [gr.Button(p, size="sm") for p in QUICK_PROMPTS[3:]]

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=460,
                show_copy_button=True,
                type="messages",
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Ask about the image",
                    placeholder="What's in this image?",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation", size="sm")

    # Wiring
    send_btn.click(
        respond,
        inputs=[msg_box, image_input, chatbot],
        outputs=[chatbot, msg_box],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, image_input, chatbot],
        outputs=[chatbot, msg_box],
    )
    clear_btn.click(clear_chat, outputs=[chatbot, msg_box])

    # Quick prompts populate the message box
    for btn, prompt_text in zip(quick_btns, QUICK_PROMPTS):
        btn.click(lambda p=prompt_text: p, outputs=msg_box)


if __name__ == "__main__":
    # show_api=False avoids the gradio_client schema-introspection path that
    # crashes on bool-valued JSON schema nodes (additionalProperties: true).
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
    )
