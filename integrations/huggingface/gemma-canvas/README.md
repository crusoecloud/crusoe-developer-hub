---
title: Gemma Canvas
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: Image-aware chat with Gemma 4 on Crusoe Foundry
---

# Gemma Canvas

Upload an image, ask questions about it. Powered by **`google/gemma-4-31b-it`** on **Crusoe Foundry**.

---

## Prerequisite: run the probe first

Gemma 4's vision support on Crusoe Foundry is unconfirmed. Before assuming the app works, run the probe:

```bash
cd gemma-canvas
pip install openai
export CRUSOE_API_KEY=your-key
python probe.py
```

- **Exits 0 + prints a description** → vision works. Launch `app.py`.
- **Exits 1 with an error about `image_url` / unsupported content** → Gemma 4 on Crusoe is text-only. Pivot to the prompt-engineer + separate image-gen pipeline.

---

## How It Works

```
User uploads image + types question
        │
        ▼
  PIL resizes (long side ≤ 1024px)
  + encodes as base64 JPEG data URL
        │
        ▼
  Crusoe chat.completions.create(
    model=google/gemma-4-31b-it,
    messages=[{role: user, content: [
      {type: image_url, image_url: {url: data:image/jpeg;base64,…}},
      {type: text,      text: "What's in this image?"}
    ]}],
    stream=True
  )
        │
        ▼
  Streamed deltas → Gradio Chatbot
```

**Context continuity:** On follow-up turns the image is re-attached to the first user message so the model always sees the visual while reading the conversation history.

---

## Features

- 🖼️ Drag-and-drop image upload (PIL-resized to ≤ 1024px long side)
- 💬 Multi-turn chat — Gemma remembers the conversation and the image
- ⚡ Streamed responses (OpenAI `stream=True`)
- 🎯 Quick-prompt buttons for common queries
- 🔑 API key input or `CRUSOE_API_KEY` env var / Space Secret

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CRUSOE_API_KEY` | *(required)* | Crusoe Foundry auth |
| `CRUSOE_API_BASE` | `https://managed-inference-api-proxy.crusoecloud.com/v1/` | Endpoint override |
| `GEMMA_MODEL` | `google/gemma-4-31b-it` | Model override (`DEBATE_MODEL` also accepted as legacy fallback) |
| `PORT` | `7860` | HF Spaces port |

---

## Running Locally

```bash
pip install -r requirements.txt
export CRUSOE_API_KEY=your-key
python app.py
```

Open http://localhost:7860.

---

## Deployment (HuggingFace Space)

1. Create a new Space with **SDK = Gradio**
2. Upload `app.py`, `requirements.txt`, `README.md`, `probe.py`
3. Add `CRUSOE_API_KEY` as a **Space Secret**

The frontmatter above is the Space metadata.

---

## Files

```
gemma-canvas/
├── app.py            # Gradio UI + chat streaming
├── probe.py          # Smoke test — does Gemma 4 accept image_url?
├── requirements.txt
└── README.md
```

---

## License

MIT
