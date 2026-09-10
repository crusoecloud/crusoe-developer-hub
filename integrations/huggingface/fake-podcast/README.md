---
title: FakePod — AI Podcast Generator
emoji: 🎙️
colorFrom: purple
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FakePod — AI Podcast Generator

Generate continuous AI podcast conversations between two hosts powered by different LLMs on **Crusoe Cloud Foundry**. Each host has a distinct personality and voice — they talk to each other for up to 20 minutes, with real text-to-speech audio.

**Live demo:** [https://huggingface.co/crusoeai/spaces](https://huggingface.co/crusoeai/spaces)

---

## How It Works

```
User picks a topic (or hits "Inspire Me")
            │
            ▼
  ┌─────────────────┐     ┌─────────────────────┐
  │  Qwen3-235B     │     │  Nemotron Super-120B │
  │  (Host A:Atlas) │◄───►│  (Host B: Nova)      │
  │  Deep-diver     │     │  Sharp reactor       │
  └────────┬────────┘     └──────────┬───────────┘
           │    Back-and-forth via SSE│
           ▼                         ▼
     ┌──────────────────────────────────┐
     │  Streamed transcript in browser  │
     │  + live subtitles overlay        │
     └──────────────┬───────────────────┘
                    │
                    ▼
     ┌──────────────────────────────────┐
     │  edge-tts (Microsoft Neural TTS) │
     │  Christopher (male) + Aria (fem) │
     │  Falls back to browser TTS       │
     └──────────────────────────────────┘
```

**Two models, two voices, one conversation.** Host A (Qwen3-235B) and Host B (Nemotron Super-120B) alternate turns indefinitely. Neither controls the full script — the conversation emerges naturally from model differences. Variety prompts are injected every few rounds to keep the discussion dynamic.

---

## Features

- **Continuous conversation** — Hosts talk non-stop for up to ~20 minutes until you press Stop
- **Dual-model dialogue** — Qwen3-235B vs Nemotron-120B alternate turns via SSE streaming
- **Real text-to-speech** — `edge-tts` neural voices (free, no API key) with automatic fallback to browser SpeechSynthesis
- **Distinct voices** — Male voice (Christopher) for Host A, female voice (Aria) for Host B
- **Live subtitles** — Auto-advancing subtitle overlay that works with or without audio
- **"Inspire Me" button** — AI-generated topic ideas filtered by category
- **Customizable hosts** — Change names, personalities, and speaking styles
- **Real-time transcript** — Lines appear as they generate with typing animations
- **Episode library** — Auto-saved to localStorage, replayable
- **Sliding context window** — Keeps last 20 turns in memory so conversations never hit token limits

---

## Architecture

```
fake-podcast/
├── backend/                        # Python FastAPI server
│   ├── main.py                     # API routes, SSE streaming, TTS, SPA serving
│   ├── config.py                   # Models, endpoints, host defaults, TTS voices
│   ├── prompts.py                  # All prompt templates + variety prompts
│   ├── requirements.txt
│   └── .env.example
├── frontend/                       # React + TypeScript + Vite + Tailwind
│   ├── src/
│   │   ├── App.tsx                 # Main state, generation loop, stop control
│   │   ├── api.ts                  # SSE streaming client, TTS fetch, helpers
│   │   ├── types.ts                # TypeScript interfaces
│   │   └── components/
│   │       ├── AudioPlayer.tsx     # TTS playback (server + browser fallback)
│   │       ├── Subtitles.tsx       # Auto-advancing subtitle overlay
│   │       ├── Transcript.tsx      # Scrolling dialogue with avatars
│   │       ├── InspireMe.tsx       # Topic idea generator with categories
│   │       ├── HostEditor.tsx      # Host personality customizer
│   │       └── EpisodeLibrary.tsx  # Saved episodes sidebar
│   ├── mic.svg                     # Favicon
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── Dockerfile                      # Multi-stage build for HF Spaces
└── README.md
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS | UI with SSE streaming |
| Backend | Python FastAPI + SSE-Starlette | API + dialogue orchestration |
| Host A LLM | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Deep-diving, analytical host |
| Host B LLM | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` | Witty, concise reactor host |
| TTS (primary) | `edge-tts` (Microsoft Neural) | Free server-side speech synthesis |
| TTS (fallback) | Browser SpeechSynthesis API | Client-side fallback when edge-tts is blocked |
| Inference API | Crusoe Cloud Foundry (OpenAI-compatible) | All LLM calls |

---

## Quick Start (Local Development)

### Prerequisites

- Python 3.9+
- Node.js 18+
- A [Crusoe Cloud](https://crusoe.ai) API key

### 1. Clone and configure

```bash
git clone https://huggingface.co/spaces/crusoeai/<space-name>
cd fake-podcast

# Set up environment
cp backend/.env.example backend/.env
# Edit backend/.env and add your CRUSOE_API_KEY
```

### 2. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start the frontend (separate terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — the Vite dev server proxies `/api` requests to the backend on port 8000.

### 4. Use it

1. Type a topic or click **Inspire Me** for ideas
2. Hit **Generate** — the two hosts start talking
3. Watch the transcript stream in real-time
4. Press **Play** on the audio player to hear them speak
5. Hit **Stop** whenever you want to end the conversation

---

## Deploy to Hugging Face Spaces

### 1. Create a new Space

- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- Select **Docker** as the SDK
- Set visibility to Public or Private

### 2. Add your secret

In your Space settings, add:

| Secret Name | Value |
|-------------|-------|
| `CRUSOE_API_KEY` | Your Crusoe Cloud Foundry API key |

### 3. Push the code

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/fake-podcast
git push hf main
```

The Dockerfile handles everything — builds the React frontend, installs Python deps (including `edge-tts`), and starts FastAPI on port 7860.

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Stream a continuous conversation (SSE) |
| `POST` | `/api/tts` | Text-to-speech for a single line (returns MP3) |
| `GET`  | `/api/tts/test` | Check if server TTS is available |
| `POST` | `/api/inspire` | Get AI-generated topic ideas |
| `GET`  | `/api/hosts` | Default host configurations |
| `GET`  | `/api/categories` | Available inspiration categories |
| `GET`  | `/api/health` | Health check + model list |

### SSE Stream Format

`POST /api/generate` streams newline-delimited JSON events:

```
data: {"host": "a", "name": "Atlas", "color": "#8B5CF6", "text": "So picture this...", "round": 1, "total_rounds": 120}

data: {"host": "b", "name": "Nova", "color": "#10B981", "text": "Hold on, you're telling me...", "round": 2, "total_rounds": 120}

...continues until stopped or round limit...

data: [DONE]
```

**Request body:**

```json
{
  "topic": "Why do cats judge us?",
  "angle": "from the cat's perspective",
  "rounds": 0,
  "host_a": { "name": "Atlas", "personality": "Enthusiastic deep-diver..." },
  "host_b": { "name": "Nova", "personality": "Sharp, witty reactor..." }
}
```

- `rounds: 0` = continuous mode (runs until client disconnects or 120-round safety limit)
- `rounds: N` = fixed number of exchanges

### TTS Endpoint

`POST /api/tts` generates MP3 audio:

```json
{ "text": "So picture this...", "host": "a" }
```

- `host: "a"` = male voice (en-US-ChristopherNeural)
- `host: "b"` = female voice (en-US-AriaNeural)
- Returns `audio/mpeg` binary stream

### Crusoe Cloud Foundry

All LLM calls use the OpenAI-compatible SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1/",
    api_key="YOUR_CRUSOE_API_KEY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=1,
    top_p=0.95,
)
```

---

## Customization Guide

### Swap LLM models

Edit `backend/config.py`:

```python
HOST_A_MODEL = "your-org/your-model-a"
HOST_B_MODEL = "your-org/your-model-b"
```

Any OpenAI-compatible endpoint works — just change `CRUSOE_API_BASE` in `.env`.

### Change TTS voices

Edit `backend/config.py`:

```python
TTS_VOICE_A = "en-US-ChristopherNeural"  # Host A voice
TTS_VOICE_B = "en-US-AriaNeural"         # Host B voice
```

Run `edge-tts --list-voices` to see all available voices. Some good options:
- Male: `en-US-ChristopherNeural`, `en-GB-RyanNeural`, `en-US-EricNeural`
- Female: `en-US-AriaNeural`, `en-US-JennyNeural`, `en-GB-SoniaNeural`

### Modify host personalities

Edit `backend/config.py`:

```python
DEFAULT_HOST_A = {
    "name": "Atlas",
    "color": "#8B5CF6",
    "personality": "Your custom personality description...",
}
```

Or change them at runtime via the Hosts panel in the UI.

### Tweak conversation dynamics

Edit `backend/prompts.py`:

- `host_system_prompt()` — Each host's base instructions
- `response_prompt()` — The standard "keep talking" prompt
- `VARIETY_PROMPTS` — List of prompts injected every ~3 rounds (devil's advocate, tangents, hot takes, etc.)
- `opening_prompt()` — How the first host kicks things off

### Adjust conversation length

Edit `backend/main.py`:

```python
MAX_CONTEXT_TURNS = 20  # How many turns to keep in LLM context (sliding window)
MAX_ROUNDS = 120        # Safety limit (~20 min at ~10s/round)
```

### Use a different frontend

The backend is framework-agnostic. Key integration points:
1. `POST /api/generate` with `rounds: 0` — read the SSE stream
2. `POST /api/tts` for each line — returns MP3 audio
3. Abort the fetch to stop generation

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CRUSOE_API_KEY` | **Yes** | — | Your Crusoe Cloud API key |
| `CRUSOE_API_BASE` | No | `https://api.inference.crusoecloud.com/v1/` | LLM inference endpoint |
| `VOICECHAT_API_BASE` | No | (placeholder) | Reserved for future Crusoe VoiceChat integration |

---

## Troubleshooting

**Audio says "Browser Voices" instead of "AI Voices"**
- `edge-tts` connects to Microsoft's servers via WebSocket. Some Docker hosts block outbound WebSocket connections. The app automatically falls back to browser SpeechSynthesis. Both work fine.

**No audio at all**
- Make sure your browser supports the Web Speech API (Chrome, Edge, Safari do)
- Check the browser console for errors on `/api/tts`

**Generation stops unexpectedly**
- Check if your Crusoe API key is valid and has quota
- Look at the backend logs for error details

**Transcript appears but is slow**
- Each round requires an LLM API call. Speed depends on Crusoe inference latency.
- The sliding context window (20 turns) prevents requests from getting slower over time.

---

## License

MIT — fork it, remix it, ship it.

Built with [Crusoe Cloud Foundry](https://crusoe.ai) | React | FastAPI | edge-tts
