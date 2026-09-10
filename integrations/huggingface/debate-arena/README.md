---
title: Debate Arena — Gemma 4 vs Gemma 4
emoji: 🎭
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Two Gemma 4 personas debate on Crusoe Foundry
---

# Debate Arena

Two cartoon characters, each backed by its own instance of **`google/gemma-4-31b-it`** on **Crusoe Foundry**, argue a topic you give them. Responses stream live into speech bubbles above their heads while their mouths animate in sync.

---

## How It Works

```
User picks topic + persona pair + rounds
                    │
                    ▼
            POST /api/debate (SSE)
                    │
                    ▼
  ┌───────────────────────────────────┐
  │ FastAPI orchestrator              │
  │  for round in 1..N:               │
  │    for speaker in [A, B]:         │
  │      Crusoe chat.completions      │
  │        (stream=True)              │
  │      for delta in stream:         │
  │        yield SSE {chunk, ...}     │
  └───────────────┬───────────────────┘
                  │ data: {...}\n\n
                  ▼
  ┌───────────────────────────────────┐
  │ React frontend (Vite + TS)        │
  │  - text streams into Bubble       │
  │  - active Character mouth-syncs   │
  │  - TranscriptLog accumulates      │
  └───────────────────────────────────┘
```

### Why this pattern

- **One model, two system prompts** — the divergent personas are the entire recipe
- **Per-token streaming** — each SSE event carries one `delta.content`; frontend appends to the active speaker's bubble and toggles the mouth animation
- **Independent histories** — each debater sees the exchange from their own POV so the conversation stays coherent even with opposing framing

---

## Persona Pairs

| ID | A | B |
|---|---|---|
| `optimist-skeptic` | 😊 Sunny (optimist) | 🧐 Dex (skeptic) |
| `philosopher-pragmatist` | 📚 Sophia (philosopher) | 🛠️ Max (pragmatist) |
| `progressive-traditionalist` | 🚀 Nova (progressive) | 🏛️ Oren (traditionalist) |
| `idealist-realist` | ✨ Iris (idealist) | 🎯 Rex (realist) |
| `scientist-artist` | 🔬 Ada (scientist) | 🎨 Juno (artist) |

Each persona has a color, emoji, and personality prompt (see `backend/prompts.py`).

---

## Configuration

- **Endpoint:** `https://api.inference.crusoecloud.com/v1/`
- **Model:** `google/gemma-4-31b-it`
- **Env vars:**
  - `CRUSOE_API_KEY` (required) — set as a **HuggingFace Space Secret** for deployment, or create a local `.env` file for development
  - `CRUSOE_API_BASE` (optional, defaults to the Crusoe proxy above)
  - `DEBATE_MODEL` (optional, defaults to `google/gemma-4-31b-it`)

Example local `.env` (create it in `backend/`):

```
CRUSOE_API_KEY=your-key
```

---

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`. Open http://localhost:5173.

### Smoke-test the stream directly

```bash
curl -N -X POST http://localhost:8000/api/debate \
  -H 'Content-Type: application/json' \
  -d '{"topic":"Is pineapple acceptable on pizza?","rounds":2,"persona_pair_id":"optimist-skeptic"}'
```

You should see `data: {...}\n\n` frames arriving in real time.

---

## Deployment (HuggingFace Space)

1. Create a new Space with **SDK = Docker**
2. Push this repo to the Space
3. Add `CRUSOE_API_KEY` as a **Space Secret**
4. The `Dockerfile` handles the multi-stage build (node frontend → python runtime)

The `app_port` in the frontmatter matches the `7860` exposed by the Dockerfile.

---

## Project Layout

```
debate-arena/
├── Dockerfile
├── README.md
├── backend/
│   ├── config.py          # env + Crusoe constants
│   ├── main.py            # FastAPI app + SSE streaming orchestrator
│   ├── prompts.py         # persona pairs + prompt builders
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tailwind.config.js
    ├── postcss.config.js
    └── src/
        ├── main.tsx
        ├── App.tsx        # top-level state + SSE consumption
        ├── api.ts         # streamDebate() async generator
        ├── index.css      # Tailwind + custom keyframes
        ├── types.ts
        └── components/
            ├── Character.tsx    # animated SVG avatar
            ├── Bubble.tsx       # speech bubble w/ streaming caret
            ├── Stage.tsx        # two-character layout
            ├── Controls.tsx     # topic/rounds/persona form
            └── TranscriptLog.tsx
```

---

## License

MIT
