---
title: Arcade Race
emoji: 🕹️
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
python_version: "3.12"
pinned: false
short_description: Race four LLMs to generate playable 80s arcade games
---

# Arcade Race

Type a prompt. Four open-source LLMs race in parallel to generate a single
self-contained HTML5 arcade game. Watch the timers tick, then play each
result side-by-side.

All inference runs on **Crusoe Managed AI**.

## Setup

Set `CRUSOE_API_KEY` as a Space secret (Settings → Variables and secrets →
Secret tab), then Factory rebuild.

## Local

```bash
export CRUSOE_API_KEY=...
pip install -r requirements.txt
python app.py
```
