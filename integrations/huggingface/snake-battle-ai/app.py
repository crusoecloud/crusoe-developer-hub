"""
Snake Battle AI - Two LLM models compete in a real-time Snake game.
Powered by Crusoe Intelligence Foundry. Built for Hugging Face Spaces.
"""

import gradio as gr
import json
import os
import time
import random
import concurrent.futures
from openai import OpenAI
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# GAME ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

DIRECTION_DELTAS = {
    Direction.UP: (0, -1), Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0),
}
OPPOSITE = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
}


class SnakeGame:
    def __init__(self, width=20, height=20, max_turns=200):
        self.width = width
        self.height = height
        self.max_turns = max_turns
        self.turn = 0
        self.game_over = False
        self.winner = None
        self.food = []

        mid = height // 2
        self.snake1 = {
            "body": [(3, mid), (2, mid), (1, mid)],
            "direction": Direction.RIGHT, "alive": True, "score": 0,
        }
        self.snake2 = {
            "body": [(width - 4, mid), (width - 3, mid), (width - 2, mid)],
            "direction": Direction.LEFT, "alive": True, "score": 0,
        }
        for _ in range(3):
            self._spawn_food()

    def _occupied(self):
        occ = set()
        for s in (self.snake1, self.snake2):
            for seg in s["body"]:
                occ.add(seg)
        for f in self.food:
            occ.add(f)
        return occ

    def _spawn_food(self):
        occ = self._occupied()
        empty = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occ]
        if empty:
            self.food.append(random.choice(empty))

    def get_state_for_llm(self, player: int) -> str:
        me = self.snake1 if player == 1 else self.snake2
        opp = self.snake2 if player == 1 else self.snake1
        head = me["body"][0]
        oh = opp["body"][0]

        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for seg in me["body"][1:]:
            if 0 <= seg[0] < self.width and 0 <= seg[1] < self.height:
                grid[seg[1]][seg[0]] = "s"
        grid[head[1]][head[0]] = "H"
        for seg in opp["body"][1:]:
            if 0 <= seg[0] < self.width and 0 <= seg[1] < self.height:
                grid[seg[1]][seg[0]] = "e"
        if 0 <= oh[0] < self.width and 0 <= oh[1] < self.height:
            grid[oh[1]][oh[0]] = "E"
        for f in self.food:
            grid[f[1]][f[0]] = "F"

        board_str = "\n".join("".join(row) for row in grid)
        blocked = {"up": "down", "down": "up", "left": "right", "right": "left"}[me["direction"].value]

        return f"""You are playing a competitive Snake game on a {self.width}x{self.height} grid.

BOARD:
{board_str}

LEGEND: H=your head, s=your body, E=enemy head, e=enemy body, F=food, .=empty
YOUR HEAD: ({head[0]}, {head[1]}) | LENGTH: {len(me['body'])} | DIRECTION: {me['direction'].value}
ENEMY HEAD: ({oh[0]}, {oh[1]}) | ENEMY LENGTH: {len(opp['body'])}
FOOD: {', '.join(f'({f[0]},{f[1]})' for f in self.food)}

RULES:
- Edges wrap around (going off the right side puts you on the left, etc.)
- You die if you hit your own body or the enemy body
- There are 3 food items on the board. First to eat 2 wins!
- You CANNOT reverse direction (cannot go {blocked})

STRATEGY: Race to the nearest food! Avoid all snake bodies. Take the shortest path.

Respond with ONLY one word: up, down, left, or right"""

    def parse_move(self, response: str, player: int) -> Direction:
        snake = self.snake1 if player == 1 else self.snake2
        resp = response.strip().lower()
        for d in Direction:
            if d.value in resp:
                if d != OPPOSITE[snake["direction"]]:
                    return d
        return snake["direction"]

    def step(self, move1: Direction, move2: Direction):
        if self.game_over:
            return
        self.turn += 1
        snakes = [self.snake1, self.snake2]
        moves = [move1, move2]
        new_heads = []

        for i, s in enumerate(snakes):
            if not s["alive"]:
                new_heads.append(None)
                continue
            d = moves[i]
            if d == OPPOSITE[s["direction"]]:
                d = s["direction"]
            s["direction"] = d
            dx, dy = DIRECTION_DELTAS[d]
            hx, hy = s["body"][0]
            # Wrap around edges
            new_heads.append(((hx + dx) % self.width, (hy + dy) % self.height))

        deaths = [False, False]
        for i, s in enumerate(snakes):
            if not s["alive"] or new_heads[i] is None:
                continue
            nx, ny = new_heads[i]
            if (nx, ny) in s["body"][:-1]:
                deaths[i] = True; continue
            j = 1 - i
            if snakes[j]["alive"] and (nx, ny) in snakes[j]["body"][:-1]:
                deaths[i] = True

        if new_heads[0] and new_heads[1] and new_heads[0] == new_heads[1]:
            deaths[0] = deaths[1] = True

        for i, s in enumerate(snakes):
            if not s["alive"] or new_heads[i] is None:
                continue
            if deaths[i]:
                s["alive"] = False; continue
            s["body"].insert(0, new_heads[i])
            ate = False
            for fi, f in enumerate(self.food):
                if new_heads[i] == f:
                    ate = True; s["score"] += 1
                    self.food.pop(fi)
                    if s["score"] >= 2:
                        self.game_over = True
                        self.winner = "Player 1" if s is self.snake1 else "Player 2"
                    else:
                        self._spawn_food()
                    break
            if not ate:
                s["body"].pop()

        if not self.game_over:
            alive = [s for s in snakes if s["alive"]]
            if len(alive) == 0:
                self.game_over = True; self.winner = "Draw"
            elif len(alive) == 1:
                self.game_over = True
                self.winner = "Player 1" if alive[0] is self.snake1 else "Player 2"
            elif self.turn >= self.max_turns:
                self.game_over = True; self.winner = "Draw"

    def to_json(self):
        return {
            "turn": self.turn,
            "snake1": {"body": list(self.snake1["body"]), "alive": self.snake1["alive"],
                       "score": self.snake1["score"], "direction": self.snake1["direction"].value},
            "snake2": {"body": list(self.snake2["body"]), "alive": self.snake2["alive"],
                       "score": self.snake2["score"], "direction": self.snake2["direction"].value},
            "food": list(self.food),
            "game_over": self.game_over,
            "winner": self.winner,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG (all from environment / defaults)
# ═══════════════════════════════════════════════════════════════════════════════

API_URL = os.environ.get("API_URL", "https://api.inference.crusoecloud.com/v1")
API_KEY = os.environ.get("API_KEY", "")
GRID_SIZE = int(os.environ.get("GRID_SIZE", "20"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "200"))

CRUSOE_MODELS = [
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "google/gemma-3-12b-it",
]

# Persistent client for connection reuse (much faster than creating per-call)
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_URL, api_key=API_KEY)
    return _client


# ═══════════════════════════════════════════════════════════════════════════════
# LLM AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def query_llm(model: str, prompt: str) -> tuple:
    """Query an LLM via OpenAI-compatible API. Returns (response_text, latency_ms)."""
    client = get_client()
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Snake game AI player. You must respond with ONLY one word: up, down, left, or right. No explanation, no punctuation, just the direction."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.2,
        )
        lat = (time.time() - start) * 1000
        return resp.choices[0].message.content.strip().lower(), lat
    except Exception as e:
        lat = (time.time() - start) * 1000
        return f"error:{e}", lat


# ═══════════════════════════════════════════════════════════════════════════════
# GAME RUNNER (generator for Gradio streaming)
# ═══════════════════════════════════════════════════════════════════════════════

# Persistent thread pool for parallel LLM calls
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def run_game(model1, model2):
    """Run a full game, yielding (game_json, log1, log2) each turn for live rendering."""
    if not API_KEY:
        yield json.dumps({"error": "API key not configured. Set the API_KEY secret in Space settings."}), "", ""
        return

    game = SnakeGame(width=GRID_SIZE, height=GRID_SIZE, max_turns=MAX_TURNS)
    m1_name = model1.split("/")[-1]
    m2_name = model2.split("/")[-1]
    p1_log, p2_log = [], []

    def make_frame(game, **extras):
        f = game.to_json()
        f["board_size"] = GRID_SIZE
        f["model1"] = m1_name
        f["model2"] = m2_name
        f.update(extras)
        return json.dumps(f)

    # Yield initial state
    yield make_frame(game), "", ""

    while not game.game_over:
        prompt1 = game.get_state_for_llm(1)
        prompt2 = game.get_state_for_llm(2)

        # Query both models in PARALLEL for faster gameplay
        f1 = _executor.submit(query_llm, model1, prompt1)
        f2 = _executor.submit(query_llm, model2, prompt2)
        resp1, lat1 = f1.result()
        resp2, lat2 = f2.result()

        move1 = game.parse_move(resp1, 1)
        move2 = game.parse_move(resp2, 2)

        p1_log.append(f"T{game.turn+1}: {resp1} -> {move1.value} ({lat1:.0f}ms)")
        p2_log.append(f"T{game.turn+1}: {resp2} -> {move2.value} ({lat2:.0f}ms)")

        game.step(move1, move2)

        yield (
            make_frame(game, p1_move=move1.value, p2_move=move2.value, p1_lat=lat1, p2_lat=lat2),
            "\n".join(p1_log[-20:]),
            "\n".join(p2_log[-20:]),
        )

    # Final yield
    yield make_frame(game), "\n".join(p1_log[-20:]), "\n".join(p2_log[-20:])


# ═══════════════════════════════════════════════════════════════════════════════
# CANVAS RENDERER (HTML + JS)
# ═══════════════════════════════════════════════════════════════════════════════

GAME_HTML = """
<div id="game-container" style="width:100%;display:flex;flex-direction:column;align-items:center;">
  <canvas id="snakeCanvas" width="600" height="600"
    style="border:2px solid #1e293b;border-radius:12px;background:#0a0a1a;image-rendering:pixelated;"></canvas>
  <div id="game-status" style="margin-top:12px;font-size:18px;font-weight:bold;color:#e2e8f0;text-align:center;min-height:32px;">
    <span style="color:#64748b;">Click Start Battle to begin!</span>
  </div>
</div>

<script>
// Global render function called by Gradio JS callbacks
window.renderSnakeGame = function(jsonStr) {
  try {
    const data = JSON.parse(jsonStr);
    if (data.error) {
      const st = document.getElementById('game-status');
      if (st) st.innerHTML = '<span style="color:#ef4444;">' + data.error + '</span>';
      return;
    }

    const canvas = document.getElementById('snakeCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const sz = data.board_size || 20;
    const cw = W / sz, ch = H / sz;

    // ── Background ──
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);

    // ── Subtle grid ──
    ctx.strokeStyle = 'rgba(30,41,59,0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= sz; i++) {
      ctx.beginPath(); ctx.moveTo(i*cw, 0); ctx.lineTo(i*cw, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i*ch); ctx.lineTo(W, i*ch); ctx.stroke();
    }

    // ── Food (glowing red dots) ──
    (data.food || []).forEach(f => {
      const fx = f[0]*cw + cw/2, fy = f[1]*ch + ch/2;
      // Glow
      const grd = ctx.createRadialGradient(fx, fy, 0, fx, fy, cw*0.6);
      grd.addColorStop(0, 'rgba(239,68,68,0.9)');
      grd.addColorStop(0.5, 'rgba(239,68,68,0.3)');
      grd.addColorStop(1, 'rgba(239,68,68,0)');
      ctx.fillStyle = grd;
      ctx.fillRect(f[0]*cw - cw*0.1, f[1]*ch - ch*0.1, cw*1.2, ch*1.2);
      // Core
      ctx.fillStyle = '#ef4444';
      ctx.beginPath(); ctx.arc(fx, fy, cw/3.5, 0, Math.PI*2); ctx.fill();
    });

    // ── Draw snake helper ──
    function drawSnake(snake, headColor, bodyR, bodyG, bodyB) {
      if (!snake || !snake.body) return;
      const body = snake.body;
      const len = body.length;

      // Body segments (tail to head for proper layering)
      for (let i = len - 1; i >= 0; i--) {
        const seg = body[i];
        const t = 1 - (i / Math.max(len, 1)) * 0.5; // fade toward tail
        const r = Math.round(bodyR * t);
        const g = Math.round(bodyG * t);
        const b = Math.round(bodyB * t);

        if (i === 0) {
          // Head: brighter with glow
          ctx.shadowColor = headColor;
          ctx.shadowBlur = 12;
          ctx.fillStyle = headColor;
          // Rounded head
          const rx = seg[0]*cw + 1, ry = seg[1]*ch + 1, rw = cw - 2, rh = ch - 2, rad = 4;
          ctx.beginPath();
          ctx.moveTo(rx + rad, ry);
          ctx.lineTo(rx + rw - rad, ry);
          ctx.quadraticCurveTo(rx + rw, ry, rx + rw, ry + rad);
          ctx.lineTo(rx + rw, ry + rh - rad);
          ctx.quadraticCurveTo(rx + rw, ry + rh, rx + rw - rad, ry + rh);
          ctx.lineTo(rx + rad, ry + rh);
          ctx.quadraticCurveTo(rx, ry + rh, rx, ry + rh - rad);
          ctx.lineTo(rx, ry + rad);
          ctx.quadraticCurveTo(rx, ry, rx + rad, ry);
          ctx.closePath();
          ctx.fill();
          ctx.shadowBlur = 0;

          // Eyes
          const ex1 = seg[0]*cw + cw*0.3, ex2 = seg[0]*cw + cw*0.7, ey = seg[1]*ch + ch*0.35;
          ctx.fillStyle = '#fff';
          ctx.beginPath(); ctx.arc(ex1, ey, 2.5, 0, Math.PI*2); ctx.fill();
          ctx.beginPath(); ctx.arc(ex2, ey, 2.5, 0, Math.PI*2); ctx.fill();
          ctx.fillStyle = '#000';
          ctx.beginPath(); ctx.arc(ex1, ey, 1.2, 0, Math.PI*2); ctx.fill();
          ctx.beginPath(); ctx.arc(ex2, ey, 1.2, 0, Math.PI*2); ctx.fill();
        } else {
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(seg[0]*cw + 1.5, seg[1]*ch + 1.5, cw - 3, ch - 3);
        }
      }

      // Death marker
      if (!snake.alive && body.length > 0) {
        const h = body[0];
        ctx.font = 'bold ' + Math.round(cw * 0.8) + 'px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#fff';
        ctx.fillText('X', h[0]*cw + cw/2, h[1]*ch + ch/2);
      }
    }

    // ── Snake 1 (Cyan) ──
    drawSnake(data.snake1, '#22d3ee', 34, 211, 238);

    // ── Snake 2 (Pink/Red) ──
    drawSnake(data.snake2, '#f472b6', 244, 114, 182);

    // ── HUD bar ──
    ctx.shadowBlur = 0;
    const hudH = 38;
    const hudGrd = ctx.createLinearGradient(0, 0, 0, hudH);
    hudGrd.addColorStop(0, 'rgba(0,0,0,0.85)');
    hudGrd.addColorStop(1, 'rgba(0,0,0,0.4)');
    ctx.fillStyle = hudGrd;
    ctx.fillRect(0, 0, W, hudH);

    // Separator line
    ctx.strokeStyle = 'rgba(100,116,139,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, hudH); ctx.lineTo(W, hudH); ctx.stroke();

    ctx.font = 'bold 14px "SF Mono", "Fira Code", monospace';

    // Player 1 info
    ctx.fillStyle = '#22d3ee';
    ctx.textAlign = 'left';
    const p1Text = (data.model1 || 'Player 1') + ': ' + (data.snake1?.score ?? 0);
    ctx.fillText(p1Text, 12, 25);

    // Turn counter
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText('Turn ' + (data.turn || 0), W/2, 25);

    // Player 2 info
    ctx.fillStyle = '#f472b6';
    ctx.textAlign = 'right';
    const p2Text = (data.model2 || 'Player 2') + ': ' + (data.snake2?.score ?? 0);
    ctx.fillText(p2Text, W - 12, 25);

    // ── Game Over overlay ──
    const statusEl = document.getElementById('game-status');
    if (data.game_over && data.winner) {
      // Semi-transparent overlay
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, H/2 - 50, W, 100);

      ctx.font = 'bold 28px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      let msg, color;
      if (data.winner === 'Draw') {
        msg = 'DRAW!'; color = '#fbbf24';
      } else if (data.winner === 'Player 1') {
        msg = (data.model1 || 'Player 1') + ' WINS!'; color = '#22d3ee';
      } else {
        msg = (data.model2 || 'Player 2') + ' WINS!'; color = '#f472b6';
      }
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 20;
      ctx.fillText(msg, W/2, H/2 - 8);
      ctx.shadowBlur = 0;

      ctx.font = '16px "Inter", sans-serif';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('Score: ' + (data.snake1?.score??0) + ' - ' + (data.snake2?.score??0) + '  |  Turns: ' + data.turn, W/2, H/2 + 24);

      if (statusEl) statusEl.innerHTML = '<span style="color:' + color + ';">' + msg + '</span>';
    } else if (data.turn > 0) {
      if (statusEl) statusEl.innerHTML = '<span style="color:#4ade80;">Game in progress...</span>';
    }

  } catch(e) { console.error('Snake render error:', e); }
};
</script>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 1400px !important; }
#game-canvas { min-height: 660px; }
.log-box textarea {
  font-family: "SF Mono","Fira Code",monospace !important;
  font-size: 12px !important;
  background: #0f172a !important;
  color: #a5f3fc !important;
  border: 1px solid #1e293b !important;
}
.log-box-p2 textarea { color: #fbcfe8 !important; }
/* Dropdown menu styling */
ul.options { background: #1e293b !important; }
ul.options li { color: #e2e8f0 !important; }
ul.options li:hover, ul.options li.active { background: #334155 !important; }
ul.options li.selected { background: #0e7490 !important; }
input.search-input { background: #1e293b !important; color: #e2e8f0 !important; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue="cyan", secondary_hue="pink", neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0a0a1a", body_background_fill_dark="#0a0a1a",
    block_background_fill="#111827", block_background_fill_dark="#111827",
    block_border_color="#1f2937", block_border_color_dark="#1f2937",
    block_label_text_color="#e2e8f0", block_label_text_color_dark="#e2e8f0",
    block_title_text_color="#e2e8f0", block_title_text_color_dark="#e2e8f0",
    body_text_color="#e2e8f0", body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8", body_text_color_subdued_dark="#94a3b8",
    input_background_fill="#1e293b", input_background_fill_dark="#1e293b",
    input_border_color="#334155", input_border_color_dark="#334155",
    button_primary_background_fill="#06b6d4", button_primary_background_fill_dark="#06b6d4",
    button_primary_text_color="#000", button_primary_text_color_dark="#000",
)

with gr.Blocks(css=CSS, title="Snake Battle AI - Crusoe Foundry", theme=THEME) as demo:

    # ── Header ──
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 12px 0;">
        <h1 style="font-size:2.4em;margin:0;color:#e2e8f0;letter-spacing:-0.02em;">
            Snake Battle AI
        </h1>
        <p style="color:#94a3b8;margin:8px 0 0 0;font-size:1.05em;">
            Two open LLM models race to eat the food &mdash; first to grab it wins!<br>
            Powered by <strong style="color:#06b6d4;">Crusoe Intelligence Foundry</strong>
        </p>
    </div>
    """)

    with gr.Row():
        # ── Left Panel: Controls ──
        with gr.Column(scale=1, min_width=320):
            gr.HTML('<h4 style="color:#22d3ee;margin:0 0 6px 0;">Player 1 (Cyan Snake)</h4>')
            model1 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[0],
                                 label="Model", allow_custom_value=True)

            gr.HTML('<h4 style="color:#f472b6;margin:12px 0 6px 0;">Player 2 (Pink Snake)</h4>')
            model2 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[2],
                                 label="Model", allow_custom_value=True)

            start_btn = gr.Button("Start Battle!", variant="primary", size="lg")

            # ── Logs ──
            gr.HTML('<h4 style="color:#22d3ee;margin:12px 0 4px 0;">Player 1 Move Log</h4>')
            log1 = gr.Textbox(lines=8, max_lines=20, interactive=False,
                              show_label=False, elem_classes=["log-box"])
            gr.HTML('<h4 style="color:#f472b6;margin:12px 0 4px 0;">Player 2 Move Log</h4>')
            log2 = gr.Textbox(lines=8, max_lines=20, interactive=False,
                              show_label=False, elem_classes=["log-box", "log-box-p2"])

        # ── Right Panel: Game Canvas ──
        with gr.Column(scale=2, min_width=620):
            game_canvas = gr.HTML(GAME_HTML, elem_id="game-canvas")
            game_state = gr.Textbox(visible=False, elem_id="game-state-holder")

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center;padding:20px 0;color:#475569;font-size:0.85em;">
                Open models running on Crusoe Intelligence Foundry
    </div>
    """)

    # ── Event wiring ──
    start_btn.click(
        fn=run_game,
        inputs=[model1, model2],
        outputs=[game_state, log1, log2],
    )

    game_state.change(
        fn=None,
        inputs=[game_state],
        outputs=None,
        js="(s) => { if (s && window.renderSnakeGame) window.renderSnakeGame(s); }",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
