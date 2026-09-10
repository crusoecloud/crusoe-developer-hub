import gradio as gr
import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

API_URL = os.environ.get("API_URL", "https://api.inference.crusoecloud.com/v1")
API_KEY = os.environ.get("API_KEY", "")

FIELD_WIDTH = 40
FIELD_HEIGHT = 20
PADDLE_HEIGHT = 4
MAX_TICKS = 500

CRUSOE_MODELS = [
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "google/gemma-3-12b-it",
]

_executor = ThreadPoolExecutor(max_workers=2)


# ============================================================================
# GAME ENGINE
# ============================================================================

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def generate_field_ascii(state):
    field = []
    for y in range(FIELD_HEIGHT):
        row = []
        for x in range(FIELD_WIDTH):
            if x == 0 and state["p1_y"] <= y < state["p1_y"] + PADDLE_HEIGHT:
                row.append("|")
            elif x == FIELD_WIDTH - 1 and state["p2_y"] <= y < state["p2_y"] + PADDLE_HEIGHT:
                row.append("|")
            elif x == state["ball_x"] and y == state["ball_y"]:
                row.append("O")
            elif y == 0 or y == FIELD_HEIGHT - 1:
                row.append("-")
            else:
                row.append(".")
        field.append("".join(row))
    return "\n".join(field)


def get_llm_move(model, state, is_left_player):
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        paddle_side = "LEFT" if is_left_player else "RIGHT"
        my_y = state["p1_y"] if is_left_player else state["p2_y"]
        opp_y = state["p2_y"] if is_left_player else state["p1_y"]
        my_score = state["p1_score"] if is_left_player else state["p2_score"]
        opp_score = state["p2_score"] if is_left_player else state["p1_score"]

        prompt = f"""You are playing Pong. You control the {paddle_side} paddle.

FIELD (40x20):
{generate_field_ascii(state)}

YOUR PADDLE: rows {my_y} to {my_y + PADDLE_HEIGHT - 1} (column {'0' if is_left_player else '39'})
BALL POSITION: ({state['ball_x']}, {state['ball_y']}) moving ({state['ball_dx']:+d}, {state['ball_dy']:+d})
OPPONENT PADDLE: rows {opp_y} to {opp_y + PADDLE_HEIGHT - 1}
SCORE: You {my_score} - Opponent {opp_score}

Move your paddle: up, down, or stay"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Pong AI. Respond with ONLY one word: up, down, or stay."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )
        move_text = response.choices[0].message.content.strip().lower()
        if "up" in move_text:
            return "up"
        elif "down" in move_text:
            return "down"
        return "stay"
    except Exception as e:
        print(f"LLM Error: {e}")
        return "stay"


def reset_ball(state):
    state["ball_x"] = FIELD_WIDTH // 2
    state["ball_y"] = FIELD_HEIGHT // 2
    state["ball_dx"] = random.choice([-1, 1])
    state["ball_dy"] = random.choice([-1, 1])


def update_state(state, p1_move, p2_move):
    if p1_move == "up":
        state["p1_y"] = clamp(state["p1_y"] - 1, 0, FIELD_HEIGHT - PADDLE_HEIGHT)
    elif p1_move == "down":
        state["p1_y"] = clamp(state["p1_y"] + 1, 0, FIELD_HEIGHT - PADDLE_HEIGHT)

    if p2_move == "up":
        state["p2_y"] = clamp(state["p2_y"] - 1, 0, FIELD_HEIGHT - PADDLE_HEIGHT)
    elif p2_move == "down":
        state["p2_y"] = clamp(state["p2_y"] + 1, 0, FIELD_HEIGHT - PADDLE_HEIGHT)

    state["ball_x"] += state["ball_dx"]
    state["ball_y"] += state["ball_dy"]

    if state["ball_y"] <= 0 or state["ball_y"] >= FIELD_HEIGHT - 1:
        state["ball_dy"] *= -1
        state["ball_y"] = clamp(state["ball_y"], 0, FIELD_HEIGHT - 1)

    if (state["ball_x"] == 1 and
        state["p1_y"] <= state["ball_y"] < state["p1_y"] + PADDLE_HEIGHT):
        state["ball_dx"] = 1
        state["ball_x"] = 1

    if (state["ball_x"] == FIELD_WIDTH - 2 and
        state["p2_y"] <= state["ball_y"] < state["p2_y"] + PADDLE_HEIGHT):
        state["ball_dx"] = -1
        state["ball_x"] = FIELD_WIDTH - 2

    if state["ball_x"] < 0:
        state["p2_score"] += 1
        reset_ball(state)
    elif state["ball_x"] >= FIELD_WIDTH:
        state["p1_score"] += 1
        reset_ball(state)

    state["tick"] += 1


# ============================================================================
# GAME RUNNER
# ============================================================================

def run_game(model1, model2, points_to_win):
    if not API_KEY:
        yield json.dumps({"error": "API_KEY not set"}), "Error: Set API_KEY secret in Space settings."
        return

    winning = int(points_to_win)
    state = {
        "p1_y": FIELD_HEIGHT // 2 - PADDLE_HEIGHT // 2,
        "p2_y": FIELD_HEIGHT // 2 - PADDLE_HEIGHT // 2,
        "ball_x": FIELD_WIDTH // 2,
        "ball_y": FIELD_HEIGHT // 2,
        "ball_dx": random.choice([-1, 1]),
        "ball_dy": random.choice([-1, 1]),
        "p1_score": 0,
        "p2_score": 0,
        "tick": 0,
        "game_over": False,
        "winner": None,
    }

    m1 = model1.split("/")[-1]
    m2 = model2.split("/")[-1]
    log_lines = []

    def make_json(s):
        return json.dumps({
            "p1_y": s["p1_y"], "p2_y": s["p2_y"],
            "ball_x": s["ball_x"], "ball_y": s["ball_y"],
            "p1_score": s["p1_score"], "p2_score": s["p2_score"],
            "field_width": FIELD_WIDTH, "field_height": FIELD_HEIGHT,
            "paddle_height": PADDLE_HEIGHT,
            "gameOver": s["game_over"], "winner": s["winner"],
            "turn": s["tick"], "model1": m1, "model2": m2,
        })

    yield make_json(state), ""

    while not state["game_over"]:
        f1 = _executor.submit(get_llm_move, model1, state, True)
        f2 = _executor.submit(get_llm_move, model2, state, False)
        p1_move = f1.result()
        p2_move = f2.result()

        log_lines.append(f"T{state['tick']}: P1={p1_move} P2={p2_move} | {state['p1_score']}-{state['p2_score']}")

        update_state(state, p1_move, p2_move)

        if state["p1_score"] >= winning:
            state["game_over"] = True
            state["winner"] = m1
        elif state["p2_score"] >= winning:
            state["game_over"] = True
            state["winner"] = m2
        elif state["tick"] >= MAX_TICKS:
            state["game_over"] = True
            if state["p1_score"] > state["p2_score"]:
                state["winner"] = m1
            elif state["p2_score"] > state["p1_score"]:
                state["winner"] = m2
            else:
                state["winner"] = "Tie"

        yield make_json(state), "\n".join(log_lines[-20:])

    yield make_json(state), "\n".join(log_lines[-20:])


# ============================================================================
# CANVAS RENDERER
# ============================================================================

GAME_HTML = """
<div style="display:flex;justify-content:center;padding:20px;">
  <canvas id="pongCanvas" width="600" height="300"
    style="border:2px solid #22d3ee;border-radius:12px;background:#0a0a1a;"></canvas>
</div>
<script>
window.renderPong = function(jsonStr) {
  try {
    const data = JSON.parse(jsonStr);
    if (data.error) return;

    const canvas = document.getElementById('pongCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const scaleX = W / data.field_width;
    const scaleY = H / data.field_height;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);

    // Center line
    ctx.strokeStyle = '#334155';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W / 2, 0);
    ctx.lineTo(W / 2, H);
    ctx.stroke();
    ctx.setLineDash([]);

    // P1 paddle (cyan)
    const pw = scaleX * 0.8;
    ctx.shadowColor = '#22d3ee';
    ctx.shadowBlur = 15;
    ctx.fillStyle = '#22d3ee';
    ctx.fillRect(scaleX * 0.5, data.p1_y * scaleY, pw, data.paddle_height * scaleY);

    // P2 paddle (pink)
    ctx.shadowColor = '#f472b6';
    ctx.fillStyle = '#f472b6';
    ctx.fillRect(W - scaleX * 1.3, data.p2_y * scaleY, pw, data.paddle_height * scaleY);

    // Ball
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 20;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(data.ball_x * scaleX, data.ball_y * scaleY, scaleX * 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Score
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(data.p1_score + '  -  ' + data.p2_score, W / 2, 10);

    // Model names
    ctx.font = 'bold 12px monospace';
    ctx.fillStyle = '#22d3ee';
    ctx.textAlign = 'left';
    ctx.fillText(data.model1, 10, H - 18);
    ctx.fillStyle = '#f472b6';
    ctx.textAlign = 'right';
    ctx.fillText(data.model2, W - 10, H - 18);

    // Turn counter
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Turn ' + data.turn, W / 2, H - 18);

    // Game over
    if (data.gameOver) {
      ctx.fillStyle = 'rgba(10,10,26,0.7)';
      ctx.fillRect(0, 0, W, H);

      ctx.font = 'bold 32px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#fbbf24';
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 20;
      ctx.fillText('GAME OVER', W / 2, H / 2 - 24);
      ctx.shadowBlur = 0;

      ctx.font = 'bold 20px monospace';
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText('Winner: ' + data.winner, W / 2, H / 2 + 16);
    }
  } catch (e) { console.error('Pong render error:', e); }
};
</script>
"""


# ============================================================================
# GRADIO UI
# ============================================================================

CSS = """
.gradio-container { max-width: 1200px !important; }
footer { display: none !important; }
.log-box textarea {
  font-family: "SF Mono","Fira Code",monospace !important;
  font-size: 12px !important;
  background: #0f172a !important;
  color: #a5f3fc !important;
  border: 1px solid #1e293b !important;
}
ul.options { background: #1e293b !important; }
ul.options li { color: #e2e8f0 !important; }
ul.options li:hover, ul.options li.active { background: #334155 !important; }
ul.options li.selected { background: #0e7490 !important; }
input.search-input { background: #1e293b !important; color: #e2e8f0 !important; }
"""

THEME = gr.themes.Base(
    primary_hue="cyan", secondary_hue="pink", neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0a0a1a", body_background_fill_dark="#0a0a1a",
    block_background_fill="#111827", block_background_fill_dark="#111827",
    block_border_color="#1f2937", block_border_color_dark="#1f2937",
    block_label_text_color="#e2e8f0", block_label_text_color_dark="#e2e8f0",
    body_text_color="#e2e8f0", body_text_color_dark="#e2e8f0",
    input_background_fill="#1e293b", input_background_fill_dark="#1e293b",
    input_border_color="#334155", input_border_color_dark="#334155",
    button_primary_background_fill="#06b6d4", button_primary_background_fill_dark="#06b6d4",
    button_primary_text_color="#000", button_primary_text_color_dark="#000",
)

with gr.Blocks(css=CSS, title="Pong AI", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px 0;">
        <h1 style="font-size:2.2em;margin:0;color:#e2e8f0;">Pong AI</h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Two LLM models compete in Pong &mdash;
            Powered by <strong style="color:#06b6d4;">Crusoe Intelligence Foundry</strong>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<h4 style="color:#22d3ee;margin:0 0 6px 0;">Player 1 (Left - Cyan)</h4>')
            model1 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[0],
                                 label="Model", allow_custom_value=True)
            gr.HTML('<h4 style="color:#f472b6;margin:12px 0 6px 0;">Player 2 (Right - Pink)</h4>')
            model2 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[2],
                                 label="Model", allow_custom_value=True)
            points_to_win = gr.Slider(minimum=3, maximum=11, value=5, step=1,
                                      label="Points to Win")
            start_btn = gr.Button("Start Battle!", variant="primary", size="lg")

            gr.HTML('<h4 style="color:#94a3b8;margin:16px 0 4px 0;">Move Log</h4>')
            move_log = gr.Textbox(lines=10, max_lines=20, interactive=False,
                                  show_label=False, elem_classes=["log-box"])

        with gr.Column(scale=2, min_width=620):
            game_canvas = gr.HTML(GAME_HTML)
            game_state = gr.Textbox(visible=False)

    gr.HTML("""
    <div style="text-align:center;padding:16px 0;color:#475569;font-size:0.85em;">
        Open models on Crusoe Intelligence Foundry
    </div>
    """)

    start_btn.click(
        fn=run_game,
        inputs=[model1, model2, points_to_win],
        outputs=[game_state, move_log],
    )

    game_state.change(
        fn=None,
        inputs=[game_state],
        outputs=None,
        js="(s) => { if (s && window.renderPong) window.renderPong(s); }",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
