import os
import json
import gradio as gr
from openai import OpenAI
from typing import List, Optional
import re
import random

API_URL = os.environ.get("API_URL", "https://api.inference.crusoecloud.com/v1")
API_KEY = os.environ.get("API_KEY", "")

CRUSOE_MODELS = [
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "google/gemma-3-12b-it",
]


# ============================================================================
# GAME ENGINE
# ============================================================================

class ConnectFourGame:
    def __init__(self):
        self.board = [[0] * 7 for _ in range(6)]
        self.current_player = 1
        self.winner = None
        self.move_count = 0
        self.game_over = False
        self.last_move = None

    def drop_piece(self, column: int) -> bool:
        if column < 0 or column >= 7 or self.board[0][column] != 0:
            return False
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                self.last_move = (row, column)
                self.move_count += 1
                self._check_winner()
                return True
        return False

    def _check_winner(self):
        if self.last_move is None:
            return
        row, col = self.last_move
        player = self.board[row][col]

        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= 4:
                self.winner = player
                self.game_over = True
                return

        if self.move_count >= 42:
            self.game_over = True
            self.winner = 0

    def get_valid_columns(self) -> List[int]:
        return [c for c in range(7) if self.board[0][c] == 0]

    def get_board_ascii(self, player_num: int) -> str:
        lines = ["Board (. = empty, X = you, O = opponent):",""]
        for row in range(6):
            line = ""
            for col in range(7):
                cell = self.board[row][col]
                if cell == 0: line += ". "
                elif cell == player_num: line += "X "
                else: line += "O "
            lines.append(line)
        lines.append("0 1 2 3 4 5 6  (columns)")
        lines.append(f"Valid columns: {self.get_valid_columns()}")
        return "\n".join(lines)

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def to_json(self, model1, model2):
        return json.dumps({
            "board": self.board,
            "current_player": self.current_player,
            "winner": self.winner,
            "move_count": self.move_count,
            "game_over": self.game_over,
            "last_move": self.last_move,
            "models": [model1, model2],
        })


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_llm_move(model: str, game: ConnectFourGame, player_num: int) -> Optional[int]:
    board_state = game.get_board_ascii(player_num)
    prompt = f"""You are a Connect Four AI.

{board_state}

You are X. Respond with ONLY a single digit 0-6 for the column. No explanation."""

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Connect Four AI. Respond with ONLY a single digit 0-6. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\d', text)
        if match:
            col = int(match.group())
            if 0 <= col <= 6:
                return col
    except Exception as e:
        print(f"LLM Error: {e}")
    return None


# ============================================================================
# GAME RUNNER (streaming generator)
# ============================================================================

def run_game(model1: str, model2: str):
    if not API_KEY:
        yield json.dumps({"error": "API_KEY not set"}), "Error: Set API_KEY secret in Space settings."
        return

    game = ConnectFourGame()
    m1 = model1.split("/")[-1]
    m2 = model2.split("/")[-1]
    log_lines = []

    yield game.to_json(model1, model2), ""

    while not game.game_over:
        current = game.current_player
        model = model1 if current == 1 else model2
        name = m1 if current == 1 else m2
        color = "Cyan" if current == 1 else "Pink"

        move = get_llm_move(model, game, current)

        if move is None or not (0 <= move <= 6) or game.board[0][move] != 0:
            valid = game.get_valid_columns()
            if not valid:
                break
            move = random.choice(valid)
            log_lines.append(f"Move {game.move_count+1} ({color}): col {move} [random] - {name}")
        else:
            log_lines.append(f"Move {game.move_count+1} ({color}): col {move} - {name}")

        game.drop_piece(move)

        if game.game_over:
            if game.winner == 0:
                log_lines.append("\nDRAW!")
            elif game.winner == 1:
                log_lines.append(f"\n{m1} (Cyan) WINS!")
            else:
                log_lines.append(f"\n{m2} (Pink) WINS!")
        else:
            game.switch_player()

        yield game.to_json(model1, model2), "\n".join(log_lines[-25:])

    yield game.to_json(model1, model2), "\n".join(log_lines[-25:])


# ============================================================================
# CANVAS RENDERER
# ============================================================================

GAME_HTML = """
<div style="display:flex;justify-content:center;padding:20px;">
  <canvas id="c4Canvas" width="600" height="560"
    style="border:2px solid #22d3ee;border-radius:12px;background:#0a0a1a;"></canvas>
</div>
<script>
window.renderC4 = function(jsonStr) {
  try {
    const state = JSON.parse(jsonStr);
    if (state.error) return;

    const canvas = document.getElementById('c4Canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const cols = 7, rows = 6;
    const cellW = 80, cellH = 80;
    const padX = (W - cols * cellW) / 2;
    const padY = 10;
    const rad = 32;

    // Background
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);

    // Board frame
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(padX - 5, padY - 5, cols * cellW + 10, rows * cellH + 10);

    // Grid + pieces
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = padX + c * cellW + cellW / 2;
        const y = padY + r * cellH + cellH / 2;
        const piece = state.board[r][c];
        const isLast = state.last_move && state.last_move[0] === r && state.last_move[1] === c;

        // Cell background hole
        ctx.fillStyle = '#0a0a1a';
        ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fill();

        if (piece === 1) {
          if (isLast) { ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 18; }
          ctx.fillStyle = '#22d3ee';
          ctx.beginPath(); ctx.arc(x, y, rad - 2, 0, Math.PI * 2); ctx.fill();
          ctx.shadowBlur = 0;
        } else if (piece === 2) {
          if (isLast) { ctx.shadowColor = '#f472b6'; ctx.shadowBlur = 18; }
          ctx.fillStyle = '#f472b6';
          ctx.beginPath(); ctx.arc(x, y, rad - 2, 0, Math.PI * 2); ctx.fill();
          ctx.shadowBlur = 0;
        } else {
          ctx.strokeStyle = 'rgba(34,211,238,0.15)';
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(x, y, rad - 2, 0, Math.PI * 2); ctx.stroke();
        }
      }
    }

    // Column numbers
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    for (let c = 0; c < cols; c++) {
      ctx.fillText(c, padX + c * cellW + cellW / 2, padY + rows * cellH + 28);
    }

    // Model names
    const models = state.models || ['Player 1', 'Player 2'];
    const m1 = models[0].split('/').pop();
    const m2 = models[1].split('/').pop();
    ctx.font = 'bold 13px monospace';
    ctx.fillStyle = '#22d3ee';
    ctx.textAlign = 'left';
    ctx.fillText('Cyan: ' + m1, 10, H - 12);
    ctx.fillStyle = '#f472b6';
    ctx.textAlign = 'right';
    ctx.fillText('Pink: ' + m2, W - 10, H - 12);
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Move ' + state.move_count, W / 2, H - 12);

    // Game over overlay
    if (state.game_over) {
      ctx.fillStyle = 'rgba(10,10,26,0.75)';
      ctx.fillRect(0, 0, W, H);

      ctx.font = 'bold 32px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      let msg, color;
      if (state.winner === 0) { msg = 'DRAW!'; color = '#fbbf24'; }
      else if (state.winner === 1) { msg = m1 + ' WINS!'; color = '#22d3ee'; }
      else { msg = m2 + ' WINS!'; color = '#f472b6'; }

      ctx.fillStyle = color;
      ctx.shadowColor = color; ctx.shadowBlur = 20;
      ctx.fillText(msg, W / 2, H / 2);
      ctx.shadowBlur = 0;
    }
  } catch (e) { console.error('C4 render error:', e); }
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

with gr.Blocks(css=CSS, title="Connect Four AI", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px 0;">
        <h1 style="font-size:2.2em;margin:0;color:#e2e8f0;">Connect Four AI</h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Two LLM models compete in Connect Four &mdash;
            Powered by <strong style="color:#06b6d4;">Crusoe Intelligence Foundry</strong>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<h4 style="color:#22d3ee;margin:0 0 6px 0;">Player 1 (Cyan)</h4>')
            model1 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[0],
                                 label="Model", allow_custom_value=True)
            gr.HTML('<h4 style="color:#f472b6;margin:12px 0 6px 0;">Player 2 (Pink)</h4>')
            model2 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[2],
                                 label="Model", allow_custom_value=True)
            start_btn = gr.Button("Start Battle!", variant="primary", size="lg")

            gr.HTML('<h4 style="color:#94a3b8;margin:16px 0 4px 0;">Move Log</h4>')
            move_log = gr.Textbox(lines=15, max_lines=25, interactive=False,
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
        inputs=[model1, model2],
        outputs=[game_state, move_log],
    )

    game_state.change(
        fn=None,
        inputs=[game_state],
        outputs=None,
        js="(s) => { if (s && window.renderC4) window.renderC4(s); }",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
