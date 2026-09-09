import os
import json
import gradio as gr
from openai import OpenAI
import random
import time

API_URL = os.environ.get("API_URL", "https://managed-inference-api-proxy.crusoecloud.com/v1")
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

class OthelloGame:
    def __init__(self):
        self.board = [[0] * 8 for _ in range(8)]
        self.board[3][3] = 2
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[4][4] = 2
        self.move_history = []
        self.last_move = None

    def get_valid_moves(self, player):
        valid = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    valid.append((row, col))
        return valid

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != 0:
            return False
        opponent = 3 - player
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            if self._has_flip(row, col, dr, dc, player, opponent):
                return True
        return False

    def _has_flip(self, row, col, dr, dc, player, opponent):
        r, c = row + dr, col + dc
        found = False
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] == 0:
                return False
            elif self.board[r][c] == opponent:
                found = True
            elif self.board[r][c] == player:
                return found
            r += dr
            c += dc
        return False

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False
        self.board[row][col] = player
        self.last_move = (row, col)
        opponent = 3 - player
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            self._flip(row, col, dr, dc, player, opponent)
        self.move_history.append((row, col, player))
        return True

    def _flip(self, row, col, dr, dc, player, opponent):
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] == 0:
                return
            elif self.board[r][c] == opponent:
                to_flip.append((r, c))
            elif self.board[r][c] == player:
                for fr, fc in to_flip:
                    self.board[fr][fc] = player
                return
            r += dr
            c += dc

    def get_disc_count(self):
        black = sum(row.count(1) for row in self.board)
        white = sum(row.count(2) for row in self.board)
        return black, white

    def is_game_over(self):
        return len(self.get_valid_moves(1)) == 0 and len(self.get_valid_moves(2)) == 0

    def get_board_ascii(self):
        s = "  A B C D E F G H\n"
        for row in range(8):
            s += f"{row + 1} "
            for col in range(8):
                if self.board[row][col] == 0:
                    s += ". "
                elif self.board[row][col] == 1:
                    s += "B "
                else:
                    s += "W "
            s += "\n"
        return s

    def get_winner(self):
        b, w = self.get_disc_count()
        if b > w: return 1
        if w > b: return 2
        return 0

    def to_json(self, model1, model2, current_player):
        b, w = self.get_disc_count()
        return json.dumps({
            "board": self.board,
            "validMoves": self.get_valid_moves(current_player),
            "lastMove": self.last_move,
            "blackCount": b, "whiteCount": w,
            "currentPlayer": current_player,
            "gameOver": self.is_game_over(),
            "winner": self.get_winner(),
            "model1": model1, "model2": model2,
        })


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_llm_move(model, game, player):
    valid_moves = game.get_valid_moves(player)
    if not valid_moves:
        return None

    valid_str = ", ".join([f"{chr(65+c)}{r+1}" for r, c in valid_moves])
    b, w = game.get_disc_count()
    side = "Black (B)" if player == 1 else "White (W)"

    prompt = (
        f"Board:\n{game.get_board_ascii()}\n"
        f"Black: {b}, White: {w}\n"
        f"You are {side}. Valid moves: {valid_str}\n"
        f"Choose your move."
    )

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an Othello AI. Respond with ONLY a position like D3 or F6. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip().upper()
        # Parse
        for i, ch in enumerate(text):
            if ch.isalpha():
                col = ord(ch) - ord('A')
                for j in range(i + 1, len(text)):
                    if text[j].isdigit():
                        row = int(text[j]) - 1
                        if (row, col) in valid_moves:
                            return (row, col)
    except Exception as e:
        print(f"LLM Error: {e}")

    return random.choice(valid_moves)


# ============================================================================
# GAME RUNNER (streaming generator)
# ============================================================================

def run_game(model1, model2):
    if not API_KEY:
        yield json.dumps({"error": "API_KEY not set"}), "Error: Set API_KEY secret in Space settings."
        return

    game = OthelloGame()
    current_player = 1
    m1 = model1.split("/")[-1]
    m2 = model2.split("/")[-1]
    log_lines = [f"Black: {m1}", f"White: {m2}", ""]
    consecutive_passes = 0

    yield game.to_json(model1, model2, current_player), "\n".join(log_lines)

    while not game.is_game_over():
        valid = game.get_valid_moves(current_player)
        side = "Black" if current_player == 1 else "White"
        model = model1 if current_player == 1 else model2

        if not valid:
            log_lines.append(f"{side} ({m1 if current_player == 1 else m2}): PASS")
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            current_player = 3 - current_player
            yield game.to_json(model1, model2, current_player), "\n".join(log_lines[-20:])
            continue

        consecutive_passes = 0
        move = get_llm_move(model, game, current_player)

        if move:
            row, col = move
            game.make_move(row, col, current_player)
            move_str = f"{chr(65+col)}{row+1}"
            b, w = game.get_disc_count()
            log_lines.append(f"{side}: {move_str} (B:{b} W:{w})")

        current_player = 3 - current_player
        yield game.to_json(model1, model2, current_player), "\n".join(log_lines[-20:])

    # Final
    b, w = game.get_disc_count()
    winner = game.get_winner()
    if winner == 1:
        log_lines.append(f"\nBlack ({m1}) wins! {b}-{w}")
    elif winner == 2:
        log_lines.append(f"\nWhite ({m2}) wins! {b}-{w}")
    else:
        log_lines.append(f"\nTie! {b}-{w}")

    yield game.to_json(model1, model2, current_player), "\n".join(log_lines[-20:])


# ============================================================================
# CANVAS RENDERER
# ============================================================================

GAME_HTML = """
<div style="display:flex;justify-content:center;padding:20px;">
  <canvas id="othelloCanvas" width="520" height="520"
    style="border:2px solid #22d3ee;border-radius:12px;"></canvas>
</div>
<script>
window.renderOthello = function(jsonStr) {
  try {
    const data = JSON.parse(jsonStr);
    if (data.error) return;

    const canvas = document.getElementById('othelloCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const sq = W / 8;

    // Board background (green felt)
    ctx.fillStyle = '#2d5016';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#4a7c1c';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {
      ctx.beginPath(); ctx.moveTo(i * sq, 0); ctx.lineTo(i * sq, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * sq); ctx.lineTo(W, i * sq); ctx.stroke();
    }

    // Valid moves (small dots)
    if (data.validMoves) {
      ctx.fillStyle = 'rgba(34, 211, 238, 0.5)';
      data.validMoves.forEach(m => {
        ctx.beginPath();
        ctx.arc(m[1] * sq + sq/2, m[0] * sq + sq/2, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Discs
    data.board.forEach((row, r) => {
      row.forEach((cell, c) => {
        if (cell === 0) return;
        const x = c * sq + sq/2, y = r * sq + sq/2;
        const rad = sq/2 - 4;

        const grad = ctx.createRadialGradient(x - 3, y - 3, 0, x, y, rad);
        if (cell === 1) {
          grad.addColorStop(0, '#555'); grad.addColorStop(1, '#111');
          ctx.strokeStyle = '#222';
        } else {
          grad.addColorStop(0, '#fff'); grad.addColorStop(1, '#ccc');
          ctx.strokeStyle = '#999';
        }
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fill();
        ctx.lineWidth = 1; ctx.stroke();
      });
    });

    // Last move highlight
    if (data.lastMove) {
      const lx = data.lastMove[1] * sq + sq/2, ly = data.lastMove[0] * sq + sq/2;
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(lx, ly, sq/2 - 2, 0, Math.PI * 2); ctx.stroke();
    }

    // HUD bar at top
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(0, 0, W, 36);
    ctx.font = 'bold 14px monospace';
    ctx.textBaseline = 'middle';

    // Black info
    ctx.fillStyle = '#e2e8f0';
    ctx.textAlign = 'left';
    const m1 = (data.model1 || 'Player 1').split('/').pop();
    ctx.fillText('Black (' + m1 + '): ' + data.blackCount, 10, 18);

    // White info
    ctx.textAlign = 'right';
    const m2 = (data.model2 || 'Player 2').split('/').pop();
    ctx.fillText('White (' + m2 + '): ' + data.whiteCount, W - 10, 18);

    // Game over
    if (data.gameOver) {
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(0, H/2 - 45, W, 90);

      ctx.font = 'bold 28px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#fbbf24';
      ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 20;
      ctx.fillText('GAME OVER', W/2, H/2 - 14);
      ctx.shadowBlur = 0;

      ctx.font = 'bold 18px monospace';
      ctx.fillStyle = '#e2e8f0';
      let result;
      if (data.winner === 1) result = 'Black (' + m1 + ') wins! ' + data.blackCount + '-' + data.whiteCount;
      else if (data.winner === 2) result = 'White (' + m2 + ') wins! ' + data.blackCount + '-' + data.whiteCount;
      else result = 'Tie! ' + data.blackCount + '-' + data.whiteCount;
      ctx.fillText(result, W/2, H/2 + 18);
    }
  } catch (e) { console.error('Othello render error:', e); }
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

with gr.Blocks(css=CSS, title="Othello AI", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px 0;">
        <h1 style="font-size:2.2em;margin:0;color:#e2e8f0;">Othello AI</h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Two LLM models compete in Othello/Reversi &mdash;
            Powered by <strong style="color:#06b6d4;">Crusoe Intelligence Foundry</strong>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<h4 style="color:#e2e8f0;margin:0 0 6px 0;">Player 1 (Black)</h4>')
            model1 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[0],
                                 label="Model", allow_custom_value=True)
            gr.HTML('<h4 style="color:#e2e8f0;margin:12px 0 6px 0;">Player 2 (White)</h4>')
            model2 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[2],
                                 label="Model", allow_custom_value=True)
            start_btn = gr.Button("Start Battle!", variant="primary", size="lg")

            gr.HTML('<h4 style="color:#94a3b8;margin:16px 0 4px 0;">Move Log</h4>')
            move_log = gr.Textbox(lines=15, max_lines=25, interactive=False,
                                  show_label=False, elem_classes=["log-box"])

        with gr.Column(scale=2, min_width=540):
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
        js="(s) => { if (s && window.renderOthello) window.renderOthello(s); }",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
