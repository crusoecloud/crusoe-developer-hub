import os
import json
import re
import random
from typing import Optional, Tuple, List
import gradio as gr
from openai import OpenAI

API_URL = os.environ.get("API_URL", "https://api.inference.crusoecloud.com/v1")
API_KEY = os.environ.get("API_KEY", "")

CRUSOE_MODELS = [
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "google/gemma-3-12b-it",
]


def get_llm_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=API_URL)


# ============================================================================
# GAME ENGINE - 3x3 Tic-Tac-Toe
# ============================================================================

class TicTacToe:
    def __init__(self):
        self.board = [[0] * 3 for _ in range(3)]
        self.turn = 1  # 1=X, 2=O
        self.game_over = False
        self.winner = None  # 1=X wins, 2=O wins, 0=draw
        self.win_line = []
        self.move_history = []
        self.move_count = 0

    def get_board_state(self) -> str:
        lines = ["    1   2   3"]
        lines.append("  +" + "-" * 11 + "+")
        for row_idx, row in enumerate(self.board):
            row_label = chr(ord('A') + row_idx)
            cells = []
            for cell in row:
                if cell == 0:
                    cells.append(" ")
                elif cell == 1:
                    cells.append("X")
                else:
                    cells.append("O")
            lines.append(f"{row_label} | " + " | ".join(cells) + " |")
            lines.append("  +" + "-" * 11 + "+")
        return "\n".join(lines)

    def get_valid_moves(self) -> List[str]:
        moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    moves.append(f"{chr(ord('A') + r)}{c + 1}")
        return moves

    def parse_move(self, move_str: str) -> Optional[Tuple[int, int]]:
        move_str = move_str.strip().upper()
        match = re.match(r'^([A-C])([1-3])$', move_str)
        if not match:
            return None
        row = ord(match.group(1)) - ord('A')
        col = int(match.group(2)) - 1
        if self.board[row][col] != 0:
            return None
        return (row, col)

    def make_move(self, row: int, col: int) -> bool:
        if self.board[row][col] != 0 or self.game_over:
            return False
        self.board[row][col] = self.turn
        self.move_count += 1
        self.move_history.append(f"{chr(ord('A') + row)}{col + 1}")

        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.turn
        elif self.move_count >= 9:
            self.game_over = True
            self.winner = 0

        self.turn = 3 - self.turn
        return True

    def check_win(self, last_row: int, last_col: int) -> bool:
        player = self.board[last_row][last_col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            coords = [(last_row, last_col)]

            r, c = last_row + dr, last_col + dc
            while 0 <= r < 3 and 0 <= c < 3 and self.board[r][c] == player:
                coords.append((r, c))
                count += 1
                r += dr
                c += dc

            r, c = last_row - dr, last_col - dc
            while 0 <= r < 3 and 0 <= c < 3 and self.board[r][c] == player:
                coords.insert(0, (r, c))
                count += 1
                r -= dr
                c -= dc

            if count >= 3:
                self.win_line = coords[:3]
                return True
        return False

    def to_json(self, model1: str, model2: str) -> str:
        return json.dumps({
            "board": self.board,
            "lastMove": self.move_history[-1] if self.move_history else None,
            "gameOver": self.game_over,
            "winner": self.winner,
            "winLine": self.win_line,
            "turn": self.turn,
            "model1": model1,
            "model2": model2,
            "moveHistory": self.move_history,
        })


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_ai_move(game: TicTacToe, model: str) -> Optional[str]:
    try:
        client = get_llm_client()
        board_state = game.get_board_state()
        valid_moves = game.get_valid_moves()

        system_prompt = (
            "You are a Tic-Tac-Toe AI on a 3x3 board. "
            "Respond with ONLY a position like A1 or C3. "
            "Rows are A-C, columns are 1-3. No explanation."
        )
        user_message = (
            f"Current board:\n{board_state}\n\n"
            f"Valid moves: {', '.join(valid_moves)}\n\n"
            f"Make your move:"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=10,
        )
        move_text = response.choices[0].message.content.strip().upper()
        match = re.search(r'([A-C][1-3])', move_text)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"Error getting AI move: {e}")
        return None


# ============================================================================
# GAME RUNNER
# ============================================================================

def play_game(model1: str, model2: str):
    """Play a complete game, yielding state each turn for streaming."""
    if not API_KEY:
        yield json.dumps({"error": "API_KEY not set"}), "Error: Set API_KEY secret in Space settings."
        return

    game = TicTacToe()
    models = [None, model1, model2]
    game_log = []

    # Yield initial empty board
    yield game.to_json(model1, model2), ""

    while not game.game_over:
        current_model = models[game.turn]
        player_label = "X" if game.turn == 1 else "O"

        move = get_ai_move(game, current_model)

        if move is None or game.parse_move(move) is None:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            game_log.append(f"Turn {game.move_count + 1} ({player_label}): random {move}")
        else:
            game_log.append(f"Turn {game.move_count + 1} ({player_label}): {move}")

        parsed = game.parse_move(move)
        if parsed:
            game.make_move(parsed[0], parsed[1])

        yield game.to_json(model1, model2), "\n".join(game_log)

    # Final result
    if game.winner == 1:
        game_log.append(f"\n{model1} (X) WINS!")
    elif game.winner == 2:
        game_log.append(f"\n{model2} (O) WINS!")
    else:
        game_log.append("\nDRAW!")

    yield game.to_json(model1, model2), "\n".join(game_log)


# ============================================================================
# CANVAS RENDERER
# ============================================================================

GAME_HTML = """
<div style="display:flex;justify-content:center;align-items:center;padding:20px;">
  <canvas id="tttCanvas" width="500" height="500"
    style="border:2px solid #22d3ee;border-radius:12px;background:#0a0a1a;"></canvas>
</div>
<script>
window.renderTTT = function(jsonStr) {
  try {
    const state = JSON.parse(jsonStr);
    if (state.error) return;

    const canvas = document.getElementById('tttCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = 500, H = 500;
    const CELL = W / 3;
    const PAD = 30;

    // Background
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 3;
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(i * CELL, PAD);
      ctx.lineTo(i * CELL, H - PAD);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PAD, i * CELL);
      ctx.lineTo(W - PAD, i * CELL);
      ctx.stroke();
    }

    // Row labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const labels = ['A', 'B', 'C'];
    for (let i = 0; i < 3; i++) {
      ctx.fillText(labels[i], 12, i * CELL + CELL / 2);
      ctx.fillText(String(i + 1), i * CELL + CELL / 2, 14);
    }

    // Pieces
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const cell = state.board[r][c];
        const cx = c * CELL + CELL / 2;
        const cy = r * CELL + CELL / 2;
        const isLast = state.lastMove === (labels[r] + String(c + 1));

        if (cell === 1) {
          // X - cyan
          const sz = CELL * 0.28;
          ctx.strokeStyle = '#22d3ee';
          ctx.lineWidth = isLast ? 6 : 4;
          ctx.lineCap = 'round';
          if (isLast) { ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 15; }
          ctx.beginPath(); ctx.moveTo(cx - sz, cy - sz); ctx.lineTo(cx + sz, cy + sz); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(cx + sz, cy - sz); ctx.lineTo(cx - sz, cy + sz); ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          // O - pink
          const rad = CELL * 0.28;
          ctx.strokeStyle = '#f472b6';
          ctx.lineWidth = isLast ? 6 : 4;
          if (isLast) { ctx.shadowColor = '#f472b6'; ctx.shadowBlur = 15; }
          ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
          ctx.shadowBlur = 0;
        }
      }
    }

    // Win line
    if (state.gameOver && state.winner && state.winLine && state.winLine.length >= 3) {
      const [r1, c1] = state.winLine[0];
      const [r2, c2] = state.winLine[2];
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 5;
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.moveTo(c1 * CELL + CELL / 2, r1 * CELL + CELL / 2);
      ctx.lineTo(c2 * CELL + CELL / 2, r2 * CELL + CELL / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // HUD bar
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(0, H - 44, W, 44);
    ctx.font = 'bold 14px monospace';
    ctx.textBaseline = 'middle';

    // Player 1
    ctx.fillStyle = '#22d3ee';
    ctx.textAlign = 'left';
    const m1 = (state.model1 || 'Player 1').split('/').pop();
    ctx.fillText('X: ' + m1, 12, H - 22);

    // Player 2
    ctx.fillStyle = '#f472b6';
    ctx.textAlign = 'right';
    const m2 = (state.model2 || 'Player 2').split('/').pop();
    ctx.fillText('O: ' + m2, W - 12, H - 22);

    // Game over overlay
    if (state.gameOver) {
      ctx.fillStyle = 'rgba(10,10,26,0.6)';
      ctx.fillRect(0, H / 2 - 40, W, 80);
      ctx.font = 'bold 28px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      let msg, color;
      if (state.winner === 1) {
        msg = m1 + ' (X) WINS!'; color = '#22d3ee';
      } else if (state.winner === 2) {
        msg = m2 + ' (O) WINS!'; color = '#f472b6';
      } else {
        msg = 'DRAW!'; color = '#fbbf24';
      }
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 20;
      ctx.fillText(msg, W / 2, H / 2);
      ctx.shadowBlur = 0;
    }
  } catch (e) { console.error('TTT render error:', e); }
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

with gr.Blocks(css=CSS, title="Tic-Tac-Toe AI", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px 0;">
        <h1 style="font-size:2.2em;margin:0;color:#e2e8f0;">Tic-Tac-Toe AI</h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Two LLM models compete in classic 3x3 Tic-Tac-Toe &mdash;
            Powered by <strong style="color:#06b6d4;">Crusoe Intelligence Foundry</strong>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<h4 style="color:#22d3ee;margin:0 0 6px 0;">Player 1 (X - Cyan)</h4>')
            model1 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[0],
                                 label="Model", allow_custom_value=True)
            gr.HTML('<h4 style="color:#f472b6;margin:12px 0 6px 0;">Player 2 (O - Pink)</h4>')
            model2 = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[2],
                                 label="Model", allow_custom_value=True)
            start_btn = gr.Button("Start Battle!", variant="primary", size="lg")

            gr.HTML('<h4 style="color:#94a3b8;margin:16px 0 4px 0;">Game Log</h4>')
            game_log = gr.Textbox(lines=12, max_lines=20, interactive=False,
                                  show_label=False, elem_classes=["log-box"])

        with gr.Column(scale=2, min_width=520):
            game_canvas = gr.HTML(GAME_HTML)
            game_state = gr.Textbox(visible=False)

    gr.HTML("""
    <div style="text-align:center;padding:16px 0;color:#475569;font-size:0.85em;">
        Open models on Crusoe Intelligence Foundry
    </div>
    """)

    start_btn.click(
        fn=play_game,
        inputs=[model1, model2],
        outputs=[game_state, game_log],
    )

    game_state.change(
        fn=None,
        inputs=[game_state],
        outputs=None,
        js="(s) => { if (s && window.renderTTT) window.renderTTT(s); }",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
