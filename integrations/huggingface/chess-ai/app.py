import os
import json
import chess
import random
import re
import time
from openai import OpenAI
import gradio as gr

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

class ChessGame:
    def __init__(self, max_moves=150):
        self.board = chess.Board()
        self.max_moves = max_moves
        self.move_count = 0
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.game_over_reason = ""

    def is_game_over(self):
        if self.board.is_checkmate():
            self.game_over = True
            self.winner = "Black" if self.board.turn else "White"
            self.game_over_reason = "Checkmate"
            return True
        if self.board.is_stalemate():
            self.game_over = True
            self.game_over_reason = "Stalemate"
            return True
        if self.board.is_repetition():
            self.game_over = True
            self.game_over_reason = "Draw by repetition"
            return True
        if self.board.halfmove_clock >= 100:
            self.game_over = True
            self.game_over_reason = "50-move rule"
            return True
        if self.move_count >= self.max_moves:
            self.game_over = True
            self.game_over_reason = f"Max moves ({self.max_moves})"
            return True
        return False

    def make_move(self, uci_move):
        try:
            move = chess.Move.from_uci(uci_move)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_count += 1
                self.move_history.append(uci_move)
                return True
        except Exception:
            pass
        return False

    def make_random_move(self):
        legal = list(self.board.legal_moves)
        if not legal:
            return None
        move = random.choice(legal)
        self.board.push(move)
        self.move_count += 1
        self.move_history.append(move.uci())
        return move.uci()

    def to_json(self, white_model, black_model):
        board_2d = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                row.append(piece_char(piece) if piece else "")
            board_2d.append(row)

        last_move = None
        if self.move_history:
            lm = self.move_history[-1]
            last_move = {"from": lm[:2], "to": lm[2:4]}

        return json.dumps({
            "board": board_2d,
            "lastMove": last_move,
            "check": self.board.is_check(),
            "gameOver": self.game_over,
            "winner": self.winner,
            "gameOverReason": self.game_over_reason,
            "moveCount": self.move_count,
            "fen": self.board.fen(),
            "whiteModel": white_model,
            "blackModel": black_model,
        })


PIECE_CHARS = {
    chess.PAWN: ("♙", "♟"), chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"), chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"), chess.KING: ("♔", "♚"),
}

def piece_char(piece):
    if piece is None:
        return ""
    w, b = PIECE_CHARS[piece.piece_type]
    return w if piece.color == chess.WHITE else b


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_llm_move(model, game, player_color):
    legal = [m.uci() for m in game.board.legal_moves]
    history = ", ".join(game.move_history[-10:]) if game.move_history else "None"

    prompt = f"""You are playing chess as {player_color}.

BOARD:
{game.board}

FEN: {game.board.fen()}
LEGAL MOVES: {", ".join(legal)}
HISTORY: {history}

Respond with ONLY your move in UCI notation (e.g., e2e4). Pick from the legal moves."""

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a chess AI. Respond with ONLY a UCI move like e2e4. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=15,
        )
        latency = time.time() - start
        text = response.choices[0].message.content.strip().lower()

        # Try exact match
        for m in legal:
            if m in text:
                return m, text, latency

        # Try regex
        matches = re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", text)
        if matches and matches[0] in legal:
            return matches[0], text, latency

        return None, text, latency
    except Exception as e:
        return None, f"Error: {e}", 0


# ============================================================================
# GAME RUNNER
# ============================================================================

def run_game(white_model, black_model, max_moves):
    if not API_KEY:
        yield json.dumps({"error": "API_KEY not set"}), "Error: Set API_KEY secret.", ""
        return

    game = ChessGame(max_moves=int(max_moves))
    wm = white_model.split("/")[-1]
    bm = black_model.split("/")[-1]
    white_log, black_log = [], []

    yield game.to_json(white_model, black_model), "", ""

    while not game.is_game_over():
        is_white = game.board.turn
        model = white_model if is_white else black_model
        color = "White" if is_white else "Black"
        name = wm if is_white else bm
        log_list = white_log if is_white else black_log

        move, resp, lat = get_llm_move(model, game, color)

        if move and game.make_move(move):
            log_list.append(f"M{game.move_count} {move} ({lat:.1f}s)")
        else:
            fallback = game.make_random_move()
            log_list.append(f"M{game.move_count} {fallback} [random] (resp: {resp[:30]})")

        yield (
            game.to_json(white_model, black_model),
            "\n".join(white_log[-20:]),
            "\n".join(black_log[-20:]),
        )

    # Final
    if game.winner:
        result = f"\n{game.winner} wins! {game.game_over_reason}"
    else:
        result = f"\nDraw: {game.game_over_reason}"
    white_log.append(result)
    black_log.append(result)

    yield (
        game.to_json(white_model, black_model),
        "\n".join(white_log[-20:]),
        "\n".join(black_log[-20:]),
    )


# ============================================================================
# CANVAS RENDERER
# ============================================================================

GAME_HTML = """
<div style="display:flex;justify-content:center;padding:20px;">
  <canvas id="chessCanvas" width="520" height="580"
    style="border:2px solid #22d3ee;border-radius:12px;background:#0a0a1a;"></canvas>
</div>
<script>
window.renderChess = function(jsonStr) {
  try {
    const state = JSON.parse(jsonStr);
    if (!state.board || state.error) return;

    const canvas = document.getElementById('chessCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const sq = 65;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const light = '#E8D0A8', dark = '#B58863', highlight = '#c8a84e';

    // Board squares
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        ctx.fillStyle = (r + c) % 2 === 0 ? light : dark;

        // Highlight last move
        if (state.lastMove) {
          const fc = state.lastMove.from.charCodeAt(0) - 97;
          const fr = 7 - (state.lastMove.from.charCodeAt(1) - 49);
          const tc = state.lastMove.to.charCodeAt(0) - 97;
          const tr = 7 - (state.lastMove.to.charCodeAt(1) - 49);
          if ((r === fr && c === fc) || (r === tr && c === tc)) {
            ctx.fillStyle = highlight;
          }
        }
        ctx.fillRect(c * sq, r * sq, sq, sq);
      }
    }

    // Pieces
    ctx.font = '46px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const piece = state.board[r][c];
        if (piece) {
          // Shadow for readability
          ctx.fillStyle = 'rgba(0,0,0,0.3)';
          ctx.fillText(piece, c * sq + sq/2 + 1, r * sq + sq/2 + 2);
          ctx.fillStyle = '#fff';
          ctx.fillText(piece, c * sq + sq/2, r * sq + sq/2);
        }

        // Check indicator
        if (state.check && state.board[r][c]) {
          const isWhiteTurn = state.fen.split(' ')[1] === 'w';
          const king = isWhiteTurn ? '\u2654' : '\u265a';
          if (state.board[r][c] === king) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 3;
            ctx.strokeRect(c * sq + 2, r * sq + 2, sq - 4, sq - 4);
          }
        }
      }
    }

    // Rank/file labels
    ctx.font = '11px monospace';
    ctx.fillStyle = '#64748b';
    for (let i = 0; i < 8; i++) {
      ctx.textAlign = 'left';
      ctx.fillText(String(8 - i), 2, i * sq + 14);
      ctx.textAlign = 'center';
      ctx.fillText(String.fromCharCode(97 + i), i * sq + sq/2, 8 * sq + 14);
    }

    // HUD
    const wm = (state.whiteModel || '').split('/').pop();
    const bm = (state.blackModel || '').split('/').pop();
    ctx.font = 'bold 13px monospace';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#e2e8f0';
    ctx.textAlign = 'left';
    ctx.fillText('White: ' + wm, 8, 8 * sq + 34);
    ctx.textAlign = 'right';
    ctx.fillText('Black: ' + bm, canvas.width - 8, 8 * sq + 34);
    ctx.textAlign = 'center';
    ctx.fillStyle = '#64748b';
    ctx.fillText('Move ' + state.moveCount, canvas.width/2, 8 * sq + 34);

    // Game over overlay
    if (state.gameOver) {
      ctx.fillStyle = 'rgba(10,10,26,0.75)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = 'bold 28px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#fbbf24';
      ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 20;
      ctx.fillText('GAME OVER', canvas.width/2, canvas.height/2 - 20);
      ctx.shadowBlur = 0;

      ctx.font = 'bold 16px monospace';
      ctx.fillStyle = '#e2e8f0';
      let msg = state.gameOverReason || 'Draw';
      if (state.winner) msg = state.winner + ' wins - ' + msg;
      ctx.fillText(msg, canvas.width/2, canvas.height/2 + 16);
    }
  } catch (e) { console.error('Chess render error:', e); }
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

with gr.Blocks(css=CSS, title="Chess AI", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px 0;">
        <h1 style="font-size:2.2em;margin:0;color:#e2e8f0;">Chess AI</h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Two LLM models compete in chess &mdash;
            Powered by <strong style="color:#06b6d4;">Crusoe Intelligence Foundry</strong>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<h4 style="color:#e2e8f0;margin:0 0 6px 0;">White</h4>')
            white_model = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[0],
                                      label="Model", allow_custom_value=True)
            gr.HTML('<h4 style="color:#94a3b8;margin:12px 0 6px 0;">Black</h4>')
            black_model = gr.Dropdown(choices=CRUSOE_MODELS, value=CRUSOE_MODELS[2],
                                      label="Model", allow_custom_value=True)
            max_moves = gr.Slider(minimum=50, maximum=300, value=150, step=10,
                                  label="Max Moves")
            start_btn = gr.Button("Start Battle!", variant="primary", size="lg")

            gr.HTML('<h4 style="color:#e2e8f0;margin:16px 0 4px 0;">White Log</h4>')
            white_log = gr.Textbox(lines=8, max_lines=20, interactive=False,
                                   show_label=False, elem_classes=["log-box"])
            gr.HTML('<h4 style="color:#94a3b8;margin:8px 0 4px 0;">Black Log</h4>')
            black_log = gr.Textbox(lines=8, max_lines=20, interactive=False,
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
        inputs=[white_model, black_model, max_moves],
        outputs=[game_state, white_log, black_log],
    )

    game_state.change(
        fn=None,
        inputs=[game_state],
        outputs=None,
        js="(s) => { if (s && window.renderChess) window.renderChess(s); }",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
