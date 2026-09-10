import os
import json
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Tuple, List, Dict, Optional
import time

API_URL = os.environ.get("API_URL", "https://api.inference.crusoecloud.com/v1")
API_KEY = os.environ.get("API_KEY", "")

# ============================================================================
# GAME ENGINE
# ============================================================================

class TronLightCyclesGame:
    """Tron Light Cycles game engine"""

    def __init__(self, grid_size: int = 30, max_turns: int = 900):
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """Initialize a new game"""
        self.grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Player 1: cyan, starts at (5, 15) going RIGHT
        self.p1_pos = (5, 15)
        self.p1_dir = (0, 1)  # (row_delta, col_delta) - RIGHT
        self.p1_trail = set()

        # Player 2: pink, starts at (24, 15) going LEFT
        self.p2_pos = (self.grid_size - 6, self.grid_size - 1)
        self.p2_dir = (0, -1)  # LEFT
        self.p2_trail = set()

        self.turn = 0
        self.game_over = False
        self.winner = None  # None, 'p1', 'p2', or 'draw'

    def get_board_state(self) -> str:
        """Return current board as ASCII grid"""
        board = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark trails
        for pos in self.p1_trail:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                board[pos[0]][pos[1]] = 't'

        for pos in self.p2_trail:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                board[pos[0]][pos[1]] = 'e'

        # Mark heads (overwrites trails)
        if 0 <= self.p1_pos[0] < self.grid_size and 0 <= self.p1_pos[1] < self.grid_size:
            board[self.p1_pos[0]][self.p1_pos[1]] = 'H'

        if 0 <= self.p2_pos[0] < self.grid_size and 0 <= self.p2_pos[1] < self.grid_size:
            board[self.p2_pos[0]][self.p2_pos[1]] = 'E'

        return '\n'.join(''.join(row) for row in board)

    def get_safe_moves(self, pos: Tuple[int, int], direction: Tuple[int, int],
                       own_trail: set, enemy_trail: set) -> List[str]:
        """Return list of safe moves from current position"""
        safe = []
        moves = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        for move_name, (dr, dc) in moves.items():
            # Can't reverse
            if (dr, dc) == (-direction[0], -direction[1]):
                continue

            new_pos = (pos[0] + dr, pos[1] + dc)

            # Check bounds
            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                continue

            # Check collision with own trail
            if new_pos in own_trail:
                continue

            # Check collision with enemy trail
            if new_pos in enemy_trail:
                continue

            safe.append(move_name)

        return safe

    def move_player(self, player: int, move: str) -> bool:
        """
        Move a player in the specified direction.
        Returns True if player survives, False if they die.
        """
        if player == 1:
            pos = self.p1_pos
            direction = self.p1_dir
            own_trail = self.p1_trail
            enemy_trail = self.p2_trail
        else:
            pos = self.p2_pos
            direction = self.p2_dir
            own_trail = self.p2_trail
            enemy_trail = self.p1_trail

        moves = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # Determine new direction
        if move not in moves:
            move = 'up' if direction[0] == 0 else 'left'

        new_dir = moves[move]

        # Can't reverse
        if new_dir == (-direction[0], -direction[1]):
            new_dir = direction

        # Add current position to trail before moving
        own_trail.add(pos)

        # Calculate new position
        new_pos = (pos[0] + new_dir[0], pos[1] + new_dir[1])

        # Check if out of bounds
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return False

        # Check collision with own trail
        if new_pos in own_trail:
            return False

        # Check collision with enemy trail
        if new_pos in enemy_trail:
            return False

        # Update player state
        if player == 1:
            self.p1_pos = new_pos
            self.p1_dir = new_dir
        else:
            self.p2_pos = new_pos
            self.p2_dir = new_dir

        return True

    def step(self, move1: str, move2: str) -> Tuple[bool, bool]:
        """Execute one game turn. Returns (p1_alive, p2_alive)"""
        self.turn += 1

        p1_alive = self.move_player(1, move1)
        p2_alive = self.move_player(2, move2)

        # Determine game state
        if not p1_alive and not p2_alive:
            self.game_over = True
            self.winner = 'draw'
        elif not p1_alive:
            self.game_over = True
            self.winner = 'p2'
        elif not p2_alive:
            self.game_over = True
            self.winner = 'p1'
        elif self.turn >= self.max_turns:
            self.game_over = True
            self.winner = 'draw'

        return p1_alive, p2_alive

    def get_state_json(self) -> str:
        """Serialize game state to JSON"""
        return json.dumps({
            'grid_size': self.grid_size,
            'turn': self.turn,
            'max_turns': self.max_turns,
            'p1_pos': self.p1_pos,
            'p1_trail': list(self.p1_trail),
            'p2_pos': self.p2_pos,
            'p2_trail': list(self.p2_trail),
            'game_over': self.game_over,
            'winner': self.winner,
            'board': self.get_board_state()
        })

# ============================================================================
# LLM INTEGRATION
# ============================================================================

class TronAI:
    """LLM-based Tron AI player"""

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=API_KEY, base_url=API_URL)

    def get_move(self, board_state: str, player: int, pos: Tuple[int, int],
                 direction: Tuple[int, int], safe_moves: List[str]) -> str:
        """Query LLM for next move"""

        dir_name = {(0, 1): 'RIGHT', (0, -1): 'LEFT', (-1, 0): 'UP', (1, 0): 'DOWN'}.get(direction, 'UNKNOWN')

        prompt = f"""Current board state (. = empty, H = your head, t = your trail, E = enemy head, e = enemy trail):

{board_state}

Your position: {pos}
Your direction: {dir_name}
Safe moves: {', '.join(safe_moves) if safe_moves else 'NONE - YOU WILL DIE'}

Strategy: Avoid walls and trails. Try to trap your opponent. Move strategically.

Respond with ONLY one word: up, down, left, or right
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Tron Light Cycle AI. Respond with ONLY one word: up, down, left, or right. No explanation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=10,
                timeout=10.0
            )

            move = response.choices[0].message.content.strip().lower()
            valid_moves = ['up', 'down', 'left', 'right']

            if move not in valid_moves:
                # Try to extract a valid move from the response
                for valid_move in valid_moves:
                    if valid_move in move:
                        return valid_move
                # Default to first safe move
                return safe_moves[0] if safe_moves else 'up'

            return move

        except Exception as e:
            # Fallback to safe move or default
            return safe_moves[0] if safe_moves else 'up'

# ============================================================================
# GRADIO APP
# ============================================================================

def create_game_ui():
    """Create the Gradio interface"""

    game = None
    ai1 = None
    ai2 = None

    def start_battle(model1: str, model2: str,
                     grid_size: int, max_turns: int):
        """Initialize and start the game"""
        nonlocal game, ai1, ai2

        # Validate inputs
        if not API_KEY:
            yield ("", "Error: API_KEY secret not configured", "", "")
            return

        try:
            game = TronLightCyclesGame(grid_size=grid_size, max_turns=max_turns)
            ai1 = TronAI(model1)
            ai2 = TronAI(model2)
        except Exception as e:
            yield ("", f"Error initializing: {str(e)}", "", "")
            return

        log1 = f"Starting {model1} vs {model2} on {grid_size}x{grid_size} grid\n"
        log2 = f"Starting {model1} vs {model2} on {grid_size}x{grid_size} grid\n"

        yield (game.get_state_json(), log1, log2, "")

        # Main game loop
        while not game.game_over:
            board = game.get_board_state()

            # Get safe moves for both players
            safe1 = game.get_safe_moves(game.p1_pos, game.p1_dir, game.p1_trail, game.p2_trail)
            safe2 = game.get_safe_moves(game.p2_pos, game.p2_dir, game.p2_trail, game.p1_trail)

            # Query both AIs in parallel
            moves = ['up', 'up']
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future1 = executor.submit(
                        ai1.get_move, board, 1, game.p1_pos, game.p1_dir, safe1
                    )
                    future2 = executor.submit(
                        ai2.get_move, board, 2, game.p2_pos, game.p2_dir, safe2
                    )

                    moves[0] = future1.result(timeout=12)
                    moves[1] = future2.result(timeout=12)
            except Exception as e:
                moves[0] = safe1[0] if safe1 else 'up'
                moves[1] = safe2[0] if safe2 else 'up'

            # Execute moves
            p1_alive, p2_alive = game.step(moves[0], moves[1])

            # Update logs
            log1 += f"Turn {game.turn}: You moved {moves[0]} | Opponent moved {moves[1]}\n"
            log2 += f"Turn {game.turn}: You moved {moves[1]} | Opponent moved {moves[0]}\n"

            if not p1_alive:
                log1 += "DIED!\n"
            if not p2_alive:
                log2 += "DIED!\n"

            yield (game.get_state_json(), log1, log2, "")

        # Game over
        if game.winner == 'p1':
            result = f"GAME OVER - {model1} WINS!"
            log1 += result + "\n"
            log2 += result + "\n"
        elif game.winner == 'p2':
            result = f"GAME OVER - {model2} WINS!"
            log1 += result + "\n"
            log2 += result + "\n"
        else:
            result = "GAME OVER - DRAW!"
            log1 += result + "\n"
            log2 += result + "\n"

        yield (game.get_state_json(), log1, log2, result)

    # ========================================================================
    # BUILD UI
    # ========================================================================

    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="pink"
        ),
        css="""
        body {
            background-color: #0a0a1a;
            color: #e2e8f0;
        }
        .gradio-container {
            background-color: #0a0a1a;
            max-width: 1400px;
        }
        .block {
            background-color: #111827;
            border-color: #1f2937;
        }
        textarea {
            background-color: #0f172a;
            color: #e2e8f0;
            border-color: #1f2937;
        }
        input {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            border-color: #1f2937 !important;
        }
        .gr-button {
            background: linear-gradient(135deg, #22d3ee 0%, #f472b6 100%);
            color: #0a0a1a;
            border: none;
            font-weight: bold;
        }
        .gr-button:hover {
            opacity: 0.9;
        }
        .canvas-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }
        """
    ) as demo:

        # Title
        gr.Markdown("""
        # Tron Light Cycles AI
        **Two LLM models compete in a light cycle battle • Powered by Crusoe Foundry**
        """)

        with gr.Row():
            # Left sidebar - controls
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Configuration")

                model_options = [
                    "deepseek-ai/DeepSeek-V3-0324",
                    "deepseek-ai/DeepSeek-R1-0528",
                    "meta-llama/Llama-3.3-70B-Instruct",
                    "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "google/gemma-3-12b-it"
                ]

                model1 = gr.Dropdown(
                    choices=model_options,
                    value="deepseek-ai/DeepSeek-V3-0324",
                    label="Player 1 Model (Cyan)",
                    interactive=True
                )

                model2 = gr.Dropdown(
                    choices=model_options,
                    value="meta-llama/Llama-3.3-70B-Instruct",
                    label="Player 2 Model (Pink)",
                    interactive=True
                )

                grid_size = gr.Slider(
                    minimum=20,
                    maximum=40,
                    value=30,
                    step=1,
                    label="Grid Size",
                    interactive=True
                )

                max_turns = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=900,
                    step=50,
                    label="Max Turns",
                    interactive=True
                )

                start_btn = gr.Button(
                    "Start Battle",
                    size="lg",
                    variant="primary"
                )

            # Center - canvas
            with gr.Column(scale=2):
                gr.Markdown("### Battle Arena")

                canvas_html = gr.HTML("""
                <div class="canvas-container">
                    <canvas id="tronCanvas" width="600" height="600"
                            style="border: 2px solid #22d3ee; background-color: #0a0a1a;
                                   box-shadow: 0 0 20px rgba(34, 211, 238, 0.3);"></canvas>
                </div>
                <script>
                const canvas = document.getElementById('tronCanvas');
                const ctx = canvas.getContext('2d');

                window.renderTron = function(jsonStr) {
                    try {
                        const state = JSON.parse(jsonStr);

                        // Clear canvas
                        ctx.fillStyle = '#0a0a1a';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);

                        // Draw subtle grid
                        ctx.strokeStyle = 'rgba(34, 211, 238, 0.1)';
                        ctx.lineWidth = 0.5;
                        const cellSize = canvas.width / state.grid_size;
                        for (let i = 0; i <= state.grid_size; i++) {
                            ctx.beginPath();
                            ctx.moveTo(i * cellSize, 0);
                            ctx.lineTo(i * cellSize, canvas.height);
                            ctx.stroke();
                            ctx.beginPath();
                            ctx.moveTo(0, i * cellSize);
                            ctx.lineTo(canvas.width, i * cellSize);
                            ctx.stroke();
                        }

                        // Draw trails
                        const cellSize2 = canvas.width / state.grid_size;

                        // Player 1 trail (cyan)
                        ctx.fillStyle = '#22d3ee';
                        ctx.shadowColor = 'rgba(34, 211, 238, 0.8)';
                        ctx.shadowBlur = 8;
                        for (const pos of state.p1_trail) {
                            const [r, c] = pos;
                            ctx.fillRect(c * cellSize2, r * cellSize2, cellSize2, cellSize2);
                        }

                        // Player 2 trail (pink)
                        ctx.fillStyle = '#f472b6';
                        ctx.shadowColor = 'rgba(244, 114, 182, 0.8)';
                        for (const pos of state.p2_trail) {
                            const [r, c] = pos;
                            ctx.fillRect(c * cellSize2, r * cellSize2, cellSize2, cellSize2);
                        }

                        // Reset shadow
                        ctx.shadowBlur = 0;

                        // Draw heads with glow
                        const headSize = cellSize2 * 0.8;
                        const headOffset = (cellSize2 - headSize) / 2;

                        // Player 1 head
                        const [r1, c1] = state.p1_pos;
                        ctx.fillStyle = '#22d3ee';
                        ctx.shadowColor = 'rgba(34, 211, 238, 1)';
                        ctx.shadowBlur = 15;
                        ctx.fillRect(c1 * cellSize2 + headOffset, r1 * cellSize2 + headOffset,
                                    headSize, headSize);

                        // Player 2 head
                        const [r2, c2] = state.p2_pos;
                        ctx.fillStyle = '#f472b6';
                        ctx.shadowColor = 'rgba(244, 114, 182, 1)';
                        ctx.shadowBlur = 15;
                        ctx.fillRect(c2 * cellSize2 + headOffset, r2 * cellSize2 + headOffset,
                                    headSize, headSize);

                        ctx.shadowBlur = 0;

                        // HUD
                        ctx.fillStyle = '#e2e8f0';
                        ctx.font = '14px monospace';
                        ctx.fillText('Turn: ' + state.turn + ' / ' + state.max_turns, 10, 20);

                    } catch(e) {
                        console.error('Render error:', e);
                    }
                };
            </script>
                """)

                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Waiting to start...",
                    lines=1
                )

            # Right sidebar - logs
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Move Logs")

                log1 = gr.Textbox(
                    label="Player 1 (Cyan)",
                    interactive=False,
                    lines=15,
                    value="Waiting for battle...",
                    max_lines=30
                )

                log2 = gr.Textbox(
                    label="Player 2 (Pink)",
                    interactive=False,
                    lines=15,
                    value="Waiting for battle...",
                    max_lines=30
                )

        # Hidden state textbox for canvas rendering
        game_state_json = gr.Textbox(
            visible=False,
            value="",
            every=0.1
        )

        # Bind canvas rendering to state changes
        game_state_json.change(
            fn=None,
            inputs=game_state_json,
            js="(s) => { if (s && window.renderTron) window.renderTron(s); }"
        )

        # Start button handler
        start_btn.click(
            fn=start_battle,
            inputs=[model1, model2, grid_size, max_turns],
            outputs=[game_state_json, log1, log2, status]
        )

    return demo

if __name__ == "__main__":
    demo = create_game_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
