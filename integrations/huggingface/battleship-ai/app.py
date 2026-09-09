import os
import json
import random
import re
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict
from enum import Enum

import gradio as gr
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = os.environ.get("API_URL", "https://managed-inference-api-proxy.crusoecloud.com/v1")
API_KEY = os.environ.get("API_KEY", "")
AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "google/gemma-3-12b-it",
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class CellState(Enum):
    UNKNOWN = 0
    HIT = 1
    MISS = 2
    WATER = 3


@dataclass
class Ship:
    name: str
    length: int
    positions: List[Tuple[int, int]]
    hits: int = 0

    def is_sunk(self) -> bool:
        return self.hits == self.length


class BattleshipGame:
    def __init__(self):
        self.grid_size = 10
        self.rows = "ABCDEFGHIJ"
        self.cols = list(range(1, 11))

        # Initialize game state
        self.p1_grid = None
        self.p2_grid = None
        self.p1_attack_board = None
        self.p2_attack_board = None

        self.p1_ships = []
        self.p2_ships = []

        self.p1_shots = []
        self.p2_shots = []

        self.current_turn = 1  # 1 for P1, 2 for P2
        self.last_shot = None
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.max_turns = 200

        self.move_log = []

        # Initialize grids
        self._reset_grids()
        self._place_all_ships()

    def _reset_grids(self):
        """Initialize empty 10x10 grids (0 = empty)"""
        self.p1_grid = [[0 for _ in range(10)] for _ in range(10)]
        self.p2_grid = [[0 for _ in range(10)] for _ in range(10)]
        self.p1_attack_board = [[0 for _ in range(10)] for _ in range(10)]
        self.p2_attack_board = [[0 for _ in range(10)] for _ in range(10)]

    def _place_all_ships(self):
        """Place all ships for both players randomly"""
        ship_specs = [
            ("Carrier", 5),
            ("Battleship", 4),
            ("Cruiser", 3),
            ("Submarine", 3),
            ("Destroyer", 2),
        ]

        self.p1_ships = self._place_ships_on_grid(self.p1_grid, ship_specs)
        self.p2_ships = self._place_ships_on_grid(self.p2_grid, ship_specs)

    def _place_ships_on_grid(
        self, grid: List[List[int]], ship_specs: List[Tuple[str, int]]
    ) -> List[Ship]:
        """Place ships on a grid randomly (returns list of Ship objects)"""
        ships = []
        for ship_name, ship_length in ship_specs:
            placed = False
            while not placed:
                # Random orientation (0 = horizontal, 1 = vertical)
                is_vertical = random.choice([True, False])
                if is_vertical:
                    row = random.randint(0, 9)
                    col = random.randint(0, 10 - ship_length)
                    positions = [(row, col + i) for i in range(ship_length)]
                else:
                    row = random.randint(0, 10 - ship_length)
                    col = random.randint(0, 9)
                    positions = [(row + i, col) for i in range(ship_length)]

                # Check if placement is valid (no overlap)
                if all(grid[r][c] == 0 for r, c in positions):
                    # Mark grid
                    for r, c in positions:
                        grid[r][c] = 1

                    # Create ship object
                    ship = Ship(name=ship_name, length=ship_length, positions=positions)
                    ships.append(ship)
                    placed = True

        return ships

    def coord_to_index(self, coord: str) -> Tuple[int, int]:
        """Convert 'A5' format to (0, 4) indices"""
        if len(coord) < 2:
            return None
        row_char = coord[0].upper()
        try:
            col_num = int(coord[1:])
            row_idx = self.rows.index(row_char)
            col_idx = col_num - 1
            if 0 <= row_idx < 10 and 0 <= col_idx < 10:
                return (row_idx, col_idx)
        except (ValueError, IndexError):
            pass
        return None

    def index_to_coord(self, row: int, col: int) -> str:
        """Convert (0, 4) indices to 'A5' format"""
        return f"{self.rows[row]}{col + 1}"

    def process_shot(self, attacker: int, target_coord: str) -> Tuple[bool, bool]:
        """
        Process a shot. Returns (is_hit, is_new_shot)
        attacker: 1 or 2
        """
        target_idx = self.coord_to_index(target_coord)
        if not target_idx:
            return False, False

        row, col = target_idx

        # Determine which grid is being attacked
        if attacker == 1:
            target_grid = self.p2_grid
            target_ships = self.p2_ships
            shot_list = self.p1_shots
            attack_board = self.p1_attack_board
        else:
            target_grid = self.p1_grid
            target_ships = self.p1_ships
            shot_list = self.p2_shots
            attack_board = self.p2_attack_board

        # Check if already shot at
        if (row, col) in shot_list:
            return False, False

        shot_list.append((row, col))

        # Check if hit
        is_hit = target_grid[row][col] == 1

        if is_hit:
            attack_board[row][col] = 1  # 1 = hit
            # Update ship hits
            for ship in target_ships:
                if (row, col) in ship.positions:
                    ship.hits += 1
                    break
        else:
            attack_board[row][col] = 2  # 2 = miss

        self.last_shot = target_coord
        self.turn_count += 1

        return is_hit, True

    def get_ships_remaining(self, player: int) -> int:
        """Count remaining (not sunk) ships"""
        ships = self.p1_ships if player == 1 else self.p2_ships
        return sum(1 for ship in ships if not ship.is_sunk())

    def check_game_over(self) -> bool:
        """Check if game is over and set winner"""
        p1_remaining = self.get_ships_remaining(1)
        p2_remaining = self.get_ships_remaining(2)

        if p1_remaining == 0:
            self.game_over = True
            self.winner = 2
            return True
        elif p2_remaining == 0:
            self.game_over = True
            self.winner = 1
            return True
        elif self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = None  # Draw
            return True

        return False

    def get_sunk_ships(self, player: int) -> List[str]:
        """Get list of sunk ship names"""
        ships = self.p1_ships if player == 1 else self.p2_ships
        return [ship.name for ship in ships if ship.is_sunk()]

    def build_attack_prompt(self, attacker: int) -> Tuple[str, str]:
        """Build prompt for LLM (returns prompt and ASCII board)"""
        if attacker == 1:
            shots = self.p1_shots
            attack_board = self.p1_attack_board
        else:
            shots = self.p2_shots
            attack_board = self.p2_attack_board

        hits = [self.index_to_coord(r, c) for r, c in shots if attack_board[r][c] == 1]
        misses = [self.index_to_coord(r, c) for r, c in shots if attack_board[r][c] == 2]
        sunk_ships = self.get_sunk_ships(3 - attacker)  # opponent

        # Build ASCII board
        ascii_board = "  1 2 3 4 5 6 7 8 9 10\n"
        for i, row_char in enumerate(self.rows):
            row_str = row_char + " "
            for j in range(10):
                if attack_board[i][j] == 1:
                    row_str += "X "
                elif attack_board[i][j] == 2:
                    row_str += "O "
                else:
                    row_str += ". "
            ascii_board += row_str + "\n"

        prompt = f"""You are playing Battleship on a 10x10 grid (A-J rows, 1-10 columns).

YOUR ATTACK BOARD (what you know about enemy):
{ascii_board}

HITS SO FAR: {', '.join(hits) if hits else 'None'}
MISSES SO FAR: {', '.join(misses) if misses else 'None'}
SHIPS SUNK: {', '.join(sunk_ships) if sunk_ships else 'None'}

Pick a coordinate to fire at. Respond with ONLY a coordinate (e.g., E5). Target near previous hits to sink ships."""

        return prompt, ascii_board

    def get_game_state_json(self, model1: str, model2: str) -> str:
        """Get current game state as JSON for rendering"""
        state = {
            "p1_attack_grid": self.p1_attack_board,
            "p2_attack_grid": self.p2_attack_board,
            "p1_ships_remaining": self.get_ships_remaining(1),
            "p2_ships_remaining": self.get_ships_remaining(2),
            "turn": self.current_turn,
            "lastShot": self.last_shot,
            "gameOver": self.game_over,
            "winner": self.winner,
            "model1": model1,
            "model2": model2,
            "turnCount": self.turn_count,
        }
        return json.dumps(state)

    def get_move_log_html(self) -> str:
        """Get move log as HTML"""
        html = "<div style='font-family: monospace; font-size: 12px; color: #0ef; max-height: 200px; overflow-y: auto;'>"
        for i, log in enumerate(self.move_log):
            html += f"<div>{log}</div>"
        html += "</div>"
        return html


# ============================================================================
# GAME CONTROLLER
# ============================================================================

game_instance = None


def initialize_game(model1: str, model2: str):
    """Initialize a new game"""
    global game_instance
    game_instance = BattleshipGame()
    return f"Game initialized! {model1} vs {model2}"


def play_turn(model1: str, model2: str):
    """Play a single turn (one player's shot)"""
    global game_instance

    if not game_instance:
        return "Game not initialized. Click 'Start Battle' first."

    if game_instance.game_over:
        return "Game is over."

    # Determine current player
    attacker = game_instance.current_turn
    model_name = model1 if attacker == 1 else model2

    # Build prompt
    prompt, ascii_board = game_instance.build_attack_prompt(attacker)

    # Call LLM
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a Battleship AI. Respond with ONLY a coordinate like A5 or J10. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=10,
        )
        move_text = response.choices[0].message.content.strip()
    except Exception as e:
        move_text = None
        game_instance.move_log.append(f"P{attacker} ({model_name}): API Error: {str(e)}")

    # Parse move
    target_coord = None
    if move_text:
        # Extract coordinate pattern (letter + digits)
        match = re.search(r"([A-Ja-j])(\d+)", move_text)
        if match:
            target_coord = (match.group(1).upper() + match.group(2)).strip()

    # Validate move
    valid = False
    if target_coord:
        idx = game_instance.coord_to_index(target_coord)
        if idx and idx not in (game_instance.p1_shots if attacker == 1 else game_instance.p2_shots):
            valid = True

    # If invalid, pick random valid move
    if not valid:
        if attacker == 1:
            used_shots = game_instance.p1_shots
        else:
            used_shots = game_instance.p2_shots

        available = [
            (r, c)
            for r in range(10)
            for c in range(10)
            if (r, c) not in used_shots
        ]
        if available:
            r, c = random.choice(available)
            target_coord = game_instance.index_to_coord(r, c)
        else:
            return "No valid moves available."

    # Process shot
    is_hit, is_new = game_instance.process_shot(attacker, target_coord)

    if not is_new:
        game_instance.move_log.append(f"P{attacker} ({model_name}): {target_coord} (ALREADY SHOT)")
    else:
        hit_str = "HIT" if is_hit else "MISS"
        game_instance.move_log.append(f"P{attacker} ({model_name}): {target_coord} - {hit_str}")

    # Switch turn
    game_instance.current_turn = 3 - attacker

    # Check for game over
    game_instance.check_game_over()

    return game_instance.get_game_state_json(model1, model2)


def auto_play(model1: str, model2: str, num_turns: int = 50):
    """Auto-play multiple turns"""
    global game_instance

    if not game_instance:
        initialize_game(model1, model2)

    results = []
    for i in range(num_turns):
        if game_instance.game_over or game_instance.turn_count >= game_instance.max_turns:
            break
        result = play_turn(model1, model2)
        results.append(result)

    return game_instance.get_game_state_json(model1, model2)


# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui():
    """Create Gradio UI"""

    with gr.Blocks(
        title="Battleship AI",
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="pink",
        ),
        css="""
        body { background: #0a0a1a; }
        .gradio-container { background: #0a0a1a; }
        .block { background: #111827; border: 1px solid #1e293b; }
        """
    ) as demo:

        gr.Markdown("# Battleship AI")
        gr.Markdown("Two LLM models compete in Battleship • Powered by Crusoe Foundry")

        with gr.Row():
            # Left: Controls
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                model1 = gr.Dropdown(
                    label="Player 1 Model (Cyan)",
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0],
                )
                model2 = gr.Dropdown(
                    label="Player 2 Model (Pink)",
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[1],
                )

                init_btn = gr.Button("Start Battle", variant="primary")
                auto_play_btn = gr.Button("Auto-Play 50 Turns", variant="secondary")

            # Center: Canvas
            with gr.Column(scale=2):
                gr.Markdown("### Game Board")
                game_state_hidden = gr.Textbox(visible=False)
                canvas_html = gr.HTML(
                    value="""
                    <canvas id="battleshipCanvas" width="620" height="320"></canvas>
                    <script>
                    window.renderBattleship = function(jsonStr) {
                        try {
                            const state = JSON.parse(jsonStr);
                            const canvas = document.getElementById('battleshipCanvas');
                            if (!canvas) return;
                            const ctx = canvas.getContext('2d');

                            // Clear canvas
                            ctx.fillStyle = '#0a1628';
                            ctx.fillRect(0, 0, 620, 320);

                            // Grid settings
                            const cellSize = 30;
                            const marginTop = 20;
                            const marginLeft = 30;
                            const boardWidth = 300;
                            const boardHeight = 300;
                            const gap = 20;

                            // Colors
                            const colors = {
                                water: '#0a1628',
                                hit: '#ef4444',
                                miss: '#475569',
                                ship: '#334155',
                                text: '#0ef',
                                gridLine: '#1e293b',
                            };

                            // Render labels
                            ctx.fillStyle = colors.text;
                            ctx.font = 'bold 12px monospace';
                            ctx.textAlign = 'center';

                            // Row labels (A-J)
                            const rows = 'ABCDEFGHIJ'.split('');
                            for (let i = 0; i < 10; i++) {
                                ctx.fillText(rows[i], marginLeft - 10, marginTop + i * cellSize + cellSize / 2 + 5);
                            }

                            // Column labels (1-10)
                            for (let i = 0; i < 10; i++) {
                                ctx.fillText((i + 1).toString(), marginLeft + i * cellSize + cellSize / 2, marginTop - 5);
                            }

                            // Render P1 attack board (left)
                            renderGrid(ctx, state.p1_attack_grid, marginLeft, marginTop, cellSize, colors, state.lastShot, 1);

                            // Render P2 attack board (right)
                            renderGrid(ctx, state.p2_attack_grid, marginLeft + boardWidth + gap, marginTop, cellSize, colors, state.lastShot, 2);

                            // Render titles
                            ctx.fillStyle = colors.text;
                            ctx.font = 'bold 14px monospace';
                            ctx.textAlign = 'left';
                            ctx.fillText('P1 Attack (' + state.model1.split('/').pop() + ')', marginLeft, marginTop + boardHeight + 15);
                            ctx.fillText('P2 Attack (' + state.model2.split('/').pop() + ')', marginLeft + boardWidth + gap, marginTop + boardHeight + 15);

                            // Render HUD
                            ctx.font = 'bold 11px monospace';
                            ctx.fillStyle = '#0ef';
                            ctx.textAlign = 'left';
                            ctx.fillText('Ships: ' + state.p1_ships_remaining, marginLeft, marginTop + boardHeight + 35);
                            ctx.fillText('Ships: ' + state.p2_ships_remaining, marginLeft + boardWidth + gap, marginTop + boardHeight + 35);

                            // Game over overlay
                            if (state.gameOver) {
                                ctx.fillStyle = 'rgba(10, 10, 26, 0.8)';
                                ctx.fillRect(0, 0, 620, 320);
                                ctx.fillStyle = '#0ef';
                                ctx.font = 'bold 20px monospace';
                                ctx.textAlign = 'center';
                                if (state.winner === 1) {
                                    ctx.fillText('Player 1 Wins!', 310, 150);
                                } else if (state.winner === 2) {
                                    ctx.fillText('Player 2 Wins!', 310, 150);
                                } else {
                                    ctx.fillText('Draw!', 310, 150);
                                }
                                ctx.font = 'bold 12px monospace';
                                ctx.fillText('Turns: ' + state.turnCount, 310, 180);
                            }
                        } catch (e) {
                            console.error('Render error:', e);
                        }
                    };

                    function renderGrid(ctx, grid, x, y, size, colors, lastShot, playerNum) {
                        // Grid background
                        ctx.fillStyle = colors.water;
                        ctx.fillRect(x, y, size * 10, size * 10);

                        // Grid lines
                        ctx.strokeStyle = colors.gridLine;
                        ctx.lineWidth = 1;
                        for (let i = 0; i <= 10; i++) {
                            ctx.beginPath();
                            ctx.moveTo(x + i * size, y);
                            ctx.lineTo(x + i * size, y + size * 10);
                            ctx.stroke();

                            ctx.beginPath();
                            ctx.moveTo(x, y + i * size);
                            ctx.lineTo(x + size * 10, y + i * size);
                            ctx.stroke();
                        }

                        // Cells
                        for (let i = 0; i < 10; i++) {
                            for (let j = 0; j < 10; j++) {
                                const cellX = x + j * size;
                                const cellY = y + i * size;

                                if (grid[i][j] === 1) { // Hit
                                    ctx.fillStyle = colors.hit;
                                    ctx.fillRect(cellX + 2, cellY + 2, size - 4, size - 4);
                                    ctx.fillStyle = '#ff0000';
                                    ctx.globalAlpha = 0.3;
                                    ctx.fillRect(cellX + 2, cellY + 2, size - 4, size - 4);
                                    ctx.globalAlpha = 1;
                                } else if (grid[i][j] === 2) { // Miss
                                    ctx.fillStyle = colors.miss;
                                    ctx.fillRect(cellX + 4, cellY + 4, size - 8, size - 8);
                                }
                            }
                        }
                    }
                    </script>
                    """
                )

        # Bottom: Logs
        with gr.Row():
            move_log = gr.HTML(
                value="<div style='font-family: monospace; font-size: 12px; color: #0ef; max-height: 150px; overflow-y: auto;'></div>"
            )

        # Event handlers
        def on_init_click():
            initialize_game(model1.value, model2.value)
            return game_instance.get_game_state_json(model1.value, model2.value), game_instance.get_move_log_html()

        def on_auto_play_click():
            auto_play(model1.value, model2.value, 50)
            return game_instance.get_game_state_json(model1.value, model2.value), game_instance.get_move_log_html()

        init_btn.click(
            fn=on_init_click,
            outputs=[game_state_hidden, move_log],
        )

        auto_play_btn.click(
            fn=on_auto_play_click,
            outputs=[game_state_hidden, move_log],
        )

        game_state_hidden.change(
            fn=lambda x: x,
            inputs=[game_state_hidden],
            outputs=[],
            js="(x) => { if (x && x.trim()) { window.renderBattleship(x); } }",
        )

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
