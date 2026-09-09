# Connect Four AI - Architecture Overview

## Project Structure

```
connect-four-ai/
├── app.py              # Main application (576 lines, single file)
├── requirements.txt    # Python dependencies
├── Procfile            # Heroku deployment configuration
├── README.md           # User documentation
└── ARCHITECTURE.md     # This file
```

## Core Components

### 1. Game Engine (`ConnectFourGame` class)

**Board Representation**
- 6 rows × 7 columns 2D array
- 0 = empty, 1 = Player 1, 2 = Player 2
- Uses tuple (row, col) for tracking last move

**Key Methods**
- `reset()` - Initialize empty board
- `drop_piece(column)` - Drop piece with gravity, return success
- `get_valid_columns()` - Return list of non-full columns
- `get_board_ascii(player_num)` - Format board for LLM with X/O notation
- `get_state_json()` - Serialize state to JSON for canvas rendering
- `_check_winner()` - Detect win/draw after each move (4 directions)
- `switch_player()` - Alternate between players 1 and 2

**Win Detection Algorithm**
- After each move, checks from the last piece position
- Four directions: horizontal, vertical, and two diagonals
- Counts consecutive pieces in both directions from placed piece
- Sets `self.winner` to 1, 2 (win), or 0 (draw)
- Sets `self.game_over = True`

### 2. LLM Integration (`ConnectFourAI` class)

**Configuration**
- AsyncOpenAI client with custom base_url (Crusoe Foundry)
- Temperature: 0.2 (deterministic but not rigid)
- max_tokens: 10 (only needs single digit)

**Flow**
1. Initialize with API URL and API key
2. `get_move(game, model, player_num)` - async method
   - Formats board as ASCII grid (. = empty, X = self, O = opponent)
   - Includes list of valid columns
   - Sends to LLM with system prompt
   - Extracts first digit (0-6) from response via regex
   - Validates column and returns, or returns None

**Error Handling**
- Try-catch wraps API call
- Returns None on API error
- Main game loop has fallback: uses first valid column if AI fails

### 3. Gradio UI

**Layout (3-column design)**
```
┌─────────────────────────────────────────────────┐
│        Connect Four AI (Title + Subtitle)       │
├─────────────────┬──────────────────┬────────────┤
│   Config        │   Canvas Board   │  Logs      │
│                 │   (600×520)      │            │
│ • API URL       │                  │ Player 1   │
│ • API Key       │  [HTML5 Canvas]  │ Moves      │
│ • Player 1 ▼    │                  │            │
│ • Player 2 ▼    │                  │ Player 2   │
│ [Start Battle]  │                  │ Moves      │
└─────────────────┴──────────────────┴────────────┘
│                   Footer (Attribution)          │
└─────────────────────────────────────────────────┘
```

**Color Scheme**
- Background: `#0a0a1a` (dark navy)
- Player 1 (Cyan): `#22d3ee`
- Player 2 (Pink): `#f472b6`
- Grid: `#1e3a3a` (subtle dark teal)
- Text: `#22d3ee` (cyan)

### 4. Canvas Rendering

**HTML5 Canvas (600×520 pixels)**

```javascript
window.renderConnectFour(jsonStr)
├── Parse JSON game state
├── Draw background (#0a0a1a)
├── Draw grid (7×6 with 80px cells)
├── Draw pieces
│   ├── Empty: transparent cyan circles
│   ├── Player 1: solid cyan
│   └── Player 2: solid pink
│   └── Last move: glow effect (shadowBlur=15)
├── Draw column numbers (0-6) at bottom
├── Draw HUD (model names, top and bottom)
└── Draw game over overlay (if game_over=true)
    └── Show "PLAYER X WINS!" or "DRAW"
```

**Event Wiring**
```
Hidden Textbox (game_state)
    ↓ onChange
JavaScript function
    ↓
window.renderConnectFour(jsonStr)
    ↓
Canvas redraws
```

### 5. Game Loop (`auto_play_generator`)

**Generator-based Async Gameplay**

```
initialize_game() 
    → yield initial state
    ↓
while not game_over and moves < 42:
    ├── play_turn()
    │   ├── Get LLM move for current player
    │   ├── Validate and drop piece
    │   ├── Check winner
    │   ├── Log move
    │   ├── Switch player
    │   └── Return state JSON
    ├── yield state, log1, log2
    ↓
Game over → stop iteration
```

## Data Flow

### State JSON Structure
```json
{
  "board": [[0,0,0,...], ...],  // 6x7 grid
  "current_player": 1,           // 1 or 2
  "winner": null|0|1|2,          // null=ongoing, 0=draw, 1/2=winner
  "move_count": 5,               // 0-42
  "game_over": false,            // true when winner or draw
  "last_move": [3, 2],           // [row, col] of last piece
  "models": ["DeepSeek-V3", "Llama-3.3"]  // Model names
}
```

### LLM Prompt Example
```
You are a Connect Four AI. The game board is shown below.

Board state (. = empty, X = your pieces, O = opponent pieces):

. . . . . . .
. . . . . . .
. . . . . . .
. . X . . . .
. O X O . . .
X O O X . . .
0 1 2 3 4 5 6  (column numbers)

Valid columns: [0, 2, 3, 4, 5, 6]

You are player X. The opponent is player O.

Respond with ONLY a single digit from 0 to 6...
```

## Deployment

### Local Development
```bash
python app.py
# Launch at http://localhost:7860
```

### Heroku Deployment
```bash
# Reads PORT from environment
demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

# git push heroku main
```

## Performance Characteristics

- **Turn Latency**: ~1-5 seconds per move (depends on model latency + API)
- **Max Game Duration**: ~45 moves before draw detected
- **Typical Game**: 15-30 moves (15-60 seconds)
- **Memory**: ~10-20 MB (minimal, state-based)
- **Canvas Redraws**: One per move (efficient immediate update)

## Available Models

1. `deepseek-ai/DeepSeek-V3-0324` - Frontier reasoning
2. `deepseek-ai/DeepSeek-R1-0528` - Enhanced reasoning
3. `meta-llama/Llama-3.3-70B-Instruct` - Strong instruct following
4. `Qwen/Qwen3-235B-A22B-Instruct-2507` - Qwen frontier
5. `google/gemma-3-12b-it` - Lightweight, efficient

## Key Design Decisions

1. **Single app.py**: Simplifies deployment, clearer dependencies
2. **Async/await**: Non-blocking LLM calls (though sequential turns)
3. **JSON state**: Canvas doesn't need Python, fully decoupled rendering
4. **Generator pattern**: Gradio-friendly, auto-updates UI each turn
5. **Fallback moves**: Ensures game always completes even if LLM misbehaves
6. **Regex extraction**: Robust to model formatting variations
7. **Temperature 0.2**: Consistent, strategic play
8. **max_tokens=10**: Prevents long responses, cheap/fast

## Extensibility

Future enhancements could include:
- Move validity checking with penalties
- Better opening/endgame strategies
- Game history/replay
- Difficulty levels via system prompt engineering
- Tournament mode (multiple games)
- Human vs AI option
- Move timing analytics
