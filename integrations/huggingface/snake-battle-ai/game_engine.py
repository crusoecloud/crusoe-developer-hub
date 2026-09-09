"""
Snake Battle Game Engine
Two snakes compete on a grid. Each snake is controlled by an LLM.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


DIRECTION_DELTAS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


@dataclass
class Snake:
    name: str
    body: list  # list of (x, y) tuples, head is body[0]
    direction: Direction
    alive: bool = True
    score: int = 0
    color: str = "#00ff00"


@dataclass
class GameState:
    width: int = 20
    height: int = 20
    snakes: list = field(default_factory=list)
    food: list = field(default_factory=list)
    turn: int = 0
    game_over: bool = False
    winner: Optional[str] = None
    max_turns: int = 300
    history: list = field(default_factory=list)


def create_game(width=20, height=20, max_turns=300) -> GameState:
    """Initialize a new game with two snakes."""
    game = GameState(width=width, height=height, max_turns=max_turns)

    # Snake 1 starts on the left side
    snake1 = Snake(
        name="Snake 1",
        body=[(3, height // 2), (2, height // 2), (1, height // 2)],
        direction=Direction.RIGHT,
        color="#4ade80",  # green
    )

    # Snake 2 starts on the right side
    snake2 = Snake(
        name="Snake 2",
        body=[
            (width - 4, height // 2),
            (width - 3, height // 2),
            (width - 2, height // 2),
        ],
        direction=Direction.LEFT,
        color="#f472b6",  # pink
    )

    game.snakes = [snake1, snake2]

    # Spawn initial food
    for _ in range(3):
        spawn_food(game)

    return game


def spawn_food(game: GameState):
    """Spawn food at a random empty position."""
    occupied = set()
    for snake in game.snakes:
        for segment in snake.body:
            occupied.add(tuple(segment))
    for f in game.food:
        occupied.add(tuple(f))

    empty = []
    for x in range(game.width):
        for y in range(game.height):
            if (x, y) not in occupied:
                empty.append((x, y))

    if empty:
        pos = random.choice(empty)
        game.food.append(pos)


def get_game_state_for_llm(game: GameState, snake_index: int) -> dict:
    """Create a simplified game state representation for the LLM."""
    snake = game.snakes[snake_index]
    opponent_index = 1 - snake_index
    opponent = game.snakes[opponent_index]

    return {
        "board": {"width": game.width, "height": game.height},
        "turn": game.turn,
        "you": {
            "head": {"x": snake.body[0][0], "y": snake.body[0][1]},
            "body": [{"x": s[0], "y": s[1]} for s in snake.body],
            "length": len(snake.body),
            "direction": snake.direction.value,
            "score": snake.score,
        },
        "opponent": {
            "head": {"x": opponent.body[0][0], "y": opponent.body[0][1]},
            "body": [{"x": s[0], "y": s[1]} for s in opponent.body],
            "length": len(opponent.body),
            "alive": opponent.alive,
        },
        "food": [{"x": f[0], "y": f[1]} for f in game.food],
    }


def build_llm_prompt(game_state: dict) -> str:
    """Build the prompt for the LLM to decide the next move."""
    head = game_state["you"]["head"]
    body = game_state["you"]["body"]
    opponent_body = game_state["opponent"]["body"]
    food = game_state["food"]
    board = game_state["board"]
    current_dir = game_state["you"]["direction"]

    # Build a simple text map
    grid = [["." for _ in range(board["width"])] for _ in range(board["height"])]

    for seg in body[1:]:
        grid[seg["y"]][seg["x"]] = "s"
    grid[head["y"]][head["x"]] = "H"

    opp_head = game_state["opponent"]["head"]
    for seg in opponent_body[1:]:
        if 0 <= seg["x"] < board["width"] and 0 <= seg["y"] < board["height"]:
            grid[seg["y"]][seg["x"]] = "e"
    if 0 <= opp_head["x"] < board["width"] and 0 <= opp_head["y"] < board["height"]:
        grid[opp_head["y"]][opp_head["x"]] = "E"

    for f in food:
        grid[f["y"]][f["x"]] = "F"

    board_str = "\n".join("".join(row) for row in grid)

    # Determine which moves are safe
    opposite_dir = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }
    blocked = opposite_dir[current_dir]

    prompt = f"""You are playing a competitive Snake game. You control snake 'H'.

BOARD ({board['width']}x{board['height']}):
{board_str}

LEGEND: H=your head, s=your body, E=enemy head, e=enemy body, F=food, .=empty

YOUR HEAD: ({head['x']}, {head['y']})
YOUR LENGTH: {len(body)}
CURRENT DIRECTION: {current_dir}
FOOD LOCATIONS: {', '.join(f"({f['x']},{f['y']})" for f in food)}
ENEMY HEAD: ({opp_head['x']}, {opp_head['y']})

RULES:
- You die if you hit a wall, your own body, or the enemy's body
- Eating food (F) makes you grow and earns a point
- You CANNOT go "{blocked}" (opposite of current direction)
- Available moves: up, down, left, right (except {blocked})

STRATEGY: Eat food, avoid walls/bodies, try to trap the opponent.

Respond with ONLY one word: up, down, left, or right"""

    return prompt


def parse_direction(response: str, current_direction: Direction) -> Direction:
    """Parse LLM response into a direction."""
    response = response.strip().lower()

    for direction in Direction:
        if direction.value in response:
            # Don't allow 180-degree turns
            if direction != OPPOSITE[current_direction]:
                return direction

    # If invalid, continue in the same direction
    return current_direction


def step(game: GameState, moves: list[Direction]) -> GameState:
    """Advance the game by one turn with the given moves for each snake."""
    if game.game_over:
        return game

    game.turn += 1
    new_heads = []

    # Calculate new head positions
    for i, snake in enumerate(game.snakes):
        if not snake.alive:
            new_heads.append(None)
            continue

        direction = moves[i]
        # Prevent 180-degree turns
        if direction == OPPOSITE[snake.direction]:
            direction = snake.direction
        snake.direction = direction

        dx, dy = DIRECTION_DELTAS[direction]
        head = snake.body[0]
        new_head = (head[0] + dx, head[1] + dy)
        new_heads.append(new_head)

    # Check collisions
    deaths = [False, False]

    for i, snake in enumerate(game.snakes):
        if not snake.alive or new_heads[i] is None:
            continue

        nx, ny = new_heads[i]

        # Wall collision
        if nx < 0 or nx >= game.width or ny < 0 or ny >= game.height:
            deaths[i] = True
            continue

        # Self collision (check against current body, excluding tail which will move)
        if (nx, ny) in snake.body[:-1]:
            deaths[i] = True
            continue

        # Collision with opponent's body
        j = 1 - i
        if game.snakes[j].alive:
            if (nx, ny) in game.snakes[j].body[:-1]:
                deaths[i] = True
                continue

    # Head-to-head collision
    if (
        new_heads[0] is not None
        and new_heads[1] is not None
        and new_heads[0] == new_heads[1]
    ):
        deaths[0] = True
        deaths[1] = True

    # Apply moves and check food
    for i, snake in enumerate(game.snakes):
        if not snake.alive or new_heads[i] is None or deaths[i]:
            snake.alive = not deaths[i] if snake.alive else False
            continue

        new_head = new_heads[i]
        snake.body.insert(0, new_head)

        # Check if food was eaten
        ate_food = False
        for f_idx, f in enumerate(game.food):
            if new_head == tuple(f):
                ate_food = True
                snake.score += 1
                game.food.pop(f_idx)
                spawn_food(game)
                break

        if not ate_food:
            snake.body.pop()

    # Record history frame
    frame = {
        "turn": game.turn,
        "snakes": [
            {
                "name": s.name,
                "body": list(s.body),
                "alive": s.alive,
                "score": s.score,
                "direction": s.direction.value,
                "color": s.color,
            }
            for s in game.snakes
        ],
        "food": list(game.food),
    }
    game.history.append(frame)

    # Check game over conditions
    alive_snakes = [s for s in game.snakes if s.alive]
    if len(alive_snakes) == 0:
        game.game_over = True
        game.winner = "Draw"
    elif len(alive_snakes) == 1:
        game.game_over = True
        game.winner = alive_snakes[0].name
    elif game.turn >= game.max_turns:
        game.game_over = True
        scores = [(s.score, s.name) for s in game.snakes]
        scores.sort(reverse=True)
        if scores[0][0] > scores[1][0]:
            game.winner = scores[0][1]
        else:
            game.winner = "Draw"

    return game
