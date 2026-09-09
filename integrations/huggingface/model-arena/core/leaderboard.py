"""
Simple JSON-based leaderboard for tracking community votes.
Persists to a local JSON file (works on HuggingFace Spaces persistent storage).
"""

import json
import os
from datetime import datetime, timezone

LEADERBOARD_FILE = "leaderboard.json"


def _load() -> dict:
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"votes": {}, "matchups": [], "total_votes": 0}
    return {"votes": {}, "matchups": [], "total_votes": 0}


def _save(data: dict):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record_vote(winner_id: str, loser_ids: list[str], prompt: str = ""):
    """Record a vote for the winning model."""
    data = _load()

    # Increment vote count
    if winner_id not in data["votes"]:
        data["votes"][winner_id] = {"wins": 0, "losses": 0, "total": 0}
    data["votes"][winner_id]["wins"] += 1
    data["votes"][winner_id]["total"] += 1

    for loser_id in loser_ids:
        if loser_id not in data["votes"]:
            data["votes"][loser_id] = {"wins": 0, "losses": 0, "total": 0}
        data["votes"][loser_id]["losses"] += 1
        data["votes"][loser_id]["total"] += 1

    # Record matchup
    data["matchups"].append({
        "winner": winner_id,
        "losers": loser_ids,
        "prompt_preview": prompt[:100] if prompt else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    data["total_votes"] = data.get("total_votes", 0) + 1
    _save(data)


def get_leaderboard() -> list[dict]:
    """Return sorted leaderboard: win rate descending, minimum 1 matchup."""
    data = _load()
    board = []

    for model_id, stats in data["votes"].items():
        total = stats["total"]
        if total > 0:
            win_rate = stats["wins"] / total
            board.append({
                "model_id": model_id,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total_matchups": total,
                "win_rate": win_rate,
            })

    board.sort(key=lambda x: (-x["win_rate"], -x["total_matchups"]))
    return board


def get_total_votes() -> int:
    data = _load()
    return data.get("total_votes", 0)
