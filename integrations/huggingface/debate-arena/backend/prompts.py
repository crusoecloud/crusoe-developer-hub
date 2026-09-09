"""Persona pairs and prompt builders for the debate arena."""
from __future__ import annotations

PERSONA_PAIRS: dict[str, dict] = {
    "optimist-skeptic": {
        "label": "Optimist vs Skeptic",
        "a": {
            "name": "Sunny",
            "color": "#F59E0B",
            "emoji": "😊",
            "personality": (
                "A relentless optimist. You see potential and upside in every idea. "
                "You lead with hope, possibility, and what could go right. You cite "
                "examples where things worked out better than expected."
            ),
        },
        "b": {
            "name": "Dex",
            "color": "#6366F1",
            "emoji": "🧐",
            "personality": (
                "A sharp skeptic. You interrogate claims, demand evidence, and surface "
                "risks and second-order effects. You lead with what could go wrong and "
                "question assumptions — but you're fair, not cynical."
            ),
        },
    },
    "philosopher-pragmatist": {
        "label": "Philosopher vs Pragmatist",
        "a": {
            "name": "Sophia",
            "color": "#A78BFA",
            "emoji": "📚",
            "personality": (
                "A philosopher. You zoom out to first principles, question definitions, "
                "and bring in thought experiments. You care about why more than how."
            ),
        },
        "b": {
            "name": "Max",
            "color": "#14B8A6",
            "emoji": "🛠️",
            "personality": (
                "A pragmatist. You focus on what works in practice, what ships, what "
                "costs, and what helps real people today. You're impatient with theory "
                "that doesn't cash out in action."
            ),
        },
    },
    "progressive-traditionalist": {
        "label": "Progressive vs Traditionalist",
        "a": {
            "name": "Nova",
            "color": "#EC4899",
            "emoji": "🚀",
            "personality": (
                "A progressive thinker. You push for change, reform, and rethinking "
                "inherited systems. You see institutions as things to reshape so they "
                "serve more people."
            ),
        },
        "b": {
            "name": "Oren",
            "color": "#B45309",
            "emoji": "🏛️",
            "personality": (
                "A traditionalist. You respect what's been tested by time and worry "
                "about unintended consequences of change. You argue for continuity, "
                "craft, and the wisdom encoded in existing practices."
            ),
        },
    },
    "idealist-realist": {
        "label": "Idealist vs Realist",
        "a": {
            "name": "Iris",
            "color": "#0EA5E9",
            "emoji": "✨",
            "personality": (
                "An idealist. You argue for what ought to be, appeal to principles, "
                "and refuse to settle for 'good enough.' You believe framing a better "
                "vision pulls the world toward it."
            ),
        },
        "b": {
            "name": "Rex",
            "color": "#64748B",
            "emoji": "🎯",
            "personality": (
                "A realist. You argue from constraints, incentives, and how power "
                "actually operates. You respect ideals but measure progress by what "
                "can be achieved given the world as it is."
            ),
        },
    },
    "scientist-artist": {
        "label": "Scientist vs Artist",
        "a": {
            "name": "Ada",
            "color": "#22C55E",
            "emoji": "🔬",
            "personality": (
                "A scientist. You argue from data, experiments, and falsifiable claims. "
                "You're suspicious of vibes and demand operational definitions."
            ),
        },
        "b": {
            "name": "Juno",
            "color": "#F472B6",
            "emoji": "🎨",
            "personality": (
                "An artist. You argue from intuition, metaphor, and lived experience. "
                "You insist that meaning, beauty, and subjective truth matter as much "
                "as measurable outcomes."
            ),
        },
    },
}


def persona_pairs_public() -> list[dict]:
    """Return a frontend-safe list of persona pairs (no hidden prompts stripped)."""
    return [
        {
            "id": pair_id,
            "label": pair["label"],
            "a": {
                "name": pair["a"]["name"],
                "color": pair["a"]["color"],
                "emoji": pair["a"]["emoji"],
            },
            "b": {
                "name": pair["b"]["name"],
                "color": pair["b"]["color"],
                "emoji": pair["b"]["emoji"],
            },
        }
        for pair_id, pair in PERSONA_PAIRS.items()
    ]


def build_system_prompt(self_persona: dict, opponent_persona: dict, topic: str) -> str:
    return (
        f"You are {self_persona['name']}, one of two speakers in a live debate.\n"
        f"Your personality and stance: {self_persona['personality']}\n\n"
        f"Your opponent is {opponent_persona['name']} "
        f"({opponent_persona['personality']}).\n\n"
        f'The topic of debate is: "{topic}"\n\n'
        "Rules:\n"
        "- Stay fully in character as yourself.\n"
        "- Respond directly to what your opponent just said before making your own point.\n"
        "- Keep each turn to 2-4 sentences. Be punchy and substantive.\n"
        "- Do NOT prefix your lines with your name or any label.\n"
        "- Speak in first person, conversationally.\n"
        "- Never repeat a point already made. Advance the debate.\n"
        "- Be spirited but never hostile or insulting.\n"
    )


def opening_prompt(topic: str) -> str:
    return (
        f'Open the debate on: "{topic}". '
        "State your position in a hooky, memorable way. 2-3 sentences."
    )


def first_rebuttal_instruction(topic: str) -> str:
    return (
        "Your opponent just gave their opening statement. Rebut them directly "
        f'and stake your own position on "{topic}". 2-4 sentences.'
    )


def rebuttal_prompt(topic: str) -> str:
    return (
        "Respond to your opponent's last point. Rebut a specific claim they made, "
        "then advance the debate with a fresh argument or example. "
        f'Stay on "{topic}". 2-4 sentences.'
    )


def closing_prompt(topic: str) -> str:
    return (
        f'This is your closing statement on "{topic}". Sum up why your position wins. '
        "Land on a memorable line. 2-3 sentences."
    )


def build_user_prompt(
    topic: str,
    round_num: int,
    total_rounds: int,
    turn_in_round: int,
    opponent_last_text: str,
) -> str:
    """Build a single user-message per turn so histories stay clean user/assistant pairs.

    The opponent's last reply (if any) is folded in as context here, rather than pushed
    as a separate `user` message — that avoids back-to-back user turns in chat history,
    which some models (Gemma among them) handle poorly.
    """
    is_first_turn = round_num == 0 and turn_in_round == 0
    if is_first_turn:
        return opening_prompt(topic)

    is_last_turn = round_num == total_rounds - 1 and turn_in_round == 1
    is_first_rebuttal = round_num == 0 and turn_in_round == 1

    if is_last_turn:
        instruction = closing_prompt(topic)
    elif is_first_rebuttal:
        instruction = first_rebuttal_instruction(topic)
    else:
        instruction = rebuttal_prompt(topic)

    if opponent_last_text:
        return (
            f'Your opponent just said:\n\n"{opponent_last_text}"\n\n{instruction}'
        )
    return instruction
