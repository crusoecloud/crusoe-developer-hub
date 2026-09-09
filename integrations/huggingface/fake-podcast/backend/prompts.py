"""Prompt templates for podcast generation."""
from __future__ import annotations

import random


def host_system_prompt(host: dict, other_host: dict) -> str:
    return (
        f"You are {host['name']}, a podcast host.\n"
        f"Your personality: {host['personality']}\n\n"
        f"You are having a conversation with {other_host['name']} "
        f"({other_host['personality']}).\n\n"
        "Rules:\n"
        "- Speak naturally as yourself — contractions, filler words, reactions.\n"
        "- React to what the other host just said before making your own points.\n"
        "- Include interruptions, tangents, and humor where natural.\n"
        "- Keep each turn to 3-6 sentences. Be substantive but conversational.\n"
        "- Do NOT prefix your lines with your name or any label.\n"
        "- Just speak directly as yourself.\n"
        "- Never repeat what you or the other host already said.\n"
        "- Bring in real facts, examples, analogies, and stories.\n"
    )


def opening_prompt(topic: str, angle: str | None) -> str:
    base = f'Start the podcast episode about: "{topic}"'
    if angle:
        base += f'\nApproach it from this angle: "{angle}"'
    base += (
        "\n\nGive an energetic, hooky opening — the kind that makes "
        "listeners stop scrolling. 2-3 sentences max."
    )
    return base


def response_prompt(topic: str) -> str:
    return (
        "Continue the conversation naturally. React to what was just said, "
        "then add your own take. Stay on the topic of "
        f'"{topic}" but tangents are welcome. 3-6 sentences.'
    )


# Variety prompts injected periodically to keep the conversation dynamic
VARIETY_PROMPTS = [
    'Play devil\'s advocate on the last point. Challenge it, poke holes in it, or offer a completely different perspective. Be provocative but fun. 3-6 sentences.',
    'Share a surprising real-world example or story related to what was just discussed. Something the audience probably hasn\'t heard before. 3-6 sentences.',
    'Go on a fascinating tangent — connect what was just said to something totally unexpected from a different field. Then tie it back. 4-6 sentences.',
    'Ask the other host a pointed, specific question that digs deeper into what they just said. Then share your own hot take on it. 3-5 sentences.',
    'Bring up a counterintuitive fact or common misconception related to the topic. Explain why most people get it wrong. 3-6 sentences.',
    'Tell a quick anecdote or hypothetical scenario that illustrates the last point in a vivid way. Make the audience picture it. 3-6 sentences.',
    'Zoom out — how does what you\'ve been discussing connect to a bigger trend or pattern? Then zoom back in with a specific detail. 4-6 sentences.',
    'Disagree with something said earlier in the conversation. Explain why you changed your mind or why that point doesn\'t hold up. 3-6 sentences.',
    'Make a bold prediction related to the topic. Something that would be controversial at a dinner party. Defend it. 3-5 sentences.',
    'Summarize the wildest thing said so far, then one-up it with something even more surprising. 3-5 sentences.',
    'Bring in a pop culture reference, historical parallel, or analogy that reframes the discussion. 3-6 sentences.',
    'React emotionally to the last point — be genuinely excited, skeptical, or blown away. Let the audience feel your reaction. 3-5 sentences.',
]


def variety_prompt(topic: str, round_num: int) -> str:
    """Pick a variety prompt to inject dynamism into the conversation."""
    prompt = random.choice(VARIETY_PROMPTS)
    return f'Topic: "{topic}". {prompt}'


def deeper_prompt(topic: str) -> str:
    return (
        "The audience wants you to go deeper on the last point discussed. "
        "Expand with more detail, examples, or a surprising fact. "
        f'Stay grounded in "{topic}". 4-6 sentences.'
    )


def wrapup_prompt(topic: str) -> str:
    return (
        f'Wrap up this episode about "{topic}". Give your final hot take '
        "or takeaway, thank the audience, and tease a possible future episode topic. "
        "3-5 sentences."
    )


def inspire_prompt(category: str | None) -> str:
    cat_line = ""
    if category and category != "all":
        cat_line = f' in the "{category}" category'
    return (
        f"Generate exactly 4 creative, unexpected, and fun podcast topic ideas{cat_line}.\n"
        "Each should be a single sentence that sounds like an intriguing episode title.\n"
        "Make them diverse — mix serious and absurd.\n"
        "Return ONLY a JSON array of 4 strings, no other text.\n"
        'Example: ["Topic one", "Topic two", "Topic three", "Topic four"]'
    )
