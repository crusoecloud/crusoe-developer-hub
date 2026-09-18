"""Interactive selection of the upload parameters (purpose, part size, workers). Each
value has a sane default; nothing here touches the network. The choices are gathered and
then recorded together as a fresh state.params."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from . import constants, runtime
from .format import human_size
from .state import State, UploadParams


@dataclass(frozen=True)
class _IntPrompt:
    """A bounded-integer question: prompt text, default, and inclusive [lo, hi]."""

    question: str
    default: int
    lo: int
    hi: int


def choose_params(
    state: State, pick: Callable[..., str] = runtime.pick, text: Callable[..., str] = runtime.text
) -> None:
    """Prompt for the upload parameters, then record them together as state.params."""
    purpose = _choose_purpose(pick)
    part_size = _choose_part_size(text)
    num_parts = math.ceil(state.file.size / part_size)
    print(f"  -> {num_parts} part(s) at {human_size(part_size)} for this {human_size(state.file.size)} file")
    workers = _choose_workers(text)
    state.params = UploadParams(purpose=purpose, part_size=part_size, workers=workers, num_parts=num_parts)
    state.save()
    print(f"  plan: {num_parts} part(s) of up to {human_size(part_size)}, {min(workers, num_parts)} worker(s)")


def _choose_purpose(pick: Callable[..., str]) -> str:
    return pick(constants.PURPOSES, "File purpose", default=constants.DEFAULT_PURPOSE)


def _choose_part_size(text: Callable[..., str]) -> int:
    max_part_mib = constants.MAX_PART_BYTES // constants.MiB
    prompt = _IntPrompt(
        f"Part size in MiB (1-{max_part_mib})",
        constants.DEFAULT_PART_BYTES // constants.MiB,
        1,
        max_part_mib,
    )
    return _prompt_int(prompt, text) * constants.MiB


def _choose_workers(text: Callable[..., str]) -> int:
    prompt = _IntPrompt(
        f"Parallel upload workers (1-{constants.MAX_WORKERS})",
        constants.DEFAULT_WORKERS,
        1,
        constants.MAX_WORKERS,
    )
    return _prompt_int(prompt, text)


def _prompt_int(prompt: _IntPrompt, text: Callable[..., str] = runtime.text) -> int:
    """Ask for a bounded integer, reprompting until the answer is in range."""
    while True:
        value = _parse_bounded_int(text(prompt.question, default=str(prompt.default)), prompt.lo, prompt.hi)
        if value is not None:
            return value
        print(f"  enter an integer between {prompt.lo} and {prompt.hi}")


def _parse_bounded_int(raw: str, lo: int, hi: int) -> int | None:
    """Parse an int within [lo, hi]; None if unparseable or out of range."""
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if lo <= value <= hi else None
