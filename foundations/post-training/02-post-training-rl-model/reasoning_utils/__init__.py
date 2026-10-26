"""Thin utilities for the post-training with RL track.

Everything here is either infrastructure (the transformer, the tokenizer,
downloads) or a function that an earlier notebook built step by step and a
later notebook wants to import rather than re-derive.
"""


def _sharper_inline_figures() -> None:
    """Inside Jupyter, render matplotlib figures at 2x pixel density so charts stay crisp."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from matplotlib_inline.backend_inline import set_matplotlib_formats

            set_matplotlib_formats("retina")
    except Exception:
        pass


_sharper_inline_figures()

from .data import load_math500  # noqa: E402
from .device import get_device, peak_memory_gb  # noqa: E402
from .evaluate import MATH_PROMPT, evaluate_math, render_math_prompt  # noqa: E402
from .generate import generate_text, generate_tokens  # noqa: E402
from .model import QWEN3_0_6B, KVCache, Qwen3Config, Qwen3Model, load_qwen3  # noqa: E402
from .tokenizer import Qwen3Tokenizer  # noqa: E402
from .verify import extract_final_answer, grade, is_equivalent, normalize_answer  # noqa: E402

__all__ = [
    "MATH_PROMPT",
    "QWEN3_0_6B",
    "KVCache",
    "Qwen3Config",
    "Qwen3Model",
    "Qwen3Tokenizer",
    "evaluate_math",
    "extract_final_answer",
    "generate_text",
    "generate_tokens",
    "get_device",
    "grade",
    "is_equivalent",
    "load_math500",
    "load_qwen3",
    "normalize_answer",
    "peak_memory_gb",
    "render_math_prompt",
]
