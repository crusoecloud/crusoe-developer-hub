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

from .data import load_math500, load_math_train  # noqa: E402
from .device import get_device, peak_memory_gb  # noqa: E402
from .evaluate import COT_PROMPT, MATH_PROMPT, evaluate_math, render_math_prompt  # noqa: E402
from .generate import generate_text, generate_tokens, scale_by_temperature, top_p_filter  # noqa: E402
from .grpo import (  # noqa: E402
    correctness_reward,
    group_advantages,
    grpo_loss,
    grpo_step,
    sample_response,
    save_checkpoint,
    sequence_logprob,
    train_grpo,
)
from .model import KVCache, Qwen3Config, Qwen3Model, load_qwen3  # noqa: E402
from .refine import (  # noqa: E402
    CRITIQUE_PROMPT,
    REFINE_PROMPT,
    render_critique_prompt,
    render_refine_prompt,
    self_refine,
)
from .scoring import avg_logprob, best_of_n, heuristic_score, token_logprobs  # noqa: E402
from .verify import extract_final_answer, grade, is_equivalent, last_boxed, normalize_answer  # noqa: E402
from .voting import majority_vote, self_consistency  # noqa: E402

__all__ = [
    "COT_PROMPT",
    "CRITIQUE_PROMPT",
    "MATH_PROMPT",
    "REFINE_PROMPT",
    "KVCache",
    "Qwen3Config",
    "Qwen3Model",
    "avg_logprob",
    "best_of_n",
    "correctness_reward",
    "evaluate_math",
    "extract_final_answer",
    "generate_text",
    "generate_tokens",
    "get_device",
    "grade",
    "group_advantages",
    "grpo_loss",
    "grpo_step",
    "heuristic_score",
    "is_equivalent",
    "last_boxed",
    "load_math500",
    "load_math_train",
    "load_qwen3",
    "majority_vote",
    "normalize_answer",
    "peak_memory_gb",
    "render_critique_prompt",
    "render_math_prompt",
    "render_refine_prompt",
    "sample_response",
    "save_checkpoint",
    "scale_by_temperature",
    "self_consistency",
    "self_refine",
    "sequence_logprob",
    "token_logprobs",
    "top_p_filter",
    "train_grpo",
]
