"""
Main orchestration pipeline.
Runs the four stages sequentially and surfaces progress via callbacks.
"""

from dataclasses import dataclass, field
from typing import Callable

from agent.intent_parser import parse_intent
from agent.code_generator import generate_code
from agent.validator import validate_code
from agent.deployer import deploy, _heal as heal_code


@dataclass
class PipelineResult:
    intent: dict = field(default_factory=dict)
    code: str = ""
    url: str = ""
    error: str = ""
    success: bool = False


def run(
    prompt: str,
    on_stage: Callable[[str, str], None] | None = None,
) -> PipelineResult:
    """
    Execute the full pipeline.

    on_stage(stage_key, message) is called at each significant step:
      stage_key in {"parsing", "parsed", "generating", "generated",
                    "validating", "validated", "deploying", "deploy_progress",
                    "done", "error"}
    """
    result = PipelineResult()

    def _notify(stage: str, msg: str = "") -> None:
        if on_stage:
            on_stage(stage, msg)

    # ── Stage 1: Intent Parsing ──────────────────────────────────────────────
    _notify("parsing", f"DeepSeek R1 is analysing your prompt...")
    try:
        result.intent = parse_intent(prompt)
        _notify(
            "parsed",
            f"Template: **{result.intent['template_type']}** · "
            f"Title: **{result.intent['title']}** · "
            f"Model: `{result.intent['model']}`",
        )
    except Exception as exc:
        result.error = f"Intent parsing failed: {exc}"
        _notify("error", result.error)
        return result

    # ── Stage 2: Code Generation ─────────────────────────────────────────────
    _notify("generating", "Kimi-K2 is writing your Streamlit app...")
    try:
        result.code = generate_code(result.intent)
        _notify("generated", f"Generated {len(result.code.splitlines())} lines of code.")
    except Exception as exc:
        result.error = f"Code generation failed: {exc}"
        _notify("error", result.error)
        return result

    # ── Stage 3: Validation + pre-flight healing ─────────────────────────────
    _notify("validating", "Validating generated code...")
    valid, validation_error = validate_code(result.code)

    if not valid:
        # Syntax error — fix it with Qwen3 before attempting deployment
        for fix_attempt in range(1, 4):
            _notify(
                "validating",
                f"Syntax error detected ({validation_error}). "
                f"Qwen3 fixing (attempt {fix_attempt}/3)...",
            )
            try:
                result.code = heal_code(result.code, validation_error)
                valid, validation_error = validate_code(result.code)
                if valid:
                    break
            except Exception:
                pass

        if not valid:
            result.error = f"Code could not be fixed after 3 attempts: {validation_error}"
            _notify("error", result.error)
            return result

    _notify("validated", f"Code valid — {len(result.code.splitlines())} lines ready to deploy.")

    # ── Stage 4: Deploy + Self-Heal ──────────────────────────────────────────
    _notify("deploying", "Deploying to Hugging Face Spaces...")
    try:
        result.url = deploy(
            result.code,
            result.intent,
            on_progress=lambda msg: _notify("deploy_progress", msg),
        )
        result.success = True
        _notify("done", result.url)
    except Exception as exc:
        result.error = f"Deployment failed: {exc}"
        _notify("error", result.error)

    return result
