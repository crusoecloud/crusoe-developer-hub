"""
Stage 3 — Validator
No LLM needed — uses Python's ast module for a fast syntax check.
Returns (is_valid: bool, error_message: str).
"""

import ast


def validate_code(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"

    issues = _lint(code)
    if issues:
        return False, "; ".join(issues)

    return True, ""


def _lint(code: str) -> list[str]:
    """Lightweight structural checks."""
    issues = []

    if "import streamlit" not in code:
        issues.append("missing 'import streamlit'")
    if "OpenAI(" not in code:
        issues.append("missing OpenAI client instantiation")
    if "CRUSOE_API_KEY" not in code:
        issues.append("missing CRUSOE_API_KEY env var reference")
    if "stream=True" not in code:
        issues.append("missing streaming (stream=True)")

    return issues
