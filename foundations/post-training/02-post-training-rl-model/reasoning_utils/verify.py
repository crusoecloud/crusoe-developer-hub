"""A math verifier: pull the final answer out of free-form text and check it against a reference.

The math verifier notebook walks through this pipeline one stage at a time.
Later notebooks import it: a verifier that says "correct" or "incorrect" is
exactly the reward signal reinforcement learning needs.

    text  --extract-->  candidate  --normalize-->  canonical string  --compare-->  True / False
"""

from __future__ import annotations

import re

from sympy import simplify
from sympy.parsing import sympy_parser as spp

# ---------------------------------------------------------------- extraction


def last_boxed(text: str) -> str | None:
    r"""Return the content of the last \boxed{...} in `text`, or None.

    Braces may nest (\boxed{\frac{1}{2}}), so this walks the string and
    counts depth instead of using a regex.
    """
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    i = start + len(r"\boxed")
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    i += 1
    begin, depth = i, 1
    while i < len(text) and depth > 0:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[begin : i - 1] if depth == 0 else None


NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def extract_final_answer(text: str) -> str:
    """Prefer the last boxed expression. Else the last number in the text. Else the text itself."""
    if not text:
        return ""
    boxed = last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    numbers = NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()


# ------------------------------------------------------------- normalization

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾", "0123456789+-()")
SUPERSCRIPT_RUN = r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+"

LATEX_REWRITES = [
    (r"\\left\s*|\\right\s*", ""),  # \left( ... \right) are sizing hints only
    (r"\\[,!;:]", ""),  # thin spaces
    (r"\\cdot|·|×", "*"),
    (r"\\[dt]frac", r"\\frac"),  # display and text fractions are plain fractions
    (r"\^\s*\{?\s*\\circ\s*\}?|°", ""),  # degrees
]


FRAC_BRACED = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
FRAC_BARE = re.compile(r"\\frac\s+([^\s{}\\]+)\s+([^\s{}\\]+)")
SQRT_BRACED = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
SQRT_BARE = re.compile(r"\\sqrt\s+([^\\\s{}]+)")


def _rewrite_fractions_and_roots(text: str) -> str:
    r"""\frac{a}{b} -> (a)/(b) and \sqrt{x} -> sqrt(x), innermost first.

    The patterns only match brace groups with no braces inside, so each pass
    rewrites the innermost fractions and roots and turns their braces into
    parentheses. Repeating until nothing changes resolves any nesting depth:
    \frac{\frac{1}{2}}{3} becomes ((1)/(2))/(3) in two passes.
    """
    while True:
        new = FRAC_BRACED.sub(r"(\1)/(\2)", text)
        new = SQRT_BRACED.sub(r"sqrt(\1)", new)
        new = FRAC_BARE.sub(r"(\1)/(\2)", new)
        new = SQRT_BARE.sub(r"sqrt(\1)", new)
        if new == text:
            return new
        text = new


def normalize_answer(text: str) -> str:
    """Rewrite the many ways of printing a value into one plain, lower-case form that SymPy can parse."""
    if not text:
        return ""
    text = re.sub(r"<\|[^>]+?\|>", "", text).strip()  # chat special tokens
    text = text.replace("\u2212", "-")  # unicode minus looks identical to a hyphen

    if m := re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text):  # multiple-choice label: "c. 3" -> "3"
        text = m.group(1)
    if m := re.match(r"^\\text\{(.+?)\}$", text):  # \text{...} wrapping the whole answer
        text = m.group(1)
    text = re.sub(r"\\[()\[\]]", "", text)  # \( \) \[ \] math delimiters

    for pattern, replacement in LATEX_REWRITES:
        text = re.sub(pattern, replacement, text)

    # 2² -> 2**2, then any superscript left standing alone
    text = re.sub(
        rf"([0-9A-Za-z)\]}}])({SUPERSCRIPT_RUN})",
        lambda m: f"{m.group(1)}**{m.group(2).translate(SUPERSCRIPTS)}",
        text,
    )
    text = text.translate(SUPERSCRIPTS)

    text = text.replace("\\%", "").replace("%", "").replace("$", "")
    text = _rewrite_fractions_and_roots(text)
    text = text.replace("^", "**")
    text = re.sub(r"(?<=\d)\s+(\d+/\d+)", r"+\1", text)  # mixed number: "1 1/2" -> "1+1/2"
    text = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)  # thousands separator: "1,234" -> "1234"
    return text.replace("{", "").replace("}", "").strip().lower()


# ---------------------------------------------------------------- comparison


def parse_math(expr: str):
    """Parse a normalized string into a SymPy expression, or None if SymPy cannot make sense of it."""
    if not expr or len(expr) > 2000:  # a runaway generation is not an answer
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(*spp.standard_transformations, spp.implicit_multiplication_application),
            evaluate=True,
        )
    except Exception:  # SymPy raises many different error types on malformed input
        return None


def is_equivalent(a: str, b: str) -> bool:
    """Same string, or two expressions whose difference simplifies to zero."""
    if a == b:
        return True
    pa, pb = parse_math(a), parse_math(b)
    if pa is None or pb is None:
        return False
    try:
        return simplify(pa - pb) == 0
    except Exception:
        return False


def split_parts(text: str) -> list[str]:
    """'(a, b)' or '[a, b]' -> ['a', 'b']; anything else -> [text]; empty -> []."""
    if not text:
        return []
    if len(text) >= 2 and text[0] in "([" and text[-1] in ")]" and "," in text[1:-1]:
        parts = [p.strip() for p in text[1:-1].split(",")]
        if all(parts):
            return parts
    return [text]


def grade(prediction: str | None, truth: str | None) -> bool:
    """True when the prediction matches the reference answer, part by part."""
    if prediction is None or truth is None:
        return False
    pred_parts = split_parts(normalize_answer(prediction))
    truth_parts = split_parts(normalize_answer(truth))
    if not pred_parts or len(pred_parts) != len(truth_parts):
        return False
    return all(is_equivalent(t, p) for t, p in zip(truth_parts, pred_parts))
