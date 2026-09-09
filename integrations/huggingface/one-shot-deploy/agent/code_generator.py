"""
Stage 2 — Code Generator
Model: Kimi-K2 (long-context, excels at code generation)
Input:  structured intent dict
Output: complete single-file Streamlit app as a string
"""

import re
from openai import OpenAI
from config import CRUSOE_API_KEY, CRUSOE_BASE_URL, CODEGEN_MODEL
from templates import get_template

client = OpenAI(api_key=CRUSOE_API_KEY, base_url=CRUSOE_BASE_URL)

SYSTEM_PROMPT = """You are an expert Streamlit developer creating polished AI demos.

You will be given a base template and a structured intent. Your job is to:
1. Fill in the template placeholders accurately
2. Add any extra UI polish or features mentioned in the intent
3. Keep the code as a single self-contained Python file
4. Ensure the openai client uses env vars for CRUSOE_API_KEY and CRUSOE_BASE_URL

Rules:
- Output ONLY valid Python code
- No markdown fences, no explanations
- All string placeholders ({title}, {description}, etc.) must be replaced
- Do not change the streaming pattern or client setup
- Available Crusoe models: deepseek-ai/DeepSeek-R1, moonshotai/Kimi-K2-Instruct, Qwen/Qwen3-235B-A22B"""


def generate_code(intent: dict) -> str:
    template = get_template(intent.get("template_type", "chatbot"))

    # Build the pre-filled template by substituting known values
    try:
        prefilled = _fill_template(template, intent)
    except KeyError:
        prefilled = template  # fall back to raw template

    user_message = f"""Fill in this Streamlit app template using the intent below.

INTENT:
- Title: {intent.get('title')}
- Description: {intent.get('description')}
- Template type: {intent.get('template_type')}
- Model: {intent.get('model')}
- System prompt: {intent.get('system_prompt')}
- Features: {', '.join(intent.get('features', []))}
- Chat placeholder: {intent.get('chat_placeholder', '')}
- Input label: {intent.get('input_label', '')}
- Input placeholder: {intent.get('input_placeholder', '')}
- Model A: {intent.get('model_a', '')} ({intent.get('model_a_label', '')})
- Model B: {intent.get('model_b', '')} ({intent.get('model_b_label', '')})
- Steps: {intent.get('steps', [])}

TEMPLATE (replace all {{placeholders}} and improve where noted):
{prefilled}"""

    response = client.chat.completions.create(
        model=CODEGEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content or ""
    return _extract_code(raw)


def _fill_template(template: str, intent: dict) -> str:
    """Best-effort template fill — remaining {placeholders} are sent to the LLM."""
    steps_repr = repr(intent.get("steps", []))
    return template.format(
        title=_esc(intent.get("title", "AI Demo")),
        description=_esc(intent.get("description", "Powered by Crusoe")),
        system_prompt=_esc(intent.get("system_prompt", "You are a helpful assistant."), multiline=True),
        model=intent.get("model", "Qwen/Qwen3-235B-A22B"),
        chat_placeholder=_esc(intent.get("chat_placeholder", "Ask me anything...")),
        input_label=_esc(intent.get("input_label", "Enter input:")),
        input_placeholder=_esc(intent.get("input_placeholder", "Type here...")),
        model_a=intent.get("model_a", "deepseek-ai/DeepSeek-R1"),
        model_a_label=_esc(intent.get("model_a_label", "Model A")),
        model_b=intent.get("model_b", "moonshotai/Kimi-K2-Instruct"),
        model_b_label=_esc(intent.get("model_b_label", "Model B")),
        steps=steps_repr,
    )


def _esc(s: str, multiline: bool = False) -> str:
    """
    Sanitise a value for safe embedding in a Python string literal.

    Single-line literals  (e.g. TITLE = "{title}"):
      - escape backslashes, double-quotes, collapse newlines to a space

    Multiline / triple-quoted literals  (e.g. SYSTEM_PROMPT = \"\"\"{system_prompt}\"\"\"):
      - only escape triple-quote sequences
    """
    s = str(s)
    if multiline:
        return s.replace('"""', "'''")
    # Single-line: make the value safe inside "..."
    s = s.replace("\\", "\\\\")   # backslashes first
    s = s.replace('"', '\\"')      # double-quotes
    s = s.replace("\n", " ").replace("\r", "")  # no literal newlines
    return s


def _extract_code(raw: str) -> str:
    """Strip markdown fences if the model wrapped the output."""
    fenced = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return raw.strip()
