"""
Stage 1 — Intent Parser
Model: DeepSeek R1 (reasoning model, ideal for structured analysis)
Input:  raw user prompt
Output: structured intent dict used by the code generator
"""

import json
import re
from openai import OpenAI
from config import CRUSOE_API_KEY, CRUSOE_BASE_URL, INTENT_MODEL

client = OpenAI(api_key=CRUSOE_API_KEY, base_url=CRUSOE_BASE_URL)

SYSTEM_PROMPT = """You are an expert at analysing AI demo requests and mapping them to the right app template.

Given a user's demo idea, output a single JSON object with these fields:

REQUIRED (all templates):
- template_type: one of "chatbot" | "comparison" | "dashboard" | "form_wizard"
- title: short, catchy demo title (max 8 words)
- description: one-line description shown below the title
- system_prompt: detailed system prompt for the AI assistant inside the demo
- model: best Crusoe model for this demo:
    • "deepseek-ai/DeepSeek-R1"        — reasoning, analysis, step-by-step thinking
    • "moonshotai/Kimi-K2-Instruct"   — coding, long documents, structured output
    • "Qwen/Qwen3-235B-A22B"           — general purpose, fast responses, multilingual
- features: list of 2-4 key features to highlight

CONDITIONAL (chatbot only):
- chat_placeholder: placeholder text shown in the chat input box

CONDITIONAL (comparison only):
- model_a: first model ID (from the list above)
- model_a_label: friendly display name for model A
- model_b: second model ID
- model_b_label: friendly display name for model B

CONDITIONAL (dashboard only):
- input_label: label for the main input text area
- input_placeholder: placeholder text for that input

CONDITIONAL (form_wizard only):
- steps: list of 3-5 objects, each {"key": "snake_case_name", "question": "Question text?"}

Template selection guide:
- chatbot:     conversational Q&A, support bots, advisors
- comparison:  show two models side-by-side on the same prompt
- dashboard:   analyze / summarize pasted text, data, or documents
- form_wizard: multi-step intake flows that end with AI recommendations

Output ONLY the JSON object. No markdown, no explanations."""


def parse_intent(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=INTENT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a demo for: {prompt}"},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    content = response.choices[0].message.content or ""

    # Strip DeepSeek R1 chain-of-thought tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Extract the JSON block if surrounded by markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    else:
        # Grab the outermost { ... }
        brace = re.search(r"\{.*\}", content, re.DOTALL)
        if brace:
            content = brace.group()

    intent = json.loads(content)
    return _apply_defaults(intent)


def _apply_defaults(intent: dict) -> dict:
    intent.setdefault("template_type", "chatbot")
    intent.setdefault("title", "AI Demo")
    intent.setdefault("description", "Powered by Crusoe Managed Inference")
    intent.setdefault("system_prompt", "You are a helpful AI assistant.")
    intent.setdefault("model", "Qwen/Qwen3-235B-A22B")
    intent.setdefault("features", [])
    # Chatbot defaults
    intent.setdefault("chat_placeholder", "Ask me anything...")
    # Comparison defaults
    intent.setdefault("model_a", "deepseek-ai/DeepSeek-R1")
    intent.setdefault("model_a_label", "DeepSeek R1")
    intent.setdefault("model_b", "moonshotai/Kimi-K2-Instruct")
    intent.setdefault("model_b_label", "Kimi K2")
    # Dashboard defaults
    intent.setdefault("input_label", "Paste your content here:")
    intent.setdefault("input_placeholder", "Enter text to analyze...")
    # Form wizard defaults
    intent.setdefault("steps", [
        {"key": "requirement", "question": "What is your main requirement?"},
        {"key": "context", "question": "Can you describe your use case in more detail?"},
        {"key": "constraints", "question": "Are there any constraints or preferences to keep in mind?"},
    ])
    return intent
