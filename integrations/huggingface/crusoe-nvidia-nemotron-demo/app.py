"""
Nemotron Duo: AI Pair Programming Session
==========================================
A Gradio demo showcasing NVIDIA Nemotron models on Crusoe Cloud Foundry.
- Nano (30B-A3B): The "Fast Coder" — rapid implementation & scaffolding
- Super (120B-A12B): The "Senior Reviewer" — code review, optimization, architecture

Built by Crusoe Cloud | Powered by NVIDIA Nemotron 3
"""

import gradio as gr
import time
import json
import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CRUSOE_BASE_URL = os.environ.get("CRUSOE_API_BASE", "https://api.inference.crusoecloud.com/v1/")
CRUSOE_API_KEY = os.environ.get("CRUSOE_API_KEY", "no-key-required")

NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SUPER_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"

client = OpenAI(base_url=CRUSOE_BASE_URL, api_key=CRUSOE_API_KEY)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
NANO_SYSTEM = """You are the "Fast Coder" in a pair programming session. Your role:
- Quickly generate working implementations based on the user's request
- Write clean, functional code with brief inline comments
- Focus on speed and getting a working solution out fast
- Keep explanations concise — let the code speak
- Use modern best practices but prioritize shipping

Format your response as:
1. A brief one-line plan
2. The full code implementation
3. A one-line summary of what you built"""

SUPER_SYSTEM = """You are the "Senior Reviewer" in a pair programming session. You are reviewing code written by a fast-coding junior developer. Your role:
- Critically review the provided code for bugs, security issues, and architectural problems
- Suggest concrete improvements with code examples
- Rewrite key sections if they can be significantly improved
- Consider edge cases, error handling, performance, and maintainability
- Be constructive but thorough — this is a teaching moment

Format your response as:
## Code Review
- List specific issues found (with severity: 🔴 Critical, 🟡 Warning, 🟢 Suggestion)

## Improved Implementation
- Show the rewritten/improved code

## Architecture Notes
- Any high-level design suggestions"""

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def call_model_streaming(model: str, system: str, user_msg: str):
    """Stream responses from a model, yielding (token, metrics) tuples."""
    start = time.time()
    tokens = 0
    full_response = ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            stream=True,
            max_tokens=4096,
            temperature=0.7,
        )

        first_token_time = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                tokens += 1
                if first_token_time is None:
                    first_token_time = time.time()
                full_response += token
                elapsed = time.time() - start
                ttft = (first_token_time - start) if first_token_time else 0
                tps = tokens / elapsed if elapsed > 0 else 0
                yield full_response, {
                    "tokens": tokens,
                    "elapsed": round(elapsed, 2),
                    "ttft": round(ttft, 3),
                    "tps": round(tps, 1),
                }

    except Exception as e:
        yield f"**Error:** {str(e)}", {
            "tokens": 0, "elapsed": 0, "ttft": 0, "tps": 0
        }


def format_metrics(label: str, metrics: dict) -> str:
    """Format metrics as a readable string."""
    if not metrics or metrics.get("tokens", 0) == 0:
        return f"**{label}** — waiting..."
    return (
        f"**{label}**  |  "
        f"⏱ TTFT: {metrics['ttft']}s  |  "
        f"🚀 {metrics['tps']} tok/s  |  "
        f"📊 {metrics['tokens']} tokens  |  "
        f"⏳ {metrics['elapsed']}s total"
    )


# ---------------------------------------------------------------------------
# Main pair programming flow
# ---------------------------------------------------------------------------
def pair_program(user_request: str, difficulty: float):
    """
    Run the pair programming session:
    1. Nano generates code fast
    2. Super reviews and improves it
    Streams both outputs in real-time.
    """
    if not user_request.strip():
        yield "", "", "", "", ""
        return

    # --- Phase 1: Nano codes ---
    nano_output = ""
    nano_metrics = {}
    super_output = "*⏳ Waiting for Fast Coder to finish...*"
    super_metrics = {}

    for text, metrics in call_model_streaming(NANO_MODEL, NANO_SYSTEM, user_request):
        nano_output = text
        nano_metrics = metrics
        yield (
            nano_output,
            super_output,
            format_metrics("⚡ Nano (Fast Coder)", nano_metrics),
            format_metrics("🧠 Super (Senior Reviewer)", super_metrics),
            format_cost_tracker(nano_metrics, super_metrics),
        )

    # --- Phase 2: Super reviews ---
    # Adjust review depth based on difficulty slider
    review_depth = {
        "light": "Do a quick review — focus only on critical bugs and one key improvement.",
        "medium": "Do a standard code review — cover bugs, security, and top 3 improvements.",
        "thorough": "Do a deep, thorough review — cover bugs, security, architecture, performance, edge cases, and provide a fully rewritten version.",
    }

    if difficulty < 0.33:
        depth = "light"
    elif difficulty < 0.66:
        depth = "medium"
    else:
        depth = "thorough"

    review_prompt = f"""{review_depth[depth]}

## User's Original Request:
{user_request}

## Fast Coder's Implementation:
{nano_output}"""

    super_output = ""
    for text, metrics in call_model_streaming(SUPER_MODEL, SUPER_SYSTEM, review_prompt):
        super_output = text
        super_metrics = metrics
        yield (
            nano_output,
            super_output,
            format_metrics("⚡ Nano (Fast Coder)", nano_metrics),
            format_metrics("🧠 Super (Senior Reviewer)", super_metrics),
            format_cost_tracker(nano_metrics, super_metrics),
        )


def format_cost_tracker(nano_m: dict, super_m: dict) -> str:
    """Estimate cost savings vs using only the large model."""
    nano_tokens = nano_m.get("tokens", 0)
    super_tokens = super_m.get("tokens", 0)
    total_tokens = nano_tokens + super_tokens

    if total_tokens == 0:
        return "**Cost Tracker:** Waiting for inference..."

    # Rough illustrative cost ratios (Nano ~4x cheaper per token than Super)
    nano_cost = nano_tokens * 0.0001
    super_cost = super_tokens * 0.0004
    duo_cost = nano_cost + super_cost
    all_super_cost = total_tokens * 0.0004

    savings = ((all_super_cost - duo_cost) / all_super_cost * 100) if all_super_cost > 0 else 0

    return (
        f"**💰 Cost Tracker**  |  "
        f"Duo approach: ~${duo_cost:.4f}  |  "
        f"All-Super baseline: ~${all_super_cost:.4f}  |  "
        f"**Savings: ~{savings:.0f}%**"
    )


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["Build a REST API with FastAPI that manages a todo list with SQLite persistence"],
    ["Create a Python CLI tool that monitors GPU utilization and alerts when usage drops below a threshold"],
    ["Implement a concurrent web scraper with rate limiting and retry logic"],
    ["Write a React component for a real-time collaborative text editor using WebSockets"],
    ["Build a Kubernetes health check dashboard in Python with Flask"],
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.header-section {
    text-align: center;
    padding: 20px 0 10px 0;
}
.header-section h1 {
    font-size: 2.2em;
    margin-bottom: 5px;
}
.header-section p {
    font-size: 1.1em;
    opacity: 0.8;
}
.nano-panel {
    border-left: 4px solid #00d4aa !important;
}
.super-panel {
    border-left: 4px solid #7c3aed !important;
}
.metrics-bar {
    font-family: monospace;
    font-size: 0.9em;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="Nemotron Duo — AI Pair Programming",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="emerald",
        secondary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # --- Header ---
    gr.HTML("""
    <div class="header-section">
        <h1>🤖 Nemotron Duo</h1>
        <p>AI Pair Programming — Fast Coder meets Senior Reviewer</p>
        <p style="font-size: 0.85em; opacity: 0.6;">
            Powered by NVIDIA Nemotron 3 on Crusoe Cloud Foundry
        </p>
    </div>
    """)

    # --- Input section ---
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                label="💬 Describe what you want to build",
                placeholder="e.g., Build a FastAPI server with WebSocket support for real-time chat...",
                lines=3,
                max_lines=6,
            )
        with gr.Column(scale=1, min_width=200):
            difficulty = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01,
                label="🎚️ Review Depth",
                info="Light → Thorough",
            )
            run_btn = gr.Button("🚀 Start Session", variant="primary", size="lg")

    # --- Metrics bar ---
    with gr.Row():
        nano_metrics = gr.Markdown("**⚡ Nano (Fast Coder)** — ready", elem_classes=["metrics-bar"])
        super_metrics = gr.Markdown("**🧠 Super (Senior Reviewer)** — ready", elem_classes=["metrics-bar"])

    # --- Cost tracker ---
    cost_tracker = gr.Markdown("**💰 Cost Tracker:** Ready to estimate savings...")

    # --- Split panel outputs ---
    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["nano-panel"]):
            gr.HTML("""
                <div style="padding: 8px 12px; background: linear-gradient(90deg, #00d4aa22, transparent); border-radius: 6px; margin-bottom: 8px;">
                    <strong>⚡ Fast Coder</strong> — Nemotron Nano 30B (3B active)
                </div>
            """)
            nano_output = gr.Markdown(
                value="*Waiting for your prompt...*",
                height=600,
                show_copy_button=True,
            )

        with gr.Column(elem_classes=["super-panel"]):
            gr.HTML("""
                <div style="padding: 8px 12px; background: linear-gradient(90deg, #7c3aed22, transparent); border-radius: 6px; margin-bottom: 8px;">
                    <strong>🧠 Senior Reviewer</strong> — Nemotron Super 120B (12B active)
                </div>
            """)
            super_output = gr.Markdown(
                value="*Waiting for Fast Coder to generate code...*",
                height=600,
                show_copy_button=True,
            )

    # --- Examples ---
    gr.Examples(
        examples=EXAMPLES,
        inputs=user_input,
        label="💡 Try these prompts",
    )

    # --- How it works ---
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown("""
        **Nemotron Duo** demonstrates a practical multi-model architecture running on Crusoe Cloud:

        1. **⚡ Nano (30B, 3B active params)** receives your request and rapidly generates a working implementation.
           It's optimized for speed — like a fast-typing junior dev who gets things done quickly.

        2. **🧠 Super (120B, 12B active params)** then reviews Nano's code like a senior engineer — finding bugs,
           suggesting improvements, and rewriting key sections.

        **Why this matters:** Instead of sending every request to the largest model, you use the right model for the
        right task. Nano handles the heavy lifting at ~4x lower cost, and Super provides the quality assurance.
        The result: faster, cheaper, and often better than using a single model alone.

        **Infrastructure:** Both models run on Crusoe Cloud's Foundry platform with NVIDIA GPU acceleration,
        giving you low-latency inference with a simple OpenAI-compatible API.
        """)

    # --- Footer ---
    gr.HTML("""
    <div style="text-align: center; padding: 20px; opacity: 0.5; font-size: 0.85em;">
        Built with ❤️ by Crusoe Cloud · NVIDIA Nemotron 3 · Gradio
    </div>
    """)

    # --- Event handlers ---
    run_btn.click(
        fn=pair_program,
        inputs=[user_input, difficulty],
        outputs=[nano_output, super_output, nano_metrics, super_metrics, cost_tracker],
    )

    user_input.submit(
        fn=pair_program,
        inputs=[user_input, difficulty],
        outputs=[nano_output, super_output, nano_metrics, super_metrics, cost_tracker],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
