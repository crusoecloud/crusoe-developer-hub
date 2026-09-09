"""
One-Shot Deploy — Main Streamlit UI
Describe a demo → watch the agent build and deploy it live.
Every request calls Crusoe Managed Inference in real time.
"""

import re
import streamlit as st
from config import CRUSOE_API_KEY, CRUSOE_BASE_URL, HF_TOKEN, HF_USERNAME

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="One-Shot Deploy | Crusoe",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Pipeline log rows */
    .stage-row  { padding: 5px 0 5px 4px; font-size: 0.92rem; border-left: 3px solid transparent; margin-bottom: 2px; }
    .stage-done { color: #22c55e; border-left-color: #22c55e; }
    .stage-run  { color: #f59e0b; border-left-color: #f59e0b; }
    .stage-err  { color: #ef4444; border-left-color: #ef4444; }
    .stage-info { color: #94a3b8; border-left-color: #475569; }

    /* Integration badge */
    .int-badge {
        display: flex; align-items: center; gap: 10px;
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;
    }
    .int-dot-green { width:10px; height:10px; border-radius:50%; background:#22c55e; flex-shrink:0; }
    .int-dot-grey  { width:10px; height:10px; border-radius:50%; background:#64748b; flex-shrink:0; }
    .int-label     { font-weight: 600; font-size: 0.88rem; }
    .int-sub       { font-size: 0.78rem; color: #94a3b8; }

    /* Pipeline flow diagram */
    .flow-wrap {
        display: flex; align-items: center; gap: 0;
        background: #0f172a; border: 1px solid #1e293b;
        border-radius: 12px; padding: 16px 20px;
        margin-bottom: 4px; overflow-x: auto;
    }
    .flow-node {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 8px 14px;
        text-align: center; min-width: 110px; flex-shrink: 0;
    }
    .flow-node .fn-label { font-size: 0.72rem; color: #94a3b8; margin-bottom: 2px; }
    .flow-node .fn-value { font-size: 0.85rem; font-weight: 600; }
    .flow-arrow { color: #475569; font-size: 1.3rem; padding: 0 10px; flex-shrink: 0; }
    .flow-node-hf { border-color: #FFD21E44; background: #1a1800; }
    .flow-node-hf .fn-value { color: #FFD21E; }
    .flow-node-crusoe { border-color: #6366f144; background: #0f0f1a; }
    .flow-node-crusoe .fn-value { color: #818cf8; }

    /* HF Space result card */
    .hf-card {
        background: #1a1800; border: 1px solid #FFD21E55;
        border-radius: 12px; padding: 20px 24px;
    }
    .hf-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .hf-title { font-size: 1.1rem; font-weight: 700; }
    .hf-row   { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 0.88rem; color: #cbd5e1; }
    .hf-pill  {
        display: inline-block; background: #FFD21E22; color: #FFD21E;
        border: 1px solid #FFD21E44; border-radius: 999px;
        padding: 1px 10px; font-size: 0.75rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:40] or "demo"


def _space_page_url(space_name: str) -> str:
    return f"https://huggingface.co/spaces/{HF_USERNAME}/{space_name}"


def _config_ok() -> bool:
    missing = []
    if not CRUSOE_API_KEY:
        missing.append("`CRUSOE_API_KEY`")
    if not HF_TOKEN:
        missing.append("`HF_TOKEN`")
    if not HF_USERNAME:
        missing.append("`HF_USERNAME`")
    if missing:
        st.error(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy `env.example` to `.env` and fill in your credentials."
        )
        return False
    return True


# ── Sidebar — Integrations ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Connected integrations")

    crusoe_dot = "int-dot-green" if CRUSOE_API_KEY else "int-dot-grey"
    crusoe_status = "Connected" if CRUSOE_API_KEY else "Missing API key"
    st.markdown(
        f"""
        <div class="int-badge">
            <div class="{crusoe_dot}"></div>
            <div>
                <div class="int-label">Crusoe Managed Inference</div>
                <div class="int-sub">{crusoe_status} · {CRUSOE_BASE_URL.replace("https://","")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hf_dot = "int-dot-green" if (HF_TOKEN and HF_USERNAME) else "int-dot-grey"
    hf_status = f"@{HF_USERNAME}" if HF_USERNAME else "Missing token / username"
    st.markdown(
        f"""
        <div class="int-badge">
            <div class="{hf_dot}"></div>
            <div>
                <div class="int-label">🤗 Hugging Face Spaces</div>
                <div class="int-sub">{hf_status} · streamlit · public</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Models in this pipeline**")
    st.markdown(
        """
| Stage | Model |
|-------|-------|
| Intent | DeepSeek R1 |
| Codegen | Kimi-K2 |
| Healer | Qwen3 |
        """
    )

    st.divider()
    st.caption("Built by Crusoe DevRel")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🚀 One-Shot Deploy")
st.caption(
    "Describe a demo in plain English. "
    "Crusoe's multi-model agent writes the code and ships it to **Hugging Face Spaces** — live in minutes."
)

# Pipeline flow diagram
st.markdown(
    """
    <div class="flow-wrap">
        <div class="flow-node">
            <div class="fn-label">You provide</div>
            <div class="fn-value">💬 Prompt</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node flow-node-crusoe">
            <div class="fn-label">DeepSeek R1</div>
            <div class="fn-value">Intent</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node flow-node-crusoe">
            <div class="fn-label">Kimi-K2</div>
            <div class="fn-value">Codegen</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node flow-node-crusoe">
            <div class="fn-label">Qwen3</div>
            <div class="fn-value">Self-heal</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node flow-node-hf">
            <div class="fn-label">🤗 HF Spaces</div>
            <div class="fn-value">Live URL</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Prompt input ──────────────────────────────────────────────────────────────
examples = [
    "A chatbot that helps users pick the right GPU for their ML workload",
    "Side-by-side comparison of DeepSeek R1 vs Kimi K2 on coding problems",
    "A dashboard that analyses a job description and surfaces key requirements",
    "A wizard that collects ML requirements and recommends the optimal Crusoe instance type",
]

col_input, col_examples = st.columns([3, 2])

with col_input:
    prompt = st.text_area(
        "What demo do you want to build?",
        height=120,
        placeholder="e.g. A chatbot that helps startups write investor updates...",
    )
    deploy_btn = st.button(
        "🚀 Deploy to Hugging Face Spaces",
        type="primary",
        disabled=not prompt,
        use_container_width=True,
    )

with col_examples:
    st.markdown("**Try an example**")
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["example_tip"] = ex

if "example_tip" in st.session_state:
    st.info(f"Paste into the prompt box: *{st.session_state['example_tip']}*")

st.divider()

# ── Pipeline runner ────────────────────────────────────────────────────────────
if deploy_btn and prompt and _config_ok():
    from agent.pipeline import run as run_pipeline

    stage_log_el = st.empty()
    code_expander = st.expander("Generated code", expanded=False)
    result_el = st.empty()

    log_lines: list[str] = []

    ICONS = {
        "parsing":         "⏳",
        "parsed":          "✅",
        "generating":      "⏳",
        "generated":       "✅",
        "validating":      "⏳",
        "validated":       "✅",
        "deploying":       "⏳",
        "deploy_progress": "   ",
        "done":            "🎉",
        "error":           "❌",
    }
    LABELS = {
        "parsing":         "Stage 1 · DeepSeek R1 — Parsing intent",
        "parsed":          "Stage 1 · Intent parsed",
        "generating":      "Stage 2 · Kimi-K2 — Generating code",
        "generated":       "Stage 2 · Code generated",
        "validating":      "Stage 3 · Validating code",
        "validated":       "Stage 3 · Validation complete",
        "deploying":       "Stage 4 · Pushing to 🤗 Hugging Face Spaces",
        "deploy_progress": "Stage 4 ·",
        "done":            "Done",
        "error":           "Error",
    }
    CSS = {
        "parsed": "stage-done", "generated": "stage-done",
        "validated": "stage-done", "done": "stage-done",
        "error": "stage-err",
        "deploy_progress": "stage-info",
    }

    def on_stage(stage: str, msg: str) -> None:
        icon = ICONS.get(stage, "  ")
        label = LABELS.get(stage, stage)
        css = CSS.get(stage, "stage-run")
        line = (
            f'<div class="stage-row {css}">'
            f'{icon}&nbsp; <b>{label}</b>'
            f'{"&nbsp;&nbsp;" + msg if msg else ""}'
            f'</div>'
        )
        log_lines.append(line)
        stage_log_el.markdown("\n".join(log_lines), unsafe_allow_html=True)

    result = run_pipeline(prompt, on_stage=on_stage)

    # Generated code preview
    if result.code:
        with code_expander:
            st.code(result.code, language="python")

    # Outcome
    if result.success:
        title = result.intent.get("title", "demo")
        space_name = f"crusoe-{_slugify(title)}"
        page_url = _space_page_url(space_name)
        app_url = result.url

        # HF Space result card
        result_el.markdown(
            f"""
            <div class="hf-card">
                <div class="hf-card-header">
                    <span style="font-size:1.6rem">🤗</span>
                    <div>
                        <div class="hf-title">Your Space is live on Hugging Face</div>
                        <div style="font-size:0.82rem;color:#94a3b8;">
                            Deployed to <b>huggingface.co/spaces/{HF_USERNAME}/{space_name}</b>
                        </div>
                    </div>
                </div>
                <div class="hf-row">
                    <span>👤</span>
                    <span><b>{HF_USERNAME}</b> / <b>{space_name}</b></span>
                    <span class="hf-pill">streamlit</span>
                    <span class="hf-pill">public</span>
                </div>
                <div class="hf-row">
                    <span>🌐</span>
                    <a href="{app_url}" target="_blank" style="color:#FFD21E;">{app_url}</a>
                </div>
                <div class="hf-row">
                    <span>📄</span>
                    <a href="{page_url}" target="_blank" style="color:#94a3b8;">{page_url}</a>
                </div>
                <div class="hf-row" style="margin-top:6px;">
                    <span>⚡</span>
                    <span style="color:#94a3b8;">Calls <b>Crusoe Managed Inference</b> on every request</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.link_button(
                "🚀 Open live demo",
                app_url,
                use_container_width=True,
                type="primary",
            )
        with col_b:
            st.link_button(
                "🤗 View Space on Hugging Face",
                page_url,
                use_container_width=True,
            )

        st.subheader("Live preview")
        st.caption(f"Embedded from `{app_url}`")
        st.components.v1.iframe(app_url, height=600, scrolling=True)

    else:
        result_el.error(f"**Deployment failed.** {result.error}")


# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("How it works", expanded=False):
    st.markdown(
        """
| Stage | Model | What happens |
|-------|-------|-------------|
| 1 · Intent | **DeepSeek R1** | Reasons about your prompt, selects a template (chatbot / comparison / dashboard / wizard), writes the AI system prompt |
| 2 · Codegen | **Kimi-K2** | Writes a complete single-file Streamlit app wired to Crusoe inference |
| 3 · Validate | *(ast parser)* | Checks syntax and structural patterns — no LLM cost |
| 4 · Deploy | **Qwen3** (healer) | Pushes `app.py` + `requirements.txt` to HF Spaces; injects `CRUSOE_API_KEY` as a Space secret; reads build logs and auto-fixes on failure (up to 3×) |

Every deployed app calls **Crusoe Managed Inference** live, so every visitor experiences Crusoe's speed firsthand.
        """
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ by Crusoe DevRel · Powered by Crusoe Managed Inference · Deployed via 🤗 Hugging Face Spaces")
