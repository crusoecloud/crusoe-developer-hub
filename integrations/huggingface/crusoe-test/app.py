import os
import json
import re
import html
from datetime import datetime
import streamlit as st
from openai import OpenAI
from newsapi import NewsApiClient

st.set_page_config(
    page_title="Crusoe AI Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CRUSOE_API_KEY = os.environ.get("CRUSOE_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1/"

# All text models available on Crusoe Managed Inference
MODELS = {
    "Qwen3 235B": {
        "id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "provider": "Qwen",
        "context": "131k",
        "note": "Flagship — best overall quality",
        "thinking": False,
    },
    "DeepSeek R1-0528": {
        "id": "deepseek-ai/DeepSeek-R1-0528",
        "provider": "DeepSeek",
        "context": "160k",
        "note": "Thinking model — slower, deeper",
        "thinking": True,
    },
    "Llama 3.3 70B": {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "provider": "Meta",
        "context": "128k",
        "note": "Fast & efficient",
        "thinking": False,
    },
    "GPT-OSS 120B": {
        "id": "openai/gpt-oss-120b",
        "provider": "OpenAI",
        "context": "128k",
        "note": "OpenAI open weights model",
        "thinking": False,
    },
    "Gemma 3 12B": {
        "id": "google/gemma-3-12b-it",
        "provider": "Google",
        "context": "128k",
        "note": "Lightweight & fast",
        "thinking": False,
    },
    "Kimi-K2 Thinking": {
        "id": "moonshotai/Kimi-K2-Thinking",
        "provider": "Moonshot AI",
        "context": "131k",
        "note": "Thinking model — complex analysis",
        "thinking": True,
    },
}

# ── Sidebar model selector ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 20px 0;">
    <div style="font-size:1rem; font-weight:700; color:#1e293b; margin-bottom:4px;">⚡ Crusoe Models</div>
    <div style="font-size:0.75rem; color:#94a3b8;">All hosted on Crusoe Managed AI</div>
</div>
""", unsafe_allow_html=True)

    model_label = st.selectbox(
        "Select model",
        options=list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected = MODELS[model_label]

    _provider = html.escape(selected["provider"])
    _note = html.escape(selected["note"])
    _ctx = html.escape(selected["context"])
    _model_id = html.escape(selected["id"])
    _thinking = "&nbsp;&nbsp;·&nbsp;&nbsp;🧠 Thinking" if selected["thinking"] else ""

    st.markdown(f"""
<div style="background:rgba(37,99,235,0.05); border:1px solid rgba(37,99,235,0.12);
     border-radius:12px; padding:14px 16px; margin-top:8px;">
    <div style="font-size:0.7rem; font-weight:600; color:#94a3b8; text-transform:uppercase;
         letter-spacing:1.5px; margin-bottom:8px;">Model Details</div>
    <div style="font-size:0.82rem; color:#1e293b; font-weight:600; margin-bottom:4px;">
        {_provider}
    </div>
    <div style="font-size:0.78rem; color:#64748b; margin-bottom:6px;">{_note}</div>
    <div style="font-size:0.72rem; color:#2563eb; font-weight:600;">
        Context: {_ctx}{_thinking}
    </div>
    <div style="font-size:0.65rem; color:#94a3b8; margin-top:8px; word-break:break-all;">
        {_model_id}
    </div>
</div>
""", unsafe_allow_html=True)

MODEL = selected["id"]
IS_THINKING = selected["thinking"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(145deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container { padding-top: 2rem; max-width: 1200px; }

.main-header { text-align: center; padding: 48px 0 32px; }

.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 50%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shine 4s linear infinite;
    letter-spacing: -0.5px;
}

@keyframes shine { to { background-position: 200% center; } }

.main-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 10px;
    letter-spacing: 0.3px;
}

.powered-by {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #2563eb;
    margin-top: 14px;
    letter-spacing: 0.5px;
}

.glass-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 28px 32px;
    margin: 10px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
}

.glass-card-accent {
    background: rgba(239,246,255,0.85);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(37,99,235,0.12);
    border-radius: 20px;
    padding: 28px 32px;
    margin: 10px 0;
    box-shadow: 0 4px 24px rgba(37,99,235,0.07), 0 1px 4px rgba(0,0,0,0.03);
}

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #94a3b8;
    margin-bottom: 14px;
}

.metric-num {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}

.metric-sublabel {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
}

.metric-desc {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 12px;
    line-height: 1.6;
}

.exec-text {
    color: #1e293b;
    font-size: 1.05rem;
    line-height: 1.8;
    margin: 0;
}

.theme-tag {
    display: inline-block;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.18);
    color: #7c3aed;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 3px;
    font-weight: 500;
}

.list-item {
    color: #334155;
    font-size: 0.88rem;
    padding: 10px 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    line-height: 1.5;
}

.list-item:last-child { border-bottom: none; }

.bar-container { margin-bottom: 18px; }

.bar-row {
    display: flex;
    justify-content: space-between;
    color: #475569;
    font-size: 0.84rem;
    margin-bottom: 7px;
    font-weight: 500;
}

.bar-track {
    height: 7px;
    border-radius: 4px;
    background: rgba(0,0,0,0.06);
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 4px;
}

.article-card {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.article-title {
    color: #1e293b;
    font-size: 0.92rem;
    font-weight: 600;
    text-decoration: none;
    line-height: 1.4;
}

.article-title:hover { color: #2563eb; }

.article-source-badge {
    display: inline-block;
    background: rgba(37,99,235,0.07);
    border: 1px solid rgba(37,99,235,0.14);
    color: #2563eb;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 10px;
    margin-right: 8px;
}

.article-timestamp {
    display: inline-block;
    color: #94a3b8;
    font-size: 0.75rem;
}

.article-desc {
    color: #64748b;
    font-size: 0.83rem;
    margin-top: 8px;
    line-height: 1.55;
}

div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    box-shadow: 0 4px 20px rgba(37,99,235,0.25);
}

div.stButton > button:hover { opacity: 0.88; }

.articles-header {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #94a3b8;
    margin: 28px 0 16px 0;
}

.competitor-card {
    background: rgba(255,255,255,0.68);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 18px;
    padding: 22px 24px;
    margin: 10px 0;
    box-shadow: 0 3px 16px rgba(0,0,0,0.05);
    height: 100%;
}

.competitor-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 14px;
}

.threat-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
    float: right;
}

.threat-high {
    background: rgba(239,68,68,0.1);
    color: #dc2626;
    border: 1px solid rgba(239,68,68,0.2);
}

.threat-medium {
    background: rgba(245,158,11,0.1);
    color: #d97706;
    border: 1px solid rgba(245,158,11,0.2);
}

.threat-low {
    background: rgba(16,185,129,0.1);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.2);
}

.competitor-narrative {
    color: #475569;
    font-size: 0.83rem;
    line-height: 1.55;
    margin: 10px 0 14px 0;
}

.comp-adv-item {
    color: #065f46;
    background: rgba(16,185,129,0.07);
    border-left: 3px solid #10b981;
    font-size: 0.85rem;
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 0 8px 8px 0;
    line-height: 1.5;
}

.comp-gap-item {
    color: #7c2d12;
    background: rgba(239,68,68,0.06);
    border-left: 3px solid #ef4444;
    font-size: 0.85rem;
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 0 8px 8px 0;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <div class="main-title">Crusoe AI Market Intelligence</div>
    <div class="main-subtitle">Real-time brand sentiment &amp; perception analysis</div>
    <div><span class="powered-by">⚡ Crusoe Managed AI · {model_label}</span></div>
</div>
""", unsafe_allow_html=True)

# ── Run Analysis button ───────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2.5, 1, 2.5])
with btn_col:
    run = st.button("Run Analysis", use_container_width=True)

if not run:
    st.stop()

# ── Error display helper ───────────────────────────────────────────────────────
def show_error(title: str, body: str, hint: str = "") -> None:
    hint_block = (
        f'<div style="margin-top:16px; padding:12px 16px; background:rgba(239,68,68,0.06);'
        f' border-radius:10px; font-size:0.78rem; color:#991b1b; font-family:monospace;'
        f' word-break:break-word; text-align:left;">{html.escape(hint)}</div>'
    ) if hint else ""
    st.markdown(f"""
<div style="background:rgba(254,242,242,0.92); border:1px solid rgba(239,68,68,0.22);
     border-radius:20px; padding:40px; margin:32px 0; text-align:center;
     box-shadow:0 4px 24px rgba(239,68,68,0.08);">
    <div style="font-size:2.2rem; margin-bottom:14px;">⚠️</div>
    <div style="font-size:1.15rem; font-weight:700; color:#dc2626; margin-bottom:10px;">{title}</div>
    <div style="font-size:0.92rem; color:#7f1d1d; line-height:1.7; max-width:560px; margin:0 auto;">
        {body}
    </div>
    {hint_block}
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Validation ────────────────────────────────────────────────────────────────
missing = [k for k, v in {"CRUSOE_API_KEY": CRUSOE_API_KEY, "NEWSAPI_KEY": NEWSAPI_KEY}.items() if not v]
if missing:
    show_error(
        "Configuration Incomplete",
        f"The following environment variables are not set: <strong>{', '.join(missing)}</strong>."
        " Please add them in the Hugging Face Space settings under <em>Settings → Variables and secrets</em>.",
    )

# ── Fetch news ────────────────────────────────────────────────────────────────
COMPETITORS = ["CoreWeave", "Lambda Labs", "Together AI", "Voltage Park", "Fluidstack", "Nebius"]

try:
    with st.spinner("Fetching latest news coverage…"):
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        results = newsapi.get_everything(
            q="Crusoe AI",
            language="en",
            sort_by="publishedAt",
            page_size=20,
        )
        comp_query = " OR ".join(f'"{c}"' for c in COMPETITORS)
        comp_results = newsapi.get_everything(
            q=comp_query,
            language="en",
            sort_by="publishedAt",
            page_size=15,
        )
except Exception as e:
    show_error(
        "News Feed Unavailable",
        "We were unable to retrieve news articles at this time. "
        "This is typically a temporary issue with the news data provider. Please try again in a moment.",
        hint=str(e),
    )

articles = results.get("articles", [])
comp_articles = comp_results.get("articles", [])

if not articles:
    show_error(
        "No Coverage Found",
        "No recent news articles mentioning Crusoe AI were found. "
        "The news feed may be rate-limited or temporarily unavailable — please try again shortly.",
    )

# ── Analyse with Qwen ─────────────────────────────────────────────────────────
def fmt_articles(arts: list, label: str) -> str:
    if not arts:
        return f"[No recent {label} articles found]"
    return "\n\n".join([
        f"Title: {a.get('title','')}\nSource: {a.get('source',{}).get('name','')}\n"
        f"Date: {a.get('publishedAt','')[:10]}\nDescription: {a.get('description','')}"
        for a in arts
    ])

crusoe_text = fmt_articles(articles, "Crusoe AI")
comp_text = fmt_articles(comp_articles, "competitor")
competitors_list = ", ".join(COMPETITORS)

PROMPT = f"""You are a senior market intelligence analyst briefing a C-suite audience.
Analyse the news articles below and return ONLY valid JSON — no markdown, no code fences.

=== CRUSOE AI ARTICLES ===
{crusoe_text}

=== COMPETITOR ARTICLES (covering: {competitors_list}) ===
{comp_text}

Return this exact JSON shape:
{{
  "executive_summary": "2-3 sentence strategic narrative for a CEO or board",
  "overall_sentiment": {{
    "score": <float -1.0 to 1.0>,
    "label": "<Very Positive|Positive|Neutral|Negative|Very Negative>",
    "summary": "One concise sentence"
  }},
  "developer_sentiment": {{
    "score": <float -1.0 to 1.0>,
    "label": "<Very Positive|Positive|Neutral|Negative|Very Negative>",
    "summary": "How the technical/developer community perceives Crusoe"
  }},
  "brand_perception": {{
    "data_center": <int 0-100, share of coverage framing Crusoe as a data center company>,
    "cloud_infrastructure": <int 0-100, share framing Crusoe as a cloud provider>,
    "managed_ai": <int 0-100, share framing Crusoe as a managed AI / inference platform>,
    "summary": "1-2 sentences on identity perception and any awareness shift"
  }},
  "key_themes": ["theme1","theme2","theme3","theme4","theme5"],
  "opportunities": ["opportunity1","opportunity2","opportunity3"],
  "risks": ["risk1","risk2","risk3"],
  "competitive_analysis": {{
    "market_summary": "2-3 sentence overview of the neocloud competitive landscape based on the articles",
    "crusoe_vs_field": "One sentence on how Crusoe positions relative to the competition",
    "competitors": [
      {{
        "name": "<competitor name>",
        "sentiment_score": <float -1.0 to 1.0 based on their press coverage>,
        "sentiment_label": "<Very Positive|Positive|Neutral|Negative|Very Negative>",
        "coverage_level": "<High|Medium|Low — relative volume of coverage>",
        "threat_level": "<High|Medium|Low — competitive threat to Crusoe>",
        "key_narrative": "One sentence on what this competitor is known for or doing right now"
      }}
    ],
    "crusoe_advantages": ["advantage vs competitors 1","advantage 2","advantage 3"],
    "crusoe_gaps": ["gap or area to close 1","gap 2"]
  }}
}}

Include an entry for each of these competitors: {competitors_list}.
If there are no articles for a competitor, infer from general market knowledge.
Return ONLY valid JSON."""

spinner_msg = f"Analysing with {model_label}…{' (thinking mode — this may take a minute)' if IS_THINKING else ' this may take 20–30 seconds'}"
try:
    with st.spinner(spinner_msg):
        client = OpenAI(api_key=CRUSOE_API_KEY, base_url=CRUSOE_BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.3,
            top_p=0.95,
        )
except Exception as e:
    err = str(e)
    if "401" in err or "authentication" in err.lower() or "api key" in err.lower():
        show_error(
            "API Authentication Failed",
            "The Crusoe API key was rejected. Please verify your <strong>CRUSOE_API_KEY</strong> "
            "in the Space settings and ensure it has not expired.",
            hint=err,
        )
    elif "429" in err or "rate limit" in err.lower():
        show_error(
            "Rate Limit Reached",
            "Too many requests have been made to the Crusoe inference endpoint. "
            "Please wait a moment and run the analysis again.",
            hint=err,
        )
    elif "404" in err or "not found" in err.lower():
        show_error(
            "Model Not Available",
            f"The selected model <strong>{html.escape(MODEL)}</strong> could not be found on Crusoe Managed AI. "
            "Please select a different model from the sidebar.",
            hint=err,
        )
    elif "timeout" in err.lower() or "timed out" in err.lower():
        show_error(
            "Request Timed Out",
            "The analysis request took too long to complete. "
            "Try selecting a faster model from the sidebar, or run the analysis again.",
            hint=err,
        )
    else:
        show_error(
            "Analysis Unavailable",
            "An error occurred while connecting to Crusoe Managed AI. "
            "Please check your API key and try again.",
            hint=err,
        )

raw_content = resp.choices[0].message.content or ""
# Thinking models wrap their answer after </think> — strip the reasoning block
if IS_THINKING and "</think>" in raw_content:
    raw_content = raw_content.split("</think>", 1)[-1]
raw = raw_content.strip()

try:
    analysis = json.loads(raw)
except json.JSONDecodeError:
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            analysis = json.loads(m.group())
        except json.JSONDecodeError:
            show_error(
                "Response Parse Error",
                "The model returned a response that could not be processed. "
                "This can happen with thinking models — please try again.",
            )
    else:
        show_error(
            "Response Parse Error",
            "The model returned an unexpected response format. "
            "Please try again or select a different model from the sidebar.",
        )

# ── Helper: sentiment colour ──────────────────────────────────────────────────
def sentiment_color(score: float) -> str:
    if score >= 0.3:
        return "#34d399"
    if score <= -0.3:
        return "#f87171"
    return "#fbbf24"

def score_to_pct(score: float) -> int:
    return int((score + 1) / 2 * 100)

# ── Executive Summary ─────────────────────────────────────────────────────────
exec_summary = analysis.get("executive_summary", "")
st.markdown(f"""
<div class="glass-card-accent">
    <div class="section-label">Executive Summary</div>
    <p class="exec-text">{exec_summary}</p>
</div>
""", unsafe_allow_html=True)

# ── Sentiment metrics row ─────────────────────────────────────────────────────
overall = analysis.get("overall_sentiment", {})
dev = analysis.get("developer_sentiment", {})

c1, c2, c3 = st.columns(3)

for col, data, label in [
    (c1, overall, "Overall Sentiment"),
    (c2, dev, "Developer Sentiment"),
]:
    score = data.get("score", 0)
    color = sentiment_color(score)
    with col:
        st.markdown(f"""
<div class="glass-card" style="text-align:center; min-height:200px;">
    <div class="section-label">{label}</div>
    <div class="metric-num" style="color:{color};">{score_to_pct(score)}%</div>
    <div class="metric-sublabel" style="color:{color};">{data.get("label","")}</div>
    <div class="metric-desc">{data.get("summary","")}</div>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
<div class="glass-card" style="text-align:center; min-height:200px;">
    <div class="section-label">Coverage Volume</div>
    <div class="metric-num" style="color:#2563eb;">{len(articles)}</div>
    <div class="metric-sublabel" style="color:#64748b;">Articles Analysed</div>
    <div class="metric-desc">Sourced from NewsAPI · sorted by most recent publication date</div>
</div>
""", unsafe_allow_html=True)

# ── Brand Perception ──────────────────────────────────────────────────────────
perception = analysis.get("brand_perception", {})
dc = perception.get("data_center", 33)
cloud = perception.get("cloud_infrastructure", 33)
ai = perception.get("managed_ai", 33)

bars = [
    ("Data Center", dc, "linear-gradient(90deg,#6366f1,#4f46e5)", "#a5b4fc"),
    ("Cloud Infrastructure", cloud, "linear-gradient(90deg,#0ea5e9,#00d4ff)", "#7dd3fc"),
    ("Managed AI", ai, "linear-gradient(90deg,#10b981,#34d399)", "#6ee7b7"),
]

bars_html = "".join([f"""
<div class="bar-container">
    <div class="bar-row">
        <span>{name}</span>
        <span style="color:{text_color}; font-weight:600;">{pct}%</span>
    </div>
    <div class="bar-track">
        <div class="bar-fill" style="width:{pct}%; background:{grad};"></div>
    </div>
</div>
""" for name, pct, grad, text_color in bars])

st.markdown(f"""
<div class="glass-card">
    <div class="section-label">Brand Perception — How is Crusoe being perceived?</div>
    <p style="color:#64748b; font-size:0.88rem; margin:0 0 22px 0; line-height:1.6;">
        {perception.get("summary","")}
    </p>
    {bars_html}
</div>
""", unsafe_allow_html=True)

# ── Key Themes + Opportunities / Risks ───────────────────────────────────────
left, right = st.columns(2)

themes = analysis.get("key_themes", [])
opps = analysis.get("opportunities", [])
risks = analysis.get("risks", [])

tags_html = "".join([f'<span class="theme-tag">{t}</span>' for t in themes])
opps_html = "".join([f'<div class="list-item">✦&nbsp; {o}</div>' for o in opps])
risks_html = "".join([f'<div class="list-item">⚠&nbsp; {r}</div>' for r in risks])

with left:
    st.markdown(f"""
<div class="glass-card">
    <div class="section-label">Key Themes</div>
    <div style="margin-bottom:24px;">{tags_html}</div>
    <div class="section-label" style="margin-top:8px;">Opportunities</div>
    {opps_html}
</div>
""", unsafe_allow_html=True)

with right:
    st.markdown(f"""
<div class="glass-card">
    <div class="section-label">Risks &amp; Watch Areas</div>
    {risks_html}
</div>
""", unsafe_allow_html=True)

# ── Competitive Analysis ──────────────────────────────────────────────────────
comp_analysis = analysis.get("competitive_analysis", {})
if comp_analysis:
    st.markdown(f"""
<div class="glass-card-accent" style="margin-top:24px;">
    <div class="section-label">Competitive Landscape</div>
    <p style="color:#1e293b; font-size:1rem; line-height:1.75; margin:0 0 6px 0;">
        {comp_analysis.get("market_summary","")}
    </p>
    <p style="color:#64748b; font-size:0.88rem; line-height:1.6; margin:0;">
        {comp_analysis.get("crusoe_vs_field","")}
    </p>
</div>
""", unsafe_allow_html=True)

    # Competitor cards — 2 or 3 per row
    competitors = comp_analysis.get("competitors", [])
    THREAT_CLASS = {"High": "threat-high", "Medium": "threat-medium", "Low": "threat-low"}
    COVERAGE_COLOR = {"High": "#2563eb", "Medium": "#7c3aed", "Low": "#94a3b8"}

    cols_per_row = 3
    for i in range(0, len(competitors), cols_per_row):
        row = competitors[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, c in zip(cols, row):
            name = c.get("name", "")
            score = c.get("sentiment_score", 0)
            s_label = c.get("sentiment_label", "")
            s_color = sentiment_color(score)
            s_pct = score_to_pct(score)
            coverage = c.get("coverage_level", "Medium")
            threat = c.get("threat_level", "Medium")
            threat_cls = THREAT_CLASS.get(threat, "threat-medium")
            narrative = c.get("key_narrative", "")
            cov_color = COVERAGE_COLOR.get(coverage, "#94a3b8")
            with col:
                st.markdown(f"""
<div class="competitor-card">
    <div>
        <span class="competitor-name">{name}</span>
        <span class="threat-badge {threat_cls}">{threat} threat</span>
    </div>
    <p class="competitor-narrative">{narrative}</p>
    <div class="bar-row" style="margin-bottom:5px;">
        <span style="font-size:0.75rem; color:#94a3b8;">Sentiment</span>
        <span style="font-size:0.78rem; color:{s_color}; font-weight:600;">{s_pct}% · {s_label}</span>
    </div>
    <div class="bar-track">
        <div class="bar-fill" style="width:{s_pct}%; background:{s_color};"></div>
    </div>
    <div style="margin-top:12px; font-size:0.75rem; color:{cov_color}; font-weight:600;">
        {coverage} coverage volume
    </div>
</div>
""", unsafe_allow_html=True)

    # Crusoe advantages vs gaps
    adv_col, gap_col = st.columns(2)
    advantages = comp_analysis.get("crusoe_advantages", [])
    gaps = comp_analysis.get("crusoe_gaps", [])

    with adv_col:
        adv_html = "".join([f'<div class="comp-adv-item">✓ &nbsp;{a}</div>' for a in advantages])
        st.markdown(f"""
<div class="glass-card">
    <div class="section-label">Crusoe Competitive Advantages</div>
    {adv_html}
</div>
""", unsafe_allow_html=True)

    with gap_col:
        gap_html = "".join([f'<div class="comp-gap-item">△ &nbsp;{g}</div>' for g in gaps])
        st.markdown(f"""
<div class="glass-card">
    <div class="section-label">Gaps to Close</div>
    {gap_html}
</div>
""", unsafe_allow_html=True)

# ── Source articles ───────────────────────────────────────────────────────────

def fmt_timestamp(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%B %-d, %Y · %H:%M UTC")
    except Exception:
        return iso[:10]

with st.expander(f"Source Articles · {len(articles)} results"):
    for a in articles:
        title = a.get("title", "No title")
        url = a.get("url", "#")
        source = a.get("source", {}).get("name", "Unknown Source")
        ts = fmt_timestamp(a.get("publishedAt", ""))
        desc = a.get("description") or ""
        st.markdown(f"""
<div class="article-card">
    <a href="{url}" target="_blank" class="article-title">{title}</a>
    <div style="margin-top:8px;">
        <span class="article-source-badge">{source}</span>
        <span class="article-timestamp">{ts}</span>
    </div>
    {"<div class='article-desc'>" + desc + "</div>" if desc else ""}
</div>
""", unsafe_allow_html=True)

