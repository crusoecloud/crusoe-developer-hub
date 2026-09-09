"""
Crusoe AI — Sentiment Intelligence Platform
Light-mode executive dashboard · Streamlit
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from database import db_session, Post, Analysis, Snapshot
from agent.models import (
    MODEL_GEMMA, MODEL_DEEPSEEK_V3, MODEL_LLAMA,
    MODEL_QWEN, MODEL_DEEPSEEK_R1, MODEL_KIMI, MODEL_GPT_OSS,
)

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crusoe AI · Sentiment Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — Light Mode
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#F8FAFC",
    "surface":   "#FFFFFF",
    "surface2":  "#F1F5F9",
    "border":    "#E2E8F0",
    "border2":   "#CBD5E1",
    "blue":      "#2563EB",
    "blue_lt":   "#EFF6FF",
    "blue_mid":  "#93C5FD",
    "green":     "#059669",
    "green_lt":  "#ECFDF5",
    "red":       "#DC2626",
    "red_lt":    "#FEF2F2",
    "amber":     "#D97706",
    "amber_lt":  "#FFFBEB",
    "gray":      "#64748B",
    "text":      "#0F172A",
    "text2":     "#334155",
    "text3":     "#64748B",
    "text4":     "#94A3B8",
    "reddit":    "#FF4500",
    "twitter":   "#1D9BF0",
    "news":      "#7C3AED",
    "shadow":    "0 1px 3px 0 rgba(0,0,0,.08), 0 1px 2px -1px rgba(0,0,0,.06)",
    "shadow_md": "0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.05)",
    "shadow_lg": "0 10px 15px -3px rgba(0,0,0,.08), 0 4px 6px -4px rgba(0,0,0,.04)",
}

SENTIMENT_COLOR = {"positive": C["green"], "negative": C["red"],
                   "neutral": C["gray"],   "mixed":    C["amber"]}
SENTIMENT_BG    = {"positive": C["green_lt"], "negative": C["red_lt"],
                   "neutral": C["surface2"],  "mixed":    C["amber_lt"]}
SENTIMENT_ICON  = {"positive": "▲", "negative": "▼", "neutral": "◆", "mixed": "◈"}

CSS = f"""
<style>
/* ── Global ─────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .main .block-container {{
    background: {C['bg']} !important;
    color: {C['text']} !important;
}}
[data-testid="stHeader"]     {{ background: {C['surface']} !important;
                                 border-bottom: 1px solid {C['border']} !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
footer, #MainMenu            {{ display: none !important; }}
.block-container             {{ padding: 0 !important; max-width: 100% !important; }}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar            {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track      {{ background: {C['surface2']}; }}
::-webkit-scrollbar-thumb      {{ background: {C['border2']}; border-radius: 99px; }}
::-webkit-scrollbar-thumb:hover{{ background: {C['gray']}; }}

/* ── Top-nav bar ─────────────────────────────────────────── */
.topbar {{
    background: {C['surface']};
    border-bottom: 1px solid {C['border']};
    padding: 0 40px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}}
.topbar-brand {{
    display: flex;
    align-items: baseline;
    gap: 10px;
}}
.topbar-title {{
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: {C['text']};
}}
.topbar-sub {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C['text4']};
}}
.topbar-meta {{
    font-size: 12px;
    color: {C['text3']};
}}

/* ── Page body ───────────────────────────────────────────── */
.page-body {{ padding: 32px 40px 60px; }}

/* ── Cards ───────────────────────────────────────────────── */
.card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 24px;
    box-shadow: {C['shadow']};
    transition: box-shadow 0.2s ease;
}}
.card:hover {{ box-shadow: {C['shadow_md']}; }}
.card-flat {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 24px;
}}

/* ── Executive banner ─────────────────────────────────────── */
.exec-banner {{
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 4px;
    display: flex;
    align-items: stretch;
    gap: 32px;
    box-shadow: {C['shadow']};
}}
.exec-divider {{
    width: 1px;
    align-self: stretch;
    flex-shrink: 0;
}}
.exec-score {{
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1;
}}
.exec-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.exec-summary {{
    font-size: 13px;
    line-height: 1.65;
    color: {C['text2']};
}}

/* ── KPI cards ───────────────────────────────────────────── */
.kpi {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: {C['shadow']};
    transition: box-shadow 0.2s, transform 0.2s;
    cursor: default;
}}
.kpi:hover {{
    box-shadow: {C['shadow_md']};
    transform: translateY(-2px);
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C['text3']};
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
}}
.kpi-sub {{
    font-size: 12px;
    color: {C['text3']};
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}}
.kpi-delta-pos {{ color: {C['green']}; font-weight: 600; }}
.kpi-delta-neg {{ color: {C['red']};   font-weight: 600; }}

/* ── Section header ──────────────────────────────────────── */
.section-hdr {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C['text3']};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.section-gap {{ margin: 28px 0 0; }}

/* ── Source bars ─────────────────────────────────────────── */
.source-table {{ width: 100%; border-collapse: collapse; }}
.source-row {{
    display: grid;
    grid-template-columns: 100px 1fr 70px 100px 70px;
    align-items: center;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid {C['border']};
    transition: background 0.15s;
}}
.source-row:last-child {{ border-bottom: none; }}
.source-row:hover {{ background: {C['surface2']}; border-radius: 8px; }}
.source-name {{
    font-size: 13px;
    font-weight: 600;
    color: {C['text2']};
    display: flex;
    align-items: center;
    gap: 7px;
}}
.source-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.source-track {{
    height: 8px;
    background: {C['surface2']};
    border-radius: 99px;
    position: relative;
    overflow: hidden;
    border: 1px solid {C['border']};
}}
.source-fill {{
    position: absolute;
    top: 0; height: 100%;
    border-radius: 99px;
}}
.source-center {{
    position: absolute;
    top: -4px; left: 50%;
    width: 2px; height: 16px;
    background: {C['border2']};
}}
.source-score {{
    font-size: 15px;
    font-weight: 700;
    text-align: right;
    letter-spacing: -0.02em;
}}
.source-badge {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    text-align: center;
    padding: 4px 10px;
    border-radius: 99px;
}}
.source-count {{
    font-size: 12px;
    color: {C['text3']};
    text-align: right;
}}

/* ── Theme chips ─────────────────────────────────────────── */
.theme-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.theme-chip {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 13px;
    font-weight: 500;
    color: {C['text2']};
    background: {C['surface2']};
    border: 1px solid {C['border']};
    transition: all 0.15s;
    cursor: default;
}}
.theme-chip:hover {{
    background: {C['blue_lt']};
    border-color: {C['blue_mid']};
    color: {C['blue']};
    transform: translateY(-1px);
    box-shadow: {C['shadow']};
}}
.theme-chip.top {{
    background: {C['blue_lt']};
    border-color: {C['blue_mid']};
    color: {C['blue']};
    font-weight: 600;
}}
.theme-rank {{
    font-size: 10px;
    font-weight: 700;
    opacity: 0.6;
}}

/* ── Report ──────────────────────────────────────────────── */
.report-wrap {{
    font-size: 14px;
    line-height: 1.8;
    color: {C['text2']};
    max-height: 440px;
    overflow-y: auto;
    padding-right: 8px;
}}
.report-wrap h4 {{
    color: {C['text']} !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 20px 0 8px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid {C['border']};
}}
.report-wrap p {{ margin-bottom: 10px; }}
.report-wrap li {{ margin-bottom: 6px; padding-left: 4px; }}

/* ── Post cards ──────────────────────────────────────────── */
.post-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    box-shadow: {C['shadow']};
    transition: box-shadow 0.2s, transform 0.15s, border-color 0.15s;
    cursor: pointer;
}}
.post-card:hover {{
    box-shadow: {C['shadow_md']};
    transform: translateY(-2px);
    border-color: {C['blue_mid']};
}}
.post-title {{
    font-size: 14px;
    font-weight: 600;
    color: {C['text']};
    line-height: 1.45;
    margin-bottom: 6px;
    text-decoration: none;
}}
.post-title a {{
    color: {C['text']};
    text-decoration: none;
}}
.post-title a:hover {{ color: {C['blue']}; }}
.post-summary {{
    font-size: 13px;
    color: {C['text3']};
    line-height: 1.55;
    margin-bottom: 10px;
}}
.post-meta {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}
.post-theme {{
    font-size: 11px;
    color: {C['blue']};
    background: {C['blue_lt']};
    padding: 2px 8px;
    border-radius: 99px;
    font-weight: 500;
}}

/* ── Badges ──────────────────────────────────────────────── */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-right: 6px;
}}

/* ── No-data ─────────────────────────────────────────────── */
.no-data {{
    background: {C['surface']};
    border: 2px dashed {C['border2']};
    border-radius: 16px;
    padding: 64px 40px;
    text-align: center;
}}
.no-data-icon  {{ font-size: 36px; margin-bottom: 16px; }}
.no-data-title {{ font-size: 17px; font-weight: 600; color: {C['text2']}; margin-bottom: 8px; }}
.no-data-sub   {{ font-size: 14px; color: {C['text3']}; }}

/* ── Streamlit native overrides ──────────────────────────── */
[data-testid="stTabs"] > div:first-child {{
    background: {C['surface']};
    border-bottom: 1px solid {C['border']};
    padding: 0 40px;
    gap: 0;
}}
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {C['text3']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 16px 20px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {C['text']} !important;
    border-bottom-color: {C['blue']} !important;
    font-weight: 600 !important;
}}
[data-testid="stTabPanel"] {{ padding: 0 !important; }}
[data-testid="stSelectbox"] > div > div {{
    background: {C['surface']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
}}
[data-testid="stNumberInput"] input {{
    background: {C['surface']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
}}
div[data-testid="stRadio"] > div {{
    background: {C['surface2']};
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
    display: inline-flex;
}}
div[data-testid="stRadio"] label {{
    padding: 6px 14px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: {C['text3']} !important;
    cursor: pointer !important;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: {C['surface']} !important;
    color: {C['text']} !important;
    box-shadow: {C['shadow']} !important;
}}
.stSpinner > div {{ border-top-color: {C['blue']} !important; }}
[data-testid="stAlert"] {{
    border-radius: 10px !important;
    border: 1px solid {C['border']} !important;
}}
</style>
"""

# Plotly shared theme for light mode
CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text3"], family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size=12),
    margin=dict(l=4, r=4, t=16, b=4),
    xaxis=dict(gridcolor=C["border"], tickfont=dict(color=C["text3"], size=11),
               zeroline=False, showline=False, ticklen=0),
    yaxis=dict(gridcolor=C["border"], tickfont=dict(color=C["text3"], size=11),
               zeroline=False, showline=False, ticklen=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text2"]), orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["border"],
                    font=dict(color=C["text"], size=12)),
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_latest_snapshot() -> dict | None:
    with db_session() as s:
        snap = s.query(Snapshot).order_by(Snapshot.run_at.desc()).first()
        if not snap:
            return None
        return dict(run_at=snap.run_at, overall_score=snap.overall_score or 0.0,
                    overall_sentiment=snap.overall_sentiment or "neutral",
                    reddit_score=snap.reddit_score, twitter_score=snap.twitter_score,
                    news_score=snap.news_score, total_posts=snap.total_posts or 0,
                    synthesis_report=snap.synthesis_report or "", top_themes=snap.top_themes or [])


@st.cache_data(ttl=120)
def load_history(days: int = 30) -> pd.DataFrame:
    since = datetime.utcnow() - timedelta(days=days)
    with db_session() as s:
        rows = s.query(Snapshot).filter(Snapshot.run_at >= since).order_by(Snapshot.run_at).all()
    return pd.DataFrame([
        {"Time": r.run_at, "Overall": r.overall_score,
         "Reddit": r.reddit_score, "Twitter": r.twitter_score, "News": r.news_score}
        for r in rows
    ])


@st.cache_data(ttl=60)
def load_source_counts() -> dict:
    with db_session() as s:
        rows = s.query(Post.source).all()
    counts = {"reddit": 0, "twitter": 0, "news": 0}
    for (src,) in rows:
        if src in counts:
            counts[src] += 1
    return counts


@st.cache_data(ttl=60)
def load_posts(source: str = "", sentiment: str = "", page: int = 1, page_size: int = 20) -> list[dict]:
    with db_session() as s:
        q = (s.query(Post, Analysis)
             .join(Analysis, Post.id == Analysis.post_id, isouter=True)
             .order_by(Post.collected_at.desc()))
        if source:
            q = q.filter(Post.source == source)
        if sentiment:
            q = q.filter(Analysis.sentiment == sentiment)
        rows = q.offset((page - 1) * page_size).limit(page_size).all()
    return [dict(source=p.source, title=p.title or (p.body[:100] if p.body else ""),
                 url=p.url, author=p.author, collected_at=p.collected_at,
                 sentiment=a.sentiment if a else None, score=a.score if a else None,
                 summary=a.summary if a else None, model_used=a.model_used if a else None,
                 key_themes=a.key_themes if a else [])
            for p, a in rows]


@st.cache_data(ttl=120)
def load_model_stats() -> pd.DataFrame:
    with db_session() as s:
        rows = s.query(Analysis.model_used, Analysis.sentiment,
                       Analysis.score, Analysis.confidence).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["model", "sentiment", "score", "confidence"])


@st.cache_data(ttl=120)
def load_prev_score() -> float | None:
    with db_session() as s:
        snaps = s.query(Snapshot.overall_score).order_by(Snapshot.run_at.desc()).limit(2).all()
    return snaps[1][0] if len(snaps) >= 2 else None


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def make_gauge(score: float, sentiment: str) -> go.Figure:
    color = SENTIMENT_COLOR.get(sentiment, C["gray"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            valueformat="+.2f",
            font=dict(size=52, color=color,
                      family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        ),
        gauge=dict(
            axis=dict(range=[-1, 1], tickwidth=0, dtick=0.5, tickformat="+.1f",
                      tickfont=dict(color=C["text3"], size=10)),
            bar=dict(color=color, thickness=0.2),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[-1.0, -0.15], color="#DC262612"),
                dict(range=[-0.15, 0.15], color="#64748B0C"),
                dict(range=[0.15,  1.0],  color="#05966912"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8, value=score,
            ),
        ),
    ))
    fig.add_annotation(
        text=f'<span style="font-weight:700;letter-spacing:0.08em">{sentiment.upper()}</span>',
        x=0.5, y=0.18, xref="paper", yref="paper",
        font=dict(size=11, color=color),
        showarrow=False,
    )
    fig.update_layout(height=280, margin=dict(l=24, r=24, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=C["text"]))
    return fig


def make_area_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    traces = [
        ("Overall", C["blue"],    3.0,  f"rgba(37,99,235,0.07)",   True),
        ("Reddit",  C["reddit"],  1.75, f"rgba(255,69,0,0.05)",    False),
        ("Twitter", C["twitter"], 1.75, f"rgba(29,155,240,0.05)",  False),
        ("News",    C["news"],    1.75, f"rgba(124,58,237,0.05)",  False),
    ]
    for col, color, width, fill_color, do_fill in traces:
        if col not in df.columns:
            continue
        kw = dict(
            x=df["Time"], y=df[col], name=col, mode="lines",
            line=dict(color=color, width=width, shape="spline", smoothing=0.7),
            hovertemplate=f"<b style='color:{color}'>{col}</b>: %{{y:+.3f}}<extra></extra>",
        )
        if do_fill:
            kw.update(fill="tozeroy", fillcolor=fill_color)
        fig.add_trace(go.Scatter(**kw))

    fig.add_hline(y=0, line_dash="dot", line_color=C["border2"], line_width=1.5)
    fig.update_layout(
        height=290,
        yaxis=dict(range=[-1.08, 1.08], tickformat="+.1f", gridcolor=C["border"],
                   tickfont=dict(color=C["text3"], size=11), zeroline=False,
                   showline=False, ticklen=0),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=C["text3"], size=11),
                   zeroline=False, showline=False, ticklen=0),
        **{k: v for k, v in CHART_BASE.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def make_donut() -> go.Figure:
    """Positive / Neutral / Negative post distribution."""
    with db_session() as s:
        rows = s.query(Analysis.sentiment).all()
    counts = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
    for (sent,) in rows:
        if sent in counts:
            counts[sent] += 1
    total = sum(counts.values()) or 1
    labels = ["Positive", "Neutral", "Negative", "Mixed"]
    values = [counts["positive"], counts["neutral"], counts["negative"], counts["mixed"]]
    colors = [C["green"], C["gray"], C["red"], C["amber"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors, line=dict(color=C["surface"], width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b>: %{value} posts (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:11px'>Posts</span>",
                       x=0.5, y=0.5, font=dict(size=18, color=C["text"]), showarrow=False)
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                                  xanchor="center", x=0.5, font=dict(color=C["text2"], size=11)))
    return fig


def make_model_chart(df: pd.DataFrame) -> go.Figure:
    dist = df.groupby(["model", "sentiment"]).size().reset_index(name="count")
    dist["model_short"] = dist["model"].str.split("/").str[-1]
    colors = {"positive": C["green"], "negative": C["red"],
              "neutral": C["gray"],   "mixed": C["amber"]}
    fig = px.bar(dist, x="model_short", y="count", color="sentiment",
                 color_discrete_map=colors, barmode="stack")
    fig.update_layout(
        height=280,
        xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(color=C["text3"], size=10), zeroline=False, showline=False, ticklen=0),
        yaxis=dict(gridcolor=C["border"], tickfont=dict(color=C["text3"], size=11),
                   zeroline=False, showline=False, ticklen=0),
        margin=dict(l=4, r=4, t=8, b=80),
        **{k: v for k, v in CHART_BASE.items() if k not in ("xaxis", "yaxis", "margin")},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(v):
    if v is None: return C["gray"]
    return C["green"] if v >= 0.15 else C["red"] if v <= -0.15 else C["gray"]

def _score_label(v):
    if v is None: return "—"
    return "Positive" if v >= 0.15 else "Negative" if v <= -0.15 else "Neutral"

def _badge(text, bg, color):
    return (f'<span class="badge" style="background:{bg};color:{color};'
            f'border:1px solid {color}35">{text}</span>')

def _source_badge(src):
    cfg = {"reddit": (C["reddit"], "#FF450015"),
           "twitter": (C["twitter"], "#1D9BF015"),
           "news": (C["news"], "#7C3AED15")}
    c, bg = cfg.get(src, (C["gray"], C["surface2"]))
    return _badge(src.capitalize(), bg, c)

def _sentiment_badge(sent, score=None):
    if not sent: return ""
    c  = SENTIMENT_COLOR.get(sent, C["gray"])
    bg = SENTIMENT_BG.get(sent, C["surface2"])
    lbl = sent.capitalize() + (f" {score:+.2f}" if score is not None else "")
    return _badge(lbl, bg, c)

def _kpi(label, value, sub="", color="", dot_color=""):
    vc = color or C["text"]
    dot = (f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
           f'background:{dot_color};margin-right:5px"></span>' if dot_color else "")
    return f"""
    <div class="kpi">
      <div class="kpi-label">{dot}{label}</div>
      <div class="kpi-value" style="color:{vc}">{value}</div>
      {"" if not sub else f'<div class="kpi-sub">{sub}</div>'}
    </div>"""

def _source_row(name, dot_color, score, count):
    if score is None:
        return f"""
        <div class="source-row">
          <div class="source-name">
            <span class="source-dot" style="background:{dot_color}"></span>{name}
          </div>
          <div class="source-track"><div class="source-center"></div></div>
          <div class="source-score" style="color:{C['text4']}">—</div>
          <div class="source-badge" style="background:{C['surface2']};color:{C['text4']}">No data</div>
          <div class="source-count">{count:,} posts</div>
        </div>"""
    pct   = (score + 1) / 2 * 100
    color = _score_color(score)
    label = _score_label(score)
    left  = 50 if score >= 0 else pct
    width = abs(pct - 50)
    return f"""
    <div class="source-row">
      <div class="source-name">
        <span class="source-dot" style="background:{dot_color}"></span>{name}
      </div>
      <div class="source-track">
        <div class="source-center"></div>
        <div class="source-fill" style="left:{left:.1f}%;width:{width:.1f}%;background:{color};opacity:0.75"></div>
      </div>
      <div class="source-score" style="color:{color}">{score:+.2f}</div>
      <div class="source-badge" style="background:{SENTIMENT_BG.get(label.lower(), C['surface2'])};color:{color}">{label}</div>
      <div class="source-count">{count:,} posts</div>
    </div>"""

def _exec_banner(snap, prev_score):
    score = snap["overall_score"]
    sent  = snap["overall_sentiment"]
    color = SENTIMENT_COLOR.get(sent, C["gray"])
    bg    = SENTIMENT_BG.get(sent, C["surface2"])
    icon  = SENTIMENT_ICON.get(sent, "◆")

    delta_html = ""
    if prev_score is not None:
        d = score - prev_score
        dc = C["green"] if d > 0 else C["red"] if d < 0 else C["gray"]
        ds = "+" if d >= 0 else ""
        arrow = "↑" if d > 0 else "↓" if d < 0 else "→"
        delta_html = (f'<span style="font-size:13px;color:{dc};font-weight:600;margin-left:14px">'
                      f'{arrow} {ds}{d:.2f} vs last run</span>')

    return f"""
    <div class="exec-banner" style="background:{bg};border:1px solid {color}25;border-left:4px solid {color}">
      <div style="flex-shrink:0;min-width:160px">
        <div class="exec-label" style="color:{color}">{icon} Overall Sentiment</div>
        <div class="exec-score" style="color:{color}">{score:+.2f}</div>
        <div style="font-size:12px;font-weight:600;color:{color};margin-top:6px;letter-spacing:.04em">
          {sent.upper()} {delta_html}
        </div>
      </div>
      <div class="exec-divider" style="background:{color}20"></div>
      <div style="flex:1">
        <div class="exec-label" style="color:{C['text3']}">Analysis coverage</div>
        <div style="font-size:26px;font-weight:700;color:{C['text']};letter-spacing:-0.03em">
          {snap['total_posts']:,} <span style="font-size:14px;font-weight:400;color:{C['text3']}">posts analyzed</span>
        </div>
        <div class="exec-summary" style="margin-top:6px">
          Reddit · Twitter/X · Global News &nbsp;·&nbsp;
          Last run: {snap['run_at'].strftime('%b %d, %Y at %I:%M %p UTC')}
        </div>
      </div>
      <div class="exec-divider" style="background:{color}20"></div>
      <div style="flex-shrink:0;min-width:140px">
        <div class="exec-label" style="color:{C['text3']}">Top Theme</div>
        <div style="font-size:18px;font-weight:600;color:{C['text']};margin-top:4px">
          {(snap['top_themes'][0] if snap['top_themes'] else '—').capitalize()}
        </div>
        <div style="font-size:12px;color:{C['text3']};margin-top:6px">
          {len(snap['top_themes'])} themes identified
        </div>
      </div>
    </div>"""

def _report_html(text):
    if not text:
        return f'<p style="color:{C["text3"]}">No report generated yet.</p>'
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            lines.append('<div style="height:6px"></div>')
        elif s.startswith("**") and s.endswith("**"):
            lines.append(f'<h4>{s.replace("**","")}</h4>')
        elif s.startswith("#"):
            lines.append(f'<h4>{s.lstrip("#").strip()}</h4>')
        elif s.startswith(("- ", "• ")):
            lines.append(f'<li>{s[2:]}</li>')
        else:
            lines.append(f'<p>{s}</p>')
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_collection():
    from agent.pipeline import run_pipeline
    log_box   = st.empty()
    log_lines: list[str] = []

    def cb(msg):
        log_lines.append(msg)
        log_box.markdown(
            "".join(
                f'<div style="font-size:12px;color:{C["text3"]};padding:2px 0;'
                f'display:flex;align-items:center;gap:8px">'
                f'<span style="color:{C["blue_mid"]}">›</span> {l}</div>'
                for l in log_lines[-6:]
            ),
            unsafe_allow_html=True,
        )

    with st.spinner("Running sentiment pipeline…"):
        try:
            result = run_pipeline(status_cb=cb)
            st.cache_data.clear()
            if "error" in result:
                st.error("No posts collected. Verify API keys in your `.env` file.")
            else:
                sent  = result["overall_sentiment"]
                score = result["overall_score"]
                color = SENTIMENT_COLOR.get(sent, C["gray"])
                bg    = SENTIMENT_BG.get(sent, C["surface2"])
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {color}30;border-radius:10px;'
                    f'padding:14px 20px;font-size:14px;color:{color};font-weight:600">'
                    f'{SENTIMENT_ICON.get(sent,"◆")} &nbsp;Complete &nbsp;·&nbsp; '
                    f'{result["total_posts"]:,} posts analyzed &nbsp;·&nbsp; '
                    f'{sent.capitalize()} ({score:+.2f})</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_topbar(snap):
    c1, _, c2 = st.columns([4, 2, 1])
    with c1:
        ts = (snap["run_at"].strftime("Last updated %b %d, %Y at %I:%M %p UTC")
              if snap else "No data yet — run a collection to get started")
        st.markdown(
            f'<div class="topbar" style="position:relative;padding:0">'
            f'<div class="topbar-brand">'
            f'<span class="topbar-title">CRUSOE AI</span>'
            f'<span class="topbar-sub">Sentiment Intelligence</span>'
            f'</div>'
            f'<div class="topbar-meta">{ts}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("▶  Run Collection", type="primary", use_container_width=True):
            run_collection()
            st.rerun()


def render_overview(snap):
    st.markdown('<div class="page-body">', unsafe_allow_html=True)

    if not snap:
        st.markdown("""
        <div class="no-data">
          <div class="no-data-icon">◈</div>
          <div class="no-data-title">No sentiment data yet</div>
          <div class="no-data-sub">Click <strong>Run Collection</strong> to pull data from
          Reddit, Twitter/X, and news — then the AI analysis pipeline will run automatically.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    prev_score = load_prev_score()
    counts     = load_source_counts()
    score      = snap["overall_score"]
    sent       = snap["overall_sentiment"]

    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown(_exec_banner(snap, prev_score), unsafe_allow_html=True)
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    def fmt(v): return f"{v:+.2f}" if v is not None else "—"
    def delta_sub(v, c):
        if prev_score is None or v is None: return f'{c:,} posts'
        d = v - prev_score
        cls = "kpi-delta-pos" if d > 0 else "kpi-delta-neg" if d < 0 else ""
        return f'<span class="{cls}">{"+" if d>=0 else ""}{d:.2f}</span> vs prev &nbsp;·&nbsp; {c:,} posts'

    kpis = [
        ("Sentiment Score",  fmt(score),                      delta_sub(score, snap["total_posts"]),   _score_color(score),               C["blue"]),
        ("Total Mentions",   f'{snap["total_posts"]:,}',      "across all channels",                   C["text"],                         C["gray"]),
        ("Reddit",           fmt(snap.get("reddit_score")),   delta_sub(snap.get("reddit_score"), counts["reddit"]),   _score_color(snap.get("reddit_score")), C["reddit"]),
        ("Twitter / X",      fmt(snap.get("twitter_score")),  delta_sub(snap.get("twitter_score"), counts["twitter"]), _score_color(snap.get("twitter_score")), C["twitter"]),
        ("News",             fmt(snap.get("news_score")),     delta_sub(snap.get("news_score"),    counts["news"]),    _score_color(snap.get("news_score")),    C["news"]),
    ]
    for col, (label, value, sub, color, dot) in zip(st.columns(5), kpis):
        with col:
            st.markdown(_kpi(label, value, sub, color, dot), unsafe_allow_html=True)

    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

    # ── Gauge + Trend ─────────────────────────────────────────────────────────
    col_g, col_t = st.columns([2, 3], gap="large")

    with col_g:
        st.markdown('<div class="section-hdr">Sentiment Score</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card-flat">', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(score, sent), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    with col_t:
        hdr_c, radio_c = st.columns([2, 2])
        with hdr_c:
            st.markdown('<div class="section-hdr">Sentiment Trend</div>', unsafe_allow_html=True)
        with radio_c:
            days = st.radio("", ["7 days", "14 days", "30 days"], horizontal=True,
                            index=1, label_visibility="collapsed")
        d_map  = {"7 days": 7, "14 days": 14, "30 days": 30}
        hist   = load_history(d_map[days])
        with st.container():
            st.markdown('<div class="card-flat">', unsafe_allow_html=True)
            if not hist.empty and len(hist) > 1:
                st.plotly_chart(make_area_chart(hist), use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.markdown(
                    f'<p style="color:{C["text3"]};font-size:13px;padding:80px 0;text-align:center">'
                    f'Run more collections to see the trend over time.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

    # ── Source breakdown ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Source Breakdown</div>', unsafe_allow_html=True)
    rows_html = (
        _source_row("Reddit",    C["reddit"],  snap.get("reddit_score"),  counts["reddit"])
        + _source_row("Twitter", C["twitter"], snap.get("twitter_score"), counts["twitter"])
        + _source_row("News",    C["news"],    snap.get("news_score"),    counts["news"])
    )
    st.markdown(f'<div class="card-flat">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

    # ── Donut + Themes ────────────────────────────────────────────────────────
    col_d, col_rep, col_th = st.columns([1, 2, 1], gap="large")

    with col_d:
        st.markdown('<div class="section-hdr">Post Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-flat">', unsafe_allow_html=True)
        st.plotly_chart(make_donut(), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rep:
        st.markdown('<div class="section-hdr">AI Synthesis Report</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-flat"><div class="report-wrap">{_report_html(snap["synthesis_report"])}</div></div>',
            unsafe_allow_html=True,
        )

    with col_th:
        st.markdown('<div class="section-hdr">Top Themes</div>', unsafe_allow_html=True)
        themes = snap.get("top_themes", [])
        if themes:
            chips = "".join(
                f'<span class="theme-chip{"  top" if i < 3 else ""}">'
                f'<span class="theme-rank">#{i+1}</span>{t}</span>'
                for i, t in enumerate(themes[:12])
            )
            st.markdown(f'<div class="card-flat"><div class="theme-chips">{chips}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:{C["text3"]};font-size:13px">No themes yet.</p>',
                        unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # page-body


def render_feed():
    st.markdown('<div class="page-body">', unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="color:{C["text"]};font-size:20px;font-weight:700;'
        f'letter-spacing:-0.02em;margin-bottom:20px">Post Feed</h3>',
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 1, 1])
    with fc1:
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["text3"]};'
                    f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Source</div>',
                    unsafe_allow_html=True)
        src = st.selectbox("src", ["All", "reddit", "twitter", "news"],
                           label_visibility="collapsed")
    with fc2:
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["text3"]};'
                    f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Sentiment</div>',
                    unsafe_allow_html=True)
        sent = st.selectbox("sent", ["All", "positive", "negative", "neutral", "mixed"],
                            label_visibility="collapsed")
    with fc3:
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["text3"]};'
                    f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Page</div>',
                    unsafe_allow_html=True)
        page = st.number_input("page", min_value=1, value=1, step=1,
                               label_visibility="collapsed")
    with fc4:
        st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        if st.button("↺  Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    posts = load_posts(
        source    = "" if src  == "All" else src,
        sentiment = "" if sent == "All" else sent,
        page      = page,
    )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    if not posts:
        st.markdown("""
        <div class="no-data" style="padding:40px">
          <div class="no-data-title">No posts match these filters</div>
          <div class="no-data-sub">Try adjusting the source or sentiment filter.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    for p in posts:
        s      = p.get("sentiment") or "neutral"
        score  = p.get("score")
        title  = p.get("title") or "—"
        url    = p.get("url")
        themes = (p.get("key_themes") or [])[:4]
        model  = (p.get("model_used") or "").split("/")[-1]
        ts     = p["collected_at"].strftime("%b %d · %H:%M UTC") if p.get("collected_at") else ""

        title_html = (f'<a href="{url}" class="post-title" target="_blank" rel="noopener">{title}</a>'
                      if url else f'<span class="post-title">{title}</span>')
        themes_html = "".join(f'<span class="post-theme">#{t}</span> ' for t in themes)
        model_html  = (f'<span style="font-family:monospace;font-size:10px;color:{C["text4"]};'
                       f'background:{C["surface2"]};padding:2px 7px;border-radius:4px">{model}</span>'
                       if model else "")

        st.markdown(f"""
        <div class="post-card">
          <div class="post-meta" style="margin-bottom:8px">
            {_source_badge(p["source"])} {_sentiment_badge(s, score)}
          </div>
          <div>{title_html}</div>
          {"" if not p.get("summary") else f'<div class="post-summary">{p["summary"]}</div>'}
          <div class="post-meta" style="margin-top:10px">
            {themes_html}
            <span style="margin-left:auto;display:flex;align-items:center;gap:10px">
              {model_html}
              <span style="font-size:11px;color:{C['text4']}">{ts}</span>
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;font-size:12px;color:{C["text4"]};padding:16px 0">'
        f'Page {page} · 20 posts per page</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_models():
    st.markdown('<div class="page-body">', unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="color:{C["text"]};font-size:20px;font-weight:700;'
        f'letter-spacing:-0.02em;margin-bottom:20px">Model Intelligence Layer</h3>',
        unsafe_allow_html=True,
    )

    routing = [
        (MODEL_GEMMA,       C["twitter"], "Tweet / short text",         "Fast classification · low latency",          "Google"),
        (MODEL_DEEPSEEK_V3, C["reddit"],  "Reddit post or comment",     "General sentiment · strong reasoning",        "DeepSeek"),
        (MODEL_LLAMA,       C["green"],   "News article < 2K tokens",   "Summarization · entity extraction",           "Meta"),
        (MODEL_QWEN,        C["news"],    "News article ≥ 2K tokens",   "Long-context · 235B parameters",              "Alibaba"),
        (MODEL_DEEPSEEK_R1, C["amber"],   "Ambiguous / sarcastic text", "Deep chain-of-thought reasoning",             "DeepSeek"),
        (MODEL_KIMI,        C["blue"],    "Trend analysis",             "Multi-step reasoning · synthesis",            "Moonshot"),
        (MODEL_GPT_OSS,     C["gray"],    "Final report generation",    "High-quality narrative output",               "OpenAI"),
    ]

    rows = "".join(f"""
    <div style="display:grid;grid-template-columns:28px 200px 200px 1fr 70px;
                gap:16px;padding:14px 0;border-bottom:1px solid {C['border']};
                align-items:center;transition:background .15s"
         onmouseover="this.style.background='{C['surface2']}'"
         onmouseout="this.style.background='transparent'">
      <div style="width:8px;height:8px;border-radius:50%;background:{color};margin:0 auto"></div>
      <div style="font-family:monospace;font-size:12px;color:{C['text2']};font-weight:600">
        {model.split('/')[-1]}
      </div>
      <div style="font-size:13px;font-weight:500;color:{C['text']}">{cond}</div>
      <div style="font-size:12px;color:{C['text3']}">{desc}</div>
      <div style="font-size:11px;color:{C['text4']};text-align:right">{provider}</div>
    </div>"""
    for model, color, cond, desc, provider in routing)

    header = f"""
    <div style="display:grid;grid-template-columns:28px 200px 200px 1fr 70px;
                gap:16px;padding:0 0 10px;border-bottom:2px solid {C['border']}">
      <div></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{C['text3']}">Model</div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{C['text3']}">Assigned to</div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{C['text3']}">Capability</div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{C['text3']};text-align:right">Provider</div>
    </div>"""

    st.markdown(f'<div class="card-flat" style="margin-bottom:28px">{header}{rows}</div>',
                unsafe_allow_html=True)

    df = load_model_stats()
    if df.empty:
        st.markdown(f"""
        <div class="no-data">
          <div class="no-data-title">No model activity yet</div>
          <div class="no-data-sub">Run a collection to see per-model performance statistics.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-hdr">Model Performance</div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2, gap="large")
    summary = (df.groupby("model")
               .agg(Posts=("score","count"), Avg_Score=("score","mean"),
                    Avg_Confidence=("confidence","mean"))
               .round(3).reset_index())
    summary["Model"] = summary["model"].str.split("/").str[-1]
    summary = summary[["Model", "Posts", "Avg_Score", "Avg_Confidence"]]
    with mc1:
        st.dataframe(summary, use_container_width=True, hide_index=True,
                     column_config={
                         "Posts":          st.column_config.NumberColumn("Posts"),
                         "Avg_Score":      st.column_config.NumberColumn("Avg Score", format="%+.3f"),
                         "Avg_Confidence": st.column_config.ProgressColumn(
                             "Confidence", min_value=0, max_value=1, format="%.2f"),
                     })
    with mc2:
        st.plotly_chart(make_model_chart(df), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    snap = load_latest_snapshot()
    render_topbar(snap)

    tab1, tab2, tab3 = st.tabs([
        "  Overview  ",
        "  Post Feed  ",
        "  Model Intelligence  ",
    ])
    with tab1:
        render_overview(snap)
    with tab2:
        render_feed()
    with tab3:
        render_models()


if __name__ == "__main__":
    main()
