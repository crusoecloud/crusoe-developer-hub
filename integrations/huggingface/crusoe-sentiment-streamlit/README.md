---
title: Crusoe AI Sentiment Intelligence
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
short_description: Real-time sentiment analysis of Crusoe AI across Reddit, Twitter, and news
---

# Crusoe AI Sentiment Intelligence

**Real-time sentiment analysis across Reddit, Twitter/X, and news — powered by a multi-model AI agent pipeline**

Collects mentions of Crusoe AI from 3 data sources, routes each post to the optimal LLM based on content type and length, then generates a unified sentiment dashboard with executive-level insights.

---

## How It Works

```
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Reddit   │  │ Twitter/X │  │   News    │
│  (PRAW)   │  │ (Tweepy)  │  │ (GNews +  │
│           │  │           │  │  NewsAPI)  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      ▼              ▼              ▼
┌────────────────────────────────────────┐
│         Smart Model Router             │
│  ┌──────────────────────────────────┐  │
│  │ Twitter → Gemma 3 12B (short)   │  │
│  │ Reddit  → DeepSeek V3 (medium)  │  │
│  │ News <2K→ Llama 3.3 70B         │  │
│  │ News 2K+→ Qwen3 235B (long)     │  │
│  │ Sarcasm → DeepSeek R1 (reason)  │  │
│  │ Synthesis→ Kimi K2              │  │
│  │ Fallback → GPT-4o               │  │
│  └──────────────────────────────────┘  │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│   SQLite/PostgreSQL                    │
│   Posts → Analyses → Snapshots         │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│   Streamlit Executive Dashboard        │
│   KPIs · Charts · Feed · Model Intel   │
└────────────────────────────────────────┘
```

---

## Features

### Overview Dashboard
- Executive KPI cards — overall sentiment, source-level scores, deltas vs previous run
- Interactive Plotly charts — sentiment distribution donut, source breakdown bars
- AI-generated synthesis report comparing current vs previous snapshot
- Top themes visualization with ranked chips

### Post Feed
- Filterable by source (Reddit/Twitter/News) and sentiment (positive/negative/neutral/mixed)
- Each post shows: summary, key themes, model used, confidence score, timestamp

### Model Intelligence Layer
- Visual routing rules showing which model handles which content type
- Model capability matrix
- Distribution chart — stacked bar of sentiment by model

---

## Architecture

```
crusoe-sentiment-streamlit/
├── app.py                  # Streamlit UI — 3-page dashboard (~1,200 lines)
├── config.py               # Pydantic settings — API keys, model config
├── database.py             # SQLAlchemy ORM — Post, Analysis, Snapshot tables
├── agent/
│   ├── models.py           # Smart router — content type → optimal model
│   └── pipeline.py         # Full orchestration — collect → route → analyze → synthesize
├── collectors/
│   ├── reddit.py           # PRAW — searches 5 subreddits for Crusoe mentions
│   ├── twitter.py          # Tweepy — 7-day search for Crusoe AI tweets
│   └── news.py             # GNews + NewsAPI — 30-day news articles
├── requirements.txt
├── runtime.txt             # Python 3.11.9
└── env.example
```

### Database Schema

| Table | Purpose |
|-------|---------|
| `Post` | Raw collected content — source, title, body, author, URL, timestamps |
| `Analysis` | AI results — sentiment, score (-1 to 1), summary, themes, entities, confidence, model used |
| `Snapshot` | Aggregate runs — overall/per-source scores, total posts, synthesis report, top themes |

---

## Quick Start

### Prerequisites

- Python 3.9+ (3.11 recommended)
- API keys for: Crusoe Cloud (via OpenRouter), Reddit, Twitter/X, GNews/NewsAPI

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crusoe-sentiment
cd crusoe-sentiment

pip install -r requirements.txt
cp env.example .env
# Edit .env with your API keys (see Environment Variables below)

streamlit run app.py
```

Open **http://localhost:8501**.

### Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Add all required secrets in Space Settings (see Environment Variables)
4. Push:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/crusoe-sentiment
git push hf main
```

---

## Use as a Framework

### Add a new data source

1. Create `collectors/your_source.py`:

```python
def collect(limit=50):
    """Return list of dicts with: title, body, url, author, published_at"""
    ...
```

2. Wire it into `agent/pipeline.py` alongside the Reddit/Twitter/News collectors

### Add or swap a model

Edit `agent/models.py` — the router maps content types to model IDs:

```python
# Route short social media posts to a fast model
if source == "twitter":
    return "google/gemma-3-12b-it"
```

### Change the analysis prompt

Edit `agent/pipeline.py` — the analysis prompt expects JSON output with: `sentiment`, `score`, `summary`, `key_themes`, `entities`, `confidence`.

### Switch from OpenRouter to direct Crusoe API

Edit `config.py` to point at Crusoe's endpoint:

```python
OPENAI_BASE_URL = "https://api.inference.crusoecloud.com/v1/"
```

And update `agent/models.py` model IDs to match Crusoe's registry.

### Add a new dashboard page

In `app.py`, add a new section to the sidebar navigation and create a rendering function:

```python
def render_your_page():
    st.header("Your New Page")
    # Query database, build charts, etc.
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (routes to 7 models) |
| `REDDIT_CLIENT_ID` | Yes | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Yes | Reddit app client secret |
| `TWITTER_BEARER_TOKEN` | Optional | Twitter/X API bearer token |
| `GNEWS_API_KEY` | Optional | GNews API key |
| `NEWS_API_KEY` | Optional | NewsAPI key (fallback) |
| `DATABASE_URL` | No | SQLite default, or PostgreSQL connection string |

---

## Dependencies

```
streamlit==1.38.0
plotly==5.24.0
langchain==0.3.0
langgraph==0.2.22
sqlalchemy==2.0.35
pg8000==1.31.2
praw==7.7.1
tweepy==4.14.0
python-dotenv==1.0.1
```

---

## License

MIT — fork it, remix it, ship it.

Built by Crusoe AI Developer Relations · [crusoe.ai](https://crusoe.ai)
