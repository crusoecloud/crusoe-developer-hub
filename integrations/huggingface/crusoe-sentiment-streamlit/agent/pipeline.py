"""
Sentiment pipeline — runs as a single synchronous function.
Called directly from Streamlit (no Celery needed).
"""
import json
import logging
import re
from collections import Counter
from datetime import datetime

from tenacity import retry, stop_after_attempt, wait_exponential

from agent.models import route_model, analysis_llm, synthesis_llm, trend_llm
from database import db_session, Post, Analysis, Snapshot

log = logging.getLogger(__name__)

ANALYSIS_PROMPT = """\
Analyze the sentiment of the following content toward Crusoe AI specifically.
Return ONLY valid JSON — no extra text.

Source: {source}
Title: {title}
Body: {body}

JSON format:
{{
  "sentiment": "positive|negative|neutral|mixed",
  "score": <float -1.0 to 1.0>,
  "summary": "<1-2 sentences about Crusoe AI>",
  "key_themes": ["<theme1>", "<theme2>"],
  "entities_mentioned": ["<name>"],
  "confidence": <float 0.0 to 1.0>
}}"""

SYNTHESIS_PROMPT = """\
You are a market intelligence analyst. Write a structured sentiment report for Crusoe AI
based on the data below. Cover: overall sentiment, key themes, source differences, notable quotes.

Data:
{data}

Previous report (for comparison):
{prev}
"""

FALLBACK = {"sentiment": "neutral", "score": 0.0, "summary": "", "key_themes": [],
            "entities_mentioned": [], "confidence": 0.0}


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
def _analyze(post: dict) -> dict:
    model_id = post["_model"]
    llm = analysis_llm(model_id)
    prompt = ANALYSIS_PROMPT.format(
        source=post.get("source", ""),
        title=post.get("title", "") or "",
        body=(post.get("body", "") or "")[:3000],
    )
    result = _extract_json(llm.invoke(prompt).content)
    if not result:
        return {**FALLBACK, "model_used": model_id}
    result["score"] = max(-1.0, min(1.0, float(result.get("score", 0.0))))
    result["model_used"] = model_id
    return result


def _weighted_score(items: list[dict]) -> float | None:
    total_w, total_s = 0.0, 0.0
    for item in items:
        s = item.get("score", 0.0) or 0.0
        w = item.get("confidence", 0.5) or 0.5
        total_s += s * w
        total_w += w
    return round(total_s / total_w, 4) if total_w else None


def _label(score: float | None) -> str:
    if score is None:
        return "neutral"
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def run_pipeline(status_cb=None) -> dict:
    """
    Full pipeline: collect → route → analyze → aggregate → synthesize → store.
    status_cb(str): optional callback for progress messages (used by Streamlit).
    """
    from collectors import reddit, twitter, news
    from config import settings

    def log_status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    # ── 1. Collect ──────────────────────────────────────────────────────────
    log_status("Collecting from Reddit...")
    reddit_posts = reddit.collect(settings.MAX_POSTS_PER_SOURCE)

    log_status("Collecting from Twitter/X...")
    twitter_posts = twitter.collect(settings.MAX_POSTS_PER_SOURCE)

    log_status("Collecting from News APIs...")
    news_posts = news.collect(settings.MAX_POSTS_PER_SOURCE)

    all_posts = reddit_posts + twitter_posts + news_posts
    log_status(f"Collected {len(all_posts)} posts total.")

    if not all_posts:
        log_status("No posts collected. Check API keys.")
        return {"error": "no_posts"}

    # ── 2. Route models ──────────────────────────────────────────────────────
    for p in all_posts:
        p["_model"] = route_model(p["source"], p.get("body", ""), p.get("title", "") or "")

    # ── 3. Analyze ───────────────────────────────────────────────────────────
    analyses: list[dict] = []
    for i, post in enumerate(all_posts):
        log_status(f"Analyzing post {i+1}/{len(all_posts)} [{post['source']}]...")
        try:
            result = _analyze(post)
        except Exception as e:
            log.error("Analysis failed for %s: %s", post.get("external_id"), e)
            result = {**FALLBACK, "model_used": post["_model"]}
        analyses.append({**post, "analysis": result})

    # ── 4. Aggregate ─────────────────────────────────────────────────────────
    by_source = {"reddit": [], "twitter": [], "news": []}
    theme_counter: Counter = Counter()
    for item in analyses:
        src = item.get("source", "other")
        a = item["analysis"]
        by_source.setdefault(src, []).append(a)
        theme_counter.update(t.lower() for t in (a.get("key_themes") or []))

    reddit_score   = _weighted_score(by_source["reddit"])
    twitter_score  = _weighted_score(by_source["twitter"])
    news_score     = _weighted_score(by_source["news"])
    overall_score  = _weighted_score([item["analysis"] for item in analyses])
    top_themes     = [t for t, _ in theme_counter.most_common(15)]

    # ── 5. Synthesize ────────────────────────────────────────────────────────
    log_status("Generating synthesis report...")
    data_lines = []
    for item in analyses[:40]:
        a = item["analysis"]
        data_lines.append(
            f"[{item['source'].upper()}] {(item.get('title') or item.get('body',''))[:80]} "
            f"| {a.get('sentiment')} {a.get('score',0):.2f} | {', '.join((a.get('key_themes') or [])[:3])}"
        )

    prev_report = "No previous report."
    with db_session() as sess:
        last = sess.query(Snapshot).order_by(Snapshot.run_at.desc()).first()
        if last and last.synthesis_report:
            prev_report = last.synthesis_report[:1000]

    synthesis_report = ""
    try:
        llm = synthesis_llm()
        synthesis_report = llm.invoke(
            SYNTHESIS_PROMPT.format(data="\n".join(data_lines), prev=prev_report)
        ).content.strip()
    except Exception as e:
        log.error("Synthesis failed: %s", e)
        synthesis_report = f"Report generation failed: {e}"

    # ── 6. Store ─────────────────────────────────────────────────────────────
    log_status("Saving to database...")
    stored = 0
    with db_session() as sess:
        for item in analyses:
            existing = sess.query(Post).filter_by(external_id=item["external_id"]).first()
            if existing:
                post_obj = existing
            else:
                post_obj = Post(
                    source=item["source"],
                    external_id=item["external_id"],
                    url=item.get("url"),
                    title=item.get("title"),
                    body=item.get("body"),
                    author=item.get("author"),
                    published_at=item.get("published_at"),
                    raw_json=item.get("raw_json"),
                )
                sess.add(post_obj)
                sess.flush()

            a = item["analysis"]
            sess.add(Analysis(
                post_id=post_obj.id,
                model_used=a.get("model_used", "unknown"),
                sentiment=a.get("sentiment", "neutral"),
                score=a.get("score", 0.0),
                summary=a.get("summary", ""),
                key_themes=a.get("key_themes", []),
                entities=a.get("entities_mentioned", []),
                confidence=a.get("confidence", 0.0),
            ))
            stored += 1

        snap = Snapshot(
            overall_score=overall_score,
            overall_sentiment=_label(overall_score),
            reddit_score=reddit_score,
            twitter_score=twitter_score,
            news_score=news_score,
            total_posts=len(analyses),
            synthesis_report=synthesis_report,
            top_themes=top_themes,
        )
        sess.add(snap)

    log_status(f"Done. {stored} posts analyzed.")
    return {
        "overall_score": overall_score,
        "overall_sentiment": _label(overall_score),
        "reddit_score": reddit_score,
        "twitter_score": twitter_score,
        "news_score": news_score,
        "total_posts": len(analyses),
        "top_themes": top_themes,
        "synthesis_report": synthesis_report,
    }
