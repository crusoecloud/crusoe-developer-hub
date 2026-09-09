import logging
from datetime import datetime, timezone
import praw
from config import settings

log = logging.getLogger(__name__)

KEYWORDS = ["Crusoe AI", "Crusoe Energy", "Crusoe cloud", "CrusoeAI"]
SUBREDDITS = ["artificial", "MachineLearning", "LocalLLaMA", "HPC", "CloudComputing"]


def collect(limit: int = 50) -> list[dict]:
    if not settings.REDDIT_CLIENT_ID:
        log.warning("Reddit credentials not configured, skipping.")
        return []

    reddit = praw.Reddit(
        client_id=settings.REDDIT_CLIENT_ID,
        client_secret=settings.REDDIT_CLIENT_SECRET,
        user_agent=settings.REDDIT_USER_AGENT,
        read_only=True,
    )

    seen, results = set(), []

    for kw in KEYWORDS:
        try:
            for sub in reddit.subreddit("all").search(kw, sort="new", limit=limit):
                eid = f"reddit_{sub.id}"
                if eid not in seen:
                    seen.add(eid)
                    results.append({
                        "source": "reddit",
                        "external_id": eid,
                        "url": f"https://reddit.com{sub.permalink}",
                        "title": sub.title,
                        "body": sub.selftext or "",
                        "author": str(sub.author) if sub.author else "[deleted]",
                        "published_at": datetime.fromtimestamp(sub.created_utc, tz=timezone.utc),
                        "raw_json": {"subreddit": sub.subreddit.display_name, "score": sub.score},
                    })
        except Exception as e:
            log.warning("Reddit search '%s' failed: %s", kw, e)

    log.info("Reddit: %d posts", len(results))
    return results
