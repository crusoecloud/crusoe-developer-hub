import logging
from datetime import datetime, timezone, timedelta
import tweepy
from config import settings

log = logging.getLogger(__name__)

QUERIES = ['"Crusoe AI" -is:retweet lang:en', '"CrusoeAI" -is:retweet lang:en']


def collect(limit: int = 50) -> list[dict]:
    if not settings.TWITTER_BEARER_TOKEN:
        log.warning("Twitter bearer token not configured, skipping.")
        return []

    client = tweepy.Client(bearer_token=settings.TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
    start_time = datetime.now(tz=timezone.utc) - timedelta(days=7)
    seen, results = set(), []

    for query in QUERIES:
        try:
            resp = client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                start_time=start_time,
                tweet_fields=["created_at", "author_id", "public_metrics"],
                expansions=["author_id"],
                user_fields=["username"],
            )
            if not resp.data:
                continue

            user_map = {}
            if resp.includes and resp.includes.get("users"):
                for u in resp.includes["users"]:
                    user_map[u.id] = u.username

            for tweet in resp.data:
                eid = f"twitter_{tweet.id}"
                if eid not in seen:
                    seen.add(eid)
                    results.append({
                        "source": "twitter",
                        "external_id": eid,
                        "url": f"https://twitter.com/i/web/status/{tweet.id}",
                        "title": None,
                        "body": tweet.text,
                        "author": user_map.get(tweet.author_id, "unknown"),
                        "published_at": tweet.created_at,
                        "raw_json": {"metrics": tweet.public_metrics},
                    })
        except Exception as e:
            log.warning("Twitter query '%s' failed: %s", query, e)

    log.info("Twitter: %d tweets", len(results))
    return results
