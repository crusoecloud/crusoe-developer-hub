import logging
from datetime import datetime, timezone, timedelta
import httpx
from config import settings

log = logging.getLogger(__name__)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def collect(limit: int = 50) -> list[dict]:
    results = []

    if settings.GNEWS_API_KEY:
        try:
            resp = httpx.get(
                "https://gnews.io/api/v4/search",
                params={"q": "Crusoe AI", "lang": "en", "sortby": "publishedAt",
                        "max": min(limit, 100), "apikey": settings.GNEWS_API_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            for a in resp.json().get("articles", []):
                results.append({
                    "source": "news",
                    "external_id": f"gnews_{abs(hash(a.get('url', '')))}",
                    "url": a.get("url"),
                    "title": a.get("title"),
                    "body": a.get("content") or a.get("description") or "",
                    "author": a.get("source", {}).get("name"),
                    "published_at": _parse_dt(a.get("publishedAt")),
                    "raw_json": {"provider": "gnews"},
                })
        except Exception as e:
            log.warning("GNews failed: %s", e)

    if not results and settings.NEWSAPI_KEY:
        try:
            from_date = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
            resp = httpx.get(
                "https://newsapi.org/v2/everything",
                params={"q": "Crusoe AI", "language": "en", "sortBy": "publishedAt",
                        "pageSize": min(limit, 100), "from": from_date, "apiKey": settings.NEWSAPI_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            for a in resp.json().get("articles", []):
                if a.get("title") == "[Removed]":
                    continue
                results.append({
                    "source": "news",
                    "external_id": f"newsapi_{abs(hash(a.get('url', '')))}",
                    "url": a.get("url"),
                    "title": a.get("title"),
                    "body": a.get("content") or a.get("description") or "",
                    "author": a.get("author") or a.get("source", {}).get("name"),
                    "published_at": _parse_dt(a.get("publishedAt")),
                    "raw_json": {"provider": "newsapi"},
                })
        except Exception as e:
            log.warning("NewsAPI failed: %s", e)

    log.info("News: %d articles", len(results))
    return results
