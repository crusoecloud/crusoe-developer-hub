import os
import httpx


def tavily(query: str, max_results: int) -> list[dict]:
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": max_results},
        timeout=15,
    )
    r.raise_for_status()
    hits = r.json().get("results", [])
    return [{"title": h.get("title", ""), "url": h["url"], "snippet": h.get("content", "")[:300]} for h in hits[:max_results]]


def brave(query: str, max_results: int) -> list[dict]:
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"], "Accept": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    hits = r.json().get("web", {}).get("results", [])
    return [{"title": h.get("title", ""), "url": h["url"], "snippet": h.get("description", "")} for h in hits[:max_results]]


ADAPTERS = {"tavily": tavily, "brave": brave}
KEYS = {"tavily": "TAVILY_API_KEY", "brave": "BRAVE_API_KEY"}


def search_adapter(query: str, max_results: int = 5) -> list[dict]:
    name = os.environ.get("SEARCH_PROVIDER", "")
    if name not in ADAPTERS or not os.environ.get(KEYS[name]):
        name = next((n for n in ADAPTERS if os.environ.get(KEYS[n])), "tavily")
    return ADAPTERS[name](query, max_results)
