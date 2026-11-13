"""Turn an item spec into the one eBay search a watch runs.

The query is built by rule, not by a model, so the same spec always searches the same way and a
person can read the plan and see why a listing was or was not found. The search is kept broad on
purpose: it never carries the size or the price ceiling. Size is proven later from what each
listing states, and a listing over the ceiling still has to be seen so it can be parked and
alerted on if its price comes down.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

from .errors import EbayError
from .spec import BUYING_OPTIONS

# eBay's limit for the q parameter of item_summary/search.
MAX_QUERY_CHARS = 100

# In eBay's keyword syntax a comma means OR and parentheses group, so anything that is not part
# of a word is turned into a space ("White/Black (Panda)" -> "White Black Panda").
_NOT_KEYWORD_RE = re.compile(r"[^\w.'&+-]+")


def _keywords(text: Any) -> List[str]:
    """Split a spec value into search words. A leading '-' would mean 'exclude', so it is dropped."""
    words = (word.lstrip("-") for word in _NOT_KEYWORD_RE.sub(" ", str(text or "")).split())
    return [word for word in words if word]


def _fit(words: List[str], limit: int) -> str:
    """Join as many whole words as fit: half a word would search for something else."""
    kept: List[str] = []
    used = 0
    for word in words:
        cost = len(word) + (1 if kept else 0)
        if used + cost > limit:
            break
        kept.append(word)
        used += cost
    return " ".join(kept)


def build_query(spec: Mapping[str, Any]) -> str:
    """Brand and model, narrowed by the first style code, or else by the colorway.

    A style code names the exact product, so it is preferred over a colorway name. The size and
    the price ceiling are never part of the query.
    """
    colorway = spec.get("colorway") or {}
    codes = spec.get("style_codes") or []
    nicknames = colorway.get("nicknames") or []
    narrowing = codes[0] if codes else (nicknames[0] if nicknames else colorway.get("official"))
    words: List[str] = []
    for piece in (spec.get("brand"), spec.get("model"), narrowing):
        for word in _keywords(piece):
            # "Nike" + "Nike Dunk Low" should not search for "Nike Nike Dunk Low".
            if word.lower() not in (kept.lower() for kept in words):
                words.append(word)
    return _fit(words, MAX_QUERY_CHARS)


def build_plan(spec: Mapping[str, Any], ebay: Optional[Any] = None) -> Dict[str, Any]:
    """The search a watch runs: {"q", "category_id", "buying_options"}.

    With an eBay client, one extra search asks eBay which category dominates the results, so
    later searches can be limited to it. That is a refinement, not a requirement: if the call
    fails for any reason the plan stays keyword-only and the watch still works.
    """
    plan: Dict[str, Any] = {
        "q": build_query(spec),
        "category_id": None,
        "buying_options": list(spec.get("buying_options") or BUYING_OPTIONS),
    }
    if ebay is None or not plan["q"]:
        return plan
    try:
        probe = ebay.search(
            plan["q"], buying_options=plan["buying_options"], limit=1,
            fieldgroups="ASPECT_REFINEMENTS", purpose="plan",
        )
    except EbayError:
        return plan
    plan["category_id"] = probe.get("dominant_category_id")
    return plan
