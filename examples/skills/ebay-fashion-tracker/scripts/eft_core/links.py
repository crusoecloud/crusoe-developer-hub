"""Read eBay links without fetching them.

eBay's User Agreement prohibits automated access to its pages, so a link is only
ever parsed as text. What it points at is then looked up through the Browse API.
Short links cannot be resolved without a request, so the user is asked for the full URL.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DOMAIN_MARKETPLACE = {
    "ebay.com": "EBAY_US",
    "ebay.co.uk": "EBAY_GB",
    "ebay.de": "EBAY_DE",
    "ebay.ca": "EBAY_CA",
    "ebay.com.au": "EBAY_AU",
    "ebay.fr": "EBAY_FR",
    "ebay.it": "EBAY_IT",
    "ebay.es": "EBAY_ES",
    "ebay.nl": "EBAY_NL",
    "ebay.ie": "EBAY_IE",
    "ebay.at": "EBAY_AT",
    "ebay.ch": "EBAY_CH",
    "ebay.pl": "EBAY_PL",
    "ebay.be": "EBAY_BE",
}
SHORT_HOSTS = ("ebay.us", "ebay.to")

_ITEM_ID_RE = re.compile(r"^\d{9,}$")


@dataclass
class EbayLink:
    """What a URL refers to.

    kind: item | product | search | short | other_ebay | not_ebay
    """
    kind: str
    original: str
    marketplace: Optional[str] = None
    legacy_item_id: Optional[str] = None
    variation_id: Optional[str] = None      # from ?var=; a hint, not a documented mapping
    epid: Optional[str] = None
    search: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ebay(self) -> bool:
        return self.kind != "not_ebay"


def _site_domain(host: str) -> Optional[str]:
    host = host.lower()
    for domain in sorted(DOMAIN_MARKETPLACE, key=len, reverse=True):
        if host == domain or host.endswith("." + domain):
            return domain
    return None


def _first(params: Dict[str, list], name: str) -> Optional[str]:
    values = params.get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def parse_link(url: str) -> EbayLink:
    """Classify a URL. Never performs a network request."""
    original = (url or "").strip()
    candidate = original if "://" in original else "https://" + original
    parts = urllib.parse.urlsplit(candidate)
    host = (parts.hostname or "").lower()
    if not host:
        return EbayLink(kind="not_ebay", original=original)

    if host in SHORT_HOSTS or any(host.endswith("." + short) for short in SHORT_HOSTS):
        return EbayLink(kind="short", original=original)

    domain = _site_domain(host)
    if domain is None:
        return EbayLink(kind="not_ebay", original=original)

    marketplace = DOMAIN_MARKETPLACE[domain]
    params = urllib.parse.parse_qs(parts.query, keep_blank_values=False)
    segments = [urllib.parse.unquote(s) for s in parts.path.split("/") if s]

    if segments and segments[0] == "itm":
        item_id = None
        for segment in reversed(segments[1:]):
            if _ITEM_ID_RE.match(segment):
                item_id = segment
                break
        if item_id is None:
            item_id = _first(params, "item")
            if item_id and not _ITEM_ID_RE.match(item_id):
                item_id = None
        if item_id:
            variation = _first(params, "var")
            if variation in (None, "0") or not variation.isdigit():
                variation = None
            return EbayLink(kind="item", original=original, marketplace=marketplace,
                            legacy_item_id=item_id, variation_id=variation)
        return EbayLink(kind="other_ebay", original=original, marketplace=marketplace)

    if len(segments) >= 2 and segments[0] == "p" and segments[-1].isdigit():
        return EbayLink(kind="product", original=original, marketplace=marketplace, epid=segments[-1])

    if segments and segments[0] == "sch":
        search: Dict[str, Any] = {}
        keywords = _first(params, "_nkw")
        if keywords:
            search["q"] = keywords
        category = _first(params, "_sacat")
        if category and category.isdigit() and category != "0":
            search["category_id"] = category
        buying = []
        if _first(params, "LH_Auction") == "1":
            buying.append("AUCTION")
        if _first(params, "LH_BIN") == "1":
            buying.append("FIXED_PRICE")
        if _first(params, "LH_BO") == "1":
            buying.append("BEST_OFFER")
        if buying:
            search["buying_options"] = buying
        for source, target in (("_udlo", "price_min"), ("_udhi", "price_max")):
            value = _first(params, source)
            if value:
                try:
                    search[target] = float(value)
                except ValueError:
                    pass
        conditions = _first(params, "LH_ItemCondition")
        if conditions:
            ids = [c for c in re.split(r"[|,]", conditions) if c.isdigit()]
            if ids:
                search["condition_ids"] = ids
        return EbayLink(kind="search", original=original, marketplace=marketplace, search=search)

    return EbayLink(kind="other_ebay", original=original, marketplace=marketplace)


def canonical_item_url(legacy_item_id: str, marketplace: str = "EBAY_US") -> str:
    """A clean listing URL for display when the API did not supply one."""
    domain = next((d for d, m in DOMAIN_MARKETPLACE.items() if m == marketplace), "ebay.com")
    return "https://www.%s/itm/%s" % (domain, legacy_item_id)
