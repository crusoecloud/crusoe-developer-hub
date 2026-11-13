"""eBay Browse API client: application token, search and listing detail.

eBay is read only through its official API. The token is an application token
limited to the public Browse scope, so this client can search and read listings
and nothing else: it has no way to bid, make an offer or buy.

Raw API responses are reduced to the Summary and Detail shapes the rest of the
tool uses. Both are built key by key from an allowlist, which is how the seller
block (and anything else not named here) is left behind.
"""
from __future__ import annotations

import base64
import hashlib
import html
import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

from .config import Settings
from .errors import (
    EbayAuthError,
    EbayBudgetExceeded,
    EbayError,
    EbayGroupRequired,
    EbayNotFound,
    EbayUnavailable,
)
from .http import Response, TransportError, UrllibTransport
from .spec import BUYING_OPTIONS
from .store import Store, iso, parse_iso

Money = Dict[str, str]
Summary = Dict[str, Any]
Detail = Dict[str, Any]

BROWSE_PATH = "/buy/browse/v1"
TOKEN_PATH = "/identity/v1/oauth2/token"
# The only scope ever requested: public Browse reads, nothing on behalf of a user.
TOKEN_SCOPE = "https://api.ebay.com/oauth/api_scope"
TOKEN_KEY = "ebay:token"
# A token this close to expiry is replaced, so a request never carries one that dies in flight.
TOKEN_MARGIN = timedelta(seconds=60)
DEFAULT_TOKEN_LIFETIME = 7200

MAX_Q_CHARS = 100
MAX_LIMIT = 200
# eBay serves at most this many results for one query, however large `total` is.
MAX_RESULTS = 10000
MAX_DESCRIPTION_CHARS = 4000
MAX_ERROR_TEXT_CHARS = 300
# A listing whose origin date is more than this much older than its creation date is a relist.
RELIST_GAP = timedelta(seconds=60)
AVAILABILITY = ("IN_STOCK", "LIMITED_STOCK", "OUT_OF_STOCK")

# Browse error ids that say more than the HTTP status does.
ERROR_ID_TYPES: Dict[int, Type[EbayError]] = {
    11001: EbayNotFound,        # item id not found
    11003: EbayNotFound,        # legacy item id not found
    11004: EbayUnavailable,     # not available for purchase; often temporary
    11006: EbayGroupRequired,   # legacy id is a multi-variation parent
}

# Top-level item fields that repeat item specifics. Sellers fill in one or the other.
# Gender is folded in twice: under its own name, and as "Department", which is the item
# specific `rules.check_department` reads.
FOLDED_ASPECTS = (
    ("brand", "Brand"), ("color", "Color"), ("size", "Size"), ("gender", "Gender"), ("gender", "Department"),
    ("mpn", "MPN"),
)

_GROUP_ID_RE = re.compile(r"item_group_id=([A-Za-z0-9_-]+)")
_REDACTED = "[redacted]"


class EbayClient:
    """Reads listings through the Browse API and counts every call against the daily cap.

    `transport` is anything with the interface of `http.UrllibTransport`. Time comes from
    `store.clock`, so tests can move it.
    """

    def __init__(self, settings: Settings, store: Store, transport: Any = None) -> None:
        self.settings = settings
        self.store = store
        self.transport = transport if transport is not None else UrllibTransport()

    # ---- token -------------------------------------------------------------

    def token(self) -> str:
        """An application access token, reused from the store until a minute before it expires."""
        cached = self.store.kv_get(TOKEN_KEY)
        if self._still_valid(cached):
            return cached["access_token"]
        return self._mint()

    def _still_valid(self, cached: Any) -> bool:
        if not isinstance(cached, dict) or not cached.get("access_token"):
            return False
        if cached.get("keyset") != self._keyset_tag():
            return False
        expires_at = parse_iso(cached.get("expires_at"))
        return expires_at is not None and self.store.clock() < expires_at - TOKEN_MARGIN

    def _keyset_tag(self) -> str:
        """Ties a cached token to the keyset and environment it was minted for.

        Without it, moving from sandbox keys to production keys would keep sending the
        old token for up to two hours. A hash is stored so the client id is not.
        """
        source = "%s\n%s" % (self.settings.ebay_api_base, self.settings.ebay_client_id)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def _basic_credential(self) -> str:
        pair = "%s:%s" % (self.settings.ebay_client_id, self.settings.ebay_client_secret)
        return base64.b64encode(pair.encode("utf-8")).decode("ascii")

    def _mint(self) -> str:
        """Request a new token with the client-credentials grant and cache it."""
        self.settings.require("ebay")
        form = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": TOKEN_SCOPE})
        headers = {
            "Authorization": "Basic " + self._basic_credential(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            response = self.transport.request(
                "POST", self.settings.ebay_api_base + TOKEN_PATH, headers=headers, body=form.encode("ascii")
            )
        except TransportError as exc:
            raise EbayUnavailable(self._redact("eBay token request failed: %s" % exc)) from None
        finally:
            # Kept apart from Browse calls: token requests do not count against the daily cap.
            self.store.ledger_add("ebay_token", "mint")
        if not response.ok:
            raise self._error(response, "token request")
        data = self._json_object(response, "token request")
        token = data.get("access_token")
        if not token or not isinstance(token, str):
            raise EbayError("eBay token request returned no access token", status=response.status)
        lifetime = _int(data.get("expires_in")) or DEFAULT_TOKEN_LIFETIME
        self.store.kv_set(TOKEN_KEY, {
            "access_token": token,
            "expires_at": iso(self.store.clock() + timedelta(seconds=lifetime)),
            "keyset": self._keyset_tag(),
        })
        return token

    # ---- search ------------------------------------------------------------

    def search(
        self,
        q: Optional[str] = None,
        *,
        category_id: Optional[str] = None,
        epid: Optional[str] = None,
        gtin: Optional[str] = None,
        buying_options: Optional[Sequence[str]] = None,
        start_from: Optional[datetime] = None,
        start_to: Optional[datetime] = None,
        extra_filters: Optional[Mapping[str, str]] = None,
        sort: Optional[str] = None,
        limit: int = MAX_LIMIT,
        offset: int = 0,
        fieldgroups: Optional[str] = None,
        purpose: str = "search",
    ) -> Dict[str, Any]:
        """One page of `item_summary/search`.

        Returns {"total", "items": [Summary], "next_offset", "dominant_category_id", "warnings"}.
        The buyingOptions filter is always sent, because without it eBay returns Buy It Now
        listings only and an auction would never be seen. `limit` is brought into eBay's
        1-200 range and `offset` down to a multiple of it. `dominant_category_id` is only
        present when `fieldgroups` asks for refinements, and a refinement-only fieldgroup
        may come back without items.
        """
        params: Dict[str, str] = {}
        notes: List[str] = []
        if q and q.strip():
            params["q"] = _clip_query(q)
            if params["q"] != q.strip():
                notes.append("q was cut to %d characters" % len(params["q"]))
        for name, value in (("category_ids", category_id), ("epid", epid), ("gtin", gtin)):
            if value:
                params[name] = str(value)
        if not params:
            raise ValueError("search needs q, category_id, epid or gtin")
        params["filter"] = _filter_value(buying_options, start_from, start_to, extra_filters)
        if sort:
            params["sort"] = sort
        limit = min(max(int(limit), 1), MAX_LIMIT)
        offset = max(int(offset), 0) // limit * limit
        params["limit"] = str(limit)
        params["offset"] = str(offset)
        if fieldgroups:
            params["fieldgroups"] = fieldgroups

        raw = self._browse("/item_summary/search", params, purpose, "search")
        total = _int(raw.get("total")) or 0
        return {
            "total": total,
            "items": [summarize(row) for row in _dicts(raw.get("itemSummaries"))],
            "next_offset": _next_offset(total, offset, limit),
            "dominant_category_id": _dominant_category_id(raw),
            "warnings": notes + _messages(raw.get("warnings")),
        }

    # ---- listing detail ----------------------------------------------------

    def get_item(self, item_id: str, *, compact: bool = False, purpose: str = "detail") -> Detail:
        """One listing by its Browse item id.

        `compact=True` asks for the COMPACT fieldgroup: price, bids and availability. eBay
        leaves out the title, buying options, end date and image in that form, so those
        keys come back empty and must not overwrite a stored snapshot.
        """
        params = {"fieldgroups": "COMPACT"} if compact else {}
        path = "/item/" + urllib.parse.quote(str(item_id), safe="")
        return detail_from(self._browse(path, params, purpose, "item lookup"))

    def get_item_by_legacy_id(
        self, legacy_item_id: str, *, legacy_variation_id: Optional[str] = None, purpose: str = "detail"
    ) -> Detail:
        """One listing by the numeric id seen in an eBay link.

        Raises EbayGroupRequired when the id is the parent of a multi-variation listing;
        the caller then reads the item group instead.
        """
        params = {"legacy_item_id": str(legacy_item_id)}
        if legacy_variation_id:
            params["legacy_variation_id"] = str(legacy_variation_id)
        return detail_from(self._browse("/item/get_item_by_legacy_id", params, purpose, "legacy item lookup"))

    def get_items_by_item_group(self, item_group_id: str, *, purpose: str = "detail") -> List[Detail]:
        """Every variation of a multi-variation listing (one per size, for example).

        eBay returns a description shared by several variations once, in
        `commonDescriptions`, so it is put back on each variation it belongs to.
        """
        raw = self._browse(
            "/item/get_items_by_item_group", {"item_group_id": str(item_group_id)}, purpose, "item group lookup"
        )
        shared = _shared_descriptions(raw)
        details: List[Detail] = []
        for item in _dicts(raw.get("items")):
            if not item.get("description"):
                item = dict(item, description=shared.get(_text(item.get("itemId")) or ""))
            detail = detail_from(item)
            detail["item_group_id"] = detail["item_group_id"] or str(item_group_id)
            details.append(detail)
        return details

    # ---- budget ------------------------------------------------------------

    def calls_today(self) -> int:
        """Browse calls sent so far today (UTC). Token requests are not included."""
        return self.store.ledger_total("ebay")

    def remaining_today(self) -> int:
        return max(0, self.settings.ebay_daily_cap - self.calls_today())

    def _check_budget(self) -> None:
        used, cap = self.calls_today(), self.settings.ebay_daily_cap
        if used >= cap:
            raise EbayBudgetExceeded("eBay daily call cap reached (%d of %d); no request was sent" % (used, cap))

    # ---- transport ---------------------------------------------------------

    def _headers(self, token: str) -> Dict[str, str]:
        headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": self.settings.ebay_marketplace,
        }
        if self.settings.ebay_zip and self.settings.ebay_marketplace == "EBAY_US":
            # Lets eBay work out the shipping cost to the buyer's area.
            location = "country=US,zip=%s" % self.settings.ebay_zip
            headers["X-EBAY-C-ENDUSERCTX"] = "contextualLocation=" + urllib.parse.quote(location, safe="")
        return headers

    def _browse(self, path: str, params: Mapping[str, str], purpose: str, action: str) -> Dict[str, Any]:
        """One GET against the Browse API: budget check, token, request, ledger, error mapping."""
        self._check_budget()
        token = self.token()
        url = self.settings.ebay_api_base + BROWSE_PATH + path
        if params:
            url += "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote, safe="")
        try:
            response = self.transport.request("GET", url, headers=self._headers(token))
        except TransportError as exc:
            raise EbayUnavailable(self._redact("eBay %s failed: %s" % (action, exc), token)) from None
        finally:
            # Counted whatever the outcome: a request that failed on the way may still have reached eBay.
            self.store.ledger_add("ebay", purpose)
        if response.status == 401:
            # eBay no longer accepts this token. Forget it so the next call mints a new one
            # instead of repeating a dead token until its two hours are up.
            self.store.kv_delete(TOKEN_KEY)
        if not response.ok:
            raise self._error(response, action, token)
        return self._json_object(response, action)

    def _json_object(self, response: Response, action: str) -> Dict[str, Any]:
        data = _json_or_none(response)
        if not isinstance(data, dict):
            # A success status with something else in the body, such as a captive portal page.
            raise EbayUnavailable(
                "eBay %s returned a response that is not a JSON object" % action, status=response.status
            )
        return data

    def _error(self, response: Response, action: str, *secrets: str) -> EbayError:
        """The exception for a failed response, with a message that is safe to log."""
        body = _json_or_none(response)
        rows = _dicts(body.get("errors")) if isinstance(body, dict) else []
        ids = _error_ids(rows)
        message = "eBay %s failed: HTTP %d" % (action, response.status)
        if ids:
            message += ", error %s" % ", ".join(str(error_id) for error_id in ids)
        # Redacted before it is shortened, so a cut can never leave half a credential behind.
        said = self._redact(_what_ebay_said(body, rows), *secrets)[:MAX_ERROR_TEXT_CHARS]
        if said:
            message += ": " + said
        kind = _error_type(response.status, ids)
        if kind is EbayGroupRequired:
            return EbayGroupRequired(
                message, item_group_id=_group_id_from_errors(rows), status=response.status, error_ids=ids
            )
        return kind(message, status=response.status, error_ids=ids)

    def _redact(self, text: str, *secrets: str) -> str:
        """Remove credentials from text that came from eBay or the network before it is raised."""
        for secret in (self.settings.ebay_client_secret, self._basic_credential()) + secrets:
            if secret:
                text = text.replace(secret, _REDACTED)
        return text


# ---- response shapes ---------------------------------------------------------

def summarize(raw: Mapping[str, Any]) -> Summary:
    """Reduce a raw search result or item to the Summary shape.

    Every key is copied by name. Nothing else in the response survives, so the seller
    block is never stored, shown or sent to a model.
    """
    created_at = _iso_or_none(raw.get("itemCreationDate"))
    origin_at = _iso_or_none(raw.get("itemOriginDate"))
    return {
        "item_id": _text(raw.get("itemId")),
        "legacy_item_id": _text(raw.get("legacyItemId")),
        "title": _text(raw.get("title")) or "",
        "price": _money(raw.get("price")),
        "current_bid": _money(raw.get("currentBidPrice")),
        "bid_count": _int(raw.get("bidCount")),
        "buying_options": _strings(raw.get("buyingOptions")),
        "shipping": _first_shipping_cost(raw),
        "end_at": _iso_or_none(raw.get("itemEndDate")),
        "created_at": created_at,
        "origin_at": origin_at,
        "relisted": _is_relisted(origin_at, created_at),
        "image_url": _image_url(raw.get("image")),
        "additional_image_urls": _strings([_image_url(image) for image in _dicts(raw.get("additionalImages"))]),
        "condition": _text(raw.get("condition")),
        "condition_id": _text(raw.get("conditionId")),
        "item_group_id": _item_group_id(raw),
        "web_url": _text(raw.get("itemWebUrl")),
        "category_ids": _category_ids(raw),
    }


def detail_from(raw: Mapping[str, Any]) -> Detail:
    """Reduce a raw item to the Detail shape: every Summary key plus what only an item carries."""
    description = strip_html(raw.get("description") or raw.get("shortDescription"))
    reserve_met = raw.get("reservePriceMet")
    detail = summarize(raw)
    detail.update({
        "aspects": _aspects(raw),
        "description": description[:MAX_DESCRIPTION_CHARS].rstrip(),
        "availability": _availability(raw),
        "min_next_bid": _money(raw.get("minimumPriceToBid")),
        "unique_bidders": _int(raw.get("uniqueBidderCount")),
        "reserve_met": reserve_met if isinstance(reserve_met, bool) else None,
    })
    return detail


# Tags that end a line or a cell become a space. Inline tags such as <b> and <span> simply
# vanish, so "DD1391<b>-100</b>" stays one word and a quote from the text can still be found.
_BLOCK_TAGS = (
    "address", "article", "aside", "blockquote", "br", "caption", "dd", "div", "dl", "dt", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "ol", "p", "pre", "section", "table",
    "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul",
)
# Scripts, style sheets and comments, closed or cut off: text a reader never sees.
_UNSEEN_RE = re.compile(r"<(script|style)\b.*?(?:</\1\s*>|\Z)|<!--.*?(?:-->|\Z)", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG_RE = re.compile(r"</?(?:%s)\b[^<>]*>" % "|".join(_BLOCK_TAGS), re.IGNORECASE)
# A tag starts with a letter, "/", "!" or "?" right after the bracket and is closed before the
# next one opens, so a stray "<" in a sentence ("waist < 32") does not swallow what follows it.
_TAG_RE = re.compile(r"<[a-zA-Z/!?][^<>]*>")


def strip_html(text: Any) -> str:
    """Seller HTML as plain text on one line: tags removed, entities decoded, whitespace collapsed.

    Done with patterns rather than a parser because listing HTML is untrusted and often
    malformed, and this must never raise.
    """
    if not text:
        return ""
    visible = _UNSEEN_RE.sub(" ", str(text))
    visible = _BLOCK_TAG_RE.sub(" ", visible)
    visible = _TAG_RE.sub("", visible)
    return " ".join(html.unescape(visible).split())


# ---- search parameters -------------------------------------------------------

def _clip_query(q: str) -> str:
    """eBay accepts at most 100 characters in q. Cut at a word boundary so no half word is searched."""
    text = q.strip()
    if len(text) <= MAX_Q_CHARS:
        return text
    clipped = text[:MAX_Q_CHARS]
    if not text[MAX_Q_CHARS].isspace() and " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip()


def _filter_value(
    buying_options: Optional[Sequence[str]],
    start_from: Optional[datetime],
    start_to: Optional[datetime],
    extra_filters: Optional[Mapping[str, str]],
) -> str:
    """The `filter` parameter: `name:value` pairs joined with commas.

    A filter passed in `extra_filters` under a name that has its own argument is ignored.
    """
    filters = {"buyingOptions": "{%s}" % "|".join(_buying_options(buying_options))}
    if start_from is not None or start_to is not None:
        filters["itemStartDate"] = "[%s..%s]" % (_filter_time(start_from), _filter_time(start_to))
    for name, value in (extra_filters or {}).items():
        filters.setdefault(name, value)
    return ",".join("%s:%s" % pair for pair in filters.items())


def _buying_options(options: Optional[Sequence[str]]) -> List[str]:
    chosen = list(dict.fromkeys(str(option).upper() for option in (options or BUYING_OPTIONS)))
    unknown = [option for option in chosen if option not in BUYING_OPTIONS]
    if unknown:
        raise ValueError("unknown buying option: %s" % ", ".join(unknown))
    return chosen


def _filter_time(moment: Optional[datetime]) -> str:
    """One end of a date range, in UTC. An open end is written as nothing."""
    if moment is None:
        return ""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("search window datetimes must be timezone-aware")
    return iso(moment)


def _next_offset(total: int, offset: int, limit: int) -> Optional[int]:
    following = offset + limit
    return following if following < min(total, MAX_RESULTS) else None


def _dominant_category_id(raw: Mapping[str, Any]) -> Optional[str]:
    refinement = raw.get("refinement")
    return _text(refinement.get("dominantCategoryId")) if isinstance(refinement, dict) else None


# ---- errors ------------------------------------------------------------------

def _error_type(status: int, error_ids: Sequence[int]) -> Type[EbayError]:
    """An error id is more specific than the status, so it is looked at first."""
    for error_id in error_ids:
        if error_id in ERROR_ID_TYPES:
            return ERROR_ID_TYPES[error_id]
    if status in (401, 403):
        return EbayAuthError
    if status == 404:
        return EbayNotFound
    if status == 429 or status >= 500:
        return EbayUnavailable
    return EbayError


def _error_ids(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    ids = (_int(row.get("errorId")) for row in rows)
    return [error_id for error_id in ids if error_id is not None]


def _what_ebay_said(body: Any, rows: Sequence[Mapping[str, Any]]) -> str:
    """One line from the error body, in either the API shape or the OAuth shape."""
    if rows:
        return str(rows[0].get("message") or rows[0].get("longMessage") or "")
    if isinstance(body, dict) and body.get("error"):
        return ": ".join(str(body[key]) for key in ("error", "error_description") if body.get(key))
    return ""


def _group_id_from_errors(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Error 11006 points at the item group with an href in its message or in a parameter."""
    for row in rows:
        texts = [row.get("message"), row.get("longMessage")]
        texts += [parameter.get("value") for parameter in _dicts(row.get("parameters"))]
        for text in texts:
            found = _group_id_in(text)
            if found:
                return found
    return None


def _group_id_in(text: Any) -> Optional[str]:
    """The id in a `...get_items_by_item_group?item_group_id=XXXX` href. The href is read, never requested."""
    match = _GROUP_ID_RE.search(str(text or ""))
    return match.group(1) if match else None


def _json_or_none(response: Response) -> Any:
    """The decoded body, or None when it is empty or not JSON (error pages are sometimes HTML)."""
    try:
        return response.json()
    except ValueError:
        return None


# ---- field readers -----------------------------------------------------------

def _dicts(value: Any) -> List[Dict[str, Any]]:
    """The dict rows of a list, or [] when the API sent something else there."""
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> List[str]:
    """The non-empty values of a list as strings, or [] when the API sent something else there."""
    texts = (_text(item) for item in value) if isinstance(value, list) else ()
    return [text for text in texts if text]


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _money(value: Any) -> Optional[Money]:
    """{"value", "currency"} with the amount kept as the string eBay sent, or None if either is missing."""
    if not isinstance(value, dict) or value.get("value") in (None, "") or not value.get("currency"):
        return None
    return {"value": str(value["value"]), "currency": str(value["currency"])}


def _messages(rows: Any) -> List[str]:
    return [str(row["message"]) for row in _dicts(rows) if row.get("message")]


def _iso_or_none(value: Any) -> Optional[str]:
    moment = parse_iso(value) if isinstance(value, str) else None
    return iso(moment) if moment else None


def _is_relisted(origin_at: Optional[str], created_at: Optional[str]) -> bool:
    """True when the item was first listed (origin) clearly before this listing was created.

    Both dates come from the live listing, so a relist is labeled without keeping any history.
    """
    origin, created = parse_iso(origin_at), parse_iso(created_at)
    return bool(origin and created and created - origin > RELIST_GAP)


def _image_url(image: Any) -> Optional[str]:
    return _text(image.get("imageUrl")) if isinstance(image, dict) else None


def _first_shipping_cost(raw: Mapping[str, Any]) -> Optional[Money]:
    """The cost of the first shipping option. None when eBay gave no figure (calculated shipping)."""
    options = _dicts(raw.get("shippingOptions"))
    return _money(options[0].get("shippingCost")) if options else None


def _item_group_id(raw: Mapping[str, Any]) -> Optional[str]:
    """Search results carry an `itemGroupHref`; a full item carries `primaryItemGroup`."""
    group = raw.get("primaryItemGroup")
    group = group if isinstance(group, dict) else {}
    return (
        _text(group.get("itemGroupId"))
        or _group_id_in(raw.get("itemGroupHref"))
        or _group_id_in(group.get("itemGroupHref"))
    )


def _category_ids(raw: Mapping[str, Any]) -> List[str]:
    """Leaf category first, then its ancestors, from whichever fields the response has."""
    found = _strings(raw.get("leafCategoryIds"))
    found += _strings([row.get("categoryId") for row in _dicts(raw.get("categories"))])
    found += _strings([raw.get("categoryId")])
    found += _strings(str(raw.get("categoryIdPath") or "").split("|"))
    return list(dict.fromkeys(found))


def _aspects(raw: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Item specifics as {name: [values]}.

    The top-level brand, color, size, gender and mpn are added under their usual names
    when no item specific of that name exists.
    """
    aspects: Dict[str, List[str]] = {}
    for row in _dicts(raw.get("localizedAspects")):
        name, value = _text(row.get("name")), _text(row.get("value"))
        if not name or not value:
            continue
        values = aspects.setdefault(name, [])
        if value not in values:
            values.append(value)
    present = {name.lower() for name in aspects}
    for key, name in FOLDED_ASPECTS:
        value = _text(raw.get(key))
        if value and name.lower() not in present:
            aspects[name] = [value]
    return aspects


def _availability(raw: Mapping[str, Any]) -> Optional[str]:
    rows = _dicts(raw.get("estimatedAvailabilities"))
    status = rows[0].get("estimatedAvailabilityStatus") if rows else None
    return status if status in AVAILABILITY else None


def _shared_descriptions(raw: Mapping[str, Any]) -> Dict[str, str]:
    """{item id: description} from an item group's `commonDescriptions`."""
    shared: Dict[str, str] = {}
    for row in _dicts(raw.get("commonDescriptions")):
        if row.get("description"):
            for item_id in _strings(row.get("itemIds")):
                shared[item_id] = row["description"]
    return shared
