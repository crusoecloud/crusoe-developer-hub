"""Alert cards and their delivery: Slack DMs, Pushover pushes, quiet hours and expiry.

Rules this module keeps (they come from eBay's API License Agreement and from what the
tool promises its user):
- Listing data always carries its "as of" time. A Slack card is rewritten without that
  data once it is six hours old, and every push with listing data expires from the
  phone after six hours.
- When a listing ends, every message about it becomes "Ended" and the stored alert
  payloads lose their listing content.
- Cards state facts from the API: no deal score, no price opinion, nothing from another
  site, and never a claim that an item is genuine. Links come from the API.
- No token or key appears in an error message.

The tool reports price and bid state. It never bids, offers or buys.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Settings
from .errors import EbayError
from .http import Response, TransportError, UrllibTransport
from .store import Store, iso, parse_iso

SIX_HOURS = 6 * 3600
SLACK_API = "https://slack.com/api/"
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

KINDS = ("new", "price", "ending", "restock", "ended", "digest", "system")
TIERS = ("P1", "P2", "P3")
# Kinds that report on a listing the user was already told about; they reply in its thread.
FOLLOW_UP_KINDS = ("price", "ending", "restock", "ended")

MAX_ATTEMPTS = 5
BURST_LIMIT = 3      # more P2 alerts than this for one watch in one run become one message
GROUP_LIMIT = 20     # entries shown in a digest or summary; Slack allows 50 blocks a message
SLACK_TEXT_MAX = 3000
PUSH_TITLE_MAX = 250
PUSH_MESSAGE_MAX = 1024
PUSH_URL_MAX = 512
PUSH_PRIORITY = {"P1": 1, "P2": 0, "P3": -1}
DIGEST_DAY_KEY = "alerts:digest_day"

WATCH_HINT = "Tap Watch on eBay for its own reminders"
EXPIRED_NOTE = "Details were removed after six hours. Open the listing for live details."
ENDED_NOTE = "This listing has ended and its details were removed."
SYSTEM_TITLE = "eBay Fashion Tracker"

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif")
_CURRENCY_SIGNS = {"USD": "$", "GBP": "£", "EUR": "€"}
_LABELS = {
    "new": "New match",
    "price": "Now within your price ceiling",
    "ending": "Ending soon",
    "restock": "Back in stock",
    "ended": "Ended",
    "digest": "Update",
}
# What is left of a payload once its listing has ended: who it was for, nothing about the listing.
_KEPT_AFTER_END = ("code", "watch_id", "watch_name", "tier", "as_of")

Payload = Dict[str, Any]
Refresher = Callable[[Dict[str, Any]], Optional[Payload]]


class DeliveryError(Exception):
    """One send on one channel failed. Names the channel and a short reason, never a secret."""

    def __init__(self, channel: str, reason: str) -> None:
        super().__init__("%s: %s" % (channel, reason))
        self.channel = channel
        self.reason = reason


# ---- payloads and tiers ------------------------------------------------------

def escape_mrkdwn(text: Any) -> str:
    """Escape the three characters Slack reads as markup control: & < >.

    Titles are written by sellers. Unescaped, one could forge a link or an @-mention
    inside a card.
    """
    value = "" if text is None else str(text)
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_payload(
    watch: Mapping[str, Any],
    candidate: Mapping[str, Any],
    listing: Optional[Mapping[str, Any]],
    *,
    kind: str,
    tier: str,
    proof: str = "",
    unconfirmed: Sequence[str] = (),
    note: Optional[str] = None,
    as_of: Optional[str] = None,
) -> Payload:
    """Everything a card needs, copied field by field from a listing snapshot.

    Fields are named one at a time so that nothing else in the snapshot, a seller
    identifier above all, can ride along into the outbox.
    """
    if kind not in KINDS:
        raise ValueError("unknown alert kind: %s" % kind)
    if tier not in TIERS:
        raise ValueError("unknown alert tier: %s" % tier)
    listing = listing or {}
    return {
        "code": candidate["code"],
        "watch_id": watch["id"],
        "watch_name": watch["name"],
        "kind": kind,
        "tier": tier,
        "title": listing.get("title") or "",
        "web_url": listing.get("web_url") or "",
        "image_url": listing.get("image_url"),
        "price": listing.get("price"),
        "shipping": listing.get("shipping"),
        "current_bid": listing.get("current_bid"),
        "min_next_bid": listing.get("min_next_bid"),
        "bid_count": listing.get("bid_count"),
        "reserve_met": listing.get("reserve_met"),
        "buying_options": list(listing.get("buying_options") or []),
        "end_at": listing.get("end_at") or candidate.get("end_at"),
        "condition": listing.get("condition"),
        "proof": proof,
        "unconfirmed": list(unconfirmed),
        "relisted": bool(listing.get("relisted")),
        "as_of": as_of or listing.get("fetched_at") or candidate.get("fetched_at"),
        "note": note,
    }


def tier_for(kind: str, watch: Mapping[str, Any], candidate: Optional[Mapping[str, Any]]) -> str:
    """How urgent an alert is.

    P1 may break quiet hours, so it is kept for what the user singled out (a hot watch
    or a starred listing) and can act on at once: a proven Buy It Now match, an auction
    about to end, a listing back in stock. Unproven matches and digest notes are P3 and
    wait for the daily digest. Everything else is P2.
    """
    candidate = candidate or {}
    singled_out = watch.get("tier") == "hot" or bool(candidate.get("starred"))
    proven = candidate.get("state") == "alerted"
    buy_now = "FIXED_PRICE" in ((candidate.get("listing") or {}).get("buying_options") or [])
    if singled_out and (kind in ("ending", "restock") or (kind == "new" and proven and buy_now)):
        return "P1"
    if kind == "digest" or candidate.get("state") == "maybe":
        return "P3"
    return "P2"


def in_quiet_hours(local_minutes: int, quiet_hours: Sequence[int]) -> bool:
    """True when a local time, in minutes after midnight, is inside the quiet window.

    The window runs from its start up to, not including, its end. 22:00-08:00 crosses
    midnight. Equal start and end means there are no quiet hours.
    """
    start, end = quiet_hours
    if start == end:
        return False
    if start < end:
        return start <= local_minutes < end
    return local_minutes >= start or local_minutes < end


# ---- wording -----------------------------------------------------------------

def _one_line(text: Any) -> str:
    """Collapse whitespace so seller text cannot start a line of its own on a card."""
    return " ".join(str("" if text is None else text).split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _with_footer(lines: Sequence[str], footer: str, limit: int) -> str:
    """Join lines under a length limit, clipping the body so the footer always survives."""
    body = "\n".join(line for line in lines if line)
    return _clip(body, limit - len(footer) - 1) + "\n" + footer


def _when(iso_text: Any, tz: Optional[tzinfo] = None) -> str:
    """'2026-11-14T21:14:00Z' -> 'Nov 14, 21:14 PST', in the machine's zone unless tz is given."""
    moment = parse_iso(iso_text if isinstance(iso_text, str) else None)
    if moment is None:
        return "an unknown time"
    local = moment.astimezone(tz)
    return ("%s %d, %s %s" % (local.strftime("%b"), local.day, local.strftime("%H:%M"), local.tzname() or "")).strip()


def _money(amount: Optional[Mapping[str, Any]]) -> str:
    """{'value': '120.00', 'currency': 'USD'} -> '$120.00'. The digits are eBay's, untouched."""
    if not amount or amount.get("value") in (None, ""):
        return ""
    currency = str(amount.get("currency") or "").upper()
    sign = _CURRENCY_SIGNS.get(currency)
    if sign:
        return "%s%s" % (sign, amount["value"])
    return ("%s %s" % (currency, amount["value"])).strip()


def _shipping(amount: Optional[Mapping[str, Any]]) -> str:
    text = _money(amount)
    if not text:
        return ""
    try:
        free = float(amount["value"]) == 0  # type: ignore[index]
    except (TypeError, ValueError):
        free = False
    return "free" if free else text


def _label(payload: Mapping[str, Any]) -> str:
    """What happened, in two or three words. An unproven match always says so."""
    kind = payload.get("kind")
    label = _LABELS.get(kind, "Update")  # type: ignore[arg-type]
    if not payload.get("unconfirmed"):
        return label
    return "Unconfirmed match" if kind == "new" else label + " (unconfirmed match)"


def _facts(payload: Mapping[str, Any], tz: Optional[tzinfo] = None) -> List[Tuple[str, str]]:
    """Price and bid state as (label, value) pairs, as eBay reported them.

    Facts only: nothing here compares the price with anything or says whether it is good.
    """
    options = payload.get("buying_options") or []
    auction = "AUCTION" in options
    price, bid = _money(payload.get("price")), _money(payload.get("current_bid"))
    next_bid, shipping = _money(payload.get("min_next_bid")), _shipping(payload.get("shipping"))
    facts: List[Tuple[str, str]] = []
    if auction:
        if bid:
            facts.append(("Current bid", bid))
        if payload.get("bid_count") is not None:
            facts.append(("Bids", str(payload["bid_count"])))
        if next_bid:
            facts.append(("Minimum next bid", next_bid))
        if payload.get("reserve_met") is not None:
            facts.append(("Reserve", "met" if payload["reserve_met"] else "not met"))
        # On an auction eBay's "price" repeats the current bid unless Buy It Now is offered too.
        if price and "FIXED_PRICE" in options:
            facts.append(("Buy It Now", price))
        elif price and not bid:
            facts.append(("Price", price))
    elif price:
        facts.append(("Price", price + (" or Best Offer" if "BEST_OFFER" in options else "")))
    if shipping:
        facts.append(("Shipping", shipping))
    if payload.get("end_at"):
        # The scheduled end is a lower bound for an auction, so the card never calls it final.
        end = "scheduled end %s" % _when(payload["end_at"], tz)
        facts.append(("Auction", end + ", may extend") if auction else ("Listing", end))
    if payload.get("condition"):
        facts.append(("Condition", _one_line(payload["condition"])))
    return facts


def _details(payload: Mapping[str, Any], tz: Optional[tzinfo] = None) -> List[Tuple[str, str]]:
    """The facts, then why the listing matched and what is still unproven."""
    details = _facts(payload, tz)
    if payload.get("proof"):
        details.append(("Proof", _one_line(payload["proof"])))
    if payload.get("unconfirmed"):
        details.append(("Not proven", ", ".join(str(name).replace("_", " ") for name in payload["unconfirmed"])))
    if payload.get("note"):
        details.append(("Note", _one_line(payload["note"])))
    return details


def _brief(payload: Mapping[str, Any], tz: Optional[tzinfo] = None) -> str:
    """The details on one line, for an entry in a digest or summary."""
    return " · ".join("%s: %s" % pair for pair in _details(payload, tz))


def _is_https(url: Any) -> bool:
    return isinstance(url, str) and url.startswith("https://")


def _link(payload: Mapping[str, Any]) -> str:
    """The listing title as a Slack link to the URL the API returned, or plain text without one."""
    title = escape_mrkdwn(_one_line(payload.get("title")))
    url = payload.get("web_url")
    if not _is_https(url):
        return title
    return "<%s|%s>" % (escape_mrkdwn(url).replace("|", "%7C"), title or "Open on eBay")


def _image_url(payload: Mapping[str, Any]) -> Optional[str]:
    """The photo URL, only when Slack can show it: a public https jpg, jpeg, png or gif."""
    url = payload.get("image_url")
    if not _is_https(url):
        return None
    path = urllib.parse.urlsplit(url).path.lower()
    return url if path.endswith(_IMAGE_SUFFIXES) else None


def _heading(payload: Mapping[str, Any]) -> str:
    """'*New match* · Panda Dunks 10 · `A17`' (Slack markup)."""
    return "*%s* · %s · `%s`" % (
        escape_mrkdwn(_label(payload)), escape_mrkdwn(payload.get("watch_name")), escape_mrkdwn(payload.get("code")),
    )


def _fallback_text(payload: Mapping[str, Any]) -> str:
    """Notification text. It carries no price: a banner on a lock screen cannot be expired."""
    text = "%s %s · %s" % (_label(payload), payload.get("code") or "", payload.get("watch_name") or "")
    title = _one_line(payload.get("title"))
    return escape_mrkdwn(text + (": " + title if title else ""))


def _section(text: str) -> Dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": _clip(text, SLACK_TEXT_MAX)}}


def _context(text: str) -> Dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": _clip(text, SLACK_TEXT_MAX)}]}


# ---- single cards --------------------------------------------------------------

def render_slack(payload: Mapping[str, Any], tz: Optional[tzinfo] = None) -> Dict[str, Any]:
    """A Slack card for one alert: {"text", "blocks"}.

    Times are shown in the machine's local zone; `tz` overrides that.
    """
    if payload.get("kind") == "ended":
        return render_slack_stub(payload, "ended")
    top = [_heading(payload), _link(payload) + (" (relisted)" if payload.get("relisted") else "")]
    blocks = [_section("\n".join(line for line in top if line))]
    details = ["*%s:* %s" % (label, escape_mrkdwn(value)) for label, value in _details(payload, tz)]
    if details:
        blocks.append(_section("\n".join(details)))
    image = _image_url(payload)
    if image:
        blocks.append({"type": "image", "image_url": image, "alt_text": "Listing photo"})
    blocks.append(_context("as of %s · not authenticated · %s" % (_when(payload.get("as_of"), tz), WATCH_HINT)))
    return {"text": _fallback_text(payload), "blocks": blocks}


def render_slack_stub(payload: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    """What a card becomes once its listing data may no longer be shown.

    "expired": the data is six hours old. The title and link stay, so the card still
    leads to the live listing. "ended": the listing is over and everything about it
    goes. Neither form has a price, a bid, a photo or an end time.
    """
    if reason == "ended":
        ended = dict(payload, kind="ended", unconfirmed=[])
        return {
            "text": escape_mrkdwn("Ended: %s · %s" % (payload.get("code") or "", payload.get("watch_name") or "")),
            "blocks": [_section("%s\n%s" % (_heading(ended), ENDED_NOTE))],
        }
    if reason != "expired":
        raise ValueError("unknown stub reason: %s" % reason)
    lines = [_heading(payload), _link(payload), "_%s_" % EXPIRED_NOTE]
    return {"text": _fallback_text(payload), "blocks": [_section("\n".join(line for line in lines if line))]}


def render_push(payload: Mapping[str, Any], tz: Optional[tzinfo] = None) -> Dict[str, Any]:
    """A Pushover message for one alert.

    The ttl makes the phone drop the push after six hours, which is how a push keeps
    the same freshness rule as a card that gets rewritten.
    """
    title = _clip("%s: %s (%s)" % (_label(payload), payload.get("watch_name") or "", payload.get("code") or ""),
                  PUSH_TITLE_MAX)
    url = ""
    if payload.get("kind") == "ended":
        message = ENDED_NOTE
    else:
        lines = [_one_line(payload.get("title"))] + ["%s: %s" % pair for pair in _details(payload, tz)]
        footer = "as of %s · not authenticated" % _when(payload.get("as_of"), tz)
        message = _with_footer(lines, footer, PUSH_MESSAGE_MAX)
        link = payload.get("web_url")
        # Pushover refuses a message whose URL is over its limit, so a long one is left out.
        if _is_https(link) and len(link) <= PUSH_URL_MAX:
            url = link
    return {
        "title": title,
        "message": message,
        "url": url,
        "url_title": "Open on eBay" if url else "",
        "priority": PUSH_PRIORITY.get(payload.get("tier"), 0),  # type: ignore[arg-type]
        "ttl": SIX_HOURS,
    }


# ---- digests and summaries -------------------------------------------------------

def _ended_payload(payload: Mapping[str, Any]) -> Payload:
    """The payload with its listing content gone, marked as ended."""
    out: Payload = {key: payload.get(key) for key in _KEPT_AFTER_END}
    out["kind"] = "ended"
    return out


def _fetch_times(payloads: Sequence[Mapping[str, Any]]) -> List[datetime]:
    """When each still-live entry was fetched, oldest first. Entries without a time are left out."""
    times = [parse_iso(p.get("as_of")) for p in payloads if p.get("kind") != "ended"]
    return sorted(moment for moment in times if moment is not None)


def _as_of_note(payloads: Sequence[Mapping[str, Any]], tz: Optional[tzinfo] = None) -> str:
    """One "as of" for a whole group: the oldest fetch time, "or later" when they differ."""
    times = _fetch_times(payloads)
    if not times:
        return "as of an unknown time"
    return "as of %s%s" % (_when(iso(times[0]), tz), " or later" if times[-1] > times[0] else "")


def _group_heading(payloads: Sequence[Mapping[str, Any]], style: str, hidden: int) -> str:
    count = len(payloads) + hidden
    if style == "digest":
        return "Daily digest: %d to review" % count
    watch_name = payloads[0].get("watch_name") if payloads else None
    return "%s: %d alerts" % (watch_name or "Watch", count)


def _group_entry(payload: Mapping[str, Any], style: str, show_data: bool, tz: Optional[tzinfo]) -> str:
    code = "`%s`" % escape_mrkdwn(payload.get("code"))
    watch = escape_mrkdwn(payload.get("watch_name"))
    if payload.get("kind") == "ended":
        return " · ".join([code, watch, "ended"])
    parts = [code, "*%s*" % escape_mrkdwn(_label(payload))]
    if style == "digest":
        parts.append(watch)
    parts.append(_link(payload))
    line = " · ".join(part for part in parts if part)
    brief = escape_mrkdwn(_brief(payload, tz)) if show_data else ""
    return line + "\n" + brief if brief else line


def _render_group(
    payloads: Sequence[Mapping[str, Any]],
    style: str,
    hidden: int = 0,
    show_data: bool = True,
    tz: Optional[tzinfo] = None,
) -> Dict[str, Any]:
    """One Slack message for several alerts: the daily digest, or a burst for one watch.

    `hidden` is how many more alerts there were than fit. With `show_data` off the
    entries keep their code and link but lose every price and bid detail.
    """
    heading = _group_heading(payloads, style, hidden)
    blocks = [_section("*%s*" % escape_mrkdwn(heading))]
    blocks.extend(_section(_group_entry(payload, style, show_data, tz)) for payload in payloads)
    notes = ["%d more not shown; ask Claude for your inbox" % hidden] if hidden else []
    if any(payload.get("kind") != "ended" for payload in payloads):
        if show_data:
            notes.extend([_as_of_note(payloads, tz), "not authenticated", WATCH_HINT])
        else:
            notes.append(EXPIRED_NOTE)
    if notes:
        blocks.append(_context(" · ".join(notes)))
    return {"text": escape_mrkdwn(heading), "blocks": blocks}


def _render_group_push(
    payloads: Sequence[Mapping[str, Any]], style: str, hidden: int = 0, tz: Optional[tzinfo] = None
) -> Dict[str, Any]:
    """The push for a digest or summary: one line per alert, and no link because there are several."""
    lines = [
        "%s %s: %s · %s" % (payload.get("code") or "", _label(payload), _one_line(payload.get("title")),
                            _brief(payload, tz))
        for payload in payloads
    ]
    if hidden:
        lines.append("+%d more" % hidden)
    footer = "%s · not authenticated" % _as_of_note(payloads, tz)
    return {
        "title": _clip(_group_heading(payloads, style, hidden), PUSH_TITLE_MAX),
        "message": _with_footer(lines, footer, PUSH_MESSAGE_MAX),
        "url": "",
        "url_title": "",
        "priority": PUSH_PRIORITY["P3" if style == "digest" else "P2"],
        "ttl": SIX_HOURS,
    }


# ---- delivery ----------------------------------------------------------------

class Notifier:
    """Sends what is in the alert outbox and keeps what was sent within the six-hour rule.

    `problems` collects one short line per failed send, so the poller can report them
    and hold back its heartbeat. Nothing in it is secret.
    """

    def __init__(
        self,
        settings: Settings,
        store: Store,
        transport: Any = None,
        local_now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.transport = transport or UrllibTransport()
        self.local_now = local_now or datetime.now
        self.problems: List[str] = []
        self._paused_until: Dict[str, datetime] = {}

    def channels(self) -> List[str]:
        """The channels that have their settings, in the order they are tried."""
        return [name for name in ("slack", "pushover") if self.settings.has(name)]

    # -- the outbox --

    def deliver_pending(self, refresher: Optional[Refresher] = None) -> Dict[str, int]:
        """Send what is due from the outbox. Returns counts of sent, held, dropped and failed.

        Timing: during quiet hours only P1 goes out, plus a P2 whose listing ends before
        the quiet window does. P3 alerts wait for the daily digest: the first run of a
        local day that is outside quiet hours and has something to put in it.

        Freshness: an alert that was held, or whose data is six hours old, is re-fetched
        through `refresher(alert)` before it is sent. That returns a fresh payload, or
        None when the listing is gone, in which case the alert is dropped. If it raises
        EbayError the listing could not be read just now and the alert stays held.

        Grouping: every P3 alert goes into one digest message. More than three P2
        alerts for one watch go out as one summary. Anything else is its own card, and
        a follow-up replies in the thread of the listing's first card.

        Failure: an alert counts as sent when at least one channel took it. Otherwise it
        gets an attempt and stays pending, and is marked failed after five. With no
        channel configured nothing is touched, so alerts wait instead of failing.

        Alerts of kind "system" are left alone: `send_system` delivers those, and their
        rows exist only so that the caller can de-duplicate them.
        """
        counts = {"sent": 0, "held": 0, "dropped": 0, "failed": 0}
        if not self.channels():
            return counts
        window = self._window()
        ready: List[Dict[str, Any]] = []
        for alert in self.store.alerts_by_status("pending", "held"):
            if alert["kind"] == "system":
                continue
            fate, alert = self._triage(alert, window, refresher)
            if fate == "ready":
                ready.append(alert)
            else:
                counts[fate] += 1
        for style, alerts in self._batches(ready):
            if not self._any_channel_open():
                break  # rate limited everywhere: the rest wait, without using up attempts
            if self._deliver(style, alerts):
                counts["sent"] += len(alerts)
                if style == "digest":
                    self.store.kv_set(DIGEST_DAY_KEY, self.local_now().date().isoformat())
            else:
                counts["failed"] += self._add_attempt(alerts)
        return counts

    def _window(self) -> Dict[str, Any]:
        """Where this run falls: in quiet hours or not, and whether today's digest is still due."""
        now = self.local_now()
        minutes = now.hour * 60 + now.minute
        end = self.settings.quiet_hours[1]
        quiet = in_quiet_hours(minutes, self.settings.quiet_hours)
        return {
            "quiet": quiet,
            "quiet_seconds_left": ((end - minutes) % 1440) * 60 - now.second if quiet else 0,
            "digest_due": not quiet and self.store.kv_get(DIGEST_DAY_KEY) != now.date().isoformat(),
        }

    def _triage(
        self, alert: Dict[str, Any], window: Mapping[str, Any], refresher: Optional[Refresher]
    ) -> Tuple[str, Dict[str, Any]]:
        """Decide one alert's fate for this run: "ready", "held" or "dropped"."""
        if self._must_wait(alert, window):
            return self._hold(alert), alert
        has_live_listing = alert["candidate_id"] is not None and alert["kind"] != "ended"
        needs_fresh_data = alert["status"] == "held" or self._is_stale(alert)
        if refresher is None or not (has_live_listing and needs_fresh_data):
            return "ready", alert
        try:
            fresh = refresher(alert)
        except EbayError as exc:
            self._note("refresh: %s" % exc.__class__.__name__)
            return self._hold(alert), alert
        if fresh is None:
            self.store.update_alert(alert["id"], status="dropped")
            return "dropped", alert
        self.store.update_alert(alert["id"], payload=fresh, as_of=fresh.get("as_of"))
        return "ready", dict(alert, payload=fresh, as_of=fresh.get("as_of") or alert["as_of"])

    def _must_wait(self, alert: Mapping[str, Any], window: Mapping[str, Any]) -> bool:
        if alert["tier"] == "P3":
            return not window["digest_due"]
        if not window["quiet"] or alert["tier"] == "P1":
            return False
        return not self._ends_within(alert["payload"], window["quiet_seconds_left"])

    def _ends_within(self, payload: Mapping[str, Any], seconds: float) -> bool:
        """True when the listing's scheduled end is still ahead but closer than `seconds`."""
        end = parse_iso(payload.get("end_at"))
        if end is None:
            return False
        return 0 <= (end - self.store.clock()).total_seconds() < seconds

    def _is_stale(self, alert: Mapping[str, Any]) -> bool:
        fetched = parse_iso(alert["payload"].get("as_of") or alert["as_of"])
        return fetched is not None and (self.store.clock() - fetched).total_seconds() >= SIX_HOURS

    def _hold(self, alert: Mapping[str, Any]) -> str:
        if alert["status"] != "held":
            self.store.update_alert(alert["id"], status="held")
        return "held"

    def _batches(self, ready: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Split the ready alerts into messages: single cards, per-watch summaries, one digest.

        P1 cards go first. The digest goes last because it is the least urgent.
        """
        digest = [alert for alert in ready if alert["tier"] == "P3"]
        cards = sorted((alert for alert in ready if alert["tier"] != "P3"), key=lambda a: a["tier"] != "P1")
        per_watch: Dict[Any, int] = {}
        for alert in cards:
            if alert["tier"] != "P1":
                per_watch[alert["watch_id"]] = per_watch.get(alert["watch_id"], 0) + 1
        batches: List[Tuple[str, List[Dict[str, Any]]]] = []
        summaries: Dict[Any, List[Dict[str, Any]]] = {}
        for alert in cards:
            watch_id = alert["watch_id"]
            if alert["tier"] == "P1" or per_watch[watch_id] <= BURST_LIMIT:
                batches.append(("single", [alert]))
                continue
            if watch_id not in summaries:
                summaries[watch_id] = []
                batches.append(("summary", summaries[watch_id]))
            summaries[watch_id].append(alert)
        if digest:
            batches.append(("digest", digest))
        return batches

    def _deliver(self, style: str, alerts: List[Dict[str, Any]]) -> bool:
        """Send one message for these alerts and record it. True when a channel took it."""
        shown = alerts[:GROUP_LIMIT]
        payloads = [alert["payload"] for alert in shown]
        group: Dict[str, Any] = {}
        thread = None
        if style == "single":
            card, push = render_slack(payloads[0]), render_push(payloads[0])
            thread = self._thread_for(alerts[0])
        else:
            hidden = len(alerts) - len(shown)
            card = _render_group(payloads, style, hidden)
            push = _render_group_push(payloads, style, hidden)
            # Kept with the message so that it can be rendered again when it is rewritten.
            group = {"alert_ids": [alert["id"] for alert in shown], "style": style, "hidden": hidden}
        sent = self._dispatch(card, push, thread)
        if sent:
            self._record(alerts, shown, sent, group)
        return bool(sent)

    def _thread_for(self, alert: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """A follow-up replies under the candidate's first Slack message: one listing, one thread."""
        if alert["kind"] not in FOLLOW_UP_KINDS or alert["candidate_id"] is None:
            return None
        for message in self.store.messages_for_candidate(alert["candidate_id"]):
            if message["channel"] == "slack":
                return message["ref"]
        return None

    def _dispatch(
        self, card: Mapping[str, Any], push: Mapping[str, Any], thread: Optional[Mapping[str, Any]] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Try every configured channel. Returns (channel, ref) for each one that worked."""
        sent: List[Tuple[str, Dict[str, Any]]] = []
        for channel in self.channels():
            try:
                ref = self.post_slack(card, thread) if channel == "slack" else self.post_push(push)
            except DeliveryError as exc:
                self._note(str(exc))
                continue
            sent.append((channel, ref))
        return sent

    def _record(
        self,
        alerts: Sequence[Dict[str, Any]],
        shown: Sequence[Dict[str, Any]],
        sent: Sequence[Tuple[str, Dict[str, Any]]],
        group: Mapping[str, Any],
    ) -> None:
        """Mark the alerts sent and remember where each shown one was posted, for expiry.

        A group message expires by its oldest entry, so no entry outlives six hours.
        """
        as_of = min(self._as_of(alert) for alert in shown)
        has_listing_data = any(alert["payload"].get("kind") != "ended" for alert in shown)
        with self.store.transaction():
            for alert in shown:
                for channel, ref in sent:
                    self.store.record_message(
                        alert["id"], channel, dict(ref, **group), candidate_id=alert["candidate_id"],
                        has_listing_data=has_listing_data, as_of=as_of,
                    )
            for alert in alerts:
                self.store.update_alert(alert["id"], status="sent")

    def _as_of(self, alert: Mapping[str, Any]) -> str:
        """When the alert's data was fetched, in the store's own time format so that it sorts."""
        fetched = parse_iso(alert["payload"].get("as_of") or alert["as_of"]) or parse_iso(alert["created_at"])
        return iso(fetched)  # type: ignore[arg-type]

    def _add_attempt(self, alerts: Sequence[Mapping[str, Any]]) -> int:
        """Count a failed send against each alert. Returns how many have now run out of attempts."""
        given_up = 0
        with self.store.transaction():
            for alert in alerts:
                exhausted = alert["attempts"] + 1 >= MAX_ATTEMPTS
                self.store.update_alert(alert["id"], status="failed" if exhausted else "pending", add_attempt=True)
                given_up += int(exhausted)
        return given_up

    # -- expiry and endings --

    def expire_messages(self) -> int:
        """Take listing data off every message that may no longer show it. Returns how many.

        That is a message whose data is six hours old, and a message whose listing has
        ended but whose "Ended" rewrite did not get through at the time. A Slack message
        is rewritten in place, and stays due if that fails. A push needs no call: its
        ttl already took it off the phone.
        """
        now = self.store.clock()
        cutoff = iso(now - timedelta(seconds=SIX_HOURS))
        still_showing_data = self.store.messages_due_for_expiry(iso(now))
        due = [m for m in still_showing_data if m["as_of"] <= cutoff or self._listing_ended(m)]
        expired, _ = self._strip(due)
        return expired

    def _listing_ended(self, message: Mapping[str, Any]) -> bool:
        alert = self.store.get_alert(message["alert_id"])
        return alert is not None and alert["payload"].get("kind") == "ended"

    def mark_ended(self, candidate_id: int) -> int:
        """A listing ended: rewrite its Slack messages as "Ended". Returns how many were rewritten.

        The stored payloads of its alerts lose their listing content here too, and any
        alert for it that was still waiting to go out is dropped.
        """
        with self.store.transaction():
            for alert in self.store.alerts_by_status():
                if alert["candidate_id"] != candidate_id:
                    continue
                # A queued "ended" notice is still worth sending; anything else would be old news.
                waiting = alert["status"] in ("pending", "held") and alert["kind"] != "ended"
                self.store.update_alert(
                    alert["id"], status="dropped" if waiting else None, payload=_ended_payload(alert["payload"])
                )
        _, rewritten = self._strip(self.store.messages_for_candidate(candidate_id))
        return rewritten

    def _strip(self, messages: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
        """Remove listing data from these messages.

        Returns (rows marked expired, Slack messages rewritten). Several rows can point
        at one Slack message (a digest); it is rewritten once.
        """
        outcome: Dict[Tuple[Any, Any], bool] = {}
        expired = 0
        for message in messages:
            if message["channel"] == "slack":
                where = (message["ref"].get("channel"), message["ref"].get("ts"))
                if where not in outcome:
                    outcome[where] = self._rewrite(message)
                if not outcome[where]:
                    continue
            self.store.mark_message_expired(message["id"])
            expired += 1
        return expired, sum(outcome.values())

    def _rewrite(self, message: Mapping[str, Any]) -> bool:
        """Resend a Slack message without its old data. True when it no longer shows any."""
        try:
            self.update_slack(message["ref"], self._replacement(message))
        except DeliveryError as exc:
            self._note(str(exc))
            # Deleted by the user: there is nothing left to take data off.
            return exc.reason == "message_not_found"
        return True

    def _replacement(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        """The card a message is rewritten to, built from what the store holds now."""
        ref = message["ref"]
        if ref.get("alert_ids"):
            alerts = [self.store.get_alert(alert_id) for alert_id in ref["alert_ids"]]
            payloads = [alert["payload"] for alert in alerts if alert]
            return _render_group(payloads, ref.get("style", "digest"), ref.get("hidden", 0),
                                 show_data=self._is_fresh(payloads))
        payload = self.store.get_alert(message["alert_id"])["payload"]
        return render_slack_stub(payload, "ended" if payload.get("kind") == "ended" else "expired")

    def _is_fresh(self, payloads: Sequence[Mapping[str, Any]]) -> bool:
        """A group may keep showing data only while its oldest live entry is under six hours old."""
        times = _fetch_times(payloads)
        live = [p for p in payloads if p.get("kind") != "ended"]
        if not times or len(times) < len(live):
            return False
        return (self.store.clock() - times[0]).total_seconds() < SIX_HOURS

    # -- safety notices --

    def send_system(self, text: str, urgent: bool = False) -> bool:
        """Send a safety notice now: push first, Slack if the push did not go. True when delivered.

        These say things like "the Crusoe key was rejected". They carry no listing
        data, so they are not expired, not recorded as messages and ignore quiet hours.
        """
        push = {"title": SYSTEM_TITLE, "message": _clip(text, PUSH_MESSAGE_MAX), "priority": 1 if urgent else 0}
        card = {"text": escape_mrkdwn(text), "blocks": [_section(escape_mrkdwn(text))]}
        for channel in ("pushover", "slack"):
            if channel not in self.channels():
                continue
            try:
                if channel == "pushover":
                    self.post_push(push)
                else:
                    self.post_slack(card)
            except DeliveryError as exc:
                self._note(str(exc))
                continue
            return True
        return False

    # -- the two channels --

    def post_slack(self, card: Mapping[str, Any], thread: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Post a card as a DM, or as a reply when `thread` is an earlier message's ref.

        Returns {"channel", "ts"}: what is needed to edit the message or thread under
        it. Raises DeliveryError. Links are never unfurled, so Slack does not fetch
        the eBay page to build a preview.
        """
        body: Dict[str, Any] = {
            "channel": thread["channel"] if thread else self.settings.slack_user_id,
            "text": card["text"],
            "blocks": card["blocks"],
            "unfurl_links": False,
            "unfurl_media": False,
        }
        if thread:
            body["thread_ts"] = thread["ts"]
        try:
            reply = self._slack("chat.postMessage", body)
        except DeliveryError as exc:
            without_image = [block for block in card["blocks"] if block.get("type") != "image"]
            if exc.reason != "invalid_blocks" or len(without_image) == len(card["blocks"]):
                raise
            # Slack could not use the photo. The card is worth sending without it.
            reply = self._slack("chat.postMessage", dict(body, blocks=without_image))
        return {"channel": reply.get("channel"), "ts": reply.get("ts")}

    def update_slack(self, ref: Mapping[str, Any], card: Mapping[str, Any]) -> None:
        """Replace a posted message. Slack wants the conversation id and the whole card again."""
        self._slack("chat.update", {
            "channel": ref.get("channel"), "ts": ref.get("ts"), "text": card["text"], "blocks": card["blocks"],
        })

    def post_push(self, push: Mapping[str, Any]) -> Dict[str, Any]:
        """Send a Pushover message. Returns {"request": id}. Raises DeliveryError."""
        if not self.settings.has("pushover"):
            raise DeliveryError("pushover", "not configured")
        form: Dict[str, Any] = {"token": self.settings.pushover_app_token, "user": self.settings.pushover_user_key}
        form.update({key: value for key, value in push.items() if value not in (None, "")})
        response, reply = self._post(
            "pushover", PUSHOVER_URL, {"Content-Type": "application/x-www-form-urlencoded"},
            urllib.parse.urlencode(form).encode("utf-8"),
        )
        if not response.ok or reply.get("status") != 1:
            raise DeliveryError("pushover", self._redact(_pushover_error(response, reply)))
        return {"request": reply.get("request")}

    def _slack(self, method: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.settings.has("slack"):
            raise DeliveryError("slack", "not configured")
        headers = {
            "Authorization": "Bearer " + self.settings.slack_bot_token,
            "Content-Type": "application/json; charset=utf-8",
        }
        response, reply = self._post("slack", SLACK_API + method, headers, json.dumps(body).encode("utf-8"))
        # Slack reports most failures as HTTP 200 with ok: false and an error code.
        if not reply.get("ok"):
            raise DeliveryError("slack", _slack_error(response, reply))
        return reply

    def _post(
        self, channel: str, url: str, headers: Dict[str, str], body: bytes
    ) -> Tuple[Response, Dict[str, Any]]:
        """POST and decode a JSON reply.

        A 429 is not retried here. The channel is left alone until its Retry-After has
        passed, which in practice means until a later poll.
        """
        if self._is_paused(channel):
            raise DeliveryError(channel, "rate limited, waiting")
        try:
            response = self.transport.request("POST", url, headers=headers, body=body)
        except TransportError:
            # The transport's own words are not repeated: they could quote the request.
            raise DeliveryError(channel, "could not be reached") from None
        if response.status == 429:
            self._pause(channel, response)
            raise DeliveryError(channel, "rate limited")
        try:
            reply = response.json()
        except ValueError:
            reply = None
        return response, reply if isinstance(reply, dict) else {}

    def _pause(self, channel: str, response: Response) -> None:
        headers = {name.lower(): value for name, value in response.headers.items()}
        wait = str(headers.get("retry-after", "")).strip()
        seconds = int(wait) if wait.isdigit() else 60
        self._paused_until[channel] = self.store.clock() + timedelta(seconds=seconds)

    def _is_paused(self, channel: str) -> bool:
        until = self._paused_until.get(channel)
        return until is not None and self.store.clock() < until

    def _any_channel_open(self) -> bool:
        return any(not self._is_paused(channel) for channel in self.channels())

    def _redact(self, text: str) -> str:
        """Remove any of our secrets from text that a remote service wrote."""
        secrets = (self.settings.slack_bot_token, self.settings.pushover_app_token, self.settings.pushover_user_key)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    def _note(self, problem: str) -> None:
        if problem not in self.problems:
            self.problems.append(problem)


def _slack_error(response: Response, reply: Mapping[str, Any]) -> str:
    """Slack's error code when it looks like one, otherwise just the HTTP status."""
    code = reply.get("error")
    if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,80}", code):
        return code
    return "HTTP %d" % response.status


def _pushover_error(response: Response, reply: Mapping[str, Any]) -> str:
    errors = reply.get("errors")
    if isinstance(errors, list) and errors:
        return _clip("; ".join(str(error) for error in errors), 200)
    return "HTTP %d" % response.status
