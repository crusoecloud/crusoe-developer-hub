"""One poll of the tracker: expire, refresh, search, judge, alert, deliver.

`run_tick` is what `eft tick` runs, by hand or from the background agent. Its order matters:

1. Expire first: listing data older than six hours comes off sent messages, and rows whose
   time is up are deleted, before anything new is fetched.
2. Refresh the tracked listings that are due, soonest-ending first, and follow up on what
   changed (a price now within the ceiling, an auction about to end, a size back in stock).
3. Search each active watch that is due and run the funnel over the new listings.
4. Every six hours per watch, sweep: the same search without the date window.
5. Deliver the outbox.
6. Send safety notices (a refused key, the call quota, a swapped model), each once a day.
7. Ping the heartbeat, but only when nothing in this tick failed.

Alerts are queued from the store, not from the funnel's return value: every tracked match
whose verdict has no alert yet gets one, in the same transaction that advances the watch's
search window. A crash at any point therefore produces a late alert, never a lost one, and
the alert's key, which is unique per verdict, means never a second one.

An API failure does not raise. It becomes a line under "problems" in the report, and the
tick carries on with the next listing or watch. A failure that says nothing about a listing
(no network, a refused key, a spent budget) is never held against one: eBay and Crusoe are
each left alone for the rest of the tick, and whatever does not need the one that failed
still runs.

Each watch has a daily allowance of eBay calls besides its searches. When it is used up, the
watch's refreshes and its new listings wait for the next UTC day, soonest-ending first. That
is not a problem, so it is reported apart, under "allowance_spent". The one kind of read
that does not wait is the one an alert waits on: the read before a held alert goes out, and
the read behind an "ending soon" reminder.
"""
from __future__ import annotations

import math
import os
import traceback
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from . import alerts, rules
from .config import ConfigError, Settings
from .errors import (
    ChatResult, CrusoeAuthError, CrusoeError, CrusoeUnavailable, EbayAuthError, EbayBudgetExceeded,
    EbayError, EbayNotFound, EbayUnreachable,
)
from .funnel import (
    FULL_READ_AT, REASON_BUDGET, UNJUDGED_REASONS, Funnel, FunnelResult, asking_price, fetch_image, snapshot,
)
from .http import BlockedUrl, TransportError, UrllibTransport
from .plan import build_plan
from .rules import MATCH, UNKNOWN
from .spec import MATCH_FIELDS, normalize_spec
from .store import Store, iso, parse_iso

# ---- the eBay call budget ----------------------------------------------------
RESERVED_CALLS = 400            # kept back for adds, dry runs and retries
CALLS_PER_WATCH = 25            # a day, for everything that is not a search
SWEEPS_PER_DAY = 4              # one every six hours; they are part of the 25
# The rest of the 25 are a watch's reads for the day: detail fetches for new listings and
# refreshes of tracked ones. Refreshes get at most 12 of them, so that keeping old matches
# up to date never uses up what judging new listings needs.
READS_PER_WATCH = CALLS_PER_WATCH - SWEEPS_PER_DAY
REFRESHES_PER_WATCH = 12
# The ledger bucket that counts those reads, per day, under "<watch id>:<purpose>".
WATCH_READS = "ebay_watch"
READ_PURPOSES = ("detail", "refresh")
MINUTES_PER_DAY = 24 * 60
HOT_MINUTES = 10
FEW_WATCHES = 10                # with this many watches or fewer, hot ones are searched more often
HOT_MINUTES_FEW = 5
NORMAL_FLOOR_MINUTES = 10
CADENCE_STEP = 5
# The scheduler never fires at an exact multiple of the cadence. Without some slack a watch
# searched 4 min 59 s ago would wait for the tick after next.
DUE_SLACK = timedelta(seconds=30)

# ---- searching ---------------------------------------------------------------
PAGE_SIZE = 200
MAX_PAGES = 3
NEWEST_FIRST = "newlyListed"
WINDOW_OVERLAP = timedelta(minutes=15)      # eBay indexes a new listing with some delay
FIRST_LOOKBACK = timedelta(hours=24)
MAX_LOOKBACK = timedelta(hours=72)
SWEEP_EVERY = timedelta(hours=6)
# What a search result may update on the snapshot of a listing that is already tracked.
SWEEP_KEYS = ("price", "current_bid", "bid_count")

# ---- refreshing --------------------------------------------------------------
# What eBay returns for fieldgroups=COMPACT. The title, the buying options, the end date and
# the photo are not part of it, so a compact refresh never touches them.
COMPACT_KEYS = ("price", "current_bid", "bid_count", "min_next_bid", "reserve_met", "availability")
# Of the facts on a card, the ones only a full read returns. Between full reads they age,
# while every compact read moves the snapshot's fetch time forward.
FULL_READ_KEYS = ("shipping", "condition", "end_at")
# How long listing data may be shown. Past it, a fact is read again or left off the card.
CARD_LIFE = timedelta(seconds=alerts.SIX_HOURS)
AUCTION_REFRESH = timedelta(hours=1)
# A card may show data for six hours. Refreshing at five leaves room for a late tick.
FIXED_PRICE_REFRESH = timedelta(hours=5)
# A listing parked over the price ceiling is re-priced by the sweep, for the whole watch in
# one call. Its own read is only there to notice that it is over, and can be rare.
PARKED_REFRESH = timedelta(hours=24)
LAST_HOUR = timedelta(hours=1)
# How long to wait after the first, second, third and fourth failed read of a tracked listing.
# The fifth read asks for the full listing, and if that fails too the listing is over.
RETRY_DELAYS = (timedelta(hours=1), timedelta(hours=6), timedelta(hours=24), timedelta(hours=72))
ENDED_KEEP = timedelta(days=14)
ENDING_MARKS = (10, 60)                     # minutes before the scheduled end, nearest first
ENDING_KEY = "ending%d:%d"                  # the alert key of a reminder: its mark, the candidate id
OUT_OF_STOCK = "OUT_OF_STOCK"

# ---- alerts ------------------------------------------------------------------
ANNOUNCED_STATES = ("alerted", "maybe")     # candidate states the user is told about
FIRST_KINDS = ("new", "price", "restock")   # alert kinds that introduce a listing
# An unproven match normally waits for the daily digest. An auction ending sooner cannot.
UNCONFIRMED_WINDOW = timedelta(hours=24)
REASON_NOT_MODEL_CHECKED = "not_model_checked"
NOT_MODEL_CHECKED_NOTE = "not model-checked"
QUOTA_NOTICE_SHARE = 0.8
NOTICE_SHELF_LIFE = timedelta(hours=24)
HEARTBEAT_TIMEOUT = 10.0

LAST_TICK_KEY = "tick:last"
OVERFLOW_KEY = "watch:%d:overflow"
RETRY_KEY = "candidate:%d:refresh"

BUDGET_MESSAGE = "the daily call cap is reached; searching resumes tomorrow (UTC)"
CRUSOE_DOWN_MESSAGE = "the service was unavailable; no more model calls were made in this tick"
EBAY_AUTH_NOTICE = (
    "eBay refused the keyset (HTTP 401 or 403). Searching is stopped until EBAY_CLIENT_ID and "
    "EBAY_CLIENT_SECRET work again. Run: eft doctor ebay"
)
CRUSOE_AUTH_NOTICE = (
    "Crusoe refused the API key (HTTP 401 or 403). Matching is stopped until CRUSOE_API_KEY "
    "works again. Run: eft doctor crusoe"
)


def cadence_minutes(n_hot: int, n_normal: int, daily_cap: int) -> Dict[str, int]:
    """Minutes between searches for hot and for normal watches, so a day's calls fit the cap.

    400 calls are held in reserve and every watch is allowed 25 a day for what is not a
    search (detail fetches, refreshes, sweeps); `_EbayMeter` holds each watch to that.
    Searches share the rest. Hot watches are searched every 10 minutes, or every 5 while
    there are at most 10 watches in all. Normal watches get what is left: never more often
    than every 10 minutes, in steps of 5. Past what the cap can carry at all, every watch is
    searched once a day.
    """
    total = n_hot + n_normal
    searches = daily_cap - RESERVED_CALLS - CALLS_PER_WATCH * total
    hot_floor = HOT_MINUTES_FEW if total <= FEW_WATCHES else HOT_MINUTES
    # Hot watches come first, but each normal watch is left at least one search a day.
    hot = _interval(n_hot, searches - n_normal, hot_floor)
    hot_searches = n_hot * MINUTES_PER_DAY / float(hot)
    return {"hot": hot, "normal": _interval(n_normal, searches - hot_searches, NORMAL_FLOOR_MINUTES)}


def _interval(watches: int, searches: float, floor: int) -> int:
    """The shortest interval, in steps of 5 minutes, at which `watches` use at most `searches` a day."""
    if watches <= 0:
        return floor
    if searches < watches:
        return MINUTES_PER_DAY
    needed = int(math.ceil(watches * MINUTES_PER_DAY / float(searches) / CADENCE_STEP)) * CADENCE_STEP
    return min(MINUTES_PER_DAY, max(floor, needed))


def window_start(watch: Mapping[str, Any], now: datetime) -> datetime:
    """Where a watch's next search window begins.

    The first search of a watch looks back a day. Later ones start a little before the
    previous window ended, because eBay indexes new listings with a delay. No window
    reaches back more than 72 hours, however long the poller was off.
    """
    previous_end = parse_iso(watch.get("window_end"))
    if previous_end is None or not watch.get("last_search_at"):
        start = now - FIRST_LOOKBACK
    else:
        start = previous_end - WINDOW_OVERLAP
    return max(start, now - MAX_LOOKBACK)


def next_refresh(listing: Mapping[str, Any], end_at: Optional[str], now: datetime,
                 parked: bool = False) -> datetime:
    """When a tracked listing is read again.

    An auction is read every tick in its last hour and hourly before that. A fixed-price
    listing is read every five hours, which keeps the data on its card under six hours old.

    `parked` is for a listing parked over the price ceiling. The sweep re-prices those, so
    such a listing is read on its own once a day at most (and when its scheduled end
    comes), and each sweep that returns it puts that read off again. In effect only a
    listing the sweep no longer sees is read, to find out whether it is over.
    """
    end = parse_iso(end_at)
    if "AUCTION" in (listing.get("buying_options") or []) and not parked:
        if end is not None and end - now <= LAST_HOUR:
            return now
        due = now + AUCTION_REFRESH
        return min(due, end - LAST_HOUR) if end is not None else due
    due = now + (PARKED_REFRESH if parked else FIXED_PRICE_REFRESH)
    return min(due, end) if end is not None and end > now else due


def last_decision(store: Store, candidate_id: int) -> Optional[Dict[str, Any]]:
    """The newest decision row in a candidate's trace, or None if it never reached the decision rule."""
    decisions = [row for row in store.verdicts_for(candidate_id) if row["stage"] == "decision"]
    return decisions[-1] if decisions else None


def stored_evidence(store: Store, candidate_id: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Per-field verdicts and where each was proven, from a candidate's last stored decision.

    Both are empty for a candidate that never reached the decision rule.
    """
    decision = last_decision(store, candidate_id)
    fields = (decision["fields"] or {}) if decision else {}
    entries = {name: entry for name, entry in fields.items() if isinstance(entry, dict)}
    verdicts = {name: entry.get("verdict", UNKNOWN) for name, entry in entries.items()}
    sources = {name: entry["source"] for name, entry in entries.items() if entry.get("source")}
    return verdicts, sources


def run_tick(
    settings: Settings,
    store: Store,
    ebay: Any,
    crusoe: Any,
    notifier: Any,
    funnel: Optional[Any] = None,
    transport: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one poll and return a report of what it did.

    `funnel` defaults to a `Funnel` over the given clients. `transport` carries the heartbeat
    ping and listing photo downloads, and defaults to `UrllibTransport()`. The report holds
    counts and a list of "problems"; it never holds listing content or a secret.
    """
    return _Tick(settings, store, ebay, crusoe, notifier, funnel, transport).run()


class _ModelMeter:
    """Passes model calls through to the Crusoe client and notes how they went.

    The funnel absorbs a failed model call (the listing just stays pending), so this is
    where the poller learns that the service was down or that another model answered than
    the one that was asked.

    It is also the circuit breaker. "Unavailable" means every candidate model was rate
    limited or down after its retries, and each of those retries waited its backoff. Once
    one call has ended that way, the rest of the tick's calls fail at once without a
    request: a tick must not sit on its lock retrying a service that is down.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.answered = 0
        self.unavailable = 0
        self.unusable = 0
        self.swaps: Dict[str, str] = {}

    def chat_json(self, role: str, system: str, parts: Sequence[Dict[str, Any]], **options: Any) -> ChatResult:
        if self.unavailable:
            raise CrusoeUnavailable("Crusoe was unavailable earlier in this tick; the %s role was not asked" % role)
        try:
            answer = self.client.chat_json(role, system, parts, **options)
        except CrusoeAuthError:
            raise
        except CrusoeUnavailable:
            self.unavailable += 1
            raise
        except CrusoeError:
            self.unusable += 1
            raise
        self.answered += 1
        if answer.swapped:
            self.swaps[answer.model_requested] = answer.model_served
        return answer

    def image_part(self, source: Any, **options: Any) -> Dict[str, Any]:
        return self.client.image_part(source, **options)

    @property
    def down(self) -> bool:
        """True when calls failed as unavailable and not one was answered."""
        return self.unavailable > 0 and self.answered == 0


class _AllowanceSpent(EbayBudgetExceeded):
    """A watch has made its reads for the day. No request was sent."""


class _EbayMeter:
    """Passes eBay calls through to the client and holds each watch to its daily reads.

    The cadence budgets the searches. Every other call a watch causes is a read, counted
    here per watch and UTC day: a detail fetch for a new listing, a refresh of a tracked
    one. The four sweeps a day are bounded by their own schedule, and with the reads they
    make up the 25 calls a watch is allowed.

    A detail fetch for a watch that has made its reads is refused with a kind of
    EbayBudgetExceeded, which the funnel already knows: it stops, and the listings it has
    not reached are found again by a later search. The poller asks `refreshes_left` itself
    before a paced refresh. A read that an alert waits on (a held alert about to go out, an
    "ending soon" reminder) is counted but never refused: an alert does not wait on a budget.

    It is also the circuit breaker for eBay. Once a call has found eBay unreachable (no
    network, no token, a 429), every later call of the tick fails the same way without a
    request: a poll that wakes up offline must not sit through a timeout per listing.
    """

    def __init__(self, client: Any, store: Store) -> None:
        self.client = client
        self.store = store
        self.watch_id = 0               # whose work the reads are; the poller sets it
        self.unreachable: Optional[str] = None      # what the call that found eBay out of reach said

    def search(self, q: Optional[str] = None, **options: Any) -> Dict[str, Any]:
        return self._call(lambda: self.client.search(q, **options))

    def get_item(self, item_id: str, *, compact: bool = False, purpose: str = "detail") -> Dict[str, Any]:
        return self._read(purpose, lambda: self.client.get_item(item_id, compact=compact, purpose=purpose))

    def get_items_by_item_group(self, item_group_id: str, *, purpose: str = "detail") -> List[Dict[str, Any]]:
        return self._read(purpose, lambda: self.client.get_items_by_item_group(item_group_id, purpose=purpose))

    def reads_left(self, watch_id: int) -> int:
        """How many more reads of any kind a watch may make today."""
        return READS_PER_WATCH - sum(self._made(watch_id).values())

    def refreshes_left(self, watch_id: int) -> int:
        """How many more of a watch's reads today may be paced refreshes."""
        return min(self.reads_left(watch_id), REFRESHES_PER_WATCH - self._made(watch_id)["refresh"])

    def _made(self, watch_id: int) -> Dict[str, int]:
        counts = self.store.ledger_breakdown(WATCH_READS)
        return {purpose: counts.get("%d:%s" % (watch_id, purpose), 0) for purpose in READ_PURPOSES}

    def _read(self, purpose: str, call: Callable[[], Any]) -> Any:
        """One call that is not a search, counted against the watch it is made for."""
        if purpose == "detail" and self.reads_left(self.watch_id) <= 0:
            raise _AllowanceSpent("watch %d has made its eBay reads for today; no request was sent" % self.watch_id)
        try:
            answer = self._call(call)
        except (EbayBudgetExceeded, EbayUnreachable):
            # Not counted: nothing came back, and an outage must not use up a watch's day.
            raise
        except EbayError:
            self._count(purpose)
            raise
        self._count(purpose)
        return answer

    def _count(self, purpose: str) -> None:
        self.store.ledger_add(WATCH_READS, "%d:%s" % (self.watch_id, purpose))

    def _call(self, call: Callable[[], Any]) -> Any:
        if self.unreachable is not None:
            raise EbayUnreachable("%s (no more eBay calls were made in this tick)" % self.unreachable)
        try:
            return call()
        except EbayUnreachable as problem:
            self.unreachable = str(problem)
            raise


class _Tick:
    """The state of one poll: the clients, what has gone wrong so far, and the report."""

    def __init__(self, settings: Settings, store: Store, ebay: Any, crusoe: Any, notifier: Any,
                 funnel: Optional[Any], transport: Optional[Any]) -> None:
        self.settings = settings
        self.store = store
        self.ebay = _EbayMeter(ebay, store)
        self.models = _ModelMeter(crusoe)
        self.notifier = notifier
        self.transport = transport if transport is not None else UrllibTransport()
        self.funnel = funnel or Funnel(
            self.ebay, self.models, store, image_fetcher=partial(fetch_image, transport=self.transport)
        )
        # Set once eBay has failed in a way that every later call of this tick would repeat:
        # a refused keyset, a spent budget, a missing setting, no network.
        self.ebay_stopped = False
        # The same for Crusoe: a refused key. The two are kept apart because one does not
        # need the other: without models, tracked listings are still read and held alerts
        # still go out.
        self.models_stopped = False
        # Tracked listings read by this tick's refresh, which a waiting alert need not read again.
        self.read_now: Set[int] = set()
        # Listings a model could not judge in this tick, for the style-code fallback.
        self.unjudged: List[Tuple[Dict[str, Any], FunnelResult]] = []
        self.notices: Dict[str, Tuple[str, bool]] = {}
        self.report: Dict[str, Any] = {
            "at": store.now_iso(),
            "expired": 0, "purged": 0,
            "refreshed": 0, "unavailable": 0, "ended": 0,
            "searched": 0, "swept": 0, "decisions": {},
            "queued": 0, "not_model_checked": 0,
            "delivery": {"sent": 0, "held": 0, "dropped": 0, "failed": 0},
            "notices": [], "cadence": {}, "heartbeat": None, "problems": [],
            # Watches whose refreshes or new listings waited because their reads for today are used.
            "allowance_spent": [],
        }

    @property
    def now(self) -> datetime:
        return self.store.clock()

    def run(self) -> Dict[str, Any]:
        self.notifier.problems = []     # delivery problems are reported per tick
        self.report["expired"] = self.notifier.expire_messages()
        self.report["purged"] = self.store.purge_expired()
        self._refresh_due()
        self._search_due()
        self._alert_on_style_codes()
        self._note_model_calls()
        self.report["delivery"] = self.notifier.deliver_pending(refresher=self._fresh_payload)
        self._send_notices()
        for line in self.notifier.problems:
            self._problem("alerts", line)
        self.report["heartbeat"] = self._heartbeat()
        self.store.kv_set(LAST_TICK_KEY, self.report)
        return self.report

    # ---- problems ----------------------------------------------------------------

    def _problem(self, area: str, message: str) -> None:
        line = "%s: %s" % (area, message)
        if line not in self.report["problems"]:
            self.report["problems"].append(line)

    def _waits(self, watch_id: int) -> None:
        """Note that some of a watch's work is waiting because its reads for today are used."""
        if watch_id not in self.report["allowance_spent"]:
            self.report["allowance_spent"].append(watch_id)

    def _ebay_is_out(self, problem: EbayError) -> None:
        """eBay failed in a way every later call of this tick would repeat: stop calling it.

        That is a spent budget, a refused keyset, or eBay out of reach. None of them says
        anything about the listing or watch that happened to run into it.
        """
        if isinstance(problem, EbayAuthError):
            self.notices["ebay_auth"] = (EBAY_AUTH_NOTICE, True)
        self.ebay_stopped = True
        self._problem("ebay", BUDGET_MESSAGE if isinstance(problem, EbayBudgetExceeded) else str(problem))

    def _guard(self, label: str, work: Callable[..., None], *args: Any) -> None:
        """Run one unit of work and turn its failure into a line of the report.

        A refused key, a spent budget, a missing setting or eBay out of reach would fail
        every later call the same way, so those also stop the calls to that service for the
        rest of the tick: eBay's or Crusoe's, not both. Anything else, a bug included, costs
        only the listing or watch it happened in: the poller runs unattended, and one bad
        watch must not silence the others.
        """
        try:
            work(*args)
        except (EbayBudgetExceeded, EbayAuthError, EbayUnreachable) as problem:
            self._ebay_is_out(problem)
        except CrusoeAuthError as problem:
            self.notices["crusoe_auth"] = (CRUSOE_AUTH_NOTICE, True)
            self.models_stopped = True
            self._problem("crusoe", str(problem))
        except ConfigError as problem:
            # eBay's settings are needed first, so a setting that is missing while eBay has
            # its own is one of Crusoe's.
            if self.settings.has("ebay"):
                self.models_stopped = True
            else:
                self.ebay_stopped = True
            self._problem("config", str(problem))
        except (EbayError, CrusoeError) as problem:
            self._problem(label, str(problem))
        except Exception as problem:  # noqa: BLE001 - see the docstring
            # The message of an unexpected error is not repeated: nothing vouches for what is in it.
            where = traceback.extract_tb(problem.__traceback__)[-1]
            self._problem(label, "unexpected %s at %s:%d" % (
                problem.__class__.__name__, os.path.basename(where.filename), where.lineno or 0))

    # ---- phase 2: refresh tracked listings -----------------------------------------

    def _refresh_due(self) -> None:
        """Read the tracked listings whose time has come, soonest-ending first.

        A watch that has made its refreshes for today is passed over. Its listings stay due,
        so tomorrow's first reads go to the ones that end soonest.

        The exception is a read that an "ending soon" reminder waits on. An auction is read
        every hour, so one that ends late in the UTC day has used the watch's refreshes long
        before its last hour, and without the exception its reminders would never be sent.
        That is two reads at most in the life of an auction, one for each reminder.
        """
        for candidate in self.store.due_for_refresh():
            if self.ebay_stopped:
                break
            if self.ebay.refreshes_left(candidate["watch_id"]) <= 0 and not self._reminder_waits(candidate):
                self._waits(candidate["watch_id"])
                continue
            self._guard("refresh %s" % candidate["code"], self._refresh, candidate)

    def _refresh(self, candidate: Dict[str, Any]) -> None:
        """Bring one tracked listing up to date, or find out that it is over.

        A compact read is enough for prices and bids. The full listing is read when only
        its end date can settle things (the scheduled end has passed, the listing has just
        gone out of stock, or four slow retries have failed) and when the last full read is
        over six hours old, so that what a card shows beside the price is never older.

        Only an answer about this listing is held against it. A spent budget, a refused
        keyset or eBay out of reach is raised for `_guard`, which stops the refresh phase:
        the listing's retry count and its next refresh time stay as they were.
        """
        retry = self.store.kv_get(RETRY_KEY % candidate["id"]) or {}
        full = retry.get("failures", 0) >= len(RETRY_DELAYS) or self._needs_full_read(candidate)
        try:
            fresh = self._read(candidate, full=full)
            if not full and _just_sold_out(candidate, fresh):
                fresh, full = self._read(candidate, full=True), True
        except (EbayAuthError, EbayBudgetExceeded, EbayUnreachable):
            raise
        except EbayError as problem:
            self._refresh_failed(candidate, retry, problem)
            return
        if full and self._is_over(fresh):
            self._end(candidate)
            return
        self._follow_up(self._keep(candidate, retry, fresh, full=full))

    def _read(self, candidate: Mapping[str, Any], full: bool) -> Dict[str, Any]:
        """One eBay call for a tracked listing. For a multi-size listing the snapshot holds
        the id of the variation that was judged, which is the one to read again."""
        item_id = (candidate["listing"] or {}).get("item_id") or candidate["item_id"]
        self.ebay.watch_id = candidate["watch_id"]
        return self.ebay.get_item(item_id, compact=not full, purpose="refresh")

    def _end_passed(self, candidate: Mapping[str, Any]) -> bool:
        end = parse_iso(candidate["end_at"])
        return end is not None and end <= self.now

    def _is_over(self, fresh: Mapping[str, Any]) -> bool:
        """True when a full read says the listing's end is behind us."""
        end = parse_iso(fresh.get("end_at"))
        return end is not None and end <= self.now

    def _full_read_is_old(self, candidate: Mapping[str, Any]) -> bool:
        """True when the snapshot was last read in full over six hours ago, or nobody knows when."""
        read = parse_iso((candidate["listing"] or {}).get(FULL_READ_AT))
        return read is None or self.now - read > CARD_LIFE

    def _needs_full_read(self, candidate: Mapping[str, Any]) -> bool:
        """True when a compact read would not do: the end date has to be seen, or the
        snapshot's shipping, condition and end time are too old to show."""
        return self._end_passed(candidate) or self._full_read_is_old(candidate)

    def _keep(self, candidate: Mapping[str, Any], retry: Mapping[str, Any], fresh: Mapping[str, Any],
              full: bool) -> Dict[str, Any]:
        """Store what a read returned, schedule the next one, and return the updated row."""
        now = self.now
        listing = dict(candidate["listing"] or {})
        listing.update(snapshot(fresh, iso(now)) if full else {key: fresh.get(key) for key in COMPACT_KEYS})
        end_at = fresh.get("end_at") if full else candidate["end_at"]
        # A listing that was unreadable for a while goes back to the state it had.
        state = retry.get("was") or candidate["state"]
        due = next_refresh(listing, end_at, now, parked=_left_to_sweep(state, candidate["reason"]))
        with self.store.transaction():
            self.store.update_candidate(
                candidate["id"], listing=listing, end_at=end_at, fetched_at=iso(now),
                next_refresh_at=iso(due), state=state,
            )
            self.store.kv_delete(RETRY_KEY % candidate["id"])
        self.report["refreshed"] += 1
        self.read_now.add(candidate["id"])
        return self.store.get_candidate_by_id(candidate["id"])  # type: ignore[return-value]

    def _refresh_failed(self, candidate: Mapping[str, Any], retry: Mapping[str, Any], problem: EbayError) -> None:
        """A paced read of a tracked listing got an answer that is about the listing: count it.

        One failure settles nothing: listings come back, and eBay has bad minutes. The
        listing is marked unavailable and read again after 1, 6, 24 and 72 hours. It is over
        on a second "not found", on a second failure once its scheduled end has passed, or
        when the full read after the fourth retry fails as well.

        Only the paced refresh counts, because only it is paced. The read made for a
        waiting alert runs on every tick and must never come here.
        """
        failures = retry.get("failures", 0) + 1
        not_found = retry.get("not_found", 0) + int(isinstance(problem, EbayNotFound))
        if not isinstance(problem, EbayNotFound):
            self._problem("refresh %s" % candidate["code"], str(problem))
        if not_found >= 2 or (failures >= 2 and self._end_passed(candidate)) or failures > len(RETRY_DELAYS):
            self._end(candidate)
            return
        self._set_aside(candidate, retry, failures, not_found, self.now + RETRY_DELAYS[failures - 1])

    def _set_aside(self, candidate: Mapping[str, Any], retry: Mapping[str, Any], failures: int, not_found: int,
                   retry_at: datetime) -> None:
        """Mark a tracked listing unavailable until `retry_at`, remembering the state it had."""
        with self.store.transaction():
            self.store.kv_set(RETRY_KEY % candidate["id"], {
                "failures": failures, "not_found": not_found, "was": retry.get("was") or candidate["state"],
            })
            self.store.update_candidate(candidate["id"], state="unavailable", next_refresh_at=iso(retry_at))
        self.report["unavailable"] += 1

    def _end(self, candidate: Mapping[str, Any]) -> None:
        """The listing is over: rewrite its messages as "Ended" and delete what was stored about it."""
        self.notifier.mark_ended(candidate["id"])
        with self.store.transaction():
            self.store.purge_listing_data(candidate["id"], state="ended")
            self.store.update_candidate(candidate["id"], purge_at=iso(self.now + ENDED_KEEP))
            self.store.kv_delete(RETRY_KEY % candidate["id"])
        self.report["ended"] += 1

    def _follow_up(self, candidate: Dict[str, Any]) -> None:
        """Act on what a refresh showed. Only active watches get follow-ups."""
        watch = self.store.get_watch(candidate["watch_id"])
        if watch is None or watch["status"] != "active":
            return
        spec = normalize_spec(watch["spec"])
        listing = candidate["listing"]
        if candidate["state"] == "parked":
            kind = _released(spec, candidate, listing)
            if kind:
                # The snapshot stands in for a search result; the funnel fetches the rest itself.
                self._recheck(watch, {candidate["id"]: kind}, [dict(listing, item_id=candidate["item_id"])])
        elif candidate["state"] in ANNOUNCED_STATES:
            if listing.get("availability") == OUT_OF_STOCK:
                self.store.update_candidate(candidate["id"], state="parked", reason="size_out_of_stock")
            else:
                self._remind_if_ending(watch, spec, candidate)

    def _remind_if_ending(self, watch: Mapping[str, Any], spec: Mapping[str, Any],
                          candidate: Mapping[str, Any]) -> None:
        """Queue an "ending soon" reply at 60 and at 10 minutes before an auction's scheduled end."""
        mark = self._ending_mark(watch, candidate)
        if mark is not None:
            self._enqueue(watch, spec, candidate, "ending", ENDING_KEY % (mark, candidate["id"]))

    def _ending_mark(self, watch: Mapping[str, Any], candidate: Mapping[str, Any]) -> Optional[int]:
        """Which "ending soon" reminder a listing is due right now: 60, 10 or None.

        Reminders are only for what the user singled out (a starred listing or a hot watch),
        and only for a listing they have already been sent: a reminder about an unknown
        listing would arrive alongside its first card.
        """
        if not (candidate["starred"] or watch["tier"] == "hot"):
            return None
        end = parse_iso(candidate["end_at"])
        if end is None or "AUCTION" not in ((candidate["listing"] or {}).get("buying_options") or []):
            return None
        minutes_left = (end - self.now).total_seconds() / 60.0
        if minutes_left <= 0 or not self._announced(candidate["id"], sent_only=True):
            return None
        # Nearest mark first, so a first look inside the last 10 minutes sends one reminder.
        return next((mark for mark in ENDING_MARKS if minutes_left <= mark), None)

    def _reminder_waits(self, candidate: Mapping[str, Any]) -> bool:
        """True when reading this listing now would queue an "ending soon" reminder.

        It asks what `_follow_up` will ask once the listing has been read, and also that the
        reminder is not in the outbox already. Otherwise a watch out of refreshes would read
        the listing on every tick of its last hour.
        """
        watch = self.store.get_watch(candidate["watch_id"])
        if watch is None or watch["status"] != "active" or candidate["state"] not in ANNOUNCED_STATES:
            return False
        mark = self._ending_mark(watch, candidate)
        if mark is None:
            return False
        key = ENDING_KEY % (mark, candidate["id"])
        return all(alert["dedupe_key"] != key for alert in self.store.alerts_for_candidate(candidate["id"]))

    # ---- phases 3 and 4: search and sweep ------------------------------------------

    def _search_due(self) -> None:
        watches = self.store.list_watches(status="active")
        hot = sum(1 for watch in watches if watch["tier"] == "hot")
        cadence = cadence_minutes(hot, len(watches) - hot, self.settings.ebay_daily_cap)
        self.report["cadence"] = cadence
        for watch in sorted(watches, key=lambda row: row["tier"] != "hot"):
            if not self.ebay_stopped and self._search_is_due(watch, cadence):
                self._guard("watch %d" % watch["id"], self._search, watch)
            # A sweep needs eBay only, so it also runs while the models are stopped.
            if not self.ebay_stopped and self._sweep_is_due(watch["id"]):
                self._guard("watch %d sweep" % watch["id"], self._sweep, watch)

    def _search_is_due(self, watch: Mapping[str, Any], cadence: Mapping[str, int]) -> bool:
        """True when the watch's cadence says so and a search could lead somewhere.

        A new listing is judged by the models and then read in full. With the models
        stopped, or the watch's reads for today used up, a search would only be repeated
        later for the same listings, so it waits. The window stays where it is and the
        next search that does run covers what this one would have.
        """
        last = parse_iso(watch["last_search_at"])
        every = timedelta(minutes=cadence["hot" if watch["tier"] == "hot" else "normal"])
        if self.models_stopped or not (last is None or self.now - last >= every - DUE_SLACK):
            return False
        if self.ebay.reads_left(watch["id"]) <= 0:
            self._waits(watch["id"])
            return False
        return True

    def _search(self, watch: Dict[str, Any]) -> None:
        """Search one watch's window, judge what is new, and queue its alerts.

        The alerts and the new window end are written together. If anything before that
        fails, the window has not moved and the next tick searches it again.
        """
        plan = watch["plan"] or build_plan(watch["spec"])
        now = self.now
        start = window_start(watch, now)
        listings, too_many = self._pages(plan, start, now)
        self._flag_overflow(watch["id"], too_many)
        results = self._judge(watch, listings)
        waiting = [result for result in results if result.decision == "pending"]
        self.unjudged.extend((watch, result) for result in waiting if result.reason in UNJUDGED_REASONS)
        with self.store.transaction():
            self._announce(watch)
            # A listing without a verdict is only looked at again when a search returns it,
            # so the window stays where it is until every listing in it has been judged.
            window_end = start + WINDOW_OVERLAP if waiting else now
            self.store.update_watch(
                watch["id"], last_search_at=iso(now), window_end=iso(window_end),
                # The six hours to a watch's first sweep count from its first search.
                last_sweep_at=watch["last_sweep_at"] or iso(now),
            )
        self.report["searched"] += 1

    def _pages(self, plan: Mapping[str, Any], start: datetime, end: datetime) -> Tuple[List[Dict[str, Any]], bool]:
        """Up to three pages of one windowed search: (summaries, whether more remain)."""
        listings: List[Dict[str, Any]] = []
        offset: Optional[int] = 0
        for _ in range(MAX_PAGES):
            page = self.ebay.search(
                plan["q"], category_id=plan.get("category_id"), buying_options=plan.get("buying_options"),
                start_from=start, start_to=end, sort=NEWEST_FIRST, limit=PAGE_SIZE, offset=offset, purpose="search",
            )
            listings.extend(page["items"])
            offset = page["next_offset"]
            if offset is None:
                return listings, False
        return listings, True

    def _flag_overflow(self, watch_id: int, too_many: bool) -> None:
        """Remember that a window held more than three pages, so `eft status` can say the query is too broad."""
        if too_many:
            self.store.kv_set(OVERFLOW_KEY % watch_id, {"at": self.store.now_iso()})
        else:
            self.store.kv_delete(OVERFLOW_KEY % watch_id)

    def _judge(self, watch: Mapping[str, Any], listings: Sequence[Dict[str, Any]],
               recheck: bool = False) -> List[FunnelResult]:
        """Run the funnel for one watch, with its detail fetches counted against that watch.

        Returns the funnel's results, after counting its decisions and picking out what it
        absorbed that the report should show.
        """
        self.ebay.watch_id = watch["id"]
        results = self.funnel.run(watch, listings, recheck=recheck)
        counts = self.report["decisions"]
        for result in results:
            counts[result.decision] = counts.get(result.decision, 0) + 1
            if result.reason in ("ebay_unavailable", "ebay_error"):
                self._problem("ebay", "a listing could not be read (%s)" % result.reason)
            elif result.reason != REASON_BUDGET:
                continue
            elif self.ebay.reads_left(watch["id"]) <= 0:
                self._waits(watch["id"])        # the watch's own allowance, not the day's cap
            else:
                self._ebay_is_out(EbayBudgetExceeded(BUDGET_MESSAGE))
        return results

    def _sweep_is_due(self, watch_id: int) -> bool:
        """Every six hours, and only for a watch with tracked listings: a sweep finds nothing new.

        The row is read again here because the search just before may have changed it.
        """
        watch = self.store.get_watch(watch_id)
        last = parse_iso(watch["last_sweep_at"]) if watch else None
        if last is None or self.now - last < SWEEP_EVERY:
            return False
        return bool(self.store.list_candidates(watch_id, tracked=True))

    def _sweep(self, watch: Dict[str, Any]) -> None:
        """One search without the date window, for what is already tracked.

        It brings the price and bids of tracked listings up to date in a single call, and it
        is how a parked listing whose price has come down to the ceiling is noticed and
        judged again. The fetch time of a snapshot is left alone: only part of it is newer.
        """
        plan = watch["plan"] or build_plan(watch["spec"])
        found = self.ebay.search(
            plan["q"], category_id=plan.get("category_id"), buying_options=plan.get("buying_options"),
            limit=PAGE_SIZE, purpose="sweep",
        )
        current = {listing["item_id"]: listing for listing in found["items"]}
        spec = normalize_spec(watch["spec"])
        released: Dict[int, str] = {}
        to_judge: List[Dict[str, Any]] = []
        with self.store.transaction():
            for candidate in self.store.list_candidates(watch["id"], tracked=True):
                listing = candidate["listing"] or {}
                seen = current.get(candidate["item_id"])
                # In a multi-size listing the search shows one size's price, which need not
                # be the price of the size that is tracked. Those wait for their own refresh.
                if seen is None or listing.get("item_id") != seen["item_id"]:
                    continue
                listing = dict(listing, **{key: seen.get(key) for key in SWEEP_KEYS})
                changes: Dict[str, Any] = {"listing": listing}
                if _left_to_sweep(candidate["state"], candidate["reason"]):
                    # Seen alive and re-priced just now, so its own read can wait another day.
                    due = next_refresh(listing, candidate["end_at"], self.now, parked=True)
                    changes["next_refresh_at"] = iso(due)
                self.store.update_candidate(candidate["id"], **changes)
                kind = _released(spec, candidate, listing) if candidate["state"] == "parked" else None
                if kind:
                    released[candidate["id"]] = kind
                    to_judge.append(seen)
            self.store.update_watch(watch["id"], last_sweep_at=self.store.now_iso())
        self.report["swept"] += 1
        if to_judge:
            self._recheck(watch, released, to_judge)

    def _recheck(self, watch: Dict[str, Any], kinds: Mapping[int, str], listings: Sequence[Dict[str, Any]]) -> None:
        """Judge parked listings again now that what held them back has changed.

        `kinds` maps candidate id to the alert kind ("price" or "restock") for a listing
        that now passes. If the tick dies between the funnel's verdict and the alert, the
        next search still announces the listing, as a plain new match.

        A listing that is rejected this time was tracked until now, so cards may be out for
        it (it was announced, sold out and came back). The funnel deletes what was stored
        about it; its cards are rewritten here, so none keeps showing a title and a price.

        Judging needs the models and one eBay read. While either is out for this tick or
        this watch, the listings stay parked; the next sweep or read finds them again.
        """
        if self.models_stopped or self.ebay_stopped:
            return
        if self.ebay.reads_left(watch["id"]) <= 0:
            self._waits(watch["id"])
            return
        results = self._judge(watch, listings, recheck=True)
        for result in results:
            if result.decision == "reject" and result.candidate is not None:
                self.notifier.mark_ended(result.candidate["id"])
        spec = normalize_spec(watch["spec"])
        with self.store.transaction():
            for candidate_id, kind in kinds.items():
                candidate = self.store.get_candidate_by_id(candidate_id)
                if candidate and candidate["tracked"] and candidate["state"] in ANNOUNCED_STATES:
                    # A match again: from here on it is read as often as any other.
                    due = next_refresh(candidate["listing"], candidate["end_at"], self.now)
                    self.store.update_candidate(candidate_id, next_refresh_at=iso(due))
                    self._enqueue(watch, spec, candidate, kind, self._verdict_key(kind, candidate_id))
            self._announce(watch)

    # ---- queueing alerts -----------------------------------------------------------

    def _announce(self, watch: Mapping[str, Any]) -> None:
        """Queue an alert for every match of this watch that the user has not been told about.

        Working from the store instead of the funnel's results is what makes a crash safe:
        a match recorded by a tick that died before queueing its alert is picked up here by
        the next one. Each tracked listing also gets its first refresh time.
        """
        spec = normalize_spec(watch["spec"])
        for candidate in self.store.list_candidates(watch["id"], tracked=True):
            if candidate["next_refresh_at"] is None:
                fetched = parse_iso(candidate["fetched_at"]) or self.now
                due = next_refresh(candidate["listing"] or {}, candidate["end_at"], fetched,
                                   parked=_left_to_sweep(candidate["state"], candidate["reason"]))
                self.store.update_candidate(candidate["id"], next_refresh_at=iso(due))
            if candidate["state"] in ANNOUNCED_STATES and not self._announced(candidate["id"]):
                self._enqueue(watch, spec, candidate, "new", self._verdict_key("new", candidate["id"]))

    def _decided_at(self, candidate_id: int) -> str:
        decision = last_decision(self.store, candidate_id)
        return decision["created_at"] if decision else ""

    def _announced(self, candidate_id: int, sent_only: bool = False) -> bool:
        """True when an alert has introduced the listing's current verdict (and, with
        `sent_only`, has gone out).

        Only alerts queued since the last decision count. A listing that was parked and has
        been judged again is news once more, even if the user was told about it long ago.
        """
        since = self._decided_at(candidate_id)
        return any(
            alert["kind"] in FIRST_KINDS and alert["created_at"] >= since
            and (alert["status"] == "sent" or not sent_only)
            for alert in self.store.alerts_for_candidate(candidate_id)
        )

    def _verdict_key(self, kind: str, candidate_id: int) -> str:
        """The key of the alert that announces a listing's current verdict: one per decision."""
        return "%s:%d:%s" % (kind, candidate_id, self._decided_at(candidate_id))

    def _enqueue(self, watch: Mapping[str, Any], spec: Mapping[str, Any], candidate: Mapping[str, Any],
                 kind: str, key: str) -> None:
        """Put one alert in the outbox. Its key is unique per event, so a repeat is a no-op."""
        verdicts, sources = stored_evidence(self.store, candidate["id"])
        unproven = _unproven(spec, verdicts) if candidate["state"] == "maybe" else []
        tier = _tier(kind, watch, candidate, self.now)
        payload = alerts.build_payload(
            watch, candidate, candidate["listing"], kind=kind, tier=tier,
            proof=rules.proof_line(spec, verdicts, sources), unconfirmed=unproven,
            note=NOT_MODEL_CHECKED_NOTE if candidate["reason"] == REASON_NOT_MODEL_CHECKED else None,
            as_of=candidate["fetched_at"],
        )
        payload.update(self._card_facts(candidate))
        queued = self.store.enqueue_alert(
            key, kind, tier, payload, watch_id=watch["id"], candidate_id=candidate["id"],
            as_of=candidate["fetched_at"],
        )
        if queued is not None:
            self.report["queued"] += 1

    def _card_facts(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        """The facts of a tracked listing that change, as a card may show them right now.

        Prices and bids are as fresh as the snapshot's last read of any kind. Shipping,
        condition and the end time come from full reads only. The poller reads in full
        before it builds or releases a card whenever the last full read is over six hours
        old; should a snapshot be that old all the same, those three are left off the card,
        so that nothing stale appears under a fresh "as of".
        """
        listing = candidate["listing"] or {}
        facts = {key: listing.get(key) for key in COMPACT_KEYS if key != "availability"}
        stale = self._full_read_is_old(candidate)
        facts.update({key: None if stale else listing.get(key) for key in FULL_READ_KEYS})
        return facts

    # ---- model-free fallback -------------------------------------------------------

    def _alert_on_style_codes(self) -> None:
        """Crusoe was down for the whole tick: alert on exact style-code matches, and say so.

        A listing whose title carries one of the watch's style codes is very likely the
        item. It is recorded as an unproven match, with the title as its only evidence, and
        its alert is labeled "not model-checked". Everything else waits for the models.
        """
        if not self.models.down:
            return
        for watch, result in self.unjudged:
            spec = normalize_spec(watch["spec"])
            listing = result.summary
            if rules.check_style_code(spec, {}, listing.get("title") or "") != MATCH:
                continue
            if rules.over_ceiling(spec, asking_price(listing), listing.get("shipping")):
                continue
            read_at = self.store.now_iso()      # the search result is this tick's own
            with self.store.transaction():
                row = self.store.upsert_candidate(
                    watch["id"], result.item_id, "maybe", reason=REASON_NOT_MODEL_CHECKED, tracked=True,
                    listing=snapshot(listing, read_at), legacy_item_id=listing.get("legacy_item_id"),
                    end_at=listing.get("end_at"), fetched_at=read_at, purge_at=None,
                )
                self.store.add_verdict(
                    row["id"], "decision", fields={"style_code": {"verdict": MATCH, "source": "title"}},
                    rule_fired=REASON_NOT_MODEL_CHECKED,
                )
                self._announce(watch)
            self.report["not_model_checked"] += 1

    def _note_model_calls(self) -> None:
        if self.models.unavailable:
            self._problem("crusoe", CRUSOE_DOWN_MESSAGE)
        if self.models.unusable:
            self._problem("crusoe", "%d model call(s) gave no usable answer" % self.models.unusable)

    # ---- phase 5: delivery ---------------------------------------------------------

    def _fresh_payload(self, alert: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Re-read the listing behind an alert that waited, so its card goes out with current data.

        Returns None when there is nothing left to alert on: the listing is no longer
        tracked, is over, or is sold out. Raises EbayError when it could not be read just
        now, and `alerts.NotReady` when it is not being read just now; both keep the alert
        waiting.

        A failed read here is never counted against the listing. This runs on every tick
        for as long as an alert waits, so counting would end a live listing within minutes
        of eBay or the network going away. Whether a listing is over is for the paced
        refresh to find out, at its own pace. While that is retrying a listing, the alert
        waits for it and no call is made here.
        """
        candidate = self.store.get_candidate_by_id(alert["candidate_id"])
        if candidate is None or not candidate["tracked"] or not candidate["listing"]:
            return None
        if candidate["id"] not in self.read_now:
            fresh, full = self._read_for_release(candidate)
            if fresh.get("availability") == OUT_OF_STOCK or (full and self._is_over(fresh)):
                return None
            candidate = self._keep(candidate, {}, fresh, full=full)
        elif candidate["listing"].get("availability") == OUT_OF_STOCK:
            return None
        return dict(alert["payload"], as_of=candidate["fetched_at"], **self._card_facts(candidate))

    def _read_for_release(self, candidate: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Read a listing whose alert is about to go out: (what eBay returned, whether in full).

        The read is in full when the snapshot's last full read is over six hours old, so
        the card's shipping, condition and end time are as current as its price.
        """
        if candidate["state"] == "unavailable":
            raise alerts.NotReady("the listing could not be read lately; its next retry decides")
        if self.ebay_stopped:
            raise alerts.NotReady("eBay is not being called for the rest of this tick")
        full = self._needs_full_read(candidate)
        try:
            return self._read(candidate, full=full), full
        except ConfigError as problem:
            self.ebay_stopped = True
            self._problem("config", str(problem))
            raise alerts.NotReady(str(problem)) from None
        except (EbayAuthError, EbayBudgetExceeded, EbayUnreachable) as problem:
            self._ebay_is_out(problem)
            raise
        except EbayError:
            # An answer about the listing itself. It is set aside with nothing counted, and
            # the paced refresh, due at once, takes it from here.
            self._set_aside(candidate, {}, 0, 0, self.now)
            raise

    # ---- phase 6: safety notices ---------------------------------------------------

    def _send_notices(self) -> None:
        """Send each safety notice once a day.

        A notice gets an outbox row keyed by its name and the day before it is sent, so a
        second tick on the same day adds nothing. One that could not be delivered stays
        pending and is tried again by the next tick.
        """
        self._note_quota()
        self._note_swaps()
        day = self.now.astimezone(timezone.utc).strftime("%Y-%m-%d")     # the ledger's day
        for name, (text, urgent) in self.notices.items():
            self.store.enqueue_alert(
                "system:%s:%s" % (name, day), "system", "P1" if urgent else "P2",
                {"kind": "system", "text": text, "urgent": urgent},
            )
        if not self.notifier.channels():
            return
        too_old = iso(self.now - NOTICE_SHELF_LIFE)
        for alert in self.store.alerts_by_status("pending"):
            if alert["kind"] != "system":
                continue
            if alert["created_at"] < too_old:
                self.store.update_alert(alert["id"], status="dropped")
            elif self.notifier.send_system(alert["payload"]["text"], urgent=bool(alert["payload"].get("urgent"))):
                self.store.update_alert(alert["id"], status="sent")
                self.report["notices"].append(alert["dedupe_key"].split(":")[1])
            else:
                gave_up = alert["attempts"] + 1 >= alerts.MAX_ATTEMPTS
                self.store.update_alert(alert["id"], status="failed" if gave_up else None, add_attempt=True)

    def _note_quota(self) -> None:
        used, cap = self.store.ledger_total("ebay"), self.settings.ebay_daily_cap
        if cap > 0 and used >= QUOTA_NOTICE_SHARE * cap:
            self.notices["ebay_quota"] = (
                "eBay calls today: %d of %d. Searching stops at the cap and resumes tomorrow (UTC)." % (used, cap),
                False,
            )

    def _note_swaps(self) -> None:
        if self.models.swaps:
            swaps = sorted(self.models.swaps.items())
            pairs = ", ".join("%s answered for %s" % (served, asked) for asked, served in swaps)
            self.notices["model_swap"] = (
                "Crusoe served a different model than the one requested (%s). Verdicts now come from the "
                "replacement. Run: eft doctor crusoe" % pairs,
                False,
            )

    # ---- phase 7: heartbeat --------------------------------------------------------

    def _heartbeat(self) -> Optional[bool]:
        """Tell the dead-man service that this tick was clean.

        None when no heartbeat is configured, False when it was withheld or could not be
        sent. The URL is a secret of sorts, so it never appears in a problem line.
        """
        if not self.settings.heartbeat_url:
            return None
        if self.report["problems"]:
            return False
        try:
            response = self.transport.request("GET", self.settings.heartbeat_url, timeout=HEARTBEAT_TIMEOUT)
        except BlockedUrl:
            self._problem("heartbeat", "EFT_HEARTBEAT_URL must be an https URL that is not an eBay page")
            return False
        except TransportError:
            self._problem("heartbeat", "could not be reached")
            return False
        if not response.ok:
            self._problem("heartbeat", "answered HTTP %d" % response.status)
            return False
        return True


# ---- helpers that need no clients ----------------------------------------------

def _just_sold_out(candidate: Mapping[str, Any], fresh: Mapping[str, Any]) -> bool:
    """True when a listing that was in stock at the last read is out of stock now.

    That can mean a size sold out or that the whole listing sold and is over. Only the
    full listing, which carries the end date, tells the two apart.
    """
    before = (candidate["listing"] or {}).get("availability")
    return fresh.get("availability") == OUT_OF_STOCK and before != OUT_OF_STOCK


def _left_to_sweep(state: str, reason: Optional[str]) -> bool:
    """True for a listing parked over the price ceiling, which the sweep re-prices.

    A listing parked because its size sold out is not: a search result does not say
    whether a size is back, so that one keeps its own reads.
    """
    return state == "parked" and reason == "over_ceiling"


def _released(spec: Mapping[str, Any], candidate: Mapping[str, Any], listing: Mapping[str, Any]) -> Optional[str]:
    """The follow-up kind when what kept a listing parked no longer holds, else None."""
    if candidate["reason"] == "over_ceiling":
        price = asking_price(listing)
        if price and not rules.over_ceiling(spec, price, listing.get("shipping")):
            return "price"
    if candidate["reason"] == "size_out_of_stock" and listing.get("availability") in ("IN_STOCK", "LIMITED_STOCK"):
        return "restock"
    return None


def _unproven(spec: Mapping[str, Any], verdicts: Mapping[str, str]) -> List[str]:
    """The required fields a stored decision did not prove."""
    requirements = spec.get("requirements") or {}
    return [name for name in MATCH_FIELDS if requirements.get(name) == "required" and verdicts.get(name) != MATCH]


def _tier(kind: str, watch: Mapping[str, Any], candidate: Mapping[str, Any], now: datetime) -> str:
    """`alerts.tier_for`, except that two kinds of unproven match do not wait for the digest:
    an auction that ends within a day, and a style-code match made while the models were down."""
    tier = alerts.tier_for(kind, watch, candidate)
    if tier != "P3" or kind not in FIRST_KINDS:
        return tier
    end = parse_iso(candidate.get("end_at"))
    auction = "AUCTION" in ((candidate.get("listing") or {}).get("buying_options") or [])
    ends_soon = auction and end is not None and now <= end <= now + UNCONFIRMED_WINDOW
    return "P2" if ends_soon or candidate.get("reason") == REASON_NOT_MODEL_CHECKED else tier
