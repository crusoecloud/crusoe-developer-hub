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
tick carries on with the next listing or watch.
"""
from __future__ import annotations

import math
import os
import traceback
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import alerts, rules
from .config import ConfigError, Settings
from .errors import (
    ChatResult, CrusoeAuthError, CrusoeError, CrusoeUnavailable, EbayAuthError, EbayBudgetExceeded,
    EbayError, EbayNotFound, EbayUnavailable,
)
from .funnel import REASON_BUDGET, REASON_MODEL_ERROR, Funnel, FunnelResult, asking_price, fetch_image, snapshot
from .http import BlockedUrl, TransportError, UrllibTransport
from .plan import build_plan
from .rules import MATCH, UNKNOWN
from .spec import MATCH_FIELDS, normalize_spec
from .store import Store, iso, parse_iso

# ---- the eBay call budget ----------------------------------------------------
RESERVED_CALLS = 400            # kept back for adds, dry runs and retries
CALLS_PER_WATCH = 25            # a day, for everything that is not a search
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
AUCTION_REFRESH = timedelta(hours=1)
# A card may show data for six hours. Refreshing at five leaves room for a late tick.
FIXED_PRICE_REFRESH = timedelta(hours=5)
LAST_HOUR = timedelta(hours=1)
# How long to wait after the first, second, third and fourth failed read of a tracked listing.
# The fifth read asks for the full listing, and if that fails too the listing is over.
RETRY_DELAYS = (timedelta(hours=1), timedelta(hours=6), timedelta(hours=24), timedelta(hours=72))
ENDED_KEEP = timedelta(days=14)
ENDING_MARKS = (10, 60)                     # minutes before the scheduled end, nearest first
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
    search (detail fetches, refreshes, sweeps). Searches share the rest. Hot watches are
    searched every 10 minutes, or every 5 while there are at most 10 watches in all. Normal
    watches get what is left: never more often than every 10 minutes, in steps of 5. Past
    what the cap can carry at all, every watch is searched once a day.
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


def next_refresh(listing: Mapping[str, Any], end_at: Optional[str], now: datetime) -> datetime:
    """When a tracked listing is read again.

    An auction is read every tick in its last hour and hourly before that. A fixed-price
    listing is read every five hours, which keeps the data on its card under six hours old.
    """
    end = parse_iso(end_at)
    if "AUCTION" in (listing.get("buying_options") or []):
        if end is not None and end - now <= LAST_HOUR:
            return now
        due = now + AUCTION_REFRESH
        return min(due, end - LAST_HOUR) if end is not None else due
    due = now + FIXED_PRICE_REFRESH
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
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.answered = 0
        self.unavailable = 0
        self.unusable = 0
        self.swaps: Dict[str, str] = {}

    def chat_json(self, role: str, system: str, parts: Sequence[Dict[str, Any]], **options: Any) -> ChatResult:
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

    def image_part(self, source: Any) -> Dict[str, Any]:
        return self.client.image_part(source)

    @property
    def down(self) -> bool:
        """True when calls failed as unavailable and not one was answered."""
        return self.unavailable > 0 and self.answered == 0


class _Tick:
    """The state of one poll: the clients, what has gone wrong so far, and the report."""

    def __init__(self, settings: Settings, store: Store, ebay: Any, crusoe: Any, notifier: Any,
                 funnel: Optional[Any], transport: Optional[Any]) -> None:
        self.settings = settings
        self.store = store
        self.ebay = ebay
        self.models = _ModelMeter(crusoe)
        self.notifier = notifier
        self.transport = transport if transport is not None else UrllibTransport()
        self.funnel = funnel or Funnel(
            ebay, self.models, store, image_fetcher=partial(fetch_image, transport=self.transport)
        )
        # Set once eBay or Crusoe has refused in a way that would repeat for every later call.
        self.stopped = False
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

    def _stop(self, area: str, message: str) -> None:
        self.stopped = True
        self._problem(area, message)

    def _guard(self, label: str, work: Callable[..., None], *args: Any) -> None:
        """Run one unit of work and turn its failure into a line of the report.

        A refused key, a spent budget or a missing setting would fail every later call the
        same way, so those also stop the eBay phases of this tick. Anything else, a bug
        included, costs only the listing or watch it happened in: the poller runs unattended,
        and one bad watch must not silence the others.
        """
        try:
            work(*args)
        except EbayBudgetExceeded:
            self._stop("ebay", BUDGET_MESSAGE)
        except EbayAuthError as problem:
            self.notices["ebay_auth"] = (EBAY_AUTH_NOTICE, True)
            self._stop("ebay", str(problem))
        except CrusoeAuthError as problem:
            self.notices["crusoe_auth"] = (CRUSOE_AUTH_NOTICE, True)
            self._stop("crusoe", str(problem))
        except ConfigError as problem:
            self._stop("config", str(problem))
        except (EbayError, CrusoeError) as problem:
            self._problem(label, str(problem))
        except Exception as problem:  # noqa: BLE001 - see the docstring
            # The message of an unexpected error is not repeated: nothing vouches for what is in it.
            where = traceback.extract_tb(problem.__traceback__)[-1]
            self._problem(label, "unexpected %s at %s:%d" % (
                problem.__class__.__name__, os.path.basename(where.filename), where.lineno or 0))

    # ---- phase 2: refresh tracked listings -----------------------------------------

    def _refresh_due(self) -> None:
        for candidate in self.store.due_for_refresh():
            if self.stopped:
                break
            self._guard("refresh %s" % candidate["code"], self._refresh, candidate)

    def _refresh(self, candidate: Dict[str, Any]) -> None:
        """Bring one tracked listing up to date, or find out that it is over.

        A compact read is enough for prices and bids. The full listing is read when only
        its end date can settle things: the scheduled end has passed, the listing has just
        gone out of stock, or four slow retries have failed.
        """
        retry = self.store.kv_get(RETRY_KEY % candidate["id"]) or {}
        confirm = self._end_passed(candidate) or retry.get("failures", 0) >= len(RETRY_DELAYS)
        try:
            fresh = self._read(candidate, full=confirm)
            if not confirm and _just_sold_out(candidate, fresh):
                fresh, confirm = self._read(candidate, full=True), True
        except (EbayAuthError, EbayBudgetExceeded):
            raise
        except EbayError as problem:
            self._refresh_failed(candidate, retry, problem)
            return
        end = parse_iso(fresh.get("end_at"))
        if confirm and end is not None and end <= self.now:
            self._end(candidate)
            return
        self._follow_up(self._keep(candidate, retry, fresh, full=confirm))

    def _read(self, candidate: Mapping[str, Any], full: bool) -> Dict[str, Any]:
        """One eBay call for a tracked listing. For a multi-size listing the snapshot holds
        the id of the variation that was judged, which is the one to read again."""
        item_id = (candidate["listing"] or {}).get("item_id") or candidate["item_id"]
        return self.ebay.get_item(item_id, compact=not full, purpose="refresh")

    def _end_passed(self, candidate: Mapping[str, Any]) -> bool:
        end = parse_iso(candidate["end_at"])
        return end is not None and end <= self.now

    def _keep(self, candidate: Mapping[str, Any], retry: Mapping[str, Any], fresh: Mapping[str, Any],
              full: bool) -> Dict[str, Any]:
        """Store what a read returned, schedule the next one, and return the updated row."""
        now = self.now
        listing = dict(candidate["listing"] or {})
        listing.update(snapshot(fresh) if full else {key: fresh.get(key) for key in COMPACT_KEYS})
        end_at = fresh.get("end_at") if full else candidate["end_at"]
        with self.store.transaction():
            self.store.update_candidate(
                candidate["id"], listing=listing, end_at=end_at, fetched_at=iso(now),
                next_refresh_at=iso(next_refresh(listing, end_at, now)),
                # A listing that was unreadable for a while goes back to the state it had.
                state=retry.get("was") or candidate["state"],
            )
            self.store.kv_delete(RETRY_KEY % candidate["id"])
        self.report["refreshed"] += 1
        return self.store.get_candidate_by_id(candidate["id"])  # type: ignore[return-value]

    def _refresh_failed(self, candidate: Mapping[str, Any], retry: Mapping[str, Any], problem: EbayError) -> bool:
        """A tracked listing could not be read. Returns True when that settles it as over.

        One failure settles nothing: listings come back, and eBay has bad minutes. The
        listing is marked unavailable and read again after 1, 6, 24 and 72 hours. It is over
        on a second "not found", on a second failure once its scheduled end has passed, or
        when the full read after the fourth retry fails as well.
        """
        failures = retry.get("failures", 0) + 1
        not_found = retry.get("not_found", 0) + int(isinstance(problem, EbayNotFound))
        if not isinstance(problem, EbayNotFound):
            self._problem("refresh %s" % candidate["code"], str(problem))
        if not_found >= 2 or (failures >= 2 and self._end_passed(candidate)) or failures > len(RETRY_DELAYS):
            self._end(candidate)
            return True
        with self.store.transaction():
            self.store.kv_set(RETRY_KEY % candidate["id"], {
                "failures": failures, "not_found": not_found, "was": retry.get("was") or candidate["state"],
            })
            self.store.update_candidate(
                candidate["id"], state="unavailable", next_refresh_at=iso(self.now + RETRY_DELAYS[failures - 1]),
            )
        self.report["unavailable"] += 1
        return False

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
        """Queue an "ending soon" reply at 60 and at 10 minutes before an auction's scheduled end.

        Only for what the user singled out (a starred listing or a hot watch), and only for
        a listing they have already been sent: a reminder about an unknown listing would
        arrive alongside its first card.
        """
        if not (candidate["starred"] or watch["tier"] == "hot"):
            return
        end = parse_iso(candidate["end_at"])
        if end is None or "AUCTION" not in (candidate["listing"].get("buying_options") or []):
            return
        minutes_left = (end - self.now).total_seconds() / 60.0
        if minutes_left <= 0 or not self._announced(candidate["id"], sent_only=True):
            return
        for mark in ENDING_MARKS:
            if minutes_left <= mark:
                # Nearest mark first, so a first look inside the last 10 minutes sends one reminder.
                self._enqueue(watch, spec, candidate, "ending", "ending%d:%d" % (mark, candidate["id"]))
                return

    # ---- phases 3 and 4: search and sweep ------------------------------------------

    def _search_due(self) -> None:
        watches = self.store.list_watches(status="active")
        hot = sum(1 for watch in watches if watch["tier"] == "hot")
        cadence = cadence_minutes(hot, len(watches) - hot, self.settings.ebay_daily_cap)
        self.report["cadence"] = cadence
        for watch in sorted(watches, key=lambda row: row["tier"] != "hot"):
            if not self.stopped and self._search_is_due(watch, cadence):
                self._guard("watch %d" % watch["id"], self._search, watch)
            if not self.stopped and self._sweep_is_due(watch["id"]):
                self._guard("watch %d sweep" % watch["id"], self._sweep, watch)

    def _search_is_due(self, watch: Mapping[str, Any], cadence: Mapping[str, int]) -> bool:
        last = parse_iso(watch["last_search_at"])
        every = timedelta(minutes=cadence["hot" if watch["tier"] == "hot" else "normal"])
        return last is None or self.now - last >= every - DUE_SLACK

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
        results = self.funnel.run(watch, listings)
        self._tally(results)
        waiting = [result for result in results if result.decision == "pending"]
        self.unjudged.extend((watch, result) for result in waiting if result.reason == REASON_MODEL_ERROR)
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

    def _tally(self, results: Sequence[FunnelResult]) -> None:
        """Count the funnel's decisions and pick out what it absorbed that the report should show."""
        counts = self.report["decisions"]
        for result in results:
            counts[result.decision] = counts.get(result.decision, 0) + 1
            if result.reason == REASON_BUDGET:
                self._stop("ebay", BUDGET_MESSAGE)
            elif result.reason in ("ebay_unavailable", "ebay_error"):
                self._problem("ebay", "a listing could not be read (%s)" % result.reason)

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
                self.store.update_candidate(candidate["id"], listing=listing)
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
        """
        self._tally(self.funnel.run(watch, listings, recheck=True))
        spec = normalize_spec(watch["spec"])
        with self.store.transaction():
            for candidate_id, kind in kinds.items():
                candidate = self.store.get_candidate_by_id(candidate_id)
                if candidate and candidate["tracked"] and candidate["state"] in ANNOUNCED_STATES:
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
                due = next_refresh(candidate["listing"] or {}, candidate["end_at"], fetched)
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
        queued = self.store.enqueue_alert(
            key, kind, tier, payload, watch_id=watch["id"], candidate_id=candidate["id"],
            as_of=candidate["fetched_at"],
        )
        if queued is not None:
            self.report["queued"] += 1

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
            with self.store.transaction():
                row = self.store.upsert_candidate(
                    watch["id"], result.item_id, "maybe", reason=REASON_NOT_MODEL_CHECKED, tracked=True,
                    listing=snapshot(listing), legacy_item_id=listing.get("legacy_item_id"),
                    end_at=listing.get("end_at"), fetched_at=self.store.now_iso(), purge_at=None,
                )
                self.store.add_verdict(
                    row["id"], "decision", fields={"style_code": {"verdict": MATCH, "source": "title"}},
                    rule_fired=REASON_NOT_MODEL_CHECKED,
                )
                self._announce(watch)
            self.report["not_model_checked"] += 1

    def _note_model_calls(self) -> None:
        if self.models.unavailable:
            self._problem("crusoe", "%d model call(s) failed: the service was unavailable" % self.models.unavailable)
        if self.models.unusable:
            self._problem("crusoe", "%d model call(s) gave no usable answer" % self.models.unusable)

    # ---- phase 5: delivery ---------------------------------------------------------

    def _fresh_payload(self, alert: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Re-read the listing behind an alert that waited, so its card goes out with current data.

        Returns None when there is nothing left to alert on: the listing is no longer
        tracked, is over, or is sold out. Raises EbayError when it could not be read just
        now, which keeps the alert waiting.
        """
        candidate = self.store.get_candidate_by_id(alert["candidate_id"])
        if candidate is None or not candidate["tracked"] or not candidate["listing"]:
            return None
        if self.stopped:
            raise EbayUnavailable("eBay is not being called for the rest of this tick")
        retry = self.store.kv_get(RETRY_KEY % candidate["id"]) or {}
        try:
            fresh = self._read(candidate, full=False)
        except ConfigError as problem:
            raise EbayUnavailable(str(problem)) from None
        except (EbayAuthError, EbayBudgetExceeded):
            raise
        except EbayError as problem:
            if self._refresh_failed(candidate, retry, problem):
                return None
            raise
        if fresh.get("availability") == OUT_OF_STOCK:
            return None
        row = self._keep(candidate, retry, fresh, full=False)
        return dict(alert["payload"], as_of=row["fetched_at"],
                    **{key: row["listing"].get(key) for key in COMPACT_KEYS if key in alert["payload"]})

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
