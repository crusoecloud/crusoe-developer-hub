import copy
import json
import re
import sqlite3
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone
from unittest import mock

import support
from eft_core import alerts, plan, spec
from eft_core.alerts import Notifier
from eft_core.errors import (
    ChatResult, CrusoeAuthError, CrusoeError, CrusoeUnavailable, EbayAuthError, EbayBudgetExceeded,
    EbayNotFound, EbayUnavailable,
)
from eft_core.http import FakeTransport, Response, json_response
from eft_core.store import iso, parse_iso
from eft_core.tick import LAST_TICK_KEY, cadence_minutes, next_refresh, run_tick, window_start

SLACK = {"SLACK_BOT_TOKEN": "fake-slack-secret-bot-token", "SLACK_USER_ID": "U0USER"}
PUSH = {"PUSHOVER_APP_TOKEN": "secret-app-token", "PUSHOVER_USER_KEY": "secret-user-key"}
POST = "https://slack.com/api/chat.postMessage"
UPDATE = "https://slack.com/api/chat.update"
PUSHOVER = "https://api.pushover.net/1/messages.json"
HEARTBEAT = "https://beat.example/ping/secret-heartbeat-id"
DETAIL_ONLY_KEYS = ("aspects", "description", "availability", "min_next_bid", "unique_bidders", "reserve_met")


def usd(value):
    return {"value": value, "currency": "USD"}


def dunk_spec(**overrides):
    base = {
        "brand": "Nike", "model": "Dunk Low", "style_code": "DD1391-100",
        "colorway": {"official": "White/Black", "nicknames": ["Panda"]},
        "size": "US 10", "department": "men", "price_ceiling": 180,
    }
    base.update(overrides)
    return spec.normalize_spec(base)


def detail(n=1, **overrides):
    """A Detail as ebay.py returns it: a men's US 10 Panda Dunk at a fixed price unless overridden."""
    base = {
        "item_id": "v1|%d|0" % n, "legacy_item_id": str(n), "title": "Nike Dunk Low Panda White Black",
        "price": usd("120.00"), "current_bid": None, "bid_count": None,
        "buying_options": ["FIXED_PRICE"], "shipping": usd("10.00"),
        "end_at": None, "created_at": "2026-11-12T17:50:00Z", "origin_at": "2026-11-12T17:50:00Z",
        "relisted": False, "image_url": "https://i.ebayimg.com/images/g/%d/s-l1600.jpg" % n,
        "additional_image_urls": [], "condition": "Pre-owned", "condition_id": "3000",
        "item_group_id": None, "web_url": "https://www.ebay.com/itm/%d" % n, "category_ids": ["15709"],
        "aspects": {"Brand": ["Nike"], "US Shoe Size": ["10"], "Department": ["Men"],
                    "Style Code": ["DD1391-100"]},
        "description": "Nike Dunk Low Panda in white and black. Worn twice, no box.",
        "availability": "IN_STOCK", "min_next_bid": None, "unique_bidders": None, "reserve_met": None,
    }
    base.update(overrides)
    return base


def auction(n=1, ends_at=None, **overrides):
    fields = {"buying_options": ["AUCTION"], "price": usd("61.00"), "current_bid": usd("61.00"),
              "bid_count": 3, "min_next_bid": usd("62.00"), "end_at": ends_at}
    fields.update(overrides)
    return detail(n, **fields)


def verify_answer(**fields):
    answer = {name: {"verdict": "unknown", "quote": ""} for name in spec.MATCH_FIELDS}
    for name, (verdict, quote) in fields.items():
        answer[name] = {"verdict": verdict, "quote": quote}
    return {"fields": answer, "flags": []}


# What the text model reads off the standard listing; code settles size, department and style code.
READS_RIGHT = verify_answer(brand=("match", "Brand: Nike"), model=("match", "Nike Dunk Low"),
                            colorway=("match", "Panda"))


def all_titles_look_right(parts):
    count = len(re.findall(r"^\d+\. ", parts[0]["text"], flags=re.MULTILINE))
    return {"results": [{"n": n, "verdict": "yes", "reason": "looks_right"} for n in range(1, count + 1)]}


def words(obj):
    return json.dumps(obj, ensure_ascii=False)


class FakeCrusoe:
    """A scripted model. `answers[role]` is a dict, an exception or a callable(parts)."""

    def __init__(self):
        self.answers = {"triage": all_titles_look_right, "verify": READS_RIGHT}
        self.swap_to = None
        self.calls = []

    def chat_json(self, role, system, parts, required=(), validator=None, max_tokens=None):
        self.calls.append(role)
        answer = self.answers[role]
        if callable(answer):
            answer = answer(parts)
        if isinstance(answer, Exception):
            raise answer
        served = self.swap_to or "served-" + role
        return ChatResult(data=copy.deepcopy(answer), model_requested="served-" + role, model_served=served,
                          swapped=self.swap_to is not None, latency_ms=40, prompt_tokens=90, completion_tokens=10)

    def image_part(self, source):
        return {"type": "image", "data": source, "mime": "image/jpeg"}


class FakeEbay:
    """eBay as the poller sees it: live listings, a search that returns all of them, scripted failures."""

    def __init__(self):
        self.live = {}
        self.fail = {}              # item id -> exceptions raised by get_item, one per call
        self.search_fail = []       # exceptions raised by search, one per call (None lets a call through)
        self.search_error = None    # raised by every search
        self.endless = False        # every page says another one follows
        self.calls = []

    def list(self, *details):
        for listing in details:
            self.live[listing["item_id"]] = listing

    def search(self, q=None, **kwargs):
        self.calls.append(("search", q, kwargs))
        problem = self.search_error or (self.search_fail.pop(0) if self.search_fail else None)
        if problem is not None:
            raise problem
        items = [{key: value for key, value in listing.items() if key not in DETAIL_ONLY_KEYS}
                 for listing in copy.deepcopy(list(self.live.values()))]
        offset, limit = kwargs.get("offset", 0), kwargs.get("limit", 200)
        return {"total": 10000 if self.endless else len(items), "items": items,
                "next_offset": offset + limit if self.endless else None,
                "dominant_category_id": None, "warnings": []}

    def get_item(self, item_id, *, compact=False, purpose="detail"):
        self.calls.append(("get_item", item_id, {"compact": compact, "purpose": purpose}))
        problem = self.fail[item_id].pop(0) if self.fail.get(item_id) else None
        if problem is not None:
            raise problem
        listing = copy.deepcopy(self.live[item_id])
        if compact:
            # What ebay.py returns for fieldgroups=COMPACT: these keys come back empty.
            listing.update(title="", buying_options=[], end_at=None, image_url=None)
        return listing

    def get_items_by_item_group(self, item_group_id, *, purpose="detail"):
        raise AssertionError("no test here lists a multi-size listing")

    def searches(self, purpose="search"):
        return [kwargs for name, _, kwargs in self.calls if name == "search" and kwargs.get("purpose") == purpose]

    def reads(self, purpose):
        return [(item_id, kwargs["compact"]) for name, item_id, kwargs in self.calls
                if name == "get_item" and kwargs["purpose"] == purpose]


class Wire:
    """Stands in for Slack, Pushover and the heartbeat service, and records what was sent."""

    def __init__(self):
        self.transport = FakeTransport(self.handle)
        self.push_ok = True
        self.heartbeat_status = 200
        self.posted = 0

    def handle(self, method, url, headers, body):
        if url == POST:
            self.posted += 1
            return json_response({"ok": True, "channel": "D0DM", "ts": "1700000000.%06d" % self.posted})
        if url == UPDATE:
            return json_response({"ok": True})
        if url == PUSHOVER:
            return json_response({"status": 1, "request": "r1"} if self.push_ok else {"status": 0, "errors": ["down"]},
                                 status=200 if self.push_ok else 500)
        if url == HEARTBEAT:
            return Response(self.heartbeat_status)
        raise AssertionError("unexpected request: %s %s" % (method, url))

    def bodies(self, url):
        return [json.loads(call["body"]) for call in self.transport.calls if call["url"] == url]

    def posts(self):
        return self.bodies(POST)

    def updates(self):
        return self.bodies(UPDATE)

    def pushes(self):
        return [dict(urllib.parse.parse_qsl(call["body"].decode())) for call in self.transport.calls
                if call["url"] == PUSHOVER]

    def beats(self):
        return [call for call in self.transport.calls if call["url"] == HEARTBEAT]


class TickTest(unittest.TestCase):
    ENV = SLACK
    START = support.FIXED_NOW      # 18:00 UTC, which the tests treat as 13:00 local

    def setUp(self):
        self.clock = support.Clock(self.START)
        self.store = support.make_store(self.clock)
        self.settings = support.make_settings(**self.ENV)
        self.ebay = FakeEbay()
        self.crusoe = FakeCrusoe()
        self.wire = Wire()
        self.notifier = Notifier(self.settings, self.store, transport=self.wire.transport, local_now=self.local_now)

    def local_now(self):
        return (self.clock() - timedelta(hours=5)).replace(tzinfo=None)

    def watch(self, name="Panda Dunks 10", tier="normal", **spec_overrides):
        wanted = dunk_spec(**spec_overrides)
        watch_id = self.store.create_watch(name, wanted, status="active", tier=tier, plan=plan.build_plan(wanted))
        return self.store.get_watch(watch_id)

    def tick(self):
        return run_tick(self.settings, self.store, self.ebay, self.crusoe, self.notifier,
                        transport=self.wire.transport)

    def candidate(self, n=1, watch_id=1):
        return self.store.get_candidate(watch_id, "v1|%d|0" % n)

    def alerts(self, kind=None):
        return [row for row in self.store.alerts_by_status() if kind is None or row["kind"] == kind]


class Cadence(unittest.TestCase):
    CAP = 4800

    def day(self, hot, normal):
        """(cadence, searches a day, all calls a day) with the reserve and the per-watch allowance."""
        cadence = cadence_minutes(hot, normal, self.CAP)
        searches = hot * 1440.0 / cadence["hot"] + normal * 1440.0 / cadence["normal"]
        return cadence, searches, searches + 25 * (hot + normal) + 400

    def test_ten_hot_watches_are_searched_every_five_minutes(self):
        cadence, searches, total = self.day(10, 0)
        self.assertEqual(cadence["hot"], 5)
        self.assertEqual(searches, 2880)                # 10 watches x 288 searches
        self.assertEqual(total, 2880 + 250 + 400)       # 3,530 of 4,800
        self.assertLessEqual(total, self.CAP)

    def test_twenty_five_watches_are_searched_every_ten_minutes(self):
        cadence, searches, total = self.day(0, 25)
        self.assertEqual(cadence["normal"], 10)
        self.assertEqual(searches, 3600)                # 25 watches x 144 searches
        self.assertEqual(total, 3600 + 625 + 400)       # 4,625 of 4,800
        self.assertLessEqual(total, self.CAP)

    def test_fifty_watches_slow_the_normal_ones_down(self):
        cadence, searches, total = self.day(10, 40)
        self.assertEqual(cadence, {"hot": 10, "normal": 35})
        # 4,800 - 400 reserve - 50 x 25 leaves 3,150 searches; the hot watches take 1,440 of them.
        self.assertAlmostEqual(searches, 1440 + 40 * 1440 / 35.0)
        self.assertLessEqual(total, self.CAP)
        # One step faster would not fit: 1,440 + 40 x 48 + 1,250 + 400 = 5,010.
        self.assertGreater(1440 + 40 * 1440 / 30.0 + 1250 + 400, self.CAP)
        self.assertEqual(self.day(0, 50)[0]["normal"], 25)

    def test_normal_watches_are_never_searched_more_often_than_every_ten_minutes(self):
        self.assertEqual(cadence_minutes(0, 1, self.CAP), {"hot": 5, "normal": 10})
        self.assertEqual(cadence_minutes(1, 10, self.CAP)["hot"], 10, "5 minutes is for 10 watches or fewer")

    def test_any_mix_the_cap_can_carry_stays_inside_it(self):
        for hot in range(0, 61, 5):
            for normal in range(0, 121, 5):
                if hot + normal == 0 or self.CAP - 400 - 25 * (hot + normal) < hot + normal:
                    continue
                cadence, _, total = self.day(hot, normal)
                self.assertLessEqual(total, self.CAP, (hot, normal, cadence))
                self.assertEqual((cadence["hot"] % 5, cadence["normal"] % 5), (0, 0))

    def test_past_what_the_cap_can_carry_every_watch_is_searched_daily(self):
        self.assertEqual(cadence_minutes(60, 120, self.CAP), {"hot": 1440, "normal": 1440})
        self.assertEqual(cadence_minutes(30, 0, self.CAP)["hot"], 15, "too many hot watches slow each other down")


class Schedules(unittest.TestCase):
    NOW = support.FIXED_NOW

    def test_first_window_looks_back_a_day_and_later_ones_overlap_by_15_minutes(self):
        never = {"window_end": iso(self.NOW), "last_search_at": None}
        self.assertEqual(window_start(never, self.NOW), self.NOW - timedelta(hours=24))
        before = {"window_end": iso(self.NOW - timedelta(minutes=10)), "last_search_at": iso(self.NOW)}
        self.assertEqual(window_start(before, self.NOW), self.NOW - timedelta(minutes=25))

    def test_no_window_reaches_back_more_than_72_hours(self):
        asleep = {"window_end": iso(self.NOW - timedelta(days=9)), "last_search_at": iso(self.NOW - timedelta(days=9))}
        self.assertEqual(window_start(asleep, self.NOW), self.NOW - timedelta(hours=72))

    def test_refresh_times(self):
        fixed, bidding = {"buying_options": ["FIXED_PRICE"]}, {"buying_options": ["AUCTION"]}
        hours = lambda n: self.NOW + timedelta(hours=n)  # noqa: E731
        self.assertEqual(next_refresh(fixed, None, self.NOW), hours(5))
        self.assertEqual(next_refresh(fixed, iso(hours(2)), self.NOW), hours(2), "read again when it ends")
        self.assertEqual(next_refresh(bidding, iso(hours(30)), self.NOW), hours(1))
        self.assertEqual(next_refresh(bidding, iso(hours(1.5)), self.NOW), hours(0.5), "the last hour starts on time")
        self.assertEqual(next_refresh(bidding, iso(hours(0.75)), self.NOW), self.NOW, "every tick in the last hour")


class SearchAndAlert(TickTest):
    def test_first_tick_finds_a_match_queues_one_alert_and_sends_it(self):
        watch = self.watch()
        self.ebay.list(detail())
        report = self.tick()
        self.assertEqual(report["problems"], [])
        self.assertEqual((report["searched"], report["decisions"], report["queued"]), (1, {"alert": 1}, 1))
        self.assertEqual(report["delivery"], {"sent": 1, "held": 0, "dropped": 0, "failed": 0})

        row = self.candidate()
        self.assertEqual((row["state"], row["tracked"]), ("alerted", 1))
        self.assertEqual(row["next_refresh_at"], iso(self.clock() + timedelta(hours=5)))
        [alert] = self.alerts()
        self.assertEqual((alert["kind"], alert["tier"], alert["status"]), ("new", "P2", "sent"))
        self.assertEqual(alert["dedupe_key"], "new:%d:%s" % (row["id"], iso(self.clock())), "one per verdict")
        self.assertEqual(alert["payload"]["web_url"], "https://www.ebay.com/itm/1")
        self.assertIn("Size US 10 (item specifics)", alert["payload"]["proof"])
        self.assertEqual(alert["payload"]["unconfirmed"], [])
        [post] = self.wire.posts()
        self.assertIn(row["code"], post["text"])
        self.assertFalse(post["unfurl_links"])

        [search] = self.ebay.searches()
        self.assertEqual(search["start_from"], self.clock() - timedelta(hours=24), "a first search looks back a day")
        self.assertEqual(search["start_to"], self.clock())
        self.assertEqual(search["buying_options"], ["FIXED_PRICE", "AUCTION", "BEST_OFFER"])
        stored = self.store.get_watch(watch["id"])
        self.assertEqual((stored["window_end"], stored["last_search_at"]), (iso(self.clock()), iso(self.clock())))

    def test_second_tick_does_not_alert_again(self):
        self.watch()
        self.ebay.list(detail())
        self.tick()
        first_search_at = self.clock()
        self.clock.advance(minutes=10)
        report = self.tick()       # the 15-minute overlap brings the same listing back
        self.assertEqual((report["queued"], report["delivery"]["sent"], report["decisions"]), (0, 0, {}))
        self.assertEqual(len(self.alerts()), 1)
        self.assertEqual(len(self.wire.posts()), 1)
        self.assertEqual(self.ebay.searches()[1]["start_from"], first_search_at - timedelta(minutes=15))

    def test_a_watch_that_is_not_due_or_not_active_is_not_searched(self):
        self.watch()
        paused = self.watch("Paused")
        self.store.update_watch(paused["id"], status="paused")
        self.tick()
        self.assertEqual(len(self.ebay.searches()), 1)
        self.clock.advance(minutes=5)
        self.tick()
        self.assertEqual(len(self.ebay.searches()), 1, "a normal watch is searched every 10 minutes")
        self.clock.advance(minutes=5)
        self.tick()
        self.assertEqual(len(self.ebay.searches()), 2)

    def test_an_unproven_match_waits_for_the_digest_unless_its_auction_ends_within_a_day(self):
        self.crusoe.answers["verify"] = verify_answer(brand=("match", "Brand: Nike"), model=("match", "Nike Dunk Low"))
        self.watch()
        self.ebay.list(detail(1), auction(2, ends_at=iso(self.clock() + timedelta(hours=20))),
                       auction(3, ends_at=iso(self.clock() + timedelta(days=3))))
        self.tick()
        tiers = {alert["payload"]["web_url"][-1]: alert["tier"] for alert in self.alerts("new")}
        self.assertEqual(tiers, {"1": "P3", "2": "P2", "3": "P3"})
        for alert in self.alerts():
            self.assertEqual(alert["payload"]["unconfirmed"], ["colorway"])
        texts = [post["text"] for post in self.wire.posts()]
        self.assertEqual(len(texts), 2, "one card for the auction, one digest for the rest")
        self.assertTrue(any(text.startswith("Unconfirmed match") for text in texts))

    def test_more_than_three_pages_flags_the_query_as_too_broad(self):
        watch = self.watch()
        self.ebay.endless = True
        self.tick()
        self.assertEqual([search["offset"] for search in self.ebay.searches()], [0, 200, 400])
        self.assertIsNotNone(self.store.kv_get("watch:%d:overflow" % watch["id"]))
        self.ebay.endless = False
        self.clock.advance(minutes=10)
        self.tick()
        self.assertIsNone(self.store.kv_get("watch:%d:overflow" % watch["id"]))

    def test_the_report_is_kept_for_status_and_holds_no_listing_content_or_secret(self):
        self.watch()
        self.ebay.list(detail())
        report = self.tick()
        self.assertEqual(self.store.kv_get(LAST_TICK_KEY), report)
        text = words(report)
        for unwanted in ("Nike", "120.00", "ebay.com/itm") + tuple(SLACK.values()):
            self.assertNotIn(unwanted, text)


class Failures(TickTest):
    def test_a_crash_in_the_middle_of_a_watch_leaves_no_alert_and_the_window_where_it_was(self):
        watch = self.watch()
        self.ebay.list(detail())
        self.crusoe.answers["verify"] = RuntimeError("boom: something nobody planned for")
        report = self.tick()
        self.assertEqual(self.alerts(), [])
        self.assertEqual(self.store.list_candidates(), [])
        stored = self.store.get_watch(watch["id"])
        self.assertEqual((stored["window_end"], stored["last_search_at"]), (None, None))
        [problem] = report["problems"]
        self.assertTrue(problem.startswith("watch 1: unexpected RuntimeError at "), problem)
        self.assertNotIn("boom", problem)

        self.crusoe.answers["verify"] = READS_RIGHT
        self.clock.advance(minutes=5)
        self.tick()
        self.assertEqual(len(self.wire.posts()), 1, "the next tick searches the same window and alerts")

    def test_a_crash_while_queueing_rolls_back_and_the_next_tick_alerts_exactly_once(self):
        watch = self.watch()
        self.ebay.list(detail())
        with mock.patch.object(self.store, "update_watch", side_effect=sqlite3.OperationalError("disk I/O error")):
            self.tick()
        # The funnel had recorded its verdict, but the alert and the window go together or not at all.
        self.assertEqual(self.candidate()["state"], "alerted")
        self.assertEqual(self.alerts(), [])
        self.assertIsNone(self.store.get_watch(watch["id"])["window_end"])
        self.assertEqual(self.wire.posts(), [])

        for _ in range(3):
            self.clock.advance(minutes=10)
            self.tick()
        self.assertEqual([alert["status"] for alert in self.alerts()], ["sent"])
        self.assertEqual(len(self.wire.posts()), 1)

    def test_one_failing_watch_does_not_stop_the_next(self):
        first, second = self.watch("Size 10"), self.watch("Size 11", size="US 11")
        self.ebay.list(detail(2, aspects={"Brand": ["Nike"], "US Shoe Size": ["11"], "Department": ["Men"]}))
        self.ebay.search_fail = [EbayUnavailable("eBay search failed: HTTP 503", status=503)]
        report = self.tick()
        self.assertEqual(report["problems"], ["watch %d: eBay search failed: HTTP 503" % first["id"]])
        self.assertIsNone(self.store.get_watch(first["id"])["last_search_at"])
        self.assertEqual(self.candidate(2, second["id"])["state"], "alerted")
        self.assertEqual([alert["watch_id"] for alert in self.alerts()], [second["id"]])
        self.assertEqual(report["searched"], 1)

    def test_a_spent_budget_ends_the_search_phase_and_the_rest_of_the_tick_still_runs(self):
        self.watch("One")
        self.watch("Two")
        self.ebay.search_error = EbayBudgetExceeded("eBay daily call cap reached (4800 of 4800); no request was sent")
        waiting = self.store.enqueue_alert("test:waiting", "digest", "P2", {
            "code": "A1", "watch_id": 1, "watch_name": "One", "kind": "digest", "tier": "P2",
            "title": "", "web_url": "", "as_of": iso(self.clock())})
        report = self.tick()
        self.assertEqual(len(self.ebay.searches()), 1, "the second watch is not even tried")
        self.assertEqual(report["problems"], ["ebay: the daily call cap is reached; searching resumes tomorrow (UTC)"])
        self.assertEqual(report["searched"], 0)
        self.assertEqual(self.store.get_alert(waiting)["status"], "sent", "the outbox is still delivered")

    def test_a_budget_that_runs_out_inside_the_funnel_keeps_the_window_for_the_unseen_listings(self):
        first, second = self.watch("One"), self.watch("Two", size="US 11")
        soon = iso(self.clock() + timedelta(days=1))
        self.ebay.list(detail(1, end_at=soon), detail(2))      # the soonest-ending listing is fetched first
        self.ebay.fail["v1|2|0"] = [EbayBudgetExceeded("eBay daily call cap reached")]
        report = self.tick()
        self.assertEqual(report["decisions"], {"alert": 1, "pending": 1})
        self.assertEqual(len(self.alerts()), 1, "what was judged before the cap still alerts")
        stored = self.store.get_watch(first["id"])
        self.assertEqual(stored["window_end"], iso(self.clock() - timedelta(hours=24) + timedelta(minutes=15)))
        self.assertIsNone(self.store.get_watch(second["id"])["last_search_at"])
        self.assertIn("ebay: the daily call cap is reached; searching resumes tomorrow (UTC)", report["problems"])
        # Once there is budget again the same window is searched and the listing is judged.
        self.clock.advance(minutes=10)
        self.tick()
        self.assertEqual(self.ebay.searches()[1]["start_from"], self.ebay.searches()[0]["start_from"])
        self.assertEqual(self.candidate(2)["state"], "alerted")

    def test_a_listing_ebay_cannot_return_is_a_problem_and_holds_the_window(self):
        watch = self.watch()
        self.ebay.list(detail())
        self.ebay.fail["v1|1|0"] = [EbayUnavailable("eBay item lookup failed: HTTP 500", status=500)]
        report = self.tick()
        self.assertEqual(report["problems"], ["ebay: a listing could not be read (ebay_unavailable)"])
        self.assertLess(self.store.get_watch(watch["id"])["window_end"], iso(self.clock()))
        self.clock.advance(minutes=10)
        self.assertEqual(self.tick()["problems"], [])
        self.assertEqual(self.candidate()["state"], "alerted")
        self.assertEqual(self.store.get_watch(watch["id"])["window_end"], iso(self.clock()))


class Refresh(TickTest):
    def first_alert(self, listing):
        self.watch()
        self.ebay.list(listing)
        self.tick()
        return self.candidate()

    def test_a_tracked_auction_is_refreshed_and_followed_up(self):
        ends = self.clock() + timedelta(hours=2)
        row = self.first_alert(auction(ends_at=iso(ends)))
        self.assertEqual(row["next_refresh_at"], iso(self.clock() + timedelta(hours=1)))
        self.store.update_candidate(row["id"], starred=True)
        self.ebay.live["v1|1|0"].update(current_bid=usd("75.00"), price=usd("75.00"), bid_count=6,
                                        min_next_bid=usd("76.00"))

        self.clock.advance(minutes=65)          # 55 minutes to go
        report = self.tick()
        self.assertEqual(self.ebay.reads("refresh"), [("v1|1|0", True)], "a compact read")
        self.assertEqual(report["refreshed"], 1)
        row = self.candidate()
        self.assertEqual((row["listing"]["current_bid"], row["listing"]["bid_count"]), (usd("75.00"), 6))
        self.assertEqual(row["listing"]["title"], "Nike Dunk Low Panda White Black", "a compact read has no title")
        self.assertEqual(row["listing"]["buying_options"], ["AUCTION"])
        self.assertEqual(row["end_at"], iso(ends))
        self.assertEqual(row["fetched_at"], iso(self.clock()))
        self.assertEqual(row["next_refresh_at"], iso(self.clock()), "every tick in the last hour")

        [ending] = self.alerts("ending")
        self.assertEqual((ending["dedupe_key"], ending["tier"], ending["status"]),
                         ("ending60:%d" % row["id"], "P1", "sent"))
        self.assertEqual(ending["payload"]["current_bid"], usd("75.00"))
        self.assertEqual(ending["payload"]["as_of"], iso(self.clock()))
        first, reply = self.wire.posts()
        self.assertNotIn("thread_ts", first)
        self.assertEqual(reply["thread_ts"], "1700000000.000001", "the follow-up replies under the first card")
        self.assertIn("Ending soon", reply["text"])

        self.clock.advance(minutes=5)
        self.tick()
        self.assertEqual(len(self.alerts("ending")), 1, "one reminder per mark")
        self.clock.advance(minutes=42)          # 8 minutes to go
        self.tick()
        self.assertEqual(sorted(alert["dedupe_key"] for alert in self.alerts("ending")),
                         ["ending10:%d" % row["id"], "ending60:%d" % row["id"]])

    def test_an_auction_nobody_singled_out_gets_no_reminder(self):
        self.first_alert(auction(ends_at=iso(self.clock() + timedelta(hours=2))))
        self.clock.advance(minutes=65)
        self.tick()
        self.assertEqual(self.alerts("ending"), [])

    def test_a_listing_that_disappears_once_is_not_treated_as_ended(self):
        row = self.first_alert(detail())
        self.ebay.fail["v1|1|0"] = [EbayNotFound("eBay item lookup failed: HTTP 404, error 11001", status=404)]
        self.clock.advance(hours=5, minutes=1)
        report = self.tick()
        self.assertEqual((report["unavailable"], report["ended"], report["problems"]), (1, 0, []))
        row = self.candidate()
        self.assertEqual((row["state"], row["tracked"]), ("unavailable", 1))
        self.assertEqual(row["listing"]["title"], "Nike Dunk Low Panda White Black")
        self.assertEqual(row["next_refresh_at"], iso(self.clock() + timedelta(hours=1)), "first retry after an hour")
        self.assertEqual(self.wire.updates(), [])

        self.clock.advance(minutes=30)
        self.tick()
        self.assertEqual(len(self.ebay.reads("refresh")), 1, "not read again before the retry is due")
        self.clock.advance(minutes=31)
        self.tick()                               # the listing is back
        row = self.candidate()
        self.assertEqual((row["state"], row["tracked"]), ("alerted", 1))
        self.assertNotIn("has ended", words(self.wire.updates()))

    def test_retries_slow_down_while_ebay_cannot_return_a_listing(self):
        row = self.first_alert(detail())
        down = EbayUnavailable("eBay item lookup failed: HTTP 503", status=503)
        self.ebay.fail["v1|1|0"] = [down, down, down, down, down]
        self.clock.advance(hours=5, minutes=1)
        waits = []
        for _ in range(4):
            report = self.tick()
            due = self.candidate()["next_refresh_at"]
            waits.append(parse_iso(due) - self.clock())
            self.assertEqual(report["ended"], 0)
            self.assertIn("refresh %s: eBay item lookup failed: HTTP 503" % row["code"], report["problems"])
            self.clock.now = parse_iso(due)
        self.assertEqual(waits, [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24), timedelta(hours=72)])
        report = self.tick()                      # the fifth read asks for the full listing
        self.assertEqual(self.ebay.reads("refresh")[-1], ("v1|1|0", False))
        self.assertEqual(report["ended"], 1)

    def test_a_listing_confirmed_ended_is_purged_and_its_messages_rewritten(self):
        row = self.first_alert(detail())
        gone = EbayNotFound("eBay item lookup failed: HTTP 404, error 11001", status=404)
        self.ebay.fail["v1|1|0"] = [gone, gone]
        del self.ebay.live["v1|1|0"]            # searches no longer return it either
        self.clock.advance(hours=5, minutes=1)
        self.tick()
        self.clock.advance(hours=1, minutes=1)
        report = self.tick()                      # not found a second time
        self.assertEqual(report["ended"], 1)

        ended = self.wire.updates()[-1]
        self.assertEqual((ended["channel"], ended["ts"]), ("D0DM", "1700000000.000001"))
        self.assertIn("has ended", words(ended))
        for gone_text in ("Nike Dunk", "120.00", "ebay.com/itm"):
            self.assertNotIn(gone_text, words(ended))
        row = self.candidate()
        self.assertEqual((row["state"], row["tracked"], row["listing"]), ("ended", 0, None))
        self.assertEqual(row["purge_at"], iso(self.clock() + timedelta(days=14)))
        self.assertIsNone(self.store.kv_get("candidate:%d:refresh" % row["id"]))
        [alert] = self.alerts("new")
        self.assertEqual(set(alert["payload"]) & {"title", "price", "web_url", "image_url"}, set())
        self.assertNotIn("Nike", words(self.store.verdicts_for(row["id"])))

        self.clock.advance(days=14, minutes=1)
        self.assertEqual(self.tick()["purged"], 1)
        self.assertIsNone(self.store.get_candidate_by_id(row["id"]))

    def test_an_auction_past_its_end_is_confirmed_with_the_full_listing(self):
        ends = self.clock() + timedelta(minutes=30)
        self.first_alert(auction(ends_at=iso(ends)))
        self.clock.advance(minutes=35)
        report = self.tick()                      # eBay still returns it, with its end date in the past
        self.assertEqual(self.ebay.reads("refresh"), [("v1|1|0", False)], "only the full listing has the end date")
        self.assertEqual(report["ended"], 1)
        self.assertEqual(self.candidate()["state"], "ended")
        self.assertIn("has ended", words(self.wire.updates()[-1]))

    def test_a_listing_that_sells_out_is_checked_for_its_end_and_parked_if_it_lives_on(self):
        self.first_alert(detail())
        self.ebay.live["v1|1|0"]["availability"] = "OUT_OF_STOCK"
        self.clock.advance(hours=5, minutes=1)
        self.tick()
        self.assertEqual(self.ebay.reads("refresh"), [("v1|1|0", True), ("v1|1|0", False)])
        row = self.candidate()
        self.assertEqual((row["state"], row["reason"], row["tracked"]), ("parked", "size_out_of_stock", 1))

        self.ebay.live["v1|1|0"]["availability"] = "IN_STOCK"
        self.clock.advance(hours=3)                 # still daytime
        self.store.update_candidate(row["id"], next_refresh_at=iso(self.clock()))
        self.tick()
        [restock] = self.alerts("restock")
        self.assertEqual(restock["status"], "sent")
        self.assertEqual(self.candidate()["state"], "alerted")
        self.assertIn("Back in stock", self.wire.posts()[-1]["text"])

    def test_listing_data_comes_off_a_card_after_six_hours(self):
        self.first_alert(detail())
        self.clock.advance(hours=6, minutes=1)
        report = self.tick()
        self.assertEqual(report["expired"], 1)
        [stub] = self.wire.updates()
        self.assertNotIn("120.00", words(stub))
        self.assertIn("removed after six hours", words(stub))


class ParkedListings(TickTest):
    def test_a_parked_listing_alerts_when_a_sweep_finds_its_price_at_the_ceiling(self):
        watch = self.watch()
        self.ebay.list(detail(price=usd("250.00")))
        self.tick()
        row = self.candidate()
        self.assertEqual((row["state"], row["reason"], row["tracked"]), ("parked", "over_ceiling", 1))
        self.assertEqual(self.alerts(), [], "an over-ceiling listing is parked, not announced")
        self.assertEqual(self.ebay.searches("sweep"), [], "no sweep in the tick of the first search")

        self.clock.advance(hours=5, minutes=5)
        self.tick()                                   # refreshed, still over the ceiling
        self.assertEqual(self.alerts(), [])
        self.ebay.live["v1|1|0"]["price"] = usd("150.00")
        self.clock.advance(hours=1)
        report = self.tick()                          # six hours after the first search: the sweep
        [sweep] = self.ebay.searches("sweep")
        self.assertNotIn("start_from", sweep)
        self.assertEqual(report["swept"], 1)
        self.assertEqual(self.ebay.reads("detail")[-1], ("v1|1|0", False), "judged again on the full listing")

        row = self.candidate()
        self.assertEqual((row["state"], row["listing"]["price"]), ("alerted", usd("150.00")))
        [alert] = self.alerts()
        self.assertEqual((alert["kind"], alert["status"]), ("price", "sent"))
        self.assertEqual(alert["dedupe_key"], "price:%d:%s" % (row["id"], iso(self.clock())))
        self.assertIn("Now within your price ceiling", self.wire.posts()[-1]["text"])
        self.assertEqual(self.store.get_watch(watch["id"])["last_sweep_at"], iso(self.clock()))

        self.clock.advance(minutes=10)
        self.tick()
        self.assertEqual(len(self.alerts()), 1)
        self.assertEqual(len(self.ebay.searches("sweep")), 1, "the next sweep is six hours away")

    def test_a_crash_after_judging_a_parked_listing_again_still_alerts_on_the_next_tick(self):
        self.watch()
        self.ebay.list(detail(price=usd("250.00")))
        self.tick()
        self.ebay.live["v1|1|0"]["price"] = usd("150.00")
        self.clock.advance(hours=5, minutes=5)
        with mock.patch.object(self.store, "enqueue_alert", side_effect=RuntimeError("died here")):
            self.tick()
        self.assertEqual(self.candidate()["state"], "alerted", "the funnel had already recorded its verdict")
        self.assertEqual(self.alerts(), [])

        for _ in range(2):
            self.clock.advance(minutes=10)
            self.tick()
        [alert] = self.alerts()
        self.assertEqual((alert["kind"], alert["status"]), ("new", "sent"), "late and plainly labeled, but not lost")
        self.assertEqual(len(self.wire.posts()), 1)

    def test_a_listing_that_sells_out_twice_is_announced_each_time_it_comes_back(self):
        self.watch()
        self.ebay.list(detail())
        self.tick()
        row = self.candidate()
        for _ in range(2):
            for availability in ("OUT_OF_STOCK", "IN_STOCK"):
                self.ebay.live["v1|1|0"]["availability"] = availability
                self.clock.advance(minutes=20)
                self.store.update_candidate(row["id"], next_refresh_at=iso(self.clock()))
                self.tick()
        self.assertEqual([alert["kind"] for alert in self.alerts()], ["new", "restock", "restock"])
        self.assertEqual(len(self.wire.posts()), 3)

    def test_a_refresh_that_finds_the_price_at_the_ceiling_alerts_without_waiting_for_the_sweep(self):
        self.watch()
        self.ebay.list(detail(price=usd("250.00")))
        self.tick()
        self.ebay.live["v1|1|0"]["price"] = usd("175.00")
        self.clock.advance(hours=5, minutes=5)
        self.tick()
        self.assertEqual([alert["kind"] for alert in self.alerts()], ["price"])
        self.assertEqual(self.candidate()["state"], "alerted")

    def test_a_sweep_updates_prices_on_tracked_listings_without_touching_their_fetch_time(self):
        self.watch()
        self.ebay.list(detail())
        self.tick()
        self.clock.advance(hours=5, minutes=5)
        self.tick()
        refreshed_at = self.candidate()["fetched_at"]
        self.ebay.live["v1|1|0"]["price"] = usd("110.00")
        self.clock.advance(hours=1)
        self.tick()
        row = self.candidate()
        self.assertEqual(row["listing"]["price"], usd("110.00"))
        self.assertEqual(row["fetched_at"], refreshed_at)


class WithoutModels(TickTest):
    CODED = "Nike Dunk Low Panda DD1391-100 Mens 10"

    def test_when_crusoe_is_down_all_tick_a_style_code_in_the_title_still_alerts(self):
        watch = self.watch()
        self.ebay.list(detail(1, title=self.CODED), detail(2))
        self.crusoe.answers["triage"] = CrusoeUnavailable("no usable answer for the triage role (HTTP 503)")
        report = self.tick()
        self.assertEqual((report["decisions"], report["not_model_checked"]), ({"pending": 2}, 1))
        self.assertEqual(report["problems"], ["crusoe: 1 model call(s) failed: the service was unavailable"])

        coded, plain = self.candidate(1), self.candidate(2)
        self.assertEqual((coded["state"], coded["reason"], coded["tracked"]), ("maybe", "not_model_checked", 1))
        self.assertEqual((plain["state"], plain["tracked"], plain["listing"]), ("pending", 0, None))
        [alert] = self.alerts()
        self.assertEqual((alert["kind"], alert["tier"], alert["status"]), ("new", "P2", "sent"))
        self.assertEqual(alert["payload"]["note"], "not model-checked")
        self.assertEqual(alert["payload"]["proof"], "DD1391-100 (title)")
        self.assertEqual(alert["payload"]["unconfirmed"], ["brand", "model", "colorway", "size", "department"])
        [post] = self.wire.posts()
        self.assertIn("not model-checked", words(post))
        self.assertTrue(post["text"].startswith("Unconfirmed match"))
        # The listing that has no verdict yet keeps the window from moving on.
        self.assertLess(self.store.get_watch(watch["id"])["window_end"], iso(self.clock()))
        self.assertEqual(self.store.verdicts_for(coded["id"])[-1]["rule_fired"], "not_model_checked")

    def test_a_listing_over_the_ceiling_is_not_alerted_on_its_style_code(self):
        self.watch()
        self.ebay.list(detail(1, title=self.CODED, price=usd("400.00")))
        self.crusoe.answers["triage"] = CrusoeUnavailable("down")
        self.tick()
        self.assertEqual(self.alerts(), [])

    def test_an_unusable_answer_is_not_an_outage(self):
        self.watch()
        self.ebay.list(detail(1, title=self.CODED))
        self.crusoe.answers["triage"] = CrusoeError("unusable after one repair attempt")
        report = self.tick()
        self.assertEqual(self.alerts(), [])
        self.assertEqual(self.candidate()["state"], "pending")
        self.assertEqual(report["problems"], ["crusoe: 1 model call(s) gave no usable answer"])

    def test_one_answered_call_means_crusoe_was_not_down_for_the_whole_tick(self):
        self.watch()
        self.ebay.list(detail(1, title=self.CODED))
        self.crusoe.answers["verify"] = CrusoeUnavailable("down")     # triage is answered, verify is not
        report = self.tick()
        self.assertEqual((report["not_model_checked"], self.alerts()), (0, []))
        self.assertEqual(self.candidate()["state"], "pending")


class SafetyNotices(TickTest):
    ENV = dict(SLACK, **PUSH)

    def test_a_refused_ebay_keyset_is_reported_once_a_day(self):
        self.watch("One")
        self.watch("Two")
        self.ebay.search_error = EbayAuthError("eBay search failed: HTTP 401", status=401)
        report = self.tick()
        self.assertEqual(len(self.ebay.searches()), 1, "every later call would be refused too")
        self.assertEqual(report["notices"], ["ebay_auth"])
        self.assertEqual(report["problems"], ["ebay: eBay search failed: HTTP 401"])
        [push] = self.wire.pushes()
        self.assertIn("eBay refused the keyset", push["message"])
        self.assertEqual(push["priority"], "1")
        self.assertNotIn("ttl", push, "a notice carries no listing data to expire")

        self.clock.advance(minutes=10)
        self.assertEqual(self.tick()["notices"], [])
        self.assertEqual(len(self.wire.pushes()), 1)
        self.clock.advance(days=1)
        self.assertEqual(self.tick()["notices"], ["ebay_auth"])
        self.assertEqual(len(self.wire.pushes()), 2)
        self.assertEqual(self.wire.posts(), [], "notices go to push first, and nothing else was sent")

    def test_a_refused_crusoe_key_is_reported_once_a_day_and_nothing_is_recorded(self):
        watch = self.watch()
        self.ebay.list(detail())
        self.crusoe.answers["triage"] = CrusoeAuthError("Crusoe refused the request (HTTP 401)")
        report = self.tick()
        self.assertEqual(report["notices"], ["crusoe_auth"])
        self.assertEqual(self.store.list_candidates(), [])
        self.assertIsNone(self.store.get_watch(watch["id"])["window_end"])
        self.clock.advance(minutes=10)
        self.tick()
        [push] = self.wire.pushes()
        self.assertIn("Crusoe refused the API key", push["message"])
        self.assertNotIn("test-crusoe-key", words(self.wire.transport.calls[-1]["body"].decode()))

    def test_eighty_percent_of_the_daily_calls_is_reported_once(self):
        self.store.ledger_add("ebay", "search", 3839)
        self.assertEqual(self.tick()["notices"], [])
        self.store.ledger_add("ebay", "search", 1)
        self.assertEqual(self.tick()["notices"], ["ebay_quota"])
        self.assertEqual(self.tick()["notices"], [])
        [push] = self.wire.pushes()
        self.assertIn("3840 of 4800", push["message"])
        self.assertEqual(push["priority"], "0")

    def test_a_swapped_model_is_reported(self):
        self.watch()
        self.ebay.list(detail())
        self.crusoe.swap_to = "some/replacement-model"
        report = self.tick()
        self.assertEqual(report["notices"], ["model_swap"])
        [notice] = [push for push in self.wire.pushes() if push["title"] == alerts.SYSTEM_TITLE]
        self.assertIn("some/replacement-model answered for served-triage", notice["message"])

    def test_a_notice_that_could_not_be_sent_is_tried_again_by_the_next_tick(self):
        settings = support.make_settings(**PUSH)            # push only, and push is down
        self.notifier = Notifier(settings, self.store, transport=self.wire.transport, local_now=self.local_now)
        self.settings = settings
        self.watch()
        self.ebay.search_error = EbayAuthError("eBay search failed: HTTP 403", status=403)
        self.wire.push_ok = False
        report = self.tick()
        self.assertEqual(report["notices"], [])
        self.assertIn("alerts: pushover: down", report["problems"])
        self.wire.push_ok = True
        self.clock.advance(minutes=10)
        self.assertEqual(self.tick()["notices"], ["ebay_auth"])


class Heartbeat(TickTest):
    ENV = dict(SLACK, EFT_HEARTBEAT_URL=HEARTBEAT)

    def test_the_heartbeat_is_requested_after_a_clean_tick(self):
        self.watch()
        self.ebay.list(detail())
        report = self.tick()
        self.assertTrue(report["heartbeat"])
        [beat] = self.wire.beats()
        self.assertEqual(beat["method"], "GET")

    def test_no_heartbeat_after_a_tick_with_a_failed_call(self):
        self.watch()
        self.ebay.search_error = EbayUnavailable("eBay search failed: HTTP 503", status=503)
        report = self.tick()
        self.assertIs(report["heartbeat"], False)
        self.assertEqual(self.wire.beats(), [])
        self.ebay.search_error = None
        self.clock.advance(minutes=10)
        self.assertTrue(self.tick()["heartbeat"])

    def test_a_heartbeat_that_fails_is_a_problem_that_does_not_show_its_url(self):
        self.wire.heartbeat_status = 500
        report = self.tick()
        self.assertIs(report["heartbeat"], False)
        self.assertEqual(report["problems"], ["heartbeat: answered HTTP 500"])
        self.assertNotIn("secret-heartbeat-id", words(report))

    def test_no_heartbeat_is_configured(self):
        self.settings = support.make_settings(**SLACK)
        self.assertIsNone(self.tick()["heartbeat"])
        self.assertEqual(self.wire.beats(), [])


class QuietHours(TickTest):
    START = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)     # 23:00 local

    def test_an_alert_held_overnight_goes_out_with_fresh_data(self):
        self.watch()
        self.ebay.list(detail())
        report = self.tick()
        self.assertEqual(report["delivery"]["held"], 1)
        self.assertEqual(self.wire.posts(), [])

        self.ebay.live["v1|1|0"]["price"] = usd("99.00")
        self.clock.advance(hours=10)                # 09:00 local
        report = self.tick()
        self.assertEqual(report["delivery"]["sent"], 1)
        [alert] = self.alerts()
        self.assertEqual((alert["payload"]["price"], alert["payload"]["as_of"]), (usd("99.00"), iso(self.clock())))
        self.assertIn("$99.00", words(self.wire.posts()))

    def test_an_alert_held_overnight_is_dropped_when_the_listing_sold(self):
        self.watch()
        self.ebay.list(detail())
        self.tick()
        self.store.update_candidate(self.candidate()["id"], next_refresh_at=iso(self.clock() + timedelta(days=1)))
        self.ebay.live["v1|1|0"]["availability"] = "OUT_OF_STOCK"
        self.clock.advance(hours=10)
        report = self.tick()
        self.assertEqual(report["delivery"]["dropped"], 1)
        self.assertEqual(self.wire.posts(), [])


if __name__ == "__main__":
    unittest.main()
