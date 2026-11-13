import json
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone

import support
from eft_core import alerts
from eft_core.alerts import DeliveryError, Notifier
from eft_core.errors import EbayUnavailable
from eft_core.http import FakeTransport, Response, TransportError, json_response
from eft_core.store import iso

UTC = timezone.utc
SLACK = {"SLACK_BOT_TOKEN": "fake-slack-secret-bot-token", "SLACK_USER_ID": "U0USER"}
PUSH = {"PUSHOVER_APP_TOKEN": "secret-app-token", "PUSHOVER_USER_KEY": "secret-user-key"}
SECRETS = (SLACK["SLACK_BOT_TOKEN"], PUSH["PUSHOVER_APP_TOKEN"], PUSH["PUSHOVER_USER_KEY"])
POST = "https://slack.com/api/chat.postMessage"
UPDATE = "https://slack.com/api/chat.update"
PUSHOVER = "https://api.pushover.net/1/messages.json"
LISTING_URL = "https://www.ebay.com/itm/256012345678"
PROOF = "Size US 10 (item specifics) · DD1391-100 (title)"
PAYLOAD_KEYS = {
    "code", "watch_id", "watch_name", "kind", "tier", "title", "web_url", "image_url", "price", "shipping",
    "current_bid", "min_next_bid", "bid_count", "reserve_met", "buying_options", "end_at", "condition",
    "proof", "unconfirmed", "relisted", "as_of", "note",
}


def usd(value):
    return {"value": value, "currency": "USD"}


def fixed_price(**overrides):
    payload = {
        "code": "A17", "watch_id": 1, "watch_name": "Panda Dunks 10", "kind": "new", "tier": "P2",
        "title": "Nike Dunk Low Panda Men's 10", "web_url": LISTING_URL,
        "image_url": "https://i.ebayimg.com/images/g/abc/s-l1600.jpg",
        "price": usd("120.00"), "shipping": usd("8.50"),
        "current_bid": None, "min_next_bid": None, "bid_count": None, "reserve_met": None,
        "buying_options": ["FIXED_PRICE"], "end_at": None, "condition": "Pre-owned",
        "proof": PROOF, "unconfirmed": [], "relisted": False,
        "as_of": "2026-11-12T18:00:00Z", "note": None,
    }
    payload.update(overrides)
    return payload


def auction(**overrides):
    fields = {
        "price": usd("95.00"), "current_bid": usd("95.00"), "min_next_bid": usd("97.50"),
        "bid_count": 7, "reserve_met": False, "buying_options": ["AUCTION"],
        "end_at": "2026-11-14T21:14:00Z",
    }
    fields.update(overrides)
    return fixed_price(**fields)


def words(obj):
    """Everything a card or request says, as one string, wherever it is laid out."""
    return json.dumps(obj, ensure_ascii=False)


class Escaping(unittest.TestCase):
    def test_escape_mrkdwn(self):
        self.assertEqual(alerts.escape_mrkdwn("<!channel> Tom & Jerry > all"),
                         "&lt;!channel&gt; Tom &amp; Jerry &gt; all")
        self.assertEqual(alerts.escape_mrkdwn("plain title"), "plain title")
        self.assertEqual(alerts.escape_mrkdwn(None), "")
        self.assertEqual(alerts.escape_mrkdwn(10), "10")

    def test_seller_text_cannot_inject_markup_or_lines(self):
        card = alerts.render_slack(fixed_price(title="<!channel> A&B <https://evil.example|free>\n*Price:* $1"), tz=UTC)
        text = words(card)
        self.assertNotIn("<!channel>", text)
        self.assertNotIn("<https://evil.example", text)
        # The whole title stays one escaped line inside the link label.
        self.assertIn("|&lt;!channel&gt; A&amp;B &lt;https://evil.example|free&gt; *Price:* $1>", text)


class Payloads(unittest.TestCase):
    WATCH = {"id": 3, "name": "Panda Dunks 10", "tier": "normal"}
    CANDIDATE = {"code": "A17", "end_at": "2026-11-14T21:14:00Z", "fetched_at": "2026-11-12T18:00:00Z"}
    LISTING = {
        "title": "Nike Dunk Low Panda", "web_url": LISTING_URL, "image_url": None,
        "price": usd("120.00"), "buying_options": ["FIXED_PRICE"], "condition": "Pre-owned", "relisted": True,
        "seller": "someone", "description": "long text", "aspects": {"Brand": ["Nike"]},
    }

    def test_build_payload_copies_only_the_card_fields(self):
        payload = alerts.build_payload(self.WATCH, self.CANDIDATE, self.LISTING, kind="new", tier="P2",
                                       proof=PROOF, unconfirmed=("size",), note="not model-checked")
        self.assertEqual(set(payload), PAYLOAD_KEYS)
        self.assertEqual((payload["code"], payload["watch_id"], payload["watch_name"]), ("A17", 3, "Panda Dunks 10"))
        self.assertEqual(payload["price"], usd("120.00"))
        self.assertEqual(payload["unconfirmed"], ["size"])
        self.assertTrue(payload["relisted"])
        self.assertEqual(payload["end_at"], "2026-11-14T21:14:00Z")
        self.assertEqual(payload["as_of"], "2026-11-12T18:00:00Z")
        self.assertNotIn("someone", words(payload))
        self.assertIn("A17", words(alerts.render_slack(payload, tz=UTC)))

    def test_explicit_as_of_wins(self):
        payload = alerts.build_payload(self.WATCH, self.CANDIDATE, self.LISTING, kind="price", tier="P2",
                                       as_of="2026-11-12T19:30:00Z")
        self.assertEqual(payload["as_of"], "2026-11-12T19:30:00Z")

    def test_unknown_kind_or_tier_is_refused(self):
        with self.assertRaises(ValueError):
            alerts.build_payload(self.WATCH, self.CANDIDATE, self.LISTING, kind="bargain", tier="P2")
        with self.assertRaises(ValueError):
            alerts.build_payload(self.WATCH, self.CANDIDATE, self.LISTING, kind="new", tier="P0")


class Tiers(unittest.TestCase):
    def test_tier_for(self):
        hot, normal = {"tier": "hot"}, {"tier": "normal"}
        buy_now = {"state": "alerted", "starred": 0, "listing": {"buying_options": ["FIXED_PRICE"]}}
        bidding = {"state": "alerted", "starred": 0, "listing": {"buying_options": ["AUCTION"]}}
        maybe = {"state": "maybe", "starred": 0, "listing": {"buying_options": ["FIXED_PRICE"]}}
        cases = [
            ("new", hot, buy_now, "P1"),
            ("new", normal, dict(buy_now, starred=1), "P1"),
            ("new", normal, buy_now, "P2"),
            ("new", hot, bidding, "P2"),
            ("new", hot, maybe, "P3"),
            ("new", normal, maybe, "P3"),
            ("ending", normal, dict(bidding, starred=1), "P1"),
            ("ending", hot, bidding, "P1"),
            ("ending", normal, bidding, "P2"),
            ("restock", hot, buy_now, "P1"),
            ("restock", normal, buy_now, "P2"),
            ("price", hot, buy_now, "P2"),
            ("ended", normal, buy_now, "P2"),
            ("digest", hot, dict(bidding, starred=1), "P3"),
        ]
        for kind, watch, candidate, expected in cases:
            self.assertEqual(alerts.tier_for(kind, watch, candidate), expected, (kind, watch, candidate))


class QuietHours(unittest.TestCase):
    def test_window_that_crosses_midnight(self):
        night = (22 * 60, 8 * 60)
        for clock, expected in (("21:59", False), ("22:00", True), ("23:30", True), ("00:00", True),
                                ("07:59", True), ("08:00", False), ("13:00", False)):
            hours, minutes = clock.split(":")
            self.assertEqual(alerts.in_quiet_hours(int(hours) * 60 + int(minutes), night), expected, clock)

    def test_window_within_one_day(self):
        siesta = (13 * 60, 15 * 60)
        for clock, expected in (("12:59", False), ("13:00", True), ("14:59", True), ("15:00", False),
                                ("23:00", False), ("03:00", False)):
            hours, minutes = clock.split(":")
            self.assertEqual(alerts.in_quiet_hours(int(hours) * 60 + int(minutes), siesta), expected, clock)

    def test_equal_start_and_end_means_no_quiet_hours(self):
        self.assertFalse(alerts.in_quiet_hours(0, (0, 0)))
        self.assertFalse(alerts.in_quiet_hours(600, (600, 600)))


class Cards(unittest.TestCase):
    def test_fixed_price_card(self):
        card = alerts.render_slack(fixed_price(), tz=UTC)
        self.assertEqual(set(card), {"text", "blocks"})
        text = words(card)
        for phrase in ("*New match*", "Panda Dunks 10", "`A17`",
                       "<%s|Nike Dunk Low Panda Men's 10>" % LISTING_URL,
                       "*Price:* $120.00", "*Shipping:* $8.50", "*Condition:* Pre-owned",
                       "*Proof:* " + PROOF, "as of Nov 12, 18:00 UTC", "not authenticated"):
            self.assertIn(phrase, text)
        for phrase in ("Current bid", "Bids", "may extend"):
            self.assertNotIn(phrase, text)

    def test_auction_card(self):
        text = words(alerts.render_slack(auction(), tz=UTC))
        for phrase in ("*Current bid:* $95.00", "*Bids:* 7", "*Minimum next bid:* $97.50", "*Reserve:* not met",
                       "scheduled end Nov 14, 21:14 UTC, may extend",
                       "`A17`", "*Proof:* " + PROOF, "as of Nov 12, 18:00 UTC", "not authenticated",
                       "<%s|" % LISTING_URL):
            self.assertIn(phrase, text)
        # eBay's "price" on an auction repeats the current bid, so it is not shown twice.
        self.assertNotIn("*Price:*", text)

    def test_price_wording(self):
        with_buy_now = words(alerts.render_slack(
            auction(price=usd("150.00"), buying_options=["AUCTION", "FIXED_PRICE"], reserve_met=True), tz=UTC))
        self.assertIn("*Buy It Now:* $150.00", with_buy_now)
        self.assertIn("*Reserve:* met", with_buy_now)
        no_bids = words(alerts.render_slack(auction(current_bid=None, bid_count=0), tz=UTC))
        self.assertIn("*Price:* $95.00", no_bids)
        self.assertIn("*Bids:* 0", no_bids)
        offers = words(alerts.render_slack(
            fixed_price(buying_options=["FIXED_PRICE", "BEST_OFFER"], shipping=usd("0.00")), tz=UTC))
        self.assertIn("*Price:* $120.00 or Best Offer", offers)
        self.assertIn("*Shipping:* free", offers)
        other_currency = words(alerts.render_slack(fixed_price(price={"value": "99.00", "currency": "CAD"}), tz=UTC))
        self.assertIn("*Price:* CAD 99.00", other_currency)
        ends = words(alerts.render_slack(fixed_price(end_at="2026-11-30T09:00:00Z"), tz=UTC))
        self.assertIn("*Listing:* scheduled end Nov 30, 09:00 UTC", ends)
        self.assertNotIn("may extend", ends)

    def test_times_are_shown_in_the_given_zone(self):
        pacific = timezone(timedelta(hours=-8), "PST")
        text = words(alerts.render_slack(auction(), tz=pacific))
        self.assertIn("scheduled end Nov 14, 13:14 PST, may extend", text)
        self.assertIn("as of Nov 12, 10:00 PST", text)

    def test_cards_state_facts_and_never_vouch_for_the_item(self):
        for payload in (fixed_price(), auction()):
            text = words(alerts.render_slack(payload, tz=UTC)).lower()
            self.assertEqual(text.count("not authenticated"), 1)
            self.assertNotIn("authentic", text.replace("not authenticated", ""))
            for word in ("deal", "score", "bargain", "worth", "cheap"):
                self.assertNotIn(word, text)

    def test_notification_text_carries_no_price(self):
        for payload in (fixed_price(), auction()):
            text = alerts.render_slack(payload, tz=UTC)["text"]
            self.assertIn("A17", text)
            self.assertNotIn("$", text)

    def test_image_only_for_types_slack_can_show(self):
        def image_blocks(url):
            card = alerts.render_slack(fixed_price(image_url=url), tz=UTC)
            return [block for block in card["blocks"] if block["type"] == "image"]

        for url in ("https://i.ebayimg.com/images/g/abc/s-l1600.jpg", "https://i.ebayimg.com/a/photo.JPEG",
                    "https://i.ebayimg.com/a/photo.png", "https://i.ebayimg.com/a/photo.gif?set_id=1"):
            blocks = image_blocks(url)
            self.assertEqual(len(blocks), 1, url)
            self.assertEqual(blocks[0]["image_url"], url)
            self.assertTrue(blocks[0]["alt_text"])
        for url in ("https://i.ebayimg.com/images/g/abc/s-l1600.webp", "https://i.ebayimg.com/images/g/abc",
                    "http://i.ebayimg.com/a/photo.jpg", "https://i.ebayimg.com/a/jpg", "", None):
            self.assertEqual(image_blocks(url), [], url)

    def test_unconfirmed_match_is_labelled(self):
        card = alerts.render_slack(auction(unconfirmed=["size", "style_code"], proof=""), tz=UTC)
        text = words(card)
        self.assertIn("*Unconfirmed match*", text)
        self.assertIn("*Not proven:* size, style code", text)
        self.assertNotIn("*New match*", text)
        self.assertIn("Unconfirmed match", card["text"])
        ending = words(alerts.render_slack(auction(kind="ending", unconfirmed=["size"]), tz=UTC))
        self.assertIn("Ending soon (unconfirmed match)", ending)

    def test_relist_and_note_are_shown(self):
        text = words(alerts.render_slack(fixed_price(relisted=True, note="not model-checked"), tz=UTC))
        self.assertIn("(relisted)", text)
        self.assertIn("*Note:* not model-checked", text)

    def test_follow_up_labels(self):
        for kind, label in (("price", "Now within your price ceiling"), ("ending", "Ending soon"),
                            ("restock", "Back in stock")):
            self.assertIn("*%s*" % label, words(alerts.render_slack(auction(kind=kind), tz=UTC)))


class Stubs(unittest.TestCase):
    def assert_no_listing_figures(self, card):
        text = words(card)
        for phrase in ("$", "120.00", "95.00", "97.50", "8.50", "bid", "price", "reserve", "shipping",
                       "scheduled end", "ebayimg"):
            self.assertNotIn(phrase, text.lower())
        self.assertEqual([block for block in card["blocks"] if block["type"] == "image"], [])

    def test_expired_stub_keeps_the_way_back_to_the_listing(self):
        for payload in (fixed_price(), auction()):
            card = alerts.render_slack_stub(payload, "expired")
            self.assert_no_listing_figures(card)
            text = words(card)
            self.assertIn("`A17`", text)
            self.assertIn("<%s|Nike Dunk Low Panda Men's 10>" % LISTING_URL, text)
            self.assertIn("removed after six hours", text)

    def test_ended_stub_keeps_nothing_about_the_listing(self):
        for payload in (fixed_price(), auction(unconfirmed=["size"])):
            card = alerts.render_slack_stub(payload, "ended")
            self.assert_no_listing_figures(card)
            text = words(card)
            self.assertIn("*Ended*", text)
            self.assertIn("`A17`", text)
            self.assertIn("Panda Dunks 10", text)
            self.assertNotIn("Nike Dunk Low", text)
            self.assertNotIn(LISTING_URL, text)

    def test_unknown_reason_is_refused(self):
        with self.assertRaises(ValueError):
            alerts.render_slack_stub(fixed_price(), "stale")

    def test_an_ended_alert_renders_without_listing_data(self):
        card = alerts.render_slack(auction(kind="ended"), tz=UTC)
        self.assert_no_listing_figures(card)
        push = alerts.render_push(auction(kind="ended"), tz=UTC)
        self.assertEqual(push["title"], "Ended: Panda Dunks 10 (A17)")
        self.assertEqual(push["url"], "")
        self.assertNotIn("$", push["message"])
        self.assertNotIn("Nike", push["message"])


class Pushes(unittest.TestCase):
    def test_fields(self):
        push = alerts.render_push(fixed_price(), tz=UTC)
        self.assertEqual(set(push), {"title", "message", "url", "url_title", "priority", "ttl"})
        self.assertEqual(push["title"], "New match: Panda Dunks 10 (A17)")
        self.assertEqual(push["url"], LISTING_URL)
        self.assertEqual(push["url_title"], "Open on eBay")
        self.assertEqual(push["ttl"], 21600)
        self.assertEqual(alerts.SIX_HOURS, 21600)
        for phrase in ("Nike Dunk Low Panda Men's 10", "Price: $120.00", "Shipping: $8.50", "Proof: " + PROOF,
                       "as of Nov 12, 18:00 UTC · not authenticated"):
            self.assertIn(phrase, push["message"])

    def test_auction_push_has_the_bid_state(self):
        message = alerts.render_push(auction(), tz=UTC)["message"]
        for phrase in ("Current bid: $95.00", "Bids: 7", "Minimum next bid: $97.50", "Reserve: not met",
                       "scheduled end Nov 14, 21:14 UTC, may extend"):
            self.assertIn(phrase, message)

    def test_priority_follows_the_tier(self):
        for tier, priority in (("P1", 1), ("P2", 0), ("P3", -1)):
            self.assertEqual(alerts.render_push(fixed_price(tier=tier), tz=UTC)["priority"], priority)

    def test_length_caps_keep_the_footer(self):
        push = alerts.render_push(
            fixed_price(watch_name="W" * 400, title="T" * 2000, proof="P" * 3000), tz=UTC)
        self.assertLessEqual(len(push["title"]), 250)
        self.assertLessEqual(len(push["message"]), 1024)
        self.assertTrue(push["message"].endswith("as of Nov 12, 18:00 UTC · not authenticated"))

    def test_url_is_dropped_when_pushover_would_refuse_it(self):
        too_long = alerts.render_push(fixed_price(web_url=LISTING_URL + "?" + "x" * 600), tz=UTC)
        self.assertEqual((too_long["url"], too_long["url_title"]), ("", ""))
        not_https = alerts.render_push(fixed_price(web_url="ftp://example.com/x"), tz=UTC)
        self.assertEqual(not_https["url"], "")


class Wire:
    """A scripted Slack and Pushover behind a FakeTransport. Keeps what each was sent."""

    def __init__(self):
        self.posts, self.updates, self.pushes = [], [], []
        self.slack_replies = []     # queued replies for the next chat.postMessage calls
        self.slack_down = None      # an error code that every Slack call fails with
        self.push_down = False
        self.transport = FakeTransport(self.handle)

    def handle(self, method, url, headers, body):
        if url == PUSHOVER:
            self.pushes.append(dict(urllib.parse.parse_qsl(body.decode("utf-8"))))
            if self.push_down:
                return json_response({"status": 0, "errors": ["application token is invalid"]}, status=400)
            return json_response({"status": 1, "request": "req-%d" % len(self.pushes)})
        sent = json.loads(body.decode("utf-8"))
        sent["_headers"] = headers
        (self.updates if url == UPDATE else self.posts).append(sent)
        if self.slack_down:
            return json_response({"ok": False, "error": self.slack_down})
        if url == UPDATE:
            return json_response({"ok": True, "channel": sent["channel"], "ts": sent["ts"]})
        if self.slack_replies:
            return self.slack_replies.pop(0)
        return json_response({"ok": True, "channel": "D0001", "ts": "1700000000.%04d" % len(self.posts)})


class OutboxCase(unittest.TestCase):
    """A store, a scripted Slack and Pushover, and a local clock set to early afternoon."""

    def setUp(self):
        self.clock = support.Clock()
        self.store = support.make_store(self.clock)
        self.wire = Wire()
        self.local = datetime(2026, 11, 12, 13, 0)
        self.watch_id = self.store.create_watch("Panda Dunks 10", {"brand": "Nike"}, status="active")
        self.items = 0

    def notifier(self, *features):
        env = {}
        for feature in features or (SLACK, PUSH):
            env.update(feature)
        return Notifier(support.make_settings(**env), self.store, transport=self.wire.transport,
                        local_now=lambda: self.local)

    def candidate(self, watch_id=None):
        self.items += 1
        return self.store.upsert_candidate(watch_id or self.watch_id, "v1|%d|0" % self.items, "alerted", tracked=True)

    def queue(self, tier="P2", kind="new", candidate=None, make=fixed_price, key=None, **fields):
        """Queue one alert, for a new candidate unless one is given. Returns the alert id."""
        candidate = candidate or self.candidate()
        payload = make(code=candidate["code"], watch_id=candidate["watch_id"], kind=kind, tier=tier, **fields)
        return self.store.enqueue_alert(
            key or "%s:%s" % (kind, candidate["code"]), kind, tier, payload,
            watch_id=candidate["watch_id"], candidate_id=candidate["id"], as_of=payload["as_of"])

    def alert(self, alert_id):
        return self.store.get_alert(alert_id)

    def messages(self, alert_id):
        candidate_id = self.alert(alert_id)["candidate_id"]
        return [m for m in self.store.messages_for_candidate(candidate_id) if m["alert_id"] == alert_id]

    def night(self):
        """04:00 local: inside the default quiet hours, which end at 08:00."""
        self.local = datetime(2026, 11, 13, 4, 0)

    def morning(self):
        """Four hours and five minutes later, so nothing held overnight is six hours old yet."""
        self.local = datetime(2026, 11, 13, 8, 5)
        self.clock.advance(hours=4, minutes=5)


class Sending(OutboxCase):
    def test_one_alert_goes_to_both_channels_and_is_recorded(self):
        alert_id = self.queue()
        self.assertEqual(self.notifier().deliver_pending(), {"sent": 1, "held": 0, "dropped": 0, "failed": 0})

        post = self.wire.posts[0]
        self.assertEqual(post["channel"], "U0USER")
        self.assertIs(post["unfurl_links"], False)
        self.assertIs(post["unfurl_media"], False)
        self.assertTrue(post["text"])
        self.assertIn("*Price:* $120.00", words(post["blocks"]))
        self.assertEqual(post["_headers"]["Authorization"], "Bearer " + SLACK["SLACK_BOT_TOKEN"])
        self.assertEqual(post["_headers"]["Content-Type"], "application/json; charset=utf-8")

        push = self.wire.pushes[0]
        self.assertEqual((push["token"], push["user"]), (PUSH["PUSHOVER_APP_TOKEN"], PUSH["PUSHOVER_USER_KEY"]))
        self.assertEqual((push["ttl"], push["priority"], push["url"]), ("21600", "0", LISTING_URL))

        stored = self.alert(alert_id)
        self.assertEqual(stored["status"], "sent")
        self.assertIsNotNone(stored["sent_at"])
        rows = {m["channel"]: m for m in self.messages(alert_id)}
        self.assertEqual(set(rows), {"slack", "pushover"})
        self.assertEqual(rows["slack"]["ref"], {"channel": "D0001", "ts": "1700000000.0001"})
        self.assertEqual(rows["pushover"]["ref"], {"request": "req-1"})
        for row in rows.values():
            self.assertEqual(row["as_of"], "2026-11-12T18:00:00Z")
            self.assertEqual(row["has_listing_data"], 1)
            self.assertEqual(row["candidate_id"], stored["candidate_id"])

    def test_nothing_is_sent_twice(self):
        self.queue()
        notifier = self.notifier()
        notifier.deliver_pending()
        self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual((len(self.wire.posts), len(self.wire.pushes)), (1, 1))

    def test_no_request_ever_goes_to_an_ebay_page(self):
        self.queue()
        self.notifier().deliver_pending()
        hosts = {urllib.parse.urlsplit(call["url"]).hostname for call in self.wire.transport.calls}
        self.assertEqual(hosts, {"slack.com", "api.pushover.net"})

    def test_without_a_channel_alerts_wait_untouched(self):
        self.night()
        alert_id = self.queue(as_of=iso(self.clock() - timedelta(hours=7)))
        notifier = Notifier(support.make_settings(), self.store, transport=self.wire.transport,
                            local_now=lambda: self.local)
        asked = []
        self.assertEqual(notifier.channels(), [])
        self.assertEqual(notifier.deliver_pending(asked.append), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual(asked, [])
        self.assertFalse(notifier.send_system("Crusoe key rejected", urgent=True))
        self.assertEqual(notifier.expire_messages(), 0)
        self.assertEqual((self.alert(alert_id)["status"], self.alert(alert_id)["attempts"]), ("pending", 0))
        self.assertEqual(self.wire.transport.calls, [])
        with self.assertRaises(DeliveryError) as refused:
            notifier.post_slack(alerts.render_slack(fixed_price()))
        self.assertEqual(str(refused.exception), "slack: not configured")

    def test_channels_lists_what_is_configured(self):
        self.assertEqual(self.notifier(SLACK).channels(), ["slack"])
        self.assertEqual(self.notifier(PUSH).channels(), ["pushover"])
        self.assertEqual(self.notifier(SLACK, PUSH).channels(), ["slack", "pushover"])
        half = self.notifier({"SLACK_BOT_TOKEN": "fake-slack-only-the-token"})
        self.assertEqual(half.channels(), [])

    def test_system_rows_are_left_for_send_system(self):
        alert_id = self.store.enqueue_alert("system:crusoe_auth:2026-11-12", "system", "P1", {"kind": "system"})
        self.assertEqual(self.notifier().deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual(self.alert(alert_id)["status"], "pending")
        self.assertEqual(self.wire.transport.calls, [])


class QuietHoursDelivery(OutboxCase):
    def test_p1_is_sent_and_p2_and_p3_are_held(self):
        self.night()
        urgent, normal, low = self.queue("P1"), self.queue("P2"), self.queue("P3")
        counts = self.notifier().deliver_pending()
        self.assertEqual(counts, {"sent": 1, "held": 2, "dropped": 0, "failed": 0})
        self.assertEqual([self.alert(i)["status"] for i in (urgent, normal, low)], ["sent", "held", "held"])
        self.assertEqual(len(self.wire.posts), 1)
        self.assertEqual(self.wire.pushes[0]["priority"], "1")

    def test_p2_goes_out_when_its_listing_ends_before_the_window_closes(self):
        self.night()  # quiet for another four hours
        now = self.clock()
        ends_tonight = self.queue(make=auction, end_at=iso(now + timedelta(hours=2)))
        ends_tomorrow = self.queue(make=auction, end_at=iso(now + timedelta(hours=12)))
        already_over = self.queue(make=auction, end_at=iso(now - timedelta(hours=1)))
        no_end = self.queue()
        counts = self.notifier().deliver_pending()
        self.assertEqual(counts, {"sent": 1, "held": 3, "dropped": 0, "failed": 0})
        self.assertEqual(self.alert(ends_tonight)["status"], "sent")
        for alert_id in (ends_tomorrow, already_over, no_end):
            self.assertEqual(self.alert(alert_id)["status"], "held")

    def test_held_alert_is_refreshed_and_released_after_quiet_hours(self):
        self.night()
        alert_id = self.queue()
        notifier = self.notifier()
        asked = []

        def refresher(alert):
            asked.append(alert["id"])
            return fixed_price(code=alert["payload"]["code"], price=usd("110.00"), as_of=self.store.now_iso())

        self.assertEqual(notifier.deliver_pending(refresher)["held"], 1)
        self.assertEqual(asked, [])
        self.morning()
        self.assertEqual(notifier.deliver_pending(refresher), {"sent": 1, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual(asked, [alert_id])
        card = words(self.wire.posts[0])
        self.assertIn("*Price:* $110.00", card)
        self.assertNotIn("$120.00", card)
        self.assertEqual(self.alert(alert_id)["payload"]["price"], usd("110.00"))
        for row in self.messages(alert_id):
            self.assertEqual(row["as_of"], "2026-11-12T22:05:00Z")

    def test_held_alert_is_dropped_when_the_listing_is_gone(self):
        self.night()
        alert_id = self.queue()
        notifier = self.notifier()
        notifier.deliver_pending(lambda alert: None)
        self.morning()
        self.assertEqual(notifier.deliver_pending(lambda alert: None),
                         {"sent": 0, "held": 0, "dropped": 1, "failed": 0})
        self.assertEqual(self.alert(alert_id)["status"], "dropped")
        self.assertEqual(self.wire.transport.calls, [])

    def test_alert_stays_held_when_the_listing_cannot_be_read_just_now(self):
        self.night()
        alert_id = self.queue()
        notifier = self.notifier()
        notifier.deliver_pending()
        self.morning()

        def refresher(alert):
            raise EbayUnavailable("try again later", status=503)

        self.assertEqual(notifier.deliver_pending(refresher), {"sent": 0, "held": 1, "dropped": 0, "failed": 0})
        self.assertEqual(self.alert(alert_id)["status"], "held")
        self.assertEqual(notifier.problems, ["refresh: EbayUnavailable"])
        self.assertEqual(self.wire.transport.calls, [])

    def test_held_alert_goes_out_as_it_is_without_a_refresher(self):
        self.night()
        alert_id = self.queue()
        notifier = self.notifier()
        notifier.deliver_pending()
        self.morning()
        self.assertEqual(notifier.deliver_pending()["sent"], 1)
        self.assertEqual(self.alert(alert_id)["status"], "sent")

    def test_an_alert_without_a_listing_is_never_sent_to_the_refresher(self):
        self.night()
        alert_id = self.store.enqueue_alert(
            "digest:note", "digest", "P2", fixed_price(kind="digest", title="", web_url="", price=None),
            watch_id=self.watch_id)
        notifier = self.notifier(SLACK)
        notifier.deliver_pending()
        self.morning()
        asked = []
        self.assertEqual(notifier.deliver_pending(asked.append)["sent"], 1)
        self.assertEqual(asked, [])
        self.assertEqual(self.alert(alert_id)["status"], "sent")

    def test_only_old_data_is_refreshed_for_an_alert_that_was_never_held(self):
        fresh = self.queue()
        old = self.queue(as_of=iso(self.clock() - timedelta(hours=7)))
        asked = []

        def refresher(alert):
            asked.append(alert["id"])
            return dict(alert["payload"], as_of=self.store.now_iso())

        self.assertEqual(self.notifier().deliver_pending(refresher)["sent"], 2)
        self.assertEqual(asked, [old])
        self.assertEqual(self.messages(old)[0]["as_of"], self.store.now_iso())
        self.assertEqual(self.messages(fresh)[0]["as_of"], "2026-11-12T18:00:00Z")


class Grouping(OutboxCase):
    def test_p3_alerts_go_out_as_one_digest(self):
        other_watch = self.store.create_watch("Chore jacket L", {"model": "chore jacket"}, status="active")
        ids = [
            self.queue("P3", unconfirmed=["size"]),
            self.queue("P3", make=auction, unconfirmed=["colorway"]),
            self.queue("P3", candidate=self.candidate(other_watch), watch_name="Chore jacket L"),
        ]
        codes = [self.alert(i)["payload"]["code"] for i in ids]
        self.assertEqual(self.notifier().deliver_pending(), {"sent": 3, "held": 0, "dropped": 0, "failed": 0})

        self.assertEqual((len(self.wire.posts), len(self.wire.pushes)), (1, 1))
        post = words(self.wire.posts[0])
        self.assertIn("Daily digest: 3 to review", post)
        for phrase in ["`%s`" % code for code in codes] + [
                "Panda Dunks 10", "Chore jacket L", "Unconfirmed match", "Not proven: size", "Not proven: colorway",
                "Price: $120.00", "Current bid: $95.00", "may extend", "as of ", "not authenticated"]:
            self.assertIn(phrase, post)
        push = self.wire.pushes[0]
        self.assertEqual((push["title"], push["priority"], push["ttl"]), ("Daily digest: 3 to review", "-1", "21600"))
        self.assertNotIn("url", push)
        self.assertIn("%s Unconfirmed match: Nike Dunk Low Panda Men's 10 · Price: $120.00" % codes[0], push["message"])
        self.assertTrue(push["message"].endswith("not authenticated"))

        slack_rows = [m for i in ids for m in self.messages(i) if m["channel"] == "slack"]
        self.assertEqual(len(slack_rows), 3)
        self.assertEqual({(m["ref"]["channel"], m["ref"]["ts"]) for m in slack_rows}, {("D0001", "1700000000.0001")})
        self.assertEqual({m["as_of"] for m in slack_rows}, {"2026-11-12T18:00:00Z"})
        self.assertEqual([self.alert(i)["status"] for i in ids], ["sent"] * 3)

    def test_digest_goes_out_once_a_day(self):
        notifier = self.notifier(SLACK)
        self.queue("P3")
        self.assertEqual(notifier.deliver_pending()["sent"], 1)
        later = self.queue("P3")
        self.clock.advance(hours=2)
        self.local = datetime(2026, 11, 12, 15, 0)
        self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 1, "dropped": 0, "failed": 0})
        self.assertEqual(self.alert(later)["status"], "held")
        self.clock.advance(hours=17)
        self.local = datetime(2026, 11, 13, 8, 0)
        self.assertEqual(notifier.deliver_pending()["sent"], 1)
        self.assertEqual(len(self.wire.posts), 2)

    def test_a_failed_digest_is_tried_again_the_same_day(self):
        notifier = self.notifier(SLACK)
        alert_id = self.queue("P3")
        self.wire.slack_down = "internal_error"
        self.assertEqual(notifier.deliver_pending()["sent"], 0)
        self.assertEqual((self.alert(alert_id)["status"], self.alert(alert_id)["attempts"]), ("pending", 1))
        self.wire.slack_down = None
        self.assertEqual(notifier.deliver_pending()["sent"], 1)

    def test_digest_expires_by_its_oldest_entry(self):
        older = self.queue("P3", as_of="2026-11-12T15:00:00Z")
        newer = self.queue("P3", as_of="2026-11-12T17:30:00Z")
        self.notifier(SLACK).deliver_pending()
        self.assertIn("as of Nov 12, ", words(self.wire.posts[0]))
        self.assertIn(" or later", words(self.wire.posts[0]))
        for alert_id in (older, newer):
            self.assertEqual(self.messages(alert_id)[0]["as_of"], "2026-11-12T15:00:00Z")

    def test_more_than_three_p2_alerts_for_one_watch_collapse(self):
        busy = [self.queue() for _ in range(4)]
        singled_out = self.queue("P1")
        other_watch = self.store.create_watch("Chore jacket L", {"model": "chore jacket"}, status="active")
        calm = [self.queue(candidate=self.candidate(other_watch), watch_name="Chore jacket L") for _ in range(3)]
        counts = self.notifier(SLACK).deliver_pending()
        self.assertEqual(counts, {"sent": 8, "held": 0, "dropped": 0, "failed": 0})

        # The P1 card first, then one summary for the busy watch, then three single cards.
        self.assertEqual(len(self.wire.posts), 5)
        self.assertIn("`%s`" % self.alert(singled_out)["payload"]["code"], words(self.wire.posts[0]))
        summary = words(self.wire.posts[1])
        self.assertIn("Panda Dunks 10: 4 alerts", summary)
        for alert_id in busy:
            self.assertIn("`%s`" % self.alert(alert_id)["payload"]["code"], summary)
            self.assertEqual(self.messages(alert_id)[0]["ref"]["ts"], "1700000000.0002")
        self.assertNotIn("`%s`" % self.alert(singled_out)["payload"]["code"], summary)
        for post, alert_id in zip(self.wire.posts[2:], calm):
            self.assertIn("`%s`" % self.alert(alert_id)["payload"]["code"], words(post))
            self.assertNotIn("alerts", post["text"])

    def test_a_long_digest_shows_twenty_and_counts_the_rest(self):
        ids = [self.queue("P3") for _ in range(23)]
        self.assertEqual(self.notifier(SLACK).deliver_pending()["sent"], 23)
        post = self.wire.posts[0]
        self.assertIn("Daily digest: 23 to review", post["text"])
        self.assertIn("3 more not shown", words(post))
        self.assertLessEqual(len(post["blocks"]), 50)
        self.assertEqual(sum(1 for i in ids if self.messages(i)), 20)
        self.assertEqual({self.alert(i)["status"] for i in ids}, {"sent"})


class Failures(OutboxCase):
    def test_attempts_are_counted_and_the_alert_fails_after_five(self):
        alert_id = self.queue()
        self.wire.slack_down, self.wire.push_down = "invalid_auth", True
        notifier = self.notifier()
        for attempt in range(1, 5):
            self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
            self.assertEqual((self.alert(alert_id)["status"], self.alert(alert_id)["attempts"]), ("pending", attempt))
        self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 1})
        self.assertEqual((self.alert(alert_id)["status"], self.alert(alert_id)["attempts"]), ("failed", 5))
        self.assertEqual(self.messages(alert_id), [])
        self.assertEqual(notifier.problems, ["slack: invalid_auth", "pushover: application token is invalid"])

        calls = len(self.wire.transport.calls)
        self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual(len(self.wire.transport.calls), calls)

    def test_one_working_channel_is_enough(self):
        alert_id = self.queue()
        self.wire.slack_down = "token_revoked"
        notifier = self.notifier()
        self.assertEqual(notifier.deliver_pending(), {"sent": 1, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual((self.alert(alert_id)["status"], self.alert(alert_id)["attempts"]), ("sent", 0))
        self.assertEqual([m["channel"] for m in self.messages(alert_id)], ["pushover"])
        self.assertEqual(notifier.problems, ["slack: token_revoked"])

    def test_push_failure_does_not_stop_the_slack_card(self):
        alert_id = self.queue()
        self.wire.push_down = True
        self.assertEqual(self.notifier().deliver_pending()["sent"], 1)
        self.assertEqual([m["channel"] for m in self.messages(alert_id)], ["slack"])

    def test_invalid_blocks_is_retried_once_without_the_image(self):
        alert_id = self.queue()
        self.wire.slack_replies = [json_response({"ok": False, "error": "invalid_blocks"})]
        self.assertEqual(self.notifier(SLACK).deliver_pending()["sent"], 1)
        first, second = self.wire.posts
        self.assertEqual([b["type"] for b in first["blocks"]].count("image"), 1)
        self.assertEqual([b["type"] for b in second["blocks"]].count("image"), 0)
        self.assertIn("*Price:* $120.00", words(second))
        self.assertEqual(len(self.messages(alert_id)), 1)

    def test_invalid_blocks_without_an_image_is_a_plain_failure(self):
        alert_id = self.queue(image_url=None)
        self.wire.slack_replies = [json_response({"ok": False, "error": "invalid_blocks"})]
        self.assertEqual(self.notifier(SLACK).deliver_pending()["sent"], 0)
        self.assertEqual(len(self.wire.posts), 1)
        self.assertEqual(self.alert(alert_id)["attempts"], 1)

    def test_rate_limited_channel_is_left_alone_until_retry_after_has_passed(self):
        first, second = self.queue(), self.queue()
        self.wire.slack_replies = [
            json_response({"ok": False, "error": "ratelimited"}, status=429, headers={"Retry-After": "120"})]
        notifier = self.notifier(SLACK)
        self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual(len(self.wire.posts), 1)
        self.assertEqual((self.alert(first)["attempts"], self.alert(second)["attempts"]), (1, 0))

        self.clock.advance(seconds=60)
        notifier.deliver_pending()
        self.assertFalse(notifier.send_system("eBay keyset disabled"))
        self.assertEqual(len(self.wire.posts), 1)
        self.assertEqual(notifier.problems, ["slack: rate limited", "slack: rate limited, waiting"])

        self.clock.advance(seconds=61)
        self.assertEqual(notifier.deliver_pending()["sent"], 2)

    def test_unreadable_reply_is_a_failure(self):
        alert_id = self.queue()
        self.wire.slack_replies = [Response(status=502, body=b"<html>bad gateway</html>")]
        notifier = self.notifier(SLACK)
        self.assertEqual(notifier.deliver_pending()["sent"], 0)
        self.assertEqual(self.alert(alert_id)["attempts"], 1)
        self.assertEqual(notifier.problems, ["slack: HTTP 502"])


class Threads(OutboxCase):
    def test_follow_ups_reply_under_the_candidates_first_slack_message(self):
        candidate = self.candidate()
        notifier = self.notifier(SLACK)
        self.queue(candidate=candidate, make=auction)
        notifier.deliver_pending()
        first = self.wire.posts[0]
        self.assertNotIn("thread_ts", first)
        self.assertEqual(first["channel"], "U0USER")

        for kind in ("price", "ending", "restock", "ended"):
            self.queue(kind=kind, candidate=candidate, make=auction)
            self.assertEqual(notifier.deliver_pending()["sent"], 1, kind)
            reply = self.wire.posts[-1]
            self.assertEqual((reply["channel"], reply["thread_ts"]), ("D0001", "1700000000.0001"), kind)
            self.assertIs(reply["unfurl_links"], False)

    def test_only_follow_ups_with_an_earlier_message_are_replies(self):
        notifier = self.notifier(SLACK)
        seen_before = self.candidate()
        self.queue(candidate=seen_before)
        notifier.deliver_pending()
        self.queue()
        self.queue(kind="ending")  # a follow-up for a candidate that has no message yet
        self.queue(candidate=seen_before, key="new:proven-after-recheck")
        self.assertEqual(notifier.deliver_pending()["sent"], 3)
        self.assertEqual(len(self.wire.posts), 4)
        for post in self.wire.posts:
            self.assertNotIn("thread_ts", post)


class Expiry(OutboxCase):
    def test_slack_card_is_rewritten_at_six_hours_and_rows_are_marked(self):
        alert_id = self.queue(make=auction)
        notifier = self.notifier()
        notifier.deliver_pending()
        self.clock.advance(hours=5, minutes=59)
        self.assertEqual(notifier.expire_messages(), 0)
        self.assertEqual(self.wire.updates, [])

        self.clock.advance(minutes=1)
        self.assertEqual(notifier.expire_messages(), 2)
        self.assertEqual(len(self.wire.updates), 1)
        update = self.wire.updates[0]
        self.assertEqual((update["channel"], update["ts"]), ("D0001", "1700000000.0001"))
        text = words({"text": update["text"], "blocks": update["blocks"]})
        for phrase in ("$", "95.00", "Current bid", "Bids", "scheduled end", "ebayimg"):
            self.assertNotIn(phrase, text)
        self.assertIn("removed after six hours", text)
        self.assertIn("<%s|" % LISTING_URL, text)
        for row in self.messages(alert_id):
            self.assertIsNotNone(row["expired_at"])
            self.assertEqual(row["has_listing_data"], 0)

        self.assertEqual(notifier.expire_messages(), 0)
        self.assertEqual(len(self.wire.updates), 1)

    def test_a_card_that_could_not_be_rewritten_stays_due(self):
        alert_id = self.queue()
        notifier = self.notifier(SLACK)
        notifier.deliver_pending()
        self.clock.advance(hours=6)
        self.wire.slack_down = "internal_error"
        self.assertEqual(notifier.expire_messages(), 0)
        self.assertIsNone(self.messages(alert_id)[0]["expired_at"])
        self.wire.slack_down = None
        self.assertEqual(notifier.expire_messages(), 1)
        self.assertIsNotNone(self.messages(alert_id)[0]["expired_at"])

    def test_a_card_the_user_deleted_has_nothing_left_to_expire(self):
        alert_id = self.queue()
        notifier = self.notifier(SLACK)
        notifier.deliver_pending()
        self.clock.advance(hours=6)
        self.wire.slack_down = "message_not_found"
        self.assertEqual(notifier.expire_messages(), 1)
        self.assertIsNotNone(self.messages(alert_id)[0]["expired_at"])

    def test_digest_is_rewritten_once_and_every_row_is_marked(self):
        ids = [self.queue("P3"), self.queue("P3", make=auction), self.queue("P3")]
        notifier = self.notifier()
        notifier.deliver_pending()
        self.clock.advance(hours=6)
        self.assertEqual(notifier.expire_messages(), 6)
        self.assertEqual(len(self.wire.updates), 1)
        text = words(self.wire.updates[0]["blocks"])
        for phrase in ("$", "Current bid", "Price", "scheduled end"):
            self.assertNotIn(phrase, text)
        for alert_id in ids:
            self.assertIn("`%s`" % self.alert(alert_id)["payload"]["code"], text)
        self.assertIn("Daily digest: 3 to review", text)
        self.assertIn("removed after six hours", text)


class Endings(OutboxCase):
    def test_mark_ended_rewrites_every_slack_message_and_empties_the_payloads(self):
        candidate, bystander = self.candidate(), self.candidate()
        notifier = self.notifier()
        first = self.queue(candidate=candidate, make=auction)
        other = self.queue(candidate=bystander)
        notifier.deliver_pending()
        follow_up = self.queue(kind="ending", candidate=candidate, make=auction)
        notifier.deliver_pending()
        waiting = self.queue(kind="price", candidate=candidate, make=auction)

        self.assertEqual(notifier.mark_ended(candidate["id"]), 2)
        self.assertEqual({u["ts"] for u in self.wire.updates}, {"1700000000.0001", "1700000000.0003"})
        for update in self.wire.updates:
            text = words({"text": update["text"], "blocks": update["blocks"]})
            self.assertIn("Ended", text)
            self.assertIn("`%s`" % candidate["code"], text)
            for phrase in ("$", "95.00", "Nike Dunk Low", LISTING_URL, "bid", "scheduled end"):
                self.assertNotIn(phrase, text)

        for alert_id in (first, follow_up, waiting):
            payload = self.alert(alert_id)["payload"]
            self.assertEqual(payload["kind"], "ended")
            self.assertEqual(payload["code"], candidate["code"])
            for key in ("title", "web_url", "image_url", "price", "current_bid", "min_next_bid", "end_at"):
                self.assertNotIn(key, payload)
        self.assertEqual(self.alert(waiting)["status"], "dropped")
        for row in self.store.messages_for_candidate(candidate["id"]):
            self.assertIsNotNone(row["expired_at"])
            self.assertEqual(row["has_listing_data"], 0)

        untouched = self.alert(other)
        self.assertEqual(untouched["payload"]["title"], "Nike Dunk Low Panda Men's 10")
        self.assertIsNone(self.messages(other)[0]["expired_at"])

    def test_an_ended_rewrite_that_failed_is_finished_on_the_next_run(self):
        candidate = self.candidate()
        notifier = self.notifier(SLACK)
        alert_id = self.queue(candidate=candidate)
        notifier.deliver_pending()
        self.wire.slack_down = "internal_error"
        self.assertEqual(notifier.mark_ended(candidate["id"]), 0)
        self.assertIsNone(self.messages(alert_id)[0]["expired_at"])

        self.wire.slack_down = None
        self.clock.advance(minutes=5)
        self.assertEqual(notifier.expire_messages(), 1)
        text = words(self.wire.updates[-1]["blocks"])
        self.assertIn("*Ended*", text)
        self.assertNotIn("$", text)
        self.assertIsNotNone(self.messages(alert_id)[0]["expired_at"])

    def test_a_dropped_alert_is_not_sent_later(self):
        candidate = self.candidate()
        notifier = self.notifier(SLACK)
        self.queue(candidate=candidate)
        notifier.mark_ended(candidate["id"])
        self.assertEqual(notifier.deliver_pending(), {"sent": 0, "held": 0, "dropped": 0, "failed": 0})
        self.assertEqual(self.wire.transport.calls, [])

    def test_a_queued_ended_notice_still_goes_out_without_listing_data(self):
        candidate = self.candidate()
        notifier = self.notifier(SLACK)
        notice = self.queue(kind="ended", candidate=candidate, make=auction)
        notifier.mark_ended(candidate["id"])
        self.assertEqual(notifier.deliver_pending()["sent"], 1)
        text = words(self.wire.posts[0])
        self.assertIn("Ended", text)
        self.assertNotIn("$", text)
        self.assertNotIn("Nike Dunk Low", text)
        self.assertEqual(self.messages(notice)[0]["has_listing_data"], 0)

    def test_ended_entry_in_a_digest_goes_and_the_others_stay_until_they_expire(self):
        gone, still_live = self.candidate(), self.candidate()
        self.queue("P3", candidate=gone, title="First listing", price=usd("111.00"))
        live_alert = self.queue("P3", candidate=still_live, title="Second listing", price=usd("222.00"))
        notifier = self.notifier(SLACK)
        notifier.deliver_pending()

        self.assertEqual(notifier.mark_ended(gone["id"]), 1)
        text = words(self.wire.updates[0]["blocks"])
        self.assertIn("`%s` · Panda Dunks 10 · ended" % gone["code"], text)
        self.assertNotIn("First listing", text)
        self.assertNotIn("$111.00", text)
        self.assertIn("Second listing", text)
        self.assertIn("Price: $222.00", text)
        self.assertIsNotNone(self.store.messages_for_candidate(gone["id"])[0]["expired_at"])
        self.assertIsNone(self.messages(live_alert)[0]["expired_at"])

        self.clock.advance(hours=6)
        self.assertEqual(notifier.expire_messages(), 1)
        text = words(self.wire.updates[1]["blocks"])
        self.assertNotIn("$", text)
        self.assertIn("Second listing", text)
        self.assertIn("`%s` · Panda Dunks 10 · ended" % gone["code"], text)


class SystemNotices(OutboxCase):
    def test_push_goes_first_and_carries_no_listing_fields(self):
        notifier = self.notifier()
        self.assertTrue(notifier.send_system("Crusoe key was rejected. Run eft doctor crusoe.", urgent=True))
        self.assertEqual(self.wire.posts, [])
        push = self.wire.pushes[0]
        self.assertEqual(push["message"], "Crusoe key was rejected. Run eft doctor crusoe.")
        self.assertEqual(push["priority"], "1")
        # No listing data, so nothing to expire: no ttl, and no link either.
        self.assertEqual(set(push), {"token", "user", "title", "message", "priority"})
        self.assertEqual(self.store.alerts_by_status(), [])

    def test_routine_notice_is_not_urgent(self):
        self.notifier().send_system("eBay calls are at 80% of today's cap.")
        self.assertEqual(self.wire.pushes[0]["priority"], "0")

    def test_slack_is_the_fallback(self):
        self.wire.push_down = True
        notifier = self.notifier()
        self.assertTrue(notifier.send_system("Model swapped: <new> & old"))
        self.assertEqual(len(self.wire.pushes), 1)
        post = self.wire.posts[0]
        self.assertEqual(post["channel"], "U0USER")
        self.assertEqual(post["text"], "Model swapped: &lt;new&gt; &amp; old")
        self.assertEqual(len(post["blocks"]), 1)

    def test_slack_alone_is_used_when_push_is_not_configured(self):
        self.assertTrue(self.notifier(SLACK).send_system("eBay keyset disabled"))
        self.assertEqual((len(self.wire.pushes), len(self.wire.posts)), (0, 1))

    def test_false_when_nothing_took_it(self):
        self.wire.push_down, self.wire.slack_down = True, "invalid_auth"
        notifier = self.notifier()
        self.assertFalse(notifier.send_system("eBay keyset disabled"))
        self.assertEqual(notifier.problems, ["pushover: application token is invalid", "slack: invalid_auth"])

    def test_quiet_hours_do_not_delay_a_safety_notice(self):
        self.night()
        self.assertTrue(self.notifier().send_system("Crusoe key was rejected", urgent=True))


class Secrets(OutboxCase):
    def leaky_notifier(self):
        """Every service here quotes our credentials back in its failure."""
        def handle(method, url, headers, body):
            if url == PUSHOVER:
                form = dict(urllib.parse.parse_qsl(body.decode("utf-8")))
                return json_response(
                    {"status": 0, "errors": ["token %s is not valid for user %s" % (form["token"], form["user"])]},
                    status=400)
            if url == UPDATE:
                return json_response({"ok": False, "error": "bad token " + headers["Authorization"]})
            raise TransportError("POST %s with %s failed" % (url, headers["Authorization"]))

        env = dict(SLACK, **PUSH)
        return Notifier(support.make_settings(**env), self.store, transport=FakeTransport(handle),
                        local_now=lambda: self.local)

    def test_no_token_appears_in_an_exception_or_a_problem_line(self):
        notifier = self.leaky_notifier()
        card = alerts.render_slack(fixed_price(), tz=UTC)
        raised = []
        for send in (lambda: notifier.post_slack(card),
                     lambda: notifier.update_slack({"channel": "D0001", "ts": "1.1"}, card),
                     lambda: notifier.post_push(alerts.render_push(fixed_price(), tz=UTC))):
            with self.assertRaises(DeliveryError) as caught:
                send()
            raised.append(caught.exception)
        self.assertEqual([str(error) for error in raised], [
            "slack: could not be reached",
            "slack: HTTP 200",
            "pushover: token [redacted] is not valid for user [redacted]",
        ])
        self.assertTrue(raised[0].__suppress_context__)

        self.queue()
        notifier.deliver_pending()
        notifier.send_system("Crusoe key was rejected")
        self.assertTrue(notifier.problems)
        for text in [str(error) for error in raised] + [repr(error) for error in raised] + notifier.problems:
            for secret in SECRETS:
                self.assertNotIn(secret, text)


if __name__ == "__main__":
    unittest.main()
