import contextlib
import copy
import io
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

import support
import test_tick as poller          # the fake eBay, the scripted model and the listing fixtures
from eft_core import cli, crusoe, doctor
from eft_core.alerts import Notifier
from eft_core.config import load_settings
from eft_core.crusoe import CrusoeClient
from eft_core.ebay import EbayClient
from eft_core.errors import CrusoeError, EbayGroupRequired, EbayNotFound
from eft_core.http import FakeTransport, json_response
from eft_core.store import Store, iso

LAUNCHER = os.path.join(support.ROOT, "scripts", "eft")
SECRETS = {
    "CRUSOE_API_KEY": "crusoe-secret-key-value",
    "EBAY_CLIENT_ID": "ebay-client-id-value",
    "EBAY_CLIENT_SECRET": "ebay-client-secret-value",
    "SLACK_BOT_TOKEN": "fake-slack-slack-secret-token",
    "SLACK_USER_ID": "U0USER",
    "PUSHOVER_APP_TOKEN": "pushover-app-secret",
    "PUSHOVER_USER_KEY": "pushover-user-secret",
    "EFT_HEARTBEAT_URL": poller.HEARTBEAT,
}
# SLACK_USER_ID is an address, not a credential; everything else must never be printed.
SECRET_VALUES = [value for name, value in SECRETS.items() if name != "SLACK_USER_ID"] + ["secret-heartbeat-id"]

DRAFT = {
    "brand": "Nike", "model": "Dunk Low", "style_codes": ["DD1391-100"],
    "colorway": {"official": "White/Black", "nicknames": ["Panda"], "colors": ["white", "black"]},
    "size": {"value": "10", "system": "US"}, "department": "men", "kids_tier": None,
    "visual_signature": ["white leather upper", "black swoosh"], "visible_text": [],
    "questions": ["Is the box important to you?"],
}
PATCH = {"set": {}, "add_must_not": ["GS"], "add_must_have": [], "add_nicknames": [], "set_requirement": {},
         "explanation": "Rules out grade-school pairs."}


def draft(**overrides):
    return dict(copy.deepcopy(DRAFT), **overrides)


class FakeCrusoe(poller.FakeCrusoe):
    """The poller's scripted model, plus draft answers and a record of every prompt."""

    def __init__(self):
        super().__init__()
        self.answers.update(parse=draft(), extract=draft())
        self.prompts = []

    def chat_json(self, role, system, parts, required=(), validator=None, max_tokens=None):
        self.prompts.append({"role": role, "system": system, "parts": parts})
        return super().chat_json(role, system, parts, required=required, validator=validator)

    def image_part(self, source):
        try:
            with open(str(source), "rb") as handle:
                return {"type": "image", "data": handle.read(), "mime": "image/jpeg"}
        except OSError:
            raise CrusoeError("cannot read the image file")

    def words(self):
        """The text of the last prompt."""
        return "\n".join(part["text"] for part in self.prompts[-1]["parts"] if part["type"] == "text")


class FakeEbay(poller.FakeEbay):
    """The poller's fake eBay, plus lookups by the numeric id found in a link."""

    def __init__(self):
        super().__init__()
        self.by_legacy = {}
        self.groups = {}

    def get_item_by_legacy_id(self, legacy_item_id, *, legacy_variation_id=None, purpose="detail"):
        self.calls.append(("get_item_by_legacy_id", legacy_item_id, {"purpose": purpose}))
        found = self.by_legacy[legacy_item_id]
        if isinstance(found, Exception):
            raise found
        return copy.deepcopy(found)

    def get_items_by_item_group(self, item_group_id, *, purpose="detail"):
        self.calls.append(("get_items_by_item_group", item_group_id, {"purpose": purpose}))
        return copy.deepcopy(self.groups[item_group_id])

    def names(self):
        return [call[0] for call in self.calls]


class CliTest(unittest.TestCase):
    ENV = SECRETS

    def setUp(self):
        # A HOME of its own, so that no test reads or writes the real ~/Library/LaunchAgents.
        self.home = tempfile.mkdtemp(prefix="eft-home-")
        patched = mock.patch.dict(os.environ, {"HOME": self.home})
        patched.start()
        self.addCleanup(patched.stop)
        self.clock = support.Clock()
        self.settings = support.make_settings(**self.ENV)
        self.store = support.make_store(self.clock)
        self.ebay = FakeEbay()
        self.crusoe = FakeCrusoe()
        self.wire = poller.Wire()
        self.notifier = Notifier(self.settings, self.store, transport=self.wire.transport,
                                 local_now=lambda: (self.clock() - timedelta(hours=5)).replace(tzinfo=None))
        self.commands = []
        self.launchctl_exit = 0
        self.rt = self.runtime()

    def runtime(self, **overrides):
        parts = dict(settings=self.settings, store=self.store, ebay=self.ebay, crusoe=self.crusoe,
                     notifier=self.notifier, transport=self.wire.transport, run=self.run_command, platform="darwin")
        parts.update(overrides)
        return cli.Runtime(**parts)

    def run_command(self, arguments):
        self.commands.append(list(arguments))
        return self.launchctl_exit, "launchctl said no" if self.launchctl_exit else ""

    def eft(self, *argv):
        """Run one command. Returns (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv), runtime=self.rt)
        self.assertNotIn("Traceback", err.getvalue())
        return code, out.getvalue(), err.getvalue()

    def eft_json(self, *argv):
        code, out, _ = self.eft("--json", *argv)
        return code, json.loads(out)

    def draft_watch(self, **spec_overrides):
        return self.store.get_watch(self.store.create_watch("Panda Dunks 10", poller.dunk_spec(**spec_overrides)))

    def active_watch(self, **spec_overrides):
        return poller.TickTest.watch(self, **spec_overrides)

    def tracked(self, n=1, **listing_overrides):
        """List a matching listing and let one poll find it and alert on it."""
        self.ebay.list(poller.detail(n, **listing_overrides))
        code, _, _ = self.eft("tick")
        self.assertEqual(code, 0)
        return self.store.get_candidate(1, "v1|%d|0" % n)


class Add(CliTest):
    def test_add_from_text_drafts_a_watch_and_keeps_only_style_codes_that_were_stated(self):
        self.crusoe.answers["parse"] = draft(style_codes=["DD1391-100", "CW1590-100"], visible_text=["CW1590-100"])
        code, data = self.eft_json("add", "--text", "Nike Dunk Low Panda DD1391-100, men's US 10")
        self.assertEqual(code, 0)
        watch = self.store.get_watch(data["watch"]["id"])
        self.assertEqual((watch["status"], watch["name"]), ("draft", "Nike Dunk Low"))
        self.assertEqual(watch["spec"]["style_codes"], ["DD1391-100"], "a code the text never stated is dropped")
        self.assertIn("CW1590-100", data["notes"][0])
        self.assertEqual(watch["spec"]["size"], {"value": "10", "system": "US", "width": None})
        self.assertIn("Is the box important to you?", data["questions"])
        self.assertTrue(any("price ceiling" in question for question in data["questions"]))
        self.assertEqual(data["problems"], [])
        self.assertEqual([prompt["role"] for prompt in self.crusoe.prompts], ["parse"])
        self.assertIn("Nike Dunk Low Panda DD1391-100", self.crusoe.words())
        self.assertEqual(self.ebay.calls, [])

    def test_add_from_an_image_reads_the_photo_and_keeps_a_copy_as_the_reference(self):
        photo = os.path.join(tempfile.mkdtemp(prefix="eft-photo-"), "My Shoes.JPG")
        with open(photo, "wb") as handle:
            handle.write(b"\xff\xd8\xff my own photo")
        self.crusoe.answers["extract"] = draft(size=None, department=None, visible_text=["NIKE", "DD1391-100"])
        code, data = self.eft_json("add", "--image", photo, "--note", "bought in 2021", "--name", "Pandas")
        self.assertEqual(code, 0)
        [prompt] = self.crusoe.prompts
        self.assertEqual(prompt["role"], "extract")
        self.assertEqual([part["type"] for part in prompt["parts"]], ["text", "image"])
        self.assertIn("bought in 2021", self.crusoe.words())
        watch = self.store.get_watch(data["watch"]["id"])
        self.assertEqual(watch["name"], "Pandas")
        self.assertEqual(watch["spec"]["style_codes"], ["DD1391-100"], "read off the photo, so it is kept")
        [reference] = watch["spec"]["reference_images"]
        self.assertEqual(os.path.dirname(reference), str(self.settings.refs_dir))
        self.assertTrue(reference.endswith(".jpg"))
        with open(reference, "rb") as handle:
            self.assertEqual(handle.read(), b"\xff\xd8\xff my own photo")
        # A photo cannot say the size, and a watch without one cannot be activated.
        self.assertTrue(any("Which size" in question for question in data["questions"]))

    def test_add_from_an_image_that_cannot_be_read_is_a_handled_problem(self):
        code, _, err = self.eft("add", "--image", "/nowhere/photo.jpg")
        self.assertEqual(code, 1)
        self.assertIn("cannot read the image", err)
        self.assertEqual(self.store.list_watches(), [])

    def test_add_from_an_ebay_item_link_looks_the_listing_up_through_the_api(self):
        self.ebay.by_legacy["256012345678"] = poller.detail(title="Nike Dunk Low Panda DD1391-100 Mens 10")
        code, data = self.eft_json("add", "--url", "https://www.ebay.com/itm/256012345678?hash=item3b9&var=0")
        self.assertEqual(code, 0)
        self.assertEqual(self.ebay.calls, [("get_item_by_legacy_id", "256012345678", {"purpose": "add"})])
        self.assertEqual(self.wire.transport.calls, [], "the link is parsed, never requested")
        self.assertIn("Title: Nike Dunk Low Panda DD1391-100 Mens 10", self.crusoe.words())
        watch = self.store.get_watch(data["watch"]["id"])
        self.assertEqual(watch["spec"]["style_codes"], ["DD1391-100"])
        self.assertEqual(watch["spec"]["provenance"], {"source": "ebay item"})
        self.assertNotIn("120.00", json.dumps(data), "no price comes along from the listing")
        self.assertNotIn("Is the box important to you?", data["questions"],
                         "free text from a model that read a seller's listing is not passed on")

    def test_an_item_link_to_a_multi_size_listing_asks_which_size(self):
        self.ebay.by_legacy["256012345678"] = EbayGroupRequired("multi-variation parent", item_group_id="900")
        self.ebay.groups["900"] = [poller.detail(1), poller.detail(2)]
        code, data = self.eft_json("add", "--url", "https://www.ebay.com/itm/256012345678")
        self.assertEqual(code, 0)
        self.assertEqual(self.ebay.names(), ["get_item_by_legacy_id", "get_items_by_item_group"])
        self.assertIsNone(self.store.get_watch(data["watch"]["id"])["spec"]["size"])
        self.assertTrue(any("several sizes" in question for question in data["questions"]))

    def test_an_item_link_ebay_does_not_know_is_a_handled_problem(self):
        self.ebay.by_legacy["256012345678"] = EbayNotFound(
            "eBay legacy item lookup failed: HTTP 404, error 11003", status=404)
        code, _, err = self.eft("add", "--url", "https://www.ebay.com/itm/256012345678")
        self.assertEqual(code, 1)
        self.assertEqual(err, "eft: eBay legacy item lookup failed: HTTP 404, error 11003\n")

    def test_add_from_a_product_link_searches_by_epid(self):
        self.ebay.list(poller.detail(1, title="Nike Dunk Low Retro Panda"),
                       poller.detail(2, title="Nike Dunk Low DD1391-100"))
        code, data = self.eft_json("add", "--url", "https://www.ebay.com/p/1234567890?iid=1")
        self.assertEqual(code, 0)
        [(name, query, kwargs)] = self.ebay.calls
        self.assertEqual((name, query), ("search", None))
        self.assertEqual((kwargs["epid"], kwargs["purpose"]), ("1234567890", "add"))
        self.assertIn("- Nike Dunk Low Retro Panda", self.crusoe.words())
        self.assertEqual(self.store.get_watch(data["watch"]["id"])["spec"]["provenance"], {"source": "ebay product"})

    def test_add_from_a_search_link_uses_its_keywords_and_filters_without_any_ebay_call(self):
        url = "https://www.ebay.com/sch/i.html?_nkw=nike+dunk+low+panda&LH_Auction=1&_udhi=150"
        self.crusoe.answers["parse"] = draft(style_codes=[])
        code, data = self.eft_json("add", "--url", url)
        self.assertEqual(code, 0)
        self.assertEqual(self.ebay.calls, [])
        self.assertIn("nike dunk low panda", self.crusoe.words())
        watch = self.store.get_watch(data["watch"]["id"])
        self.assertEqual(watch["spec"]["buying_options"], ["AUCTION"])
        self.assertEqual(watch["spec"]["price_ceiling"]["amount"], 150.0)
        self.assertTrue(any("150" in question for question in data["questions"]))

    def test_a_short_link_is_refused_with_a_request_for_the_full_url(self):
        code, out, err = self.eft("add", "--url", "https://ebay.us/m/AbC123")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("full URL", err)
        self.assertEqual((self.ebay.calls, self.crusoe.prompts, self.wire.transport.calls), ([], [], []))
        self.assertEqual(self.store.list_watches(), [])

    def test_a_link_that_is_not_ebay_is_refused_and_never_fetched(self):
        code, _, err = self.eft("add", "--url", "https://www.nike.com/t/dunk-low-retro-DD1391-100")
        self.assertEqual(code, 1)
        self.assertIn("eft add --text", err)
        self.assertEqual((self.ebay.calls, self.crusoe.prompts, self.wire.transport.calls), ([], [], []))

    def test_an_ebay_link_that_is_not_an_item_product_or_search_is_refused(self):
        code, _, err = self.eft("add", "--url", "https://www.ebay.com/usr/some_store")
        self.assertEqual(code, 1)
        self.assertIn("not an item, product or search link", err)

    def test_a_link_in_model_written_text_is_never_shown(self):
        self.crusoe.answers["parse"] = draft(questions=["Confirm the size at https://evil.example/size-chart\nplease",
                                                        "See www.evil.example now"])
        _, data = self.eft_json("add", "--text", "Nike Dunk Low Panda")
        self.assertIn("Confirm the size at [link removed] please", data["questions"])
        self.assertNotIn("evil.example", json.dumps(data))

    def test_add_needs_exactly_one_source(self):
        self.assertEqual(self.eft("add")[0], 2)
        self.assertEqual(self.eft("add", "--text", "a", "--url", "b")[0], 2)


class Watches(CliTest):
    def test_watch_set_changes_the_spec_and_makes_a_newly_set_field_required(self):
        watch = self.draft_watch(size=None, department=None, price_ceiling=None)
        self.assertEqual(watch["spec"]["requirements"]["size"], "na")
        code, data = self.eft_json("watch", "set", str(watch["id"]), "size=US 10", "department=men",
                                   "price_ceiling=180 incl", "must_not=Disrupt, GS", "strictness=strict",
                                   "require.style_code=required", "colorway=White/Black")
        self.assertEqual(code, 0)
        stored = self.store.get_watch(watch["id"])["spec"]
        self.assertEqual(stored, data["spec"])
        self.assertEqual(stored["size"], {"value": "10", "system": "US", "width": None})
        self.assertEqual(stored["price_ceiling"], {"amount": 180.0, "currency": "USD", "incl_shipping": True})
        self.assertEqual(stored["must_not"], ["Disrupt", "GS"])
        self.assertEqual(stored["strictness"], "strict")
        self.assertEqual(stored["colorway"]["nicknames"], ["Panda"], "setting the official name keeps the nicknames")
        self.assertEqual({name: stored["requirements"][name] for name in ("size", "department", "style_code")},
                         {"size": "required", "department": "required", "style_code": "required"})
        self.assertEqual(data["problems"], [])

    def test_clearing_a_field_stops_requiring_it(self):
        watch = self.draft_watch()
        self.eft("watch", "set", str(watch["id"]), "department=")
        stored = self.store.get_watch(watch["id"])["spec"]
        self.assertEqual((stored["department"], stored["requirements"]["department"]), (None, "na"))

    def test_a_setting_that_is_not_key_value_or_not_known_is_a_usage_error(self):
        watch = self.draft_watch()
        for bad in ("size", "colour=red", "require.size=sometimes", "strictness=loose", "price_ceiling=cheap"):
            code, _, err = self.eft("watch", "set", str(watch["id"]), bad)
            self.assertEqual(code, 2, bad)
            self.assertTrue(err.startswith("eft: error: "), err)
        self.assertEqual(self.store.get_watch(watch["id"])["spec"], watch["spec"])

    def test_activate_refuses_a_spec_with_problems(self):
        watch = self.draft_watch(size="10")         # a size without a size system proves nothing
        code, _, err = self.eft("watch", "activate", str(watch["id"]))
        self.assertEqual(code, 1)
        self.assertIn("size 10 has no size system", err)
        self.assertEqual(self.store.get_watch(watch["id"])["status"], "draft")
        self.assertEqual(self.ebay.calls, [])

    def test_activate_builds_the_plan_and_starts_the_window_now(self):
        watch = self.draft_watch()
        code, data = self.eft_json("watch", "activate", "Panda")        # a watch can be named by a fragment
        self.assertEqual(code, 0)
        stored = self.store.get_watch(watch["id"])
        self.assertEqual((stored["status"], stored["window_end"]), ("active", iso(self.clock())))
        self.assertEqual(stored["plan"]["q"], "Nike Dunk Low DD1391-100")
        self.assertEqual(self.ebay.searches("plan")[0]["fieldgroups"], "ASPECT_REFINEMENTS")
        self.assertEqual(data["search_every_minutes"], {"hot": 5, "normal": 10})

    def test_activate_without_ebay_keys_names_the_variables(self):
        self.settings = support.make_settings(EBAY_CLIENT_ID="", EBAY_CLIENT_SECRET="")
        self.rt = self.runtime()
        watch = self.draft_watch()
        code, _, err = self.eft("watch", "activate", str(watch["id"]))
        self.assertEqual(code, 1)
        self.assertIn("EBAY_CLIENT_ID, EBAY_CLIENT_SECRET", err)

    def test_pause_hot_and_normal(self):
        watch = self.active_watch()
        self.assertEqual(self.eft("watch", "pause", "1")[0], 0)
        self.assertEqual(self.store.get_watch(watch["id"])["status"], "paused")
        self.eft("watch", "hot", "1")
        self.assertEqual(self.store.get_watch(watch["id"])["tier"], "hot")
        self.eft("watch", "normal", "1")
        self.assertEqual(self.store.get_watch(watch["id"])["tier"], "normal")
        self.assertEqual(self.eft("watch", "pause", "99")[0], 1)

    def test_a_changed_spec_changes_the_search_of_an_active_watch(self):
        watch = self.active_watch()
        self.eft("watch", "set", "1", "style_codes=")
        self.assertEqual(self.store.get_watch(watch["id"])["plan"]["q"], "Nike Dunk Low Panda")
        code, _, err = self.eft("watch", "set", "1", "size=10")
        self.assertEqual(code, 1, "an active watch is not left with a spec that cannot match")
        self.assertIn("Pause it first", err)

    def test_remove_rewrites_sent_cards_then_deletes_the_watch_and_its_reference_photo(self):
        self.settings.ensure_state_dir()
        reference = os.path.join(str(self.settings.refs_dir), "abc.jpg")
        with open(reference, "wb") as handle:
            handle.write(b"photo")
        self.active_watch(reference_images=[reference])
        self.crusoe.answers["triage"] = poller.all_titles_look_right
        row = self.tracked()
        self.assertEqual(len(self.wire.posts()), 1)
        code, data = self.eft_json("watch", "remove", "1")
        self.assertEqual((code, data["removed"]["id"]), (0, 1))
        self.assertIn("has ended", json.dumps(self.wire.updates()[-1]))
        for rows in (self.store.list_watches(), self.store.list_candidates(), self.store.alerts_by_status()):
            self.assertEqual(rows, [])
        self.assertIsNone(self.store.get_candidate_by_id(row["id"]))
        self.assertFalse(os.path.exists(reference))

    def test_remove_waits_when_a_card_still_showing_data_cannot_be_rewritten(self):
        self.active_watch()
        row = self.tracked()
        self.wire.transport.handler = lambda *request: json_response({"ok": False, "error": "ratelimited"})
        code, _, err = self.eft("watch", "remove", "1")
        self.assertEqual(code, 1)
        self.assertIn("run this command again", err)
        self.assertEqual(self.store.get_watch(1)["status"], "paused")
        self.assertIsNone(self.store.get_candidate_by_id(row["id"])["listing"], "the stored listing data is gone")
        showing = [message["channel"] for message in self.store.messages_for_candidate(row["id"])
                   if message["has_listing_data"]]
        self.assertEqual(showing, ["slack"], "still on record, so the poller can strip it later")

    def test_list_and_show(self):
        self.active_watch()
        self.tracked()
        self.store.kv_set("watch:1:overflow", {"at": iso(self.clock())})
        code, data = self.eft_json("watch", "list")
        self.assertEqual(code, 0)
        [view] = data["watches"]
        self.assertEqual((view["tracked"], view["too_broad"]), ({"alerted": 1}, True))
        code, out, _ = self.eft("watch", "show", "1")
        self.assertIn("requirements: brand required", out)
        self.assertIn("search: Nike Dunk Low DD1391-100", out)
        self.assertNotIn("120.00", out)


class DryRun(CliTest):
    def test_dryrun_shows_decisions_and_leaves_nothing_behind(self):
        watch = self.draft_watch()              # a dry run is how a draft is checked before it goes live
        self.ebay.list(poller.detail(1), poller.detail(2, title="Nike Dunk Low Panda GS 6Y"))
        code, data = self.eft_json("dryrun", str(watch["id"]), "--limit", "5")
        self.assertEqual(code, 0)
        [search] = self.ebay.searches("dryrun")
        self.assertEqual((search["limit"], search["sort"]), (5, "newlyListed"))
        self.assertNotIn("start_from", search)
        self.assertEqual(data["decisions"], {"alert": 1, "reject": 1})
        alert = next(result for result in data["results"] if result["decision"] == "alert")
        self.assertEqual(set(alert), {"decision", "reason", "proof", "unproven", "title", "url", "price", "ends_at",
                                      "as_of"})
        self.assertEqual(alert["as_of"], iso(self.clock()))

        self.assertEqual(self.store.list_candidates(), [])
        self.assertEqual(self.store.alerts_by_status(), [])
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM verdict").fetchone()[0], 0)
        self.assertEqual(self.wire.transport.calls, [], "a dry run sends nothing")
        self.assertIsNone(self.store.get_watch(watch["id"])["window_end"])

    def test_dryrun_limit_is_checked(self):
        watch = self.draft_watch()
        self.assertEqual(self.eft("dryrun", str(watch["id"]), "--limit", "0")[0], 2)
        self.assertEqual(self.eft("dryrun", str(watch["id"]), "--limit", "many")[0], 2)


class StatusAndDoctor(CliTest):
    def test_status_json_is_valid_and_holds_no_secret(self):
        self.active_watch()
        self.tracked()
        self.store.kv_set("ebay:token", {"access_token": "v^1.1#live-token-value",
                                         "expires_at": "2026-11-12T20:00:00Z"})
        code, out, err = self.eft("--json", "status")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(set(data), {"version", "config", "watches", "last_tick", "calls_today", "waiting"})
        self.assertTrue(all(feature["configured"] for feature in data["config"]["features"].values()))
        self.assertEqual(data["last_tick"]["at"], iso(self.clock()))
        self.assertEqual(data["watches"][0]["tracked"], {"alerted": 1})
        for secret in SECRET_VALUES + ["live-token-value"]:
            self.assertNotIn(secret, out + err)
        self.assertNotIn("Nike Dunk Low Panda White Black", out, "status carries no listing data")

    def test_status_says_what_is_still_to_set_up(self):
        self.settings = load_settings(environ={"HOME": tempfile.mkdtemp(), "EFT_STATE_DIR": tempfile.mkdtemp()},
                                      skill_root=tempfile.mkdtemp())
        self.rt = self.runtime()
        self.draft_watch()
        code, data = self.eft_json("status")
        self.assertEqual(code, 0)
        waiting = " | ".join(data["waiting"])
        for expected in ("Set CRUSOE_API_KEY", "Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET", "Add an alert channel",
                         "Draft watches waiting to be activated: 1"):
            self.assertIn(expected, waiting)
        self.assertNotIn("eft agent install", waiting, "no poller is needed before a watch is active")
        self.assertIn("still to do:", self.eft("status")[1])

    def test_status_asks_for_the_poller_once_a_watch_is_active_and_until_it_is_installed(self):
        self.active_watch()
        self.assertIn("Start the background poller: eft agent install", self.eft_json("status")[1]["waiting"])
        self.eft("agent", "install")
        self.assertEqual(self.eft_json("status")[1]["waiting"], [])

    def test_doctor_env_json_names_variables_and_holds_no_secret(self):
        code, out, err = self.eft("--json", "doctor", "env")
        data = json.loads(out)
        self.assertEqual((code, data["ok"]), (0, True))
        for row in data["checks"]:
            self.assertEqual(set(row), {"area", "name", "ok", "detail"})
            self.assertEqual(row["area"], "env")
        details = {row["name"]: row["detail"] for row in data["checks"]}
        self.assertEqual(details["crusoe"], "set: CRUSOE_API_KEY")
        self.assertEqual(details["state directory"], str(self.settings.state_dir))
        for secret in SECRET_VALUES:
            self.assertNotIn(secret, out + err)

    def test_doctor_env_fails_when_a_required_key_is_missing_or_a_feature_is_half_set(self):
        self.settings = support.make_settings(CRUSOE_API_KEY="", SLACK_BOT_TOKEN="fake-slack-only-half")
        self.rt = self.runtime()
        code, data = self.eft_json("doctor", "env")
        self.assertEqual(code, 1)
        rows = {row["name"]: row for row in data["checks"]}
        self.assertEqual((rows["crusoe"]["ok"], rows["crusoe"]["detail"]), (False, "not set: CRUSOE_API_KEY"))
        self.assertEqual((rows["slack"]["ok"], rows["slack"]["detail"]), (False, "not set: SLACK_USER_ID"))
        self.assertIsNone(rows["pushover"]["ok"])
        self.assertIs(rows["alert channel"]["ok"], False)
        self.assertIsNone(rows[".env file"]["ok"])

    def test_doctor_env_reports_an_unreadable_env_file_and_an_unwritable_state_directory(self):
        blocked = os.path.join(tempfile.mkdtemp(), "a-file")
        open(blocked, "w").close()
        settings = load_settings(environ={"HOME": tempfile.mkdtemp(), "EFT_STATE_DIR": os.path.join(blocked, "state"),
                                          "EFT_ENV_FILE": tempfile.mkdtemp()}, skill_root=tempfile.mkdtemp())
        rows = {row["name"]: row for row in doctor.run_checks("env", settings, None)}
        self.assertIs(rows[".env file"]["ok"], False)
        self.assertIn("export the variables", rows[".env file"]["detail"])
        self.assertIs(rows["state directory"]["ok"], False)
        # The same directory as a command's state: a handled problem, not a traceback.
        self.rt = cli.Runtime(settings=settings)
        code, _, err = self.eft("status")
        self.assertEqual(code, 1)
        self.assertIn("EFT_STATE_DIR", err)

    def test_doctor_env_points_out_an_env_file_that_other_users_can_read(self):
        root = tempfile.mkdtemp()
        env_path = os.path.join(root, ".env")
        with open(env_path, "w") as handle:
            handle.write("CRUSOE_API_KEY=from-file\n")
        for mode, ok in ((0o644, None), (0o600, True)):
            os.chmod(env_path, mode)
            settings = load_settings(environ={"HOME": root, "EFT_ENV_FILE": env_path}, skill_root=root)
            row = {row["name"]: row for row in doctor.check_env(settings)}[".env file"]
            self.assertIs(row["ok"], ok)
            self.assertEqual("chmod 600" in row["detail"], ok is None)

    def test_doctor_rejects_an_unknown_area(self):
        self.assertEqual(self.eft("doctor", "sleep")[0], 2)
        with self.assertRaises(ValueError):
            doctor.run_checks("sleep", self.settings, self.store)


class DoctorProbes(unittest.TestCase):
    """The network checks, run against the real clients over a fake transport."""

    GEMMA, GPT_OSS = "google/gemma-4-31b-it", "openai/gpt-oss-120b"

    def setUp(self):
        self.settings = support.make_settings(**SECRETS)
        self.store = support.make_store()
        self.transport = FakeTransport(self.handle)
        self.status = {}        # url fragment -> (status, body) to answer with instead of success

    def handle(self, method, url, headers, body):
        for fragment, (status, reply) in self.status.items():
            if fragment in url:
                return json_response(reply, status=status)
        if url.endswith("/models"):
            return json_response({"data": [
                {"id": self.GEMMA, "architecture": {"modality": "multimodal"}, "supported_parameters": []},
                {"id": self.GPT_OSS, "architecture": {"modality": "text"}, "supported_parameters": []},
            ]})
        if url.endswith("/chat/completions"):
            request = json.loads(body)
            content = '{"same": false}' if "two images" in request["messages"][0]["content"] else '{"ok": true}'
            return json_response({"model": request["model"], "choices": [
                {"finish_reason": "stop", "message": {"content": content}}], "usage": {}})
        if url.endswith("/identity/v1/oauth2/token"):
            return json_response({"access_token": "v^1.1#minted-token-value", "expires_in": 7200})
        if "/item_summary/search" in url:
            return json_response({"total": 4321, "itemSummaries": []})
        if url.endswith("chat.postMessage"):
            return json_response({"ok": True, "channel": "D0DM", "ts": "1.2"})
        if url.endswith("messages.json"):
            return json_response({"status": 0, "errors": ["application token is invalid"]}, status=400)
        raise AssertionError("unexpected request: %s %s" % (method, url))

    def rows(self, area):
        clients = {
            "crusoe": CrusoeClient(self.settings, self.store, self.transport, sleep=lambda seconds: None),
            "ebay": EbayClient(self.settings, self.store, self.transport),
            "notifier": Notifier(self.settings, self.store, transport=self.transport),
        }
        with mock.patch.object(crusoe, "_can_use_sips", return_value=False):
            found = doctor.run_checks(area, self.settings, self.store, **clients)
        for secret in SECRET_VALUES + ["minted-token-value"]:
            self.assertNotIn(secret, json.dumps(found))
        return {row["name"]: row for row in found}

    def test_crusoe_probes_resolve_every_role_and_try_a_two_image_request(self):
        rows = self.rows("crusoe")
        self.assertEqual(rows["catalog"]["detail"], "2 models listed")
        self.assertEqual({name: row["detail"] for name, row in rows.items() if name.startswith("role ")}, {
            "role extract": self.GEMMA, "role compare": self.GEMMA, "role tag_read": self.GEMMA,
            "role parse": self.GPT_OSS, "role triage": self.GPT_OSS, "role verify": self.GPT_OSS})
        self.assertTrue(all(row["ok"] for row in rows.values()), rows)
        self.assertIn(self.GPT_OSS + " answered in ", rows["text model answers in JSON"]["detail"])
        last = json.loads(self.transport.calls[-1]["body"])
        kinds = [part["type"] for part in last["messages"][1]["content"]]
        self.assertEqual(kinds, ["text", "image_url", "image_url"])
        self.assertTrue(last["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_a_refused_crusoe_key_fails_the_catalog_check_without_showing_the_key(self):
        self.status["/models"] = (401, {"error": {"message": "invalid key crusoe-secret-key-value"}})
        rows = self.rows("crusoe")
        self.assertEqual(list(rows), ["catalog"])
        self.assertIs(rows["catalog"]["ok"], False)
        self.assertIn("HTTP 401", rows["catalog"]["detail"])

    def test_a_redirected_model_id_is_pointed_out(self):
        self.status["/chat/completions"] = (200, {"model": "another/model", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"ok": true, "same": true}'}}]})
        row = self.rows("crusoe")["text model answers in JSON"]
        self.assertIsNone(row["ok"])
        self.assertIn("another/model answered", row["detail"])
        self.assertIn(self.GPT_OSS + " was requested", row["detail"])

    def test_ebay_probes_mint_a_token_search_and_count_calls(self):
        rows = self.rows("ebay")
        self.assertTrue(all(row["ok"] for row in rows.values()), rows)
        self.assertEqual(rows["application token"]["detail"], "issued for EBAY_US (production)")
        self.assertEqual(rows["search"]["detail"], "HTTP 200, 4321 listings found for a keyword search")
        self.assertIn("buyingOptions%3A%7BAUCTION%7D", self.transport.calls[-1]["url"])
        self.assertEqual(rows["calls today"]["detail"], "2 of 4800 today (UTC) (doctor 2)")

    def test_a_failed_ebay_search_reports_status_and_error_ids(self):
        self.status["/item_summary/search"] = (403, {"errors": [{"errorId": 1100, "message": "Access denied"}]})
        rows = self.rows("ebay")
        self.assertIs(rows["search"]["ok"], False)
        self.assertTrue(rows["search"]["detail"].startswith("HTTP 403, error ids 1100: "), rows["search"]["detail"])

    def test_a_refused_keyset_stops_at_the_token(self):
        self.status["/oauth2/token"] = (401, {"error": "invalid_client",
                                              "error_description": "client authentication failed"})
        rows = self.rows("ebay")
        self.assertEqual(list(rows), ["application token"])
        self.assertIn("HTTP 401", rows["application token"]["detail"])

    def test_alert_probes_send_a_test_message_on_each_channel(self):
        rows = self.rows("alerts")
        self.assertEqual((rows["slack"]["ok"], rows["slack"]["detail"]), (True, "test message sent"))
        self.assertIs(rows["pushover"]["ok"], False)
        self.assertIn("application token is invalid", rows["pushover"]["detail"])
        slack = json.loads(next(call for call in self.transport.calls if "chat.postMessage" in call["url"])["body"])
        self.assertIn("test message", slack["text"])

    def test_no_channel_and_no_keys_are_reported_without_any_request(self):
        self.settings = load_settings(environ={"HOME": tempfile.mkdtemp(), "EFT_STATE_DIR": tempfile.mkdtemp()},
                                      skill_root=tempfile.mkdtemp())
        found = doctor.run_checks("all", self.settings, self.store)
        self.assertEqual(self.transport.calls, [])
        failed = {(row["area"], row["name"]) for row in found if row["ok"] is False}
        self.assertTrue({("crusoe", "api key"), ("ebay", "keyset"), ("alerts", "channel")} <= failed)


class Tracking(CliTest):
    def setUp(self):
        super().setUp()
        self.active_watch()

    def test_tick_reports_what_it_did_and_exits_1_when_something_failed(self):
        self.ebay.list(poller.detail())
        code, data = self.eft_json("tick")
        self.assertEqual((code, data["queued"], data["delivery"]["sent"], data["problems"]), (0, 1, 1, []))
        self.assertTrue(os.path.exists(os.path.join(str(self.settings.state_dir), "tick.lock")))
        self.clock.advance(minutes=10)
        self.ebay.search_error = poller.EbayUnavailable("eBay search failed: HTTP 503", status=503)
        code, out, _ = self.eft("tick")
        self.assertEqual(code, 1)
        self.assertIn("watch 1: eBay search failed: HTTP 503", out)

    def test_a_second_tick_at_the_same_time_does_nothing(self):
        with cli.tick_lock(self.settings) as free:
            self.assertTrue(free)
            code, data = self.eft_json("tick")
        self.assertEqual((code, data), (0, {"skipped": "another eft tick is running"}))
        self.assertEqual(self.ebay.calls, [])

    def test_inbox_shows_what_was_sent_since_the_last_look_with_fresh_data_only(self):
        row = self.tracked()
        code, data = self.eft_json("inbox")
        self.assertEqual(code, 0)
        [entry] = data["sent"]
        self.assertEqual((entry["code"], entry["kind"], entry["state"]), (row["code"], "new", "alerted"))
        self.assertEqual((entry["price"], entry["as_of"]), (poller.usd("120.00"), iso(self.clock())))
        self.assertEqual(entry["url"], "https://www.ebay.com/itm/1")
        self.clock.advance(minutes=1)
        code, data = self.eft_json("inbox")
        self.assertEqual((data["sent"], data["waiting"]), ([], []), "already seen")

    def test_inbox_does_not_show_listing_facts_older_than_six_hours(self):
        self.tracked()
        self.clock.advance(hours=6, minutes=1)          # no poll ran in between to refresh the listing
        code, data = self.eft_json("inbox")
        [entry] = data["sent"]
        self.assertNotIn("price", entry)
        self.assertIn("older than six hours", entry["note"])
        self.assertEqual(entry["url"], "https://www.ebay.com/itm/1")

    def test_star_and_unstar(self):
        row = self.tracked()
        self.assertEqual(self.eft("star", row["code"].lower())[0], 0)
        self.assertEqual(self.store.get_candidate_by_id(row["id"])["starred"], 1)
        self.assertEqual(self.eft("unstar", row["code"])[0], 0)
        self.assertEqual(self.store.get_candidate_by_id(row["id"])["starred"], 0)
        code, _, err = self.eft("star", "Z99")
        self.assertEqual(code, 1)
        self.assertIn("no listing has the code", err)

    def test_explain_prints_the_stored_trace(self):
        row = self.tracked()
        code, out, _ = self.eft("explain", row["code"])
        self.assertEqual(code, 0)
        for expected in ("alerted (all_required_fields_proven)", "title_triage", "served-triage", "detail_verify",
                         "size: match (item specifics) \"US 10\"", "decision", "rule: all_required_fields_proven"):
            self.assertIn(expected, out)
        code, data = self.eft_json("explain", row["code"])
        self.assertEqual([step["stage"] for step in data["trace"]],
                         ["rules", "title_triage", "detail_fetch", "spec_check", "detail_verify", "decision"])
        self.assertEqual(data["listing"]["as_of"], iso(self.clock()))
        self.assertEqual(self.eft("explain", "Z99")[0], 1)


class Feedback(CliTest):
    def setUp(self):
        super().setUp()
        self.watch = self.active_watch()
        self.adult = self.tracked(1)
        self.kids = self.tracked_as("Nike Dunk Low Panda GS 6Y", 2)
        self.wrong = self.tracked_as("Nike Dunk Low Panda Big Kids", 3)
        self.crusoe.answers["parse"] = copy.deepcopy(PATCH)
        self.crusoe.prompts = []

    def tracked_as(self, title, n):
        """A tracked match whose stored title is `title` (as if the funnel had let it through)."""
        row = self.store.upsert_candidate(self.watch["id"], "v1|%d|0" % n, "alerted", tracked=True,
                                          listing={"title": title, "web_url": "https://www.ebay.com/itm/%d" % n},
                                          fetched_at=iso(self.clock()))
        proven = ("brand", "model", "colorway", "size", "department")
        self.store.add_verdict(row["id"], "decision", fields={
            name: {"verdict": "match", "evidence": "quoted text"} for name in proven})
        return row

    def test_feedback_marks_the_listing_rejected_and_proposes_a_patch_without_applying_it(self):
        code, data = self.eft_json("feedback", self.wrong["code"], "--reason", "that is the kids version")
        self.assertEqual(code, 0)
        row = self.store.get_candidate_by_id(self.wrong["id"])
        self.assertEqual((row["state"], row["reason"], row["tracked"], row["listing"]),
                         ("rejected", "user_feedback", 0, None))
        self.assertEqual(self.crusoe.prompts[-1]["role"], "parse")
        self.assertIn("Nike Dunk Low Panda Big Kids", self.crusoe.words())
        self.assertIn("that is the kids version", self.crusoe.words())

        self.assertEqual(data["patch"], {"add_must_not": ["GS"]})
        self.assertEqual(data["explanation"], "Rules out grade-school pairs.")
        self.assertEqual((data["would_remove"], data["would_keep"]), ([self.kids["code"]], [self.adult["code"]]))
        self.assertEqual(data["apply_with"], "eft feedback --apply %d" % data["feedback_id"])
        self.assertEqual(self.store.get_watch(self.watch["id"])["spec"]["must_not"], [],
                         "nothing changes until it is applied")
        [stored] = self.store.list_feedback()
        self.assertEqual((stored["label"], stored["approved"], stored["candidate_code"]),
                         ("not_a_match", 0, self.wrong["code"]))

    def test_apply_changes_the_spec_and_removes_what_the_replay_said_it_would(self):
        _, proposed = self.eft_json("feedback", self.wrong["code"], "--reason", "that is the kids version")
        code, data = self.eft_json("feedback", "--apply", str(proposed["feedback_id"]))
        self.assertEqual(code, 0)
        self.assertEqual(self.store.get_watch(self.watch["id"])["spec"]["must_not"], ["GS"])
        self.assertEqual((data["removed"], data["kept"]), ([self.kids["code"]], [self.adult["code"]]))
        gone = self.store.get_candidate_by_id(self.kids["id"])
        self.assertEqual((gone["state"], gone["reason"], gone["listing"]), ("rejected", "spec_changed", None))
        self.assertEqual(self.store.get_candidate_by_id(self.adult["id"])["state"], "alerted")
        self.assertEqual([(row["label"], row["approved"]) for row in self.store.list_feedback()],
                         [("not_a_match", 0), ("patch_applied", 1)])

        code, _, err = self.eft("feedback", "--apply", str(proposed["feedback_id"]))
        self.assertEqual(code, 1)
        self.assertIn("already been applied", err)
        self.assertEqual(self.eft("feedback", "--apply", "99")[0], 1)

    def test_a_model_failure_still_records_the_rejection(self):
        self.crusoe.answers["parse"] = CrusoeError("no usable answer for the parse role")
        code, _, err = self.eft("feedback", self.wrong["code"], "--reason", "wrong shoe")
        self.assertEqual(code, 1)
        self.assertIn("marked as not a match", err)
        self.assertEqual(self.store.get_candidate_by_id(self.wrong["id"])["state"], "rejected")
        self.assertIsNone(self.store.list_feedback()[0]["spec_patch"])

    def test_an_alert_still_waiting_for_a_rejected_listing_is_dropped(self):
        waiting = self.store.enqueue_alert("new:%d" % self.wrong["id"], "new", "P3", {"code": self.wrong["code"]},
                                           watch_id=self.watch["id"], candidate_id=self.wrong["id"], status="held")
        self.eft("feedback", self.wrong["code"], "--reason", "wrong shoe")
        self.assertEqual(self.store.get_alert(waiting)["status"], "dropped")

    def test_feedback_needs_a_code_and_a_reason_or_an_id_to_apply(self):
        for argv in (["feedback"], ["feedback", "A1"], ["feedback", "--reason", "why"],
                     ["feedback", "A1", "--apply", "1"], ["feedback", "--apply", "one"]):
            self.assertEqual(self.eft(*argv)[0], 2, argv)


class Agent(CliTest):
    def setUp(self):
        super().setUp()
        self.plist = os.path.join(self.home, "Library", "LaunchAgents", "com.crusoe.eft.plist")

    def test_install_renders_the_plist_with_absolute_paths_and_loads_it(self):
        code, data = self.eft_json("agent", "install", "--interval", "10")
        self.assertEqual((code, data["plist"]), (0, self.plist))
        with open(self.plist, "rb") as handle:
            job = plistlib.load(handle)
        python, launcher, command = job["ProgramArguments"]
        self.assertEqual((python, command), (os.path.abspath(sys.executable), "tick"))
        self.assertEqual(launcher, LAUNCHER)
        self.assertTrue(all(os.path.isabs(path) for path in (python, launcher)))
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertEqual((job["Label"], job["StartInterval"], job["RunAtLoad"]), ("com.crusoe.eft", 600, True))
        self.assertEqual(job["EnvironmentVariables"], {"EFT_STATE_DIR": str(self.settings.state_dir),
                                                       "EFT_ENV_FILE": str(self.settings.env_file)})
        self.assertEqual(job["StandardErrorPath"], os.path.join(str(self.settings.state_dir), "tick.log"))
        with open(self.plist) as handle:
            text = handle.read()
        self.assertNotIn("{{", text)
        for secret in SECRET_VALUES:
            self.assertNotIn(secret, text)
        domain = "gui/%d" % os.getuid()
        self.assertEqual(self.commands, [["/bin/launchctl", "bootout", domain + "/com.crusoe.eft"],
                                         ["/bin/launchctl", "bootstrap", domain, self.plist]])

    def test_a_path_with_markup_characters_is_escaped(self):
        odd = support.make_settings(EFT_STATE_DIR=os.path.join(self.home, "R&D <state>"))
        job = plistlib.loads(cli.render_plist(odd, 5).encode())
        self.assertEqual(job["EnvironmentVariables"]["EFT_STATE_DIR"], os.path.join(self.home, "R&D <state>"))
        self.assertEqual(job["StartInterval"], 300)

    def test_install_says_so_when_launchctl_will_not_load_the_agent(self):
        self.launchctl_exit = 5
        code, _, err = self.eft("agent", "install")
        self.assertEqual(code, 1)
        self.assertIn("launchctl bootstrap gui/", err)
        self.assertTrue(os.path.exists(self.plist))

    def test_status_and_uninstall(self):
        self.assertEqual(self.eft_json("agent", "status")[1]["installed"], False)
        self.eft("agent", "install")
        code, data = self.eft_json("agent", "status")
        self.assertEqual((code, data["installed"], data["loaded"]), (0, True, True))
        code, data = self.eft_json("agent", "uninstall")
        self.assertEqual((code, data), (0, {"removed": True}))
        self.assertFalse(os.path.exists(self.plist))
        self.assertEqual(self.commands[-1][:2], ["/bin/launchctl", "bootout"])
        self.assertEqual(self.eft_json("agent", "uninstall")[1], {"removed": False})

    def test_other_platforms_get_a_cron_line_and_nothing_is_installed(self):
        self.rt = self.runtime(platform="linux")
        code, data = self.eft_json("agent", "install", "--interval", "15")
        self.assertEqual((code, data["installed"]), (0, False))
        self.assertTrue(data["cron"].startswith("*/15 * * * * EFT_STATE_DIR="))
        self.assertIn("%s %s tick" % (os.path.abspath(sys.executable), LAUNCHER), data["cron"])
        self.assertEqual(self.commands, [])
        self.assertFalse(os.path.exists(self.plist))
        self.assertIsNone(self.eft_json("agent", "status")[1]["installed"])

    def test_interval_must_be_a_positive_number_of_minutes(self):
        self.assertEqual(self.eft("agent", "install", "--interval", "0")[0], 2)
        self.assertEqual(self.commands, [])


class Uninstall(CliTest):
    def setUp(self):
        super().setUp()
        # A store on disk, opened by the runtime itself, as in real use.
        self.rt = self.runtime(store=None, notifier=None)
        self.rt.store.create_watch("Panda Dunks 10", poller.dunk_spec())
        self.db = str(self.settings.db_path)

    def test_uninstall_asks_for_yes_and_deletes_nothing_without_it(self):
        code, _, err = self.eft("uninstall")
        self.assertEqual(code, 1)
        self.assertIn("--yes", err)
        self.assertTrue(os.path.exists(self.db))
        self.assertEqual(len(self.rt.store.list_watches()), 1)

    def test_uninstall_unloads_the_agent_wipes_the_store_and_removes_the_state_directory(self):
        self.eft("agent", "install")
        with open(os.path.join(str(self.settings.refs_dir), "0123456789ab.jpg"), "wb") as handle:
            handle.write(b"photo")
        code, data = self.eft_json("uninstall", "--yes")
        self.assertEqual(code, 0)
        self.assertEqual((data["agent_removed"], data["store_wiped"], data["state_dir_removed"]), (True, True, True))
        self.assertFalse(os.path.exists(str(self.settings.state_dir)))
        self.assertEqual(self.commands[-1][:2], ["/bin/launchctl", "bootout"])
        self.assertTrue(any("revoke" in step for step in data["manual_steps"]))
        fresh = Store(self.db)
        self.addCleanup(fresh.close)
        self.assertEqual(fresh.list_watches(), [])

    def test_a_state_directory_with_someone_elses_files_is_emptied_of_ours_but_kept(self):
        other = os.path.join(str(self.settings.state_dir), "notes.txt")
        with open(other, "w") as handle:
            handle.write("not ours")
        code, data = self.eft_json("uninstall", "--yes")
        self.assertEqual((code, data["state_dir_removed"]), (0, False))
        self.assertEqual(os.listdir(str(self.settings.state_dir)), ["notes.txt"])

    def test_a_refs_folder_with_someone_elses_files_loses_only_the_copies_this_tool_made(self):
        refs = str(self.settings.refs_dir)
        for name in ("0123456789ab.jpg", "family.jpg"):
            with open(os.path.join(refs, name), "wb") as handle:
                handle.write(b"photo")
        os.mkdir(os.path.join(refs, "abcdef012345"))
        code, data = self.eft_json("uninstall", "--yes")
        self.assertEqual((code, data["state_dir_removed"]), (0, False))
        self.assertEqual(sorted(os.listdir(refs)), ["abcdef012345", "family.jpg"])

    def test_the_database_and_the_state_directory_are_private_to_the_user(self):
        self.assertEqual(os.stat(self.db).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(str(self.settings.state_dir)).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(str(self.settings.refs_dir)).st_mode & 0o777, 0o700)

    def test_an_existing_database_that_others_can_read_is_closed_to_them(self):
        self.rt.close()
        os.chmod(self.db, 0o644)
        reopened = Store(self.db)
        self.addCleanup(reopened.close)
        self.assertEqual(os.stat(self.db).st_mode & 0o777, 0o600)


class ExitCodes(CliTest):
    def test_success_is_0(self):
        self.assertEqual(self.eft("status")[0], 0)
        self.assertEqual(self.eft("watch", "list"), (0, "no watches yet\n", ""))

    def test_a_handled_problem_is_1_with_one_line_on_stderr_and_json_on_request(self):
        code, out, err = self.eft("watch", "show", "42")
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(err.count("\n"), 1)
        self.assertTrue(err.startswith("eft: no watch matches"))
        code, out, _ = self.eft("--json", "watch", "show", "42")
        self.assertEqual((code, json.loads(out)["ok"]), (1, False))

    def test_a_configuration_or_api_error_is_1_and_never_a_traceback(self):
        self.settings = support.make_settings(CRUSOE_API_KEY="")
        real = CrusoeClient(self.settings, self.store, self.wire.transport)
        self.rt = self.runtime(crusoe=real)
        code, _, err = self.eft("add", "--text", "Nike Dunk Low")
        self.assertEqual(code, 1)
        self.assertIn("set CRUSOE_API_KEY", err)
        self.crusoe.answers["parse"] = CrusoeError("no usable answer for the parse role (HTTP 503)")
        self.rt = self.runtime()
        code, _, err = self.eft("add", "--text", "Nike Dunk Low")
        self.assertEqual((code, err), (1, "eft: no usable answer for the parse role (HTTP 503)\n"))

    def test_a_usage_error_is_2(self):
        for argv in ([], ["frobnicate"], ["watch"], ["watch", "set", "1"], ["doctor", "everything"],
                     ["dryrun"], ["star"]):
            self.assertEqual(self.eft(*argv)[0], 2, argv)

    def test_json_flag_works_before_and_after_the_command(self):
        for argv in (["--json", "watch", "list"], ["watch", "list", "--json"]):
            code, out, _ = self.eft(*argv)
            self.assertEqual((code, json.loads(out)), (0, {"watches": []}))

    def test_help_exits_0(self):
        code, out, _ = self.eft("--help")
        self.assertEqual(code, 0)
        self.assertIn("never bids", out)


class Launcher(unittest.TestCase):
    """The real `scripts/eft`, in a subprocess, with no keys and an empty state directory."""

    def launch(self, *arguments):
        scratch = tempfile.mkdtemp(prefix="eft-launch-")
        environ = {"HOME": scratch, "PATH": os.environ.get("PATH", ""),
                   "EFT_STATE_DIR": os.path.join(scratch, "state"), "EFT_ENV_FILE": os.path.join(scratch, "no.env")}
        done = subprocess.run([sys.executable, LAUNCHER] + list(arguments), env=environ, timeout=60,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertNotIn("Traceback", done.stderr + done.stdout)
        return done

    def test_status_starts_cleanly_without_credentials(self):
        done = self.launch("--json", "status")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        data = json.loads(done.stdout)
        self.assertEqual(data["watches"], [])
        self.assertFalse(any(feature["configured"] for feature in data["config"]["features"].values()))
        self.assertIn("Set CRUSOE_API_KEY, then run: eft doctor crusoe", data["waiting"])

    def test_doctor_env_reports_the_missing_keys_by_name(self):
        done = self.launch("doctor", "env")
        self.assertEqual(done.returncode, 1, "a failed check is a handled problem")
        self.assertIn("not set: CRUSOE_API_KEY", done.stdout)
        self.assertIn("not set: EBAY_CLIENT_ID, EBAY_CLIENT_SECRET", done.stdout)
        self.assertIn("state directory", done.stdout)

    def test_a_poll_with_nothing_to_do_runs_cleanly(self):
        done = self.launch("--json", "tick")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        report = json.loads(done.stdout)
        self.assertEqual((report["searched"], report["problems"], report["heartbeat"]), (0, [], None))

    def test_the_launcher_is_executable(self):
        self.assertTrue(os.access(LAUNCHER, os.X_OK))
        with open(LAUNCHER) as handle:
            self.assertTrue(handle.readline().startswith("#!/usr/bin/env python3"))


if __name__ == "__main__":
    unittest.main()
