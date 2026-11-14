import os
import stat
import tempfile
import unittest
from datetime import timedelta

import support
from eft_core.store import Store, code_for, iso, parse_iso, strip_evidence

SPEC = {"brand": "Nike", "model": "Dunk Low"}


class TimeHelpers(unittest.TestCase):
    def test_parse_iso_accepts_ebay_forms(self):
        self.assertEqual(iso(parse_iso("2026-11-12T22:02:00.000Z")), "2026-11-12T22:02:00Z")
        self.assertEqual(iso(parse_iso("2026-11-12T22:02:00Z")), "2026-11-12T22:02:00Z")
        self.assertEqual(iso(parse_iso("2026-11-12T14:02:00-08:00")), "2026-11-12T22:02:00Z")
        self.assertIsNone(parse_iso(""))
        self.assertIsNone(parse_iso("not a date"))


class Watches(unittest.TestCase):
    def setUp(self):
        self.clock = support.Clock()
        self.store = support.make_store(self.clock)

    def test_create_get_update_find_delete(self):
        watch_id = self.store.create_watch("Panda Dunks 10", SPEC)
        watch = self.store.get_watch(watch_id)
        self.assertEqual(watch["status"], "draft")
        self.assertEqual(watch["spec"], SPEC)
        self.store.update_watch(watch_id, status="active", plan={"q": "nike dunk low"})
        watch = self.store.get_watch(watch_id)
        self.assertEqual(watch["status"], "active")
        self.assertEqual(watch["plan"], {"q": "nike dunk low"})
        self.assertEqual(self.store.find_watch("panda")["id"], watch_id)
        self.assertEqual(self.store.find_watch(str(watch_id))["id"], watch_id)
        self.assertIsNone(self.store.find_watch("jordan"))
        self.assertEqual(len(self.store.list_watches("active")), 1)
        with self.assertRaises(ValueError):
            self.store.update_watch(watch_id, nonsense=1)
        self.store.delete_watch(watch_id)
        self.assertIsNone(self.store.get_watch(watch_id))


class Candidates(unittest.TestCase):
    def setUp(self):
        self.clock = support.Clock()
        self.store = support.make_store(self.clock)
        self.watch_id = self.store.create_watch("w", SPEC)

    def test_upsert_assigns_a_code_once_and_updates_in_place(self):
        first = self.store.upsert_candidate(self.watch_id, "v1|1|0", "pending")
        again = self.store.upsert_candidate(self.watch_id, "v1|1|0", "rejected", reason="mismatch:size")
        other = self.store.upsert_candidate(self.watch_id, "v1|2|0", "pending")
        self.assertEqual(first["code"], "A1")
        self.assertEqual(again["code"], "A1")
        self.assertEqual(again["state"], "rejected")
        self.assertEqual(again["reason"], "mismatch:size")
        self.assertEqual(other["code"], "A2")
        self.assertEqual(self.store.seen_item_ids(self.watch_id), {"v1|1|0", "v1|2|0"})
        self.assertEqual(self.store.get_candidate_by_code("a2")["item_id"], "v1|2|0")

    def test_codes_never_repeat_however_many_candidates_there_have_been(self):
        self.assertEqual([code_for(n) for n in (1, 99, 100, 2574, 2575, 2673, 2674, 5149)],
                         ["A1", "A99", "B1", "Z99", "A100", "A198", "B100", "A199"])
        self.assertEqual(len({code_for(n) for n in range(1, 20001)}), 20000)
        tracked = self.store.upsert_candidate(self.watch_id, "v1|1|0", "alerted", tracked=True)
        # 2,574 more listings later (every judged listing gets a row, rejects included)...
        self.store.kv_set("code_counter", 2574)
        later = self.store.upsert_candidate(self.watch_id, "v1|2|0", "rejected")
        self.assertEqual((tracked["code"], later["code"]), ("A1", "A100"))
        self.assertEqual(self.store.get_candidate_by_code("A1")["item_id"], "v1|1|0")

    def test_a_code_a_stored_candidate_still_holds_is_skipped(self):
        first = self.store.upsert_candidate(self.watch_id, "v1|1|0", "alerted", tracked=True)
        second = self.store.upsert_candidate(self.watch_id, "v1|2|0", "alerted", tracked=True)
        self.store.kv_delete("code_counter")     # a store whose counter was lost
        third = self.store.upsert_candidate(self.watch_id, "v1|3|0", "rejected")
        self.assertEqual([row["code"] for row in (first, second, third)], ["A1", "A2", "A3"])
        self.assertEqual(self.store.upsert_candidate(self.watch_id, "v1|4|0", "rejected")["code"], "A4")

    def test_lookup_by_code_prefers_the_tracked_row(self):
        # A store written when codes started again at A1 can hold one code twice.
        tracked = self.store.upsert_candidate(self.watch_id, "v1|1|0", "alerted", tracked=True)
        newer = self.store.upsert_candidate(self.watch_id, "v1|2|0", "rejected")
        self.store.conn.execute("UPDATE candidate SET code = ? WHERE id = ?", (tracked["code"], newer["id"]))
        self.assertEqual(self.store.get_candidate_by_code("a1")["item_id"], "v1|1|0")
        self.store.update_candidate(tracked["id"], tracked=False)
        self.assertEqual(self.store.get_candidate_by_code("A1")["item_id"], "v1|2|0", "else the newest row")

    def test_rejects_unknown_state_and_fields(self):
        with self.assertRaises(ValueError):
            self.store.upsert_candidate(self.watch_id, "x", "bogus")
        with self.assertRaises(ValueError):
            self.store.upsert_candidate(self.watch_id, "x", "pending", seller="someone")

    def test_due_for_refresh_orders_by_end_time(self):
        now = self.clock()
        late = self.store.upsert_candidate(
            self.watch_id, "late", "alerted", tracked=True,
            next_refresh_at=iso(now - timedelta(minutes=1)), end_at=iso(now + timedelta(hours=5)))
        soon = self.store.upsert_candidate(
            self.watch_id, "soon", "alerted", tracked=True,
            next_refresh_at=iso(now - timedelta(minutes=1)), end_at=iso(now + timedelta(minutes=30)))
        self.store.upsert_candidate(
            self.watch_id, "notyet", "alerted", tracked=True, next_refresh_at=iso(now + timedelta(hours=1)))
        due = [c["item_id"] for c in self.store.due_for_refresh()]
        self.assertEqual(due, ["soon", "late"])
        self.assertTrue(late["tracked"] and soon["tracked"])

    def test_purge_listing_data_removes_content_and_evidence(self):
        candidate = self.store.upsert_candidate(
            self.watch_id, "v1|9|0", "alerted", tracked=True,
            listing={"title": "Nike Dunk Low Panda", "price": {"value": "120.00", "currency": "USD"}},
            fetched_at=self.store.now_iso())
        self.store.add_verdict(candidate["id"], "detail_verify",
                               fields={"size": {"verdict": "match", "evidence": "US Shoe Size: 10"}})
        self.store.purge_listing_data(candidate["id"], state="ended")
        stored = self.store.get_candidate_by_id(candidate["id"])
        self.assertIsNone(stored["listing"])
        self.assertEqual(stored["tracked"], 0)
        self.assertEqual(stored["state"], "ended")
        verdict = self.store.verdicts_for(candidate["id"])[0]
        self.assertEqual(verdict["fields"], {"size": {"verdict": "match"}})

    def test_purge_listing_data_empties_alert_payloads(self):
        candidate = self.store.upsert_candidate(self.watch_id, "v1|8|0", "alerted", tracked=True)
        alert_id = self.store.enqueue_alert(
            "new:1:8", "new", "P2",
            {"code": candidate["code"], "watch_id": self.watch_id, "watch_name": "w", "kind": "new", "tier": "P2",
             "title": "Nike Dunk Low Panda", "price": {"value": "120.00", "currency": "USD"},
             "web_url": "https://www.ebay.com/itm/8"},
            watch_id=self.watch_id, candidate_id=candidate["id"])
        self.store.purge_listing_data(candidate["id"], state="ended")
        payload = self.store.get_alert(alert_id)["payload"]
        self.assertEqual(payload, {"code": candidate["code"], "watch_id": self.watch_id, "watch_name": "w",
                                   "kind": "new", "tier": "P2"})

    def test_alerts_and_messages_follow_their_watch_and_candidate(self):
        candidate = self.store.upsert_candidate(self.watch_id, "v1|7|0", "alerted")
        by_candidate = self.store.enqueue_alert("a", "new", "P2", {}, watch_id=self.watch_id,
                                                candidate_id=candidate["id"])
        self.store.record_message(by_candidate, "slack", {"channel": "D1", "ts": "1.0"}, candidate_id=candidate["id"])
        system = self.store.enqueue_alert("system:x", "system", "P1", {})
        self.store.delete_watch(self.watch_id)
        self.assertIsNone(self.store.get_alert(by_candidate))
        self.assertEqual(self.store.messages_for_candidate(candidate["id"]), [])
        self.assertIsNotNone(self.store.get_alert(system), "system notices belong to no watch and stay")

    def test_purge_expired_deletes_rows_and_their_trace(self):
        now = self.clock()
        old = self.store.upsert_candidate(self.watch_id, "old", "rejected", purge_at=iso(now - timedelta(days=1)))
        self.store.add_verdict(old["id"], "title_triage", fields={"verdict": "no"})
        self.store.upsert_candidate(self.watch_id, "keep", "rejected", purge_at=iso(now + timedelta(days=1)))
        self.assertEqual(self.store.purge_expired(), 1)
        self.assertEqual(self.store.seen_item_ids(self.watch_id), {"keep"})
        self.assertEqual(self.store.verdicts_for(old["id"]), [])

    def test_purge_expired_waits_for_a_message_that_still_shows_listing_data(self):
        now = self.clock()
        shown = self.store.upsert_candidate(self.watch_id, "shown", "rejected", purge_at=iso(now - timedelta(hours=1)))
        alert_id = self.store.enqueue_alert("new:shown", "new", "P2", {}, watch_id=self.watch_id,
                                            candidate_id=shown["id"])
        message_id = self.store.record_message(alert_id, "slack", {"channel": "D1", "ts": "1.0"},
                                               candidate_id=shown["id"], as_of=iso(now - timedelta(hours=2)))
        self.assertEqual(self.store.purge_expired(), 0)
        self.assertEqual(len(self.store.messages_for_candidate(shown["id"])), 1,
                         "deleting the row would lose track of a card that still shows a title and a price")
        self.store.mark_message_expired(message_id)
        self.assertEqual(self.store.purge_expired(), 1)
        self.assertEqual(self.store.messages_for_candidate(shown["id"]), [])

    def test_deleting_a_watch_removes_its_candidates(self):
        self.store.upsert_candidate(self.watch_id, "a", "pending")
        self.store.delete_watch(self.watch_id)
        self.assertEqual(self.store.list_candidates(), [])


class DatabaseFile(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.umask, os.umask(0o022))    # the usual default, whatever this shell has

    @staticmethod
    def mode(path):
        return stat.S_IMODE(os.stat(path).st_mode)

    def test_the_database_and_its_folder_are_for_the_owner_only(self):
        folder = os.path.join(tempfile.mkdtemp(prefix="eft-db-"), "state")
        path = os.path.join(folder, "tracker.db")
        store = Store(path)
        self.addCleanup(store.close)
        store.kv_set("ebay:token", {"access_token": "secret"})
        self.assertEqual(self.mode(folder), 0o700)
        written = [path + suffix for suffix in ("", "-wal", "-shm") if os.path.exists(path + suffix)]
        self.assertIn(path, written)
        for name in written:
            self.assertEqual(self.mode(name), 0o600, name)

    def test_a_database_left_readable_by_others_is_tightened_when_opened(self):
        path = os.path.join(tempfile.mkdtemp(prefix="eft-db-"), "tracker.db")
        Store(path).close()
        os.chmod(path, 0o644)
        store = Store(path)
        self.addCleanup(store.close)
        self.assertEqual(self.mode(path), 0o600)


class AlertsAndMessages(unittest.TestCase):
    def setUp(self):
        self.clock = support.Clock()
        self.store = support.make_store(self.clock)

    def test_duplicate_alert_is_refused(self):
        first = self.store.enqueue_alert("new:1:123", "new", "P2", {"code": "A1"})
        second = self.store.enqueue_alert("new:1:123", "new", "P2", {"code": "A1"})
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(self.store.alerts_by_status("pending")), 1)

    def test_status_updates(self):
        alert_id = self.store.enqueue_alert("k", "new", "P2", {"a": 1}, status="held")
        self.store.update_alert(alert_id, status="sent", add_attempt=True)
        alert = self.store.get_alert(alert_id)
        self.assertEqual(alert["status"], "sent")
        self.assertEqual(alert["attempts"], 1)
        self.assertIsNotNone(alert["sent_at"])
        with self.assertRaises(ValueError):
            self.store.update_alert(alert_id, status="nope")

    def test_message_expiry_window(self):
        alert_id = self.store.enqueue_alert("k", "new", "P2", {})
        now = self.clock()
        old = self.store.record_message(alert_id, "slack", {"channel": "D1", "ts": "1.1"},
                                        as_of=iso(now - timedelta(hours=7)))
        self.store.record_message(alert_id, "slack", {"channel": "D1", "ts": "1.2"},
                                  as_of=iso(now - timedelta(hours=1)))
        self.store.record_message(alert_id, "slack", {"channel": "D1", "ts": "1.3"},
                                  has_listing_data=False, as_of=iso(now - timedelta(hours=9)))
        cutoff = iso(now - timedelta(hours=6))
        due = self.store.messages_due_for_expiry(cutoff)
        self.assertEqual([m["id"] for m in due], [old])
        self.store.mark_message_expired(old)
        self.assertEqual(self.store.messages_due_for_expiry(cutoff), [])
        self.store.refresh_message(old, iso(now))
        self.assertEqual(self.store.messages_due_for_expiry(iso(now + timedelta(hours=7))).__len__(), 2)


class LedgerKvAndTransactions(unittest.TestCase):
    def setUp(self):
        self.clock = support.Clock()
        self.store = support.make_store(self.clock)

    def test_ledger_counts_per_day_and_purpose(self):
        self.store.ledger_add("ebay", "search")
        self.store.ledger_add("ebay", "search", calls=2)
        self.store.ledger_add("ebay", "detail")
        self.store.ledger_add("crusoe", "title_triage")
        self.assertEqual(self.store.ledger_total("ebay"), 4)
        self.assertEqual(self.store.ledger_breakdown("ebay"), {"detail": 1, "search": 3})
        self.clock.advance(days=1)
        self.assertEqual(self.store.ledger_total("ebay"), 0)

    def test_kv_round_trip(self):
        self.assertEqual(self.store.kv_get("missing", "fallback"), "fallback")
        self.store.kv_set("token", {"value": "abc", "expires": 123})
        self.assertEqual(self.store.kv_get("token")["expires"], 123)
        self.store.kv_delete("token")
        self.assertIsNone(self.store.kv_get("token"))

    def test_transaction_rolls_back_everything_on_error(self):
        watch_id = self.store.create_watch("w", SPEC)
        with self.assertRaises(RuntimeError):
            with self.store.transaction():
                self.store.enqueue_alert("k", "new", "P2", {})
                self.store.update_watch(watch_id, window_end="2026-11-12T18:00:00Z")
                raise RuntimeError("crash mid-tick")
        self.assertEqual(self.store.alerts_by_status(), [])
        self.assertIsNone(self.store.get_watch(watch_id)["window_end"])

    def test_wipe_removes_everything(self):
        watch_id = self.store.create_watch("w", SPEC)
        self.store.upsert_candidate(watch_id, "a", "pending")
        self.store.kv_set("x", 1)
        self.store.wipe()
        self.assertEqual(self.store.list_watches(), [])
        self.assertEqual(self.store.list_candidates(), [])
        self.assertIsNone(self.store.kv_get("x"))

    def test_strip_evidence(self):
        fields = {"size": {"verdict": "match", "evidence": "quote"}, "note": "plain"}
        self.assertEqual(strip_evidence(fields), {"size": {"verdict": "match"}, "note": "plain"})


if __name__ == "__main__":
    unittest.main()
