"""SQLite state: watches, candidates, the funnel trace, the alert outbox, and the call ledger.

Data rules this module enforces (they come from eBay's API License Agreement):
- A candidate keeps listing content (title, price, image URL) only while it is tracked.
- Rejected and merely-seen listings are an id, a state and a reason code. Nothing else.
- When a listing ends, its content and any quoted evidence are removed.
- No seller identifier is ever stored.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Union

SCHEMA_VERSION = 1

CANDIDATE_STATES = ("pending", "parked", "maybe", "alerted", "rejected", "unavailable", "ended")
# Candidate codes run A1 to A99, then B1 to B99, and so on through the alphabet.
CODES_PER_LETTER = 99
ALERT_STATUSES = ("pending", "held", "sent", "failed", "dropped")

SCHEMA = """
CREATE TABLE IF NOT EXISTS watch (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'model',
  status TEXT NOT NULL DEFAULT 'draft',
  tier TEXT NOT NULL DEFAULT 'normal',
  spec TEXT NOT NULL,
  plan TEXT,
  pinned_item_id TEXT,
  window_end TEXT,
  last_search_at TEXT,
  last_sweep_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate (
  id INTEGER PRIMARY KEY,
  watch_id INTEGER NOT NULL REFERENCES watch(id) ON DELETE CASCADE,
  item_id TEXT NOT NULL,
  legacy_item_id TEXT,
  state TEXT NOT NULL,
  reason TEXT,
  code TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  tracked INTEGER NOT NULL DEFAULT 0,
  starred INTEGER NOT NULL DEFAULT 0,
  listing TEXT,
  end_at TEXT,
  fetched_at TEXT,
  next_refresh_at TEXT,
  purge_at TEXT,
  first_seen_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (watch_id, item_id)
);
CREATE INDEX IF NOT EXISTS candidate_code ON candidate(code);
CREATE TABLE IF NOT EXISTS verdict (
  id INTEGER PRIMARY KEY,
  candidate_id INTEGER NOT NULL REFERENCES candidate(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  model_served TEXT,
  fields TEXT,
  rule_fired TEXT,
  latency_ms INTEGER,
  tokens INTEGER,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alert (
  id INTEGER PRIMARY KEY,
  dedupe_key TEXT NOT NULL UNIQUE,
  watch_id INTEGER REFERENCES watch(id) ON DELETE CASCADE,
  candidate_id INTEGER REFERENCES candidate(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  tier TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  payload TEXT NOT NULL,
  as_of TEXT,
  created_at TEXT NOT NULL,
  sent_at TEXT
);
CREATE TABLE IF NOT EXISTS message (
  id INTEGER PRIMARY KEY,
  alert_id INTEGER NOT NULL REFERENCES alert(id) ON DELETE CASCADE,
  candidate_id INTEGER,
  channel TEXT NOT NULL,
  ref TEXT NOT NULL,
  has_listing_data INTEGER NOT NULL DEFAULT 1,
  as_of TEXT,
  expired_at TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY,
  watch_id INTEGER,
  candidate_code TEXT,
  label TEXT NOT NULL,
  reason TEXT,
  spec_patch TEXT,
  approved INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
  day TEXT NOT NULL,
  api TEXT NOT NULL,
  purpose TEXT NOT NULL,
  calls INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, api, purpose)
);
CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""

_JSON_COLUMNS = {"spec", "plan", "listing", "fields", "payload", "ref", "spec_patch"}
_WATCH_FIELDS = {
    "name", "kind", "status", "tier", "spec", "plan", "pinned_item_id",
    "window_end", "last_search_at", "last_sweep_at",
}
_CANDIDATE_FIELDS = {
    "legacy_item_id", "state", "reason", "attempts", "tracked", "starred", "listing",
    "end_at", "fetched_at", "next_refresh_at", "purge_at",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: Optional[str]) -> Optional[datetime]:
    """Parse the ISO 8601 forms eBay and this store use. Returns an aware UTC datetime."""
    if not text:
        return None
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def code_for(counter: int) -> str:
    """The short code of the nth candidate: A1..A99, B1..Z99, then A100..A198, B100.. and so on.

    After Z99 the numbers carry on instead of starting again at A1, so however many
    candidates a store has seen, no two of them share a code.
    """
    block, offset = divmod(counter - 1, CODES_PER_LETTER)
    lap, letter = divmod(block, 26)
    return "%s%d" % (chr(ord("A") + letter), lap * CODES_PER_LETTER + offset + 1)


def _keep_private(db_path: str) -> None:
    """Make sure the database file exists and that only its owner can read it.

    The file holds listing snapshots and the cached eBay application token. It is created
    here, before SQLite opens it, because SQLite gives the -wal and -shm files it writes
    beside a database the mode of the database itself. Files left by an earlier run are
    tightened too. A filesystem that has no permissions to set is left as it is.
    """
    os.close(os.open(db_path, os.O_CREAT | os.O_RDWR, 0o600))
    for suffix in ("", "-wal", "-shm"):
        try:
            os.chmod(db_path + suffix, 0o600)
        except OSError:
            pass


def _encode(column: str, value: Any) -> Any:
    if column in _JSON_COLUMNS and value is not None:
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, datetime):
        return iso(value)
    return value


def _decode(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    out: Dict[str, Any] = {}
    for column in row.keys():
        value = row[column]
        if column in _JSON_COLUMNS and value is not None:
            value = json.loads(value)
        out[column] = value
    return out


class Store:
    """All persistent state. Pass `clock` in tests to control time."""

    def __init__(self, db_path: Union[str, Path], clock: Callable[[], datetime] = utcnow) -> None:
        self.clock = clock
        self._depth = 0
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with suppress(OSError):
                _keep_private(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        try:
            self.conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA user_version = %d" % SCHEMA_VERSION)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def now_iso(self) -> str:
        return iso(self.clock())

    # ---- transactions ------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator["Store"]:
        """Group several writes so they commit together or not at all."""
        self._depth += 1
        try:
            yield self
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.rollback()
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self.conn.commit()

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        cursor = self.conn.execute(sql, params)
        if self._depth == 0:
            self.conn.commit()
        return cursor

    def _one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        return _decode(self.conn.execute(sql, params).fetchone())

    def _all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        return [_decode(row) for row in self.conn.execute(sql, params).fetchall()]  # type: ignore[misc]

    def _update(self, table: str, row_id: int, allowed: Set[str], fields: Dict[str, Any]) -> None:
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError("unknown %s fields: %s" % (table, ", ".join(sorted(unknown))))
        if not fields:
            return
        columns = sorted(fields)
        assignments = ", ".join("%s = ?" % c for c in columns) + ", updated_at = ?"
        params = [_encode(c, fields[c]) for c in columns] + [self.now_iso(), row_id]
        self._exec("UPDATE %s SET %s WHERE id = ?" % (table, assignments), params)

    # ---- watches -----------------------------------------------------------

    def create_watch(
        self,
        name: str,
        spec: Dict[str, Any],
        kind: str = "model",
        status: str = "draft",
        tier: str = "normal",
        pinned_item_id: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = self.now_iso()
        cursor = self._exec(
            "INSERT INTO watch (name, kind, status, tier, spec, plan, pinned_item_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, kind, status, tier, _encode("spec", spec), _encode("plan", plan), pinned_item_id, now, now),
        )
        return int(cursor.lastrowid)

    def get_watch(self, watch_id: int) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM watch WHERE id = ?", (watch_id,))

    def find_watch(self, ref: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Look a watch up by id, exact name, or unique name fragment (case-insensitive)."""
        text = str(ref).strip()
        if text.isdigit():
            found = self.get_watch(int(text))
            if found:
                return found
        exact = self._all("SELECT * FROM watch WHERE lower(name) = lower(?)", (text,))
        if len(exact) == 1:
            return exact[0]
        partial = self._all("SELECT * FROM watch WHERE lower(name) LIKE '%' || lower(?) || '%'", (text,))
        return partial[0] if len(partial) == 1 else None

    def list_watches(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            return self._all("SELECT * FROM watch WHERE status = ? ORDER BY id", (status,))
        return self._all("SELECT * FROM watch ORDER BY id")

    def update_watch(self, watch_id: int, **fields: Any) -> None:
        self._update("watch", watch_id, _WATCH_FIELDS, fields)

    def delete_watch(self, watch_id: int) -> None:
        self._exec("DELETE FROM watch WHERE id = ?", (watch_id,))

    # ---- candidates --------------------------------------------------------

    def _next_code(self) -> str:
        """Allocate the code for a new candidate: one that no stored candidate holds.

        A code is what the user reads off a card and types back (`eft star A17`), so it must
        never come to mean a second listing. The counter only moves forward, and a code that
        is still held (in a store whose counter was lost or reset) is skipped.
        """
        counter = int(self.kv_get("code_counter", 0))
        while True:
            counter += 1
            code = code_for(counter)
            if self.conn.execute("SELECT 1 FROM candidate WHERE code = ? LIMIT 1", (code,)).fetchone() is None:
                break
        self.kv_set("code_counter", counter)
        return code

    def seen_item_ids(self, watch_id: int) -> Set[str]:
        rows = self.conn.execute("SELECT item_id FROM candidate WHERE watch_id = ?", (watch_id,)).fetchall()
        return {row["item_id"] for row in rows}

    def get_candidate(self, watch_id: int, item_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM candidate WHERE watch_id = ? AND item_id = ?", (watch_id, item_id))

    def get_candidate_by_id(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM candidate WHERE id = ?", (candidate_id,))

    def get_candidate_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Look a candidate up by the code on its card.

        Codes are unique as they are handed out now. A store written before that was so can
        hold the same code twice, and then the tracked row (the one with a card) is the one meant.
        """
        return self._one(
            "SELECT * FROM candidate WHERE upper(code) = upper(?) ORDER BY tracked DESC, id DESC LIMIT 1",
            (code.strip(),),
        )

    def upsert_candidate(self, watch_id: int, item_id: str, state: str, **fields: Any) -> Dict[str, Any]:
        """Insert a candidate or update the existing one. Returns the stored row."""
        if state not in CANDIDATE_STATES:
            raise ValueError("unknown candidate state: %s" % state)
        unknown = set(fields) - _CANDIDATE_FIELDS
        if unknown:
            raise ValueError("unknown candidate fields: %s" % ", ".join(sorted(unknown)))
        with self.transaction():
            existing = self.get_candidate(watch_id, item_id)
            if existing is None:
                now = self.now_iso()
                columns = ["watch_id", "item_id", "state", "code", "first_seen_at", "updated_at"]
                values: List[Any] = [watch_id, item_id, state, self._next_code(), now, now]
                for column in sorted(fields):
                    columns.append(column)
                    values.append(_encode(column, fields[column]))
                self._exec(
                    "INSERT INTO candidate (%s) VALUES (%s)" % (", ".join(columns), ", ".join("?" * len(columns))),
                    values,
                )
            else:
                update = dict(fields)
                update["state"] = state
                self._update("candidate", existing["id"], _CANDIDATE_FIELDS, update)
        stored = self.get_candidate(watch_id, item_id)
        assert stored is not None
        return stored

    def update_candidate(self, candidate_id: int, **fields: Any) -> None:
        if "state" in fields and fields["state"] not in CANDIDATE_STATES:
            raise ValueError("unknown candidate state: %s" % fields["state"])
        self._update("candidate", candidate_id, _CANDIDATE_FIELDS, fields)

    def list_candidates(
        self,
        watch_id: Optional[int] = None,
        states: Optional[Sequence[str]] = None,
        tracked: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []  # type: List[str], List[Any]
        if watch_id is not None:
            clauses.append("watch_id = ?")
            params.append(watch_id)
        if states:
            clauses.append("state IN (%s)" % ", ".join("?" * len(states)))
            params.extend(states)
        if tracked is not None:
            clauses.append("tracked = ?")
            params.append(int(tracked))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._all("SELECT * FROM candidate%s ORDER BY id" % where, params)

    def due_for_refresh(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Tracked candidates whose refresh time has come, soonest-ending first."""
        cutoff = iso(now or self.clock())
        return self._all(
            "SELECT * FROM candidate WHERE tracked = 1 AND next_refresh_at IS NOT NULL AND next_refresh_at <= ?"
            " ORDER BY (end_at IS NULL), end_at, id",
            (cutoff,),
        )

    def purge_listing_data(self, candidate_id: int, state: Optional[str] = None) -> None:
        """Remove listing content and quoted evidence. Called when a listing ends or is untracked."""
        with self.transaction():
            fields: Dict[str, Any] = {"listing": None, "tracked": 0, "next_refresh_at": None, "fetched_at": None}
            if state:
                fields["state"] = state
            self.update_candidate(candidate_id, **fields)
            for row in self.verdicts_for(candidate_id):
                stripped = strip_evidence(row.get("fields"))
                self._exec("UPDATE verdict SET fields = ? WHERE id = ?", (_encode("fields", stripped), row["id"]))
            for alert in self.alerts_for_candidate(candidate_id):
                kept = strip_payload(alert.get("payload"))
                self._exec("UPDATE alert SET payload = ? WHERE id = ?", (_encode("payload", kept), alert["id"]))

    def purge_expired(self, now: Optional[datetime] = None) -> int:
        """Delete candidates whose purge date has passed. Returns how many were removed.

        A candidate with a sent message that still shows listing data is kept for now.
        Deleting it would delete the record of where that message is, and the message could
        then never be rewritten. It goes on the first purge after its messages are stripped.
        """
        cutoff = iso(now or self.clock())
        cursor = self._exec(
            "DELETE FROM candidate WHERE purge_at IS NOT NULL AND purge_at <= ? AND id NOT IN ("
            " SELECT alert.candidate_id FROM message JOIN alert ON alert.id = message.alert_id"
            " WHERE message.has_listing_data = 1 AND alert.candidate_id IS NOT NULL)",
            (cutoff,),
        )
        return int(cursor.rowcount)

    # ---- funnel trace ------------------------------------------------------

    def add_verdict(
        self,
        candidate_id: int,
        stage: str,
        fields: Optional[Dict[str, Any]] = None,
        rule_fired: Optional[str] = None,
        model_served: Optional[str] = None,
        latency_ms: Optional[int] = None,
        tokens: Optional[int] = None,
    ) -> int:
        cursor = self._exec(
            "INSERT INTO verdict (candidate_id, stage, model_served, fields, rule_fired, latency_ms, tokens, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (candidate_id, stage, model_served, _encode("fields", fields), rule_fired, latency_ms, tokens, self.now_iso()),
        )
        return int(cursor.lastrowid)

    def verdicts_for(self, candidate_id: int) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM verdict WHERE candidate_id = ? ORDER BY id", (candidate_id,))

    # ---- alerts and messages -----------------------------------------------

    def enqueue_alert(
        self,
        dedupe_key: str,
        kind: str,
        tier: str,
        payload: Dict[str, Any],
        watch_id: Optional[int] = None,
        candidate_id: Optional[int] = None,
        as_of: Optional[str] = None,
        status: str = "pending",
    ) -> Optional[int]:
        """Queue an alert. Returns None when the same event was already queued."""
        if status not in ALERT_STATUSES:
            raise ValueError("unknown alert status: %s" % status)
        try:
            cursor = self._exec(
                "INSERT INTO alert (dedupe_key, watch_id, candidate_id, kind, tier, status, payload, as_of, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (dedupe_key, watch_id, candidate_id, kind, tier, status, _encode("payload", payload), as_of, self.now_iso()),
            )
        except sqlite3.IntegrityError:
            return None
        return int(cursor.lastrowid)

    def get_alert(self, alert_id: int) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM alert WHERE id = ?", (alert_id,))

    def alerts_for_candidate(self, candidate_id: int) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM alert WHERE candidate_id = ? ORDER BY id", (candidate_id,))

    def alerts_by_status(self, *statuses: str) -> List[Dict[str, Any]]:
        if not statuses:
            return self._all("SELECT * FROM alert ORDER BY id")
        marks = ", ".join("?" * len(statuses))
        return self._all("SELECT * FROM alert WHERE status IN (%s) ORDER BY id" % marks, statuses)

    def update_alert(
        self,
        alert_id: int,
        status: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        as_of: Optional[str] = None,
        add_attempt: bool = False,
    ) -> None:
        if status is not None and status not in ALERT_STATUSES:
            raise ValueError("unknown alert status: %s" % status)
        sets, params = [], []  # type: List[str], List[Any]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
            if status == "sent":
                sets.append("sent_at = ?")
                params.append(self.now_iso())
        if payload is not None:
            sets.append("payload = ?")
            params.append(_encode("payload", payload))
        if as_of is not None:
            sets.append("as_of = ?")
            params.append(as_of)
        if add_attempt:
            sets.append("attempts = attempts + 1")
        if not sets:
            return
        params.append(alert_id)
        self._exec("UPDATE alert SET %s WHERE id = ?" % ", ".join(sets), params)

    def record_message(
        self,
        alert_id: int,
        channel: str,
        ref: Dict[str, Any],
        candidate_id: Optional[int] = None,
        has_listing_data: bool = True,
        as_of: Optional[str] = None,
    ) -> int:
        """Remember where a message was posted so it can be expired or rewritten later."""
        cursor = self._exec(
            "INSERT INTO message (alert_id, candidate_id, channel, ref, has_listing_data, as_of, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (alert_id, candidate_id, channel, _encode("ref", ref), int(has_listing_data), as_of, self.now_iso()),
        )
        return int(cursor.lastrowid)

    def messages_due_for_expiry(self, cutoff_iso: str) -> List[Dict[str, Any]]:
        """Messages that still show listing data captured at or before the cutoff."""
        return self._all(
            "SELECT * FROM message WHERE has_listing_data = 1 AND expired_at IS NULL"
            " AND as_of IS NOT NULL AND as_of <= ? ORDER BY id",
            (cutoff_iso,),
        )

    def messages_for_candidate(self, candidate_id: int) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM message WHERE candidate_id = ? ORDER BY id", (candidate_id,))

    def mark_message_expired(self, message_id: int) -> None:
        self._exec(
            "UPDATE message SET expired_at = ?, has_listing_data = 0 WHERE id = ?", (self.now_iso(), message_id)
        )

    def refresh_message(self, message_id: int, as_of: str) -> None:
        """Put a message back among those that show listing data, read at `as_of`.

        For a message that was stripped and then needs another rewrite: its listing ended
        and the "Ended" rewrite did not get through, so the next expiry run must retry it.
        """
        self._exec(
            "UPDATE message SET as_of = ?, expired_at = NULL, has_listing_data = 1 WHERE id = ?", (as_of, message_id)
        )

    # ---- feedback ----------------------------------------------------------

    def add_feedback(
        self,
        label: str,
        watch_id: Optional[int] = None,
        candidate_code: Optional[str] = None,
        reason: Optional[str] = None,
        spec_patch: Optional[Dict[str, Any]] = None,
        approved: bool = False,
    ) -> int:
        cursor = self._exec(
            "INSERT INTO feedback (watch_id, candidate_code, label, reason, spec_patch, approved, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (watch_id, candidate_code, label, reason, _encode("spec_patch", spec_patch), int(approved), self.now_iso()),
        )
        return int(cursor.lastrowid)

    def list_feedback(self, watch_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if watch_id is None:
            return self._all("SELECT * FROM feedback ORDER BY id")
        return self._all("SELECT * FROM feedback WHERE watch_id = ? ORDER BY id", (watch_id,))

    # ---- call ledger -------------------------------------------------------

    def _day(self, day: Optional[str] = None) -> str:
        # Counted per UTC day. eBay's own reset time may differ; the cap leaves headroom for that.
        return day or self.clock().astimezone(timezone.utc).strftime("%Y-%m-%d")

    def ledger_add(self, api: str, purpose: str, calls: int = 1, day: Optional[str] = None) -> None:
        self._exec(
            "INSERT INTO ledger (day, api, purpose, calls) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(day, api, purpose) DO UPDATE SET calls = calls + excluded.calls",
            (self._day(day), api, purpose, calls),
        )

    def ledger_total(self, api: str, day: Optional[str] = None) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(calls), 0) AS total FROM ledger WHERE day = ? AND api = ?", (self._day(day), api)
        ).fetchone()
        return int(row["total"])

    def ledger_breakdown(self, api: str, day: Optional[str] = None) -> Dict[str, int]:
        rows = self.conn.execute(
            "SELECT purpose, calls FROM ledger WHERE day = ? AND api = ? ORDER BY purpose", (self._day(day), api)
        ).fetchall()
        return {row["purpose"]: int(row["calls"]) for row in rows}

    # ---- key/value ---------------------------------------------------------

    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def kv_set(self, key: str, value: Any) -> None:
        self._exec(
            "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, json.dumps(value), self.now_iso()),
        )

    def kv_delete(self, key: str) -> None:
        self._exec("DELETE FROM kv WHERE key = ?", (key,))

    # ---- teardown ----------------------------------------------------------

    def wipe(self) -> None:
        """Delete everything. Used by `eft uninstall`."""
        with self.transaction():
            for table in ("message", "alert", "verdict", "candidate", "feedback", "ledger", "kv", "watch"):
                self._exec("DELETE FROM %s" % table)


# What an alert row may keep once its listing has ended or been untracked.
_PAYLOAD_KEEP = ("code", "watch_id", "watch_name", "kind", "tier")


def strip_payload(payload: Any) -> Any:
    """Drop listing content (title, price, bids, image, link) from a stored alert payload."""
    if not isinstance(payload, dict):
        return payload
    return {key: payload[key] for key in _PAYLOAD_KEEP if key in payload}


def strip_evidence(fields: Any) -> Any:
    """Keep per-field results but drop quoted listing text: {'size': {'verdict': 'match'}}."""
    if not isinstance(fields, dict):
        return fields
    out: Dict[str, Any] = {}
    for name, value in fields.items():
        if isinstance(value, dict):
            out[name] = {k: v for k, v in value.items() if k not in ("evidence", "quote", "text")}
        else:
            out[name] = value
    return out
