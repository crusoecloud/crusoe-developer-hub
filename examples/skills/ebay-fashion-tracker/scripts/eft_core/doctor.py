"""`eft doctor`: checks that say what is set up, what works and what to fix.

Nothing here was verified against the live services when the tool was written, so these checks
are also the probes that settle how each service really behaves for one account: which model
ids exist, whether a two-image request is accepted, whether auctions come back from a search.

Every check returns a row {"area", "name", "ok", "detail"}. `ok` is True, False, or None for
something worth knowing that is neither. A detail names variables, never their values, and
repeats only error text the clients have already cleared of secrets.
"""
from __future__ import annotations

import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .alerts import DeliveryError, Notifier
from .config import FEATURE_VARS, ConfigError, Settings
from .crusoe import ROLE_FAMILY, CrusoeClient
from .ebay import EbayClient
from .errors import ChatResult, CrusoeError, EbayError
from .store import Store

AREAS = ("env", "crusoe", "ebay", "alerts")
# Without these two the tool can do nothing. The others add a channel or a safeguard.
REQUIRED_FEATURES = ("crusoe", "ebay")
QUOTA_WARNING_SHARE = 0.8
# A common fashion search, so that an empty result means something is wrong with the keyset.
PROBE_QUERY = "nike air force 1"
PROBE_SYSTEM = 'Reply with one JSON object and nothing else: {"ok": true}'
PROBE_TWO_IMAGES = (
    "You are given two images, each one plain color. Reply with one JSON object and nothing else: "
    '{"same": true} if they are the same color, {"same": false} if not.'
)
TEST_MESSAGE = "eft doctor: this is a test message. Alerts from the eBay Fashion Tracker will arrive here."

Row = Dict[str, Any]


def run_checks(
    area: str,
    settings: Settings,
    store: Optional[Store],
    ebay: Optional[Any] = None,
    crusoe: Optional[Any] = None,
    notifier: Optional[Any] = None,
) -> List[Row]:
    """Run the checks of one area ("env", "crusoe", "ebay", "alerts") or of "all".

    Clients that are not passed in are built from the settings. `env` needs no store, so that
    it still reports when the state directory cannot be opened.
    """
    if area != "all" and area not in AREAS:
        raise ValueError("unknown doctor area: %s" % area)
    rows: List[Row] = []
    for name in AREAS if area == "all" else (area,):
        if name == "env":
            rows += check_env(settings)
        elif name == "crusoe":
            rows += check_crusoe(settings, store, crusoe)
        elif name == "ebay":
            rows += check_ebay(settings, store, ebay)
        else:
            rows += check_alerts(settings, store, notifier)
    return rows


def _row(area: str, name: str, ok: Optional[bool], detail: str) -> Row:
    return {"area": area, "name": name, "ok": ok, "detail": detail}


# ---- env ---------------------------------------------------------------------

def check_env(settings: Settings) -> List[Row]:
    """Which features have their variables, whether the .env file was read, and where state goes."""
    rows = [_feature_row(settings, feature) for feature in FEATURE_VARS]
    if not (settings.has("slack") or settings.has("pushover")):
        rows.append(_row("env", "alert channel", False,
                         "neither Slack nor Pushover is configured, so no alert can be sent"))
    rows.append(_env_file_row(settings))
    rows.append(_state_dir_row(settings))
    return rows


def _feature_row(settings: Settings, feature: str) -> Row:
    names = FEATURE_VARS[feature]
    missing = settings.missing(feature)
    if not missing:
        return _row("env", feature, True, "set: " + ", ".join(names))
    # A feature with only some of its variables set is a mistake; one with none is a choice.
    half_set = len(missing) < len(names)
    ok = False if feature in REQUIRED_FEATURES or half_set else None
    return _row("env", feature, ok, "not set: " + ", ".join(missing))


def _readable_by_others(path: Path) -> bool:
    if os.name != "posix":
        return False
    try:
        return bool(path.stat().st_mode & 0o077)
    except OSError:
        return False


def _env_file_row(settings: Settings) -> Row:
    path, problem = settings.env_file, settings.env_file_problem
    if problem is None:
        if _readable_by_others(path):
            return _row("env", ".env file", None,
                        "read %s; other users on this machine can read it too, run: chmod 600 %s" % (path, path))
        return _row("env", ".env file", True, "read %s" % path)
    if problem == "missing":
        return _row("env", ".env file", None, "no file at %s; variables come from the environment only" % path)
    return _row("env", ".env file", False,
                "%s is %s from this shell: export the variables instead, or run eft from your own terminal"
                % (path, problem))


def _state_dir_row(settings: Settings) -> Row:
    """Create the state directory if need be and prove a file can be written in it."""
    try:
        settings.ensure_state_dir()
        with tempfile.NamedTemporaryFile(dir=str(settings.state_dir), prefix=".eft-write-test-"):
            pass
    except OSError as problem:
        return _row("env", "state directory", False,
                    "%s cannot be written (%s); point EFT_STATE_DIR at a directory that can"
                    % (settings.state_dir, problem.__class__.__name__))
    return _row("env", "state directory", True, str(settings.state_dir))


# ---- crusoe ------------------------------------------------------------------

def check_crusoe(settings: Settings, store: Optional[Store], client: Optional[Any] = None) -> List[Row]:
    """The live catalog, the model each role resolves to, and whether those models answer."""
    if not settings.has("crusoe"):
        return [_row("crusoe", "api key", False, "not set: " + ", ".join(settings.missing("crusoe")))]
    client = client or CrusoeClient(settings, store)  # type: ignore[arg-type]
    try:
        models = client.list_models(refresh=True)
    except (CrusoeError, ConfigError) as problem:
        return [_row("crusoe", "catalog", False, str(problem))]
    rows = [_row("crusoe", "catalog", True, "%d models listed" % len(models))]
    rows += [_role_row(client, role) for role in ROLE_FAMILY]
    text = [{"type": "text", "text": "Reply now."}]
    rows.append(_answer_row(client, "text model answers in JSON", "parse", PROBE_SYSTEM, text, ("ok",)))
    rows.append(_answer_row(client, "vision model answers in JSON", "compare", PROBE_SYSTEM, text, ("ok",)))
    rows.append(_two_image_row(client))
    return rows


def _role_row(client: Any, role: str) -> Row:
    try:
        ids = client.resolve_role(role)
    except CrusoeError as problem:
        return _row("crusoe", "role " + role, False, str(problem))
    fallbacks = " (then %s)" % ", ".join(ids[1:]) if len(ids) > 1 else ""
    return _row("crusoe", "role " + role, True, ids[0] + fallbacks)


def _answer_row(client: Any, name: str, role: str, system: str, parts: Sequence[Dict[str, Any]],
                required: Sequence[str]) -> Row:
    """One small request on a role: did a model answer with usable JSON, which one, and how fast."""
    try:
        answer = client.chat_json(role, system, list(parts), required=required)
    except CrusoeError as problem:
        return _row("crusoe", name, False, str(problem))
    return _row("crusoe", name, None if answer.swapped else True, _served(answer))


def _served(answer: ChatResult) -> str:
    said = "%s answered in %d ms" % (answer.model_served, answer.latency_ms)
    if answer.swapped:
        said += "; %s was requested, so that id has been redirected" % answer.model_requested
    return said


def _two_image_row(client: Any) -> Row:
    """The photo comparison sends a reference photo and a listing photo in one request."""
    name = "two images in one request"
    try:
        parts = [{"type": "text", "text": "Compare the two images."},
                 client.image_part(solid_png((200, 30, 30))), client.image_part(solid_png((30, 30, 200)))]
    except CrusoeError as problem:
        return _row("crusoe", name, False, str(problem))
    return _answer_row(client, name, "compare", PROBE_TWO_IMAGES, parts, ("same",))


def solid_png(rgb: Sequence[int], size: int = 16) -> bytes:
    """A small one-color PNG, built here so that the probe needs no image file and no eBay photo."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)      # 8-bit RGB, no interlace
    pixels = zlib.compress(row * size)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


# ---- ebay --------------------------------------------------------------------

def check_ebay(settings: Settings, store: Optional[Store], client: Optional[Any] = None) -> List[Row]:
    """Token, one search, whether auctions come back, and today's call count."""
    if not settings.has("ebay"):
        return [_row("ebay", "keyset", False, "not set: " + ", ".join(settings.missing("ebay")))]
    client = client or EbayClient(settings, store)  # type: ignore[arg-type]
    try:
        client.token()
    except (EbayError, ConfigError) as problem:
        return [_row("ebay", "application token", False, _ebay_failure(problem))]
    rows = [_row("ebay", "application token", True,
                 "issued for %s (%s)" % (settings.ebay_marketplace, settings.ebay_env))]
    rows.append(_search_row(client, "search", None))
    # Without the buyingOptions filter eBay returns Buy It Now listings only. The client always
    # sends it; this proves that auctions really do come back for this keyset.
    rows.append(_search_row(client, "auctions in search results", ["AUCTION"]))
    if store is not None:
        rows.append(_calls_row(settings, store))
    return rows


def _ebay_failure(problem: Exception) -> str:
    """Status and error ids first, because they are what eBay's own documentation is indexed by."""
    status = getattr(problem, "status", None)
    error_ids = getattr(problem, "error_ids", None) or []
    return "HTTP %s, error ids %s: %s" % (
        status if status is not None else "-", ", ".join(str(error_id) for error_id in error_ids) or "none", problem)


def _search_row(client: Any, name: str, buying_options: Optional[List[str]]) -> Row:
    try:
        found = client.search(PROBE_QUERY, buying_options=buying_options, limit=1, purpose="doctor")
    except EbayError as problem:
        return _row("ebay", name, False, _ebay_failure(problem))
    total = found.get("total", 0)
    kind = "auctions" if buying_options else "listings"
    return _row("ebay", name, total > 0, "HTTP 200, %d %s found for a keyword search" % (total, kind))


def _calls_row(settings: Settings, store: Store) -> Row:
    used, cap = store.ledger_total("ebay"), settings.ebay_daily_cap
    by_purpose = ", ".join("%s %d" % pair for pair in sorted(store.ledger_breakdown("ebay").items()))
    detail = "%d of %d today (UTC)%s" % (used, cap, " (%s)" % by_purpose if by_purpose else "")
    if used >= cap:
        return _row("ebay", "calls today", False, detail + "; the cap is reached and searching has stopped")
    return _row("ebay", "calls today", None if used >= QUOTA_WARNING_SHARE * cap else True, detail)


# ---- alerts ------------------------------------------------------------------

def check_alerts(settings: Settings, store: Optional[Store], notifier: Optional[Any] = None) -> List[Row]:
    """Send a test message on every configured channel."""
    notifier = notifier or Notifier(settings, store)  # type: ignore[arg-type]
    channels = notifier.channels()
    if not channels:
        return [_row("alerts", "channel", False,
                     "no channel is configured: set SLACK_BOT_TOKEN and SLACK_USER_ID, or "
                     "PUSHOVER_APP_TOKEN and PUSHOVER_USER_KEY")]
    return [_test_message_row(notifier, channel) for channel in channels]


def _test_message_row(notifier: Any, channel: str) -> Row:
    try:
        if channel == "slack":
            block = {"type": "section", "text": {"type": "mrkdwn", "text": TEST_MESSAGE}}
            notifier.post_slack({"text": TEST_MESSAGE, "blocks": [block]})
        else:
            notifier.post_push({"title": "eBay Fashion Tracker", "message": TEST_MESSAGE, "priority": 0})
    except DeliveryError as problem:
        return _row("alerts", channel, False, "the test message was not delivered: %s" % problem.reason)
    return _row("alerts", channel, True, "test message sent")
