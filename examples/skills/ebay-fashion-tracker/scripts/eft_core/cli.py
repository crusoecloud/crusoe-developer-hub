"""The `eft` command line: what Claude Code and the background agent both run.

Every command returns data and a plain-text rendering of it. `--json` prints the data as one
JSON document for Claude to read. Exit codes: 0 on success, 1 on a handled problem (one line
on stderr, never a traceback for a configuration or API error), 2 on a usage error.

Three rules shape what is printed:
- A key, token or webhook URL is never printed. Settings are reported by variable name.
- Listing data is kept to a few facts (code, short title, price or bid, end time, "as of" time,
  link), because the transcript of a Claude Code session is kept on disk. Facts older than six
  hours are not printed at all.
- An eBay link is read as text and looked up through the API. Nothing here opens an eBay page.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import plistlib
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape as xml_escape

from . import __version__, alerts, doctor, links, prompts, rules
from .alerts import Notifier
from .config import FEATURE_VARS, SKILL_ROOT, ConfigError, Settings, load_settings
from .crusoe import CrusoeClient
from .ebay import EbayClient
from .errors import CrusoeError, EbayError, EbayGroupRequired
from .funnel import Funnel, FunnelResult, asking_price, fetch_image
from .http import BlockedUrl, TransportError, UrllibTransport
from .plan import build_plan, build_query
from .spec import (
    MATCH_FIELDS, REQUIREMENTS, STRICTNESS, apply_patch, default_requirements, normalize_spec, spec_summary,
    validate_spec,
)
from .store import Store, iso, parse_iso
from .tick import LAST_TICK_KEY, OVERFLOW_KEY, RETRY_KEY, cadence_minutes, run_tick, stored_evidence

AGENT_LABEL = "com.crusoe.eft"
PLIST_TEMPLATE = SKILL_ROOT / "deploy" / "launchd" / (AGENT_LABEL + ".plist.tmpl")
LAUNCHER = SKILL_ROOT / "scripts" / "eft"
LAUNCHCTL = "/bin/launchctl"
REFERENCE_NAME = re.compile(r"[0-9a-f]{12}(\..*)?", re.DOTALL)
DEFAULT_INTERVAL_MINUTES = 5
LOCK_FILE = "tick.lock"
LOG_FILE = "tick.log"

INBOX_KEY = "inbox:last"
FIRST_INBOX_LOOKBACK = timedelta(hours=24)
DRYRUN_LIMIT = 10
DRYRUN_MAX = 50
PRODUCT_TITLES = 5
TITLE_CHARS = 60
# How long an untracked row is kept, so that the same listing is not judged and alerted again.
UNTRACKED_KEEP = timedelta(days=14)
STALE_NOTE = "details older than six hours are not shown; open the listing for live details"
MODEL_WORDS_CHARS = 240
_LINK_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

SETTABLE = ("size", "department", "kids_tier", "price_ceiling", "brand", "model", "style_codes", "colorway",
            "must_not", "must_have", "strictness")
LIST_SETTINGS = ("style_codes", "must_not", "must_have")
PATCH_KEYS = ("set", "add_must_not", "add_must_have", "add_nicknames", "set_requirement")


class Problem(Exception):
    """Something the user can act on. Printed as one line; the exit code is 1."""


class UsageError(Exception):
    """The command line itself is wrong. The exit code is 2."""


@dataclass
class Output:
    """What a command produced: data for --json, text for a person, and the exit code."""
    data: Dict[str, Any]
    text: str
    code: int = 0


def run_command(arguments: Sequence[str]) -> Tuple[int, str]:
    """Run a system command (launchctl) and return (exit code, what it printed)."""
    try:
        done = subprocess.run(list(arguments), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
                              check=False)
    except (OSError, subprocess.SubprocessError) as problem:
        return 127, problem.__class__.__name__
    return done.returncode, done.stdout.decode("utf-8", errors="replace").strip()


class Runtime:
    """What a command works with: the settings, the store and the API clients.

    Each part is built the first time it is asked for, so a command that needs no network
    (status, watch list, doctor env) runs without any key. Tests pass ready-made parts in.
    `run` executes system commands and `platform` is `sys.platform`; both exist so that the
    agent commands can be tested without touching launchd.
    """

    def __init__(self, settings: Optional[Settings] = None, store: Optional[Store] = None, ebay: Any = None,
                 crusoe: Any = None, notifier: Any = None, transport: Any = None,
                 run: Optional[Callable[[Sequence[str]], Tuple[int, str]]] = None,
                 platform: Optional[str] = None) -> None:
        self._settings = settings
        self._store = store
        self._ebay = ebay
        self._crusoe = crusoe
        self._notifier = notifier
        self._transport = transport
        self._owns_store = store is None
        self.run = run or run_command
        self.platform = platform or sys.platform

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def transport(self) -> Any:
        if self._transport is None:
            self._transport = UrllibTransport()
        return self._transport

    @property
    def store(self) -> Store:
        if self._store is None:
            try:
                self.settings.ensure_state_dir()
                self._store = Store(self.settings.db_path)
            except (OSError, sqlite3.Error) as problem:
                raise Problem(
                    "cannot open the state directory %s (%s). Point EFT_STATE_DIR at a directory this shell "
                    "can write to; `eft doctor env` shows what is set."
                    % (self.settings.state_dir, problem.__class__.__name__)
                )
        return self._store

    @property
    def ebay(self) -> Any:
        if self._ebay is None:
            self._ebay = EbayClient(self.settings, self.store, self.transport)
        return self._ebay

    @property
    def crusoe(self) -> Any:
        if self._crusoe is None:
            self._crusoe = CrusoeClient(self.settings, self.store, self.transport)
        return self._crusoe

    @property
    def notifier(self) -> Any:
        if self._notifier is None:
            self._notifier = Notifier(self.settings, self.store, self.transport)
        return self._notifier

    def funnel(self, store: Optional[Store] = None) -> Funnel:
        """A funnel over this runtime's clients, recording into `store` (the real one by default)."""
        return Funnel(self.ebay, self.crusoe, store or self.store,
                      image_fetcher=partial(fetch_image, transport=self.transport))

    def close(self) -> None:
        """Close the store if this runtime opened it. One that was passed in belongs to the caller."""
        if self._owns_store:
            self.discard_store()

    def discard_store(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


# ---- small helpers -------------------------------------------------------------

def _find_watch(store: Store, ref: str) -> Dict[str, Any]:
    watch = store.find_watch(ref)
    if watch is None:
        raise Problem("no watch matches %r; `eft watch list` shows them" % ref)
    return watch


def _find_candidate(store: Store, code: str) -> Dict[str, Any]:
    candidate = store.get_candidate_by_code(code)
    if candidate is None:
        raise Problem("no listing has the code %r; `eft inbox` shows recent ones" % code)
    return candidate


def _money(amount: Optional[Mapping[str, Any]]) -> str:
    return "%s %s" % (amount.get("currency") or "", amount.get("value")) if amount else ""


def _short(text: Any, limit: int = TITLE_CHARS) -> str:
    """One line, cut to `limit` characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def model_words(text: Any) -> str:
    """A sentence a model wrote, made fit to show: one line, bounded, and without links.

    Links shown to the user come from the eBay API only. A model that has read a seller's
    text could be talked into writing one, so anything that looks like a link is removed.
    """
    return _short(_LINK_RE.sub("[link removed]", str(text or "")), MODEL_WORDS_CHARS)


def listing_view(listing: Optional[Mapping[str, Any]], as_of: Optional[str], now: datetime) -> Dict[str, Any]:
    """The few facts about a listing that may be printed, or less when they are too old.

    Data shown to the user is never more than six hours old and always says when it was
    fetched. Past that only the title and the link remain, as on an expired card.
    """
    if not listing:
        return {}
    view: Dict[str, Any] = {"title": _short(listing.get("title")), "url": listing.get("web_url")}
    fetched = parse_iso(as_of)
    if fetched is None or (now - fetched).total_seconds() >= alerts.SIX_HOURS:
        view["note"] = STALE_NOTE
        return view
    price = asking_price(listing)
    if "AUCTION" in (listing.get("buying_options") or []):
        view["current_bid"], view["bids"] = price, listing.get("bid_count")
    else:
        view["price"] = price
    view["ends_at"] = listing.get("end_at")
    view["as_of"] = as_of
    return view


def _listing_text(view: Mapping[str, Any]) -> str:
    """A listing view on one line."""
    parts = [view.get("title") or ""]
    if view.get("price"):
        parts.append(_money(view["price"]))
    if view.get("current_bid"):
        bids = view.get("bids")
        parts.append("bid %s (%s bids)" % (_money(view["current_bid"]), "?" if bids is None else bids))
    if view.get("ends_at"):
        parts.append("ends %s" % view["ends_at"])
    if view.get("as_of"):
        parts.append("as of %s" % view["as_of"])
    if view.get("note"):
        parts.append(view["note"])
    if view.get("url"):
        parts.append(view["url"])
    return " · ".join(part for part in parts if part)


def untrack(store: Store, candidate: Mapping[str, Any], state: str, reason: str) -> None:
    """Stop tracking a listing: delete what was stored about it and drop alerts still waiting to go out.

    The row itself stays for two weeks, as an id and a reason, so that the same listing is not
    judged and alerted on again.
    """
    with store.transaction():
        for alert in store.alerts_for_candidate(candidate["id"]):
            if alert["status"] in ("pending", "held"):
                store.update_alert(alert["id"], status="dropped")
        store.purge_listing_data(candidate["id"], state=state)
        store.update_candidate(candidate["id"], reason=reason, end_at=None,
                               purge_at=iso(store.clock() + UNTRACKED_KEEP))
        store.kv_delete(RETRY_KEY % candidate["id"])


def _cards_showing_data(store: Store, candidate_id: int) -> List[Dict[str, Any]]:
    return [message for message in store.messages_for_candidate(candidate_id) if message["has_listing_data"]]


def strip_cards(rt: Runtime, candidates: Sequence[Mapping[str, Any]]) -> int:
    """Rewrite the sent messages of these listings so that none shows listing data any more.

    Returns how many messages could not be rewritten (Slack unreachable, for example). Those
    still have their rows, so the poller strips them at six hours like any other card.
    """
    left = 0
    for candidate in candidates:
        if _cards_showing_data(rt.store, candidate["id"]):
            rt.notifier.mark_ended(candidate["id"])
            left += len(_cards_showing_data(rt.store, candidate["id"]))
    return left


def settle_requirements(old: Mapping[str, Any], new: Dict[str, Any], explicit: Set[str]) -> Dict[str, Any]:
    """Give a field that just gained or lost its value the default requirement for its new state.

    A spec remembers "size: na" from when it had no size. Setting a size later has to make
    it required, unless the same change said otherwise (`explicit`).
    """
    was, now = default_requirements(old), default_requirements(new)
    for name in MATCH_FIELDS:
        if name not in explicit and (was[name] == "na") != (now[name] == "na"):
            new["requirements"][name] = now[name]
    return new


def _replan(watch: Mapping[str, Any], spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Keep an activated watch's search in step with its spec. A draft has no plan yet."""
    if not watch["plan"]:
        return None
    return dict(watch["plan"], q=build_query(spec), buying_options=list(spec["buying_options"]))


def _block(title: str, lines: Sequence[str]) -> List[str]:
    return [title] + ["  " + line for line in lines] if lines else []


# ---- status and doctor ---------------------------------------------------------

def _config_view(settings: Settings) -> Dict[str, Any]:
    """The configuration by variable name. No value of a key, token or URL goes in here."""
    start, end = settings.quiet_hours
    return {
        "features": {name: {"configured": settings.has(name), "missing": settings.missing(name)}
                     for name in FEATURE_VARS},
        "env_file": {"path": str(settings.env_file), "problem": settings.env_file_problem},
        "state_dir": str(settings.state_dir),
        "marketplace": settings.ebay_marketplace,
        "ebay_env": settings.ebay_env,
        "quiet_hours": "%02d:%02d-%02d:%02d" % (start // 60, start % 60, end // 60, end % 60),
        "model_overrides": dict(settings.model_overrides),
    }


def _watch_view(store: Store, watch: Mapping[str, Any]) -> Dict[str, Any]:
    spec = normalize_spec(watch["spec"])
    tracked: Dict[str, int] = {}
    for candidate in store.list_candidates(watch["id"], tracked=True):
        tracked[candidate["state"]] = tracked.get(candidate["state"], 0) + 1
    return {
        "id": watch["id"], "name": watch["name"], "status": watch["status"], "tier": watch["tier"],
        "summary": spec_summary(spec), "tracked": tracked, "problems": validate_spec(spec),
        "too_broad": store.kv_get(OVERFLOW_KEY % watch["id"]) is not None,
        "last_search_at": watch["last_search_at"],
    }


def _watch_line(view: Mapping[str, Any]) -> str:
    tracked = ", ".join("%d %s" % (count, state) for state, count in sorted(view["tracked"].items()))
    extras = [tracked] if tracked else []
    if view["too_broad"]:
        extras.append("query too broad: add a style code, a colorway nickname or a must_have term")
    return "%d  %-7s %-6s %s: %s%s" % (
        view["id"], view["status"], view["tier"], view["name"], view["summary"],
        " [%s]" % "; ".join(extras) if extras else "")


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / (AGENT_LABEL + ".plist")


def _waiting_steps(rt: Runtime, watches: Sequence[Mapping[str, Any]]) -> List[str]:
    """Setup steps that are still open, in the order to do them."""
    settings, steps = rt.settings, []
    if settings.env_file_problem not in (None, "missing"):
        steps.append("The .env file cannot be read from this shell: export the variables, or run eft from "
                     "your own terminal.")
    for feature, then in (("crusoe", "eft doctor crusoe"), ("ebay", "eft doctor ebay")):
        if not settings.has(feature):
            steps.append("Set %s, then run: %s" % (" and ".join(settings.missing(feature)), then))
    if not (settings.has("slack") or settings.has("pushover")):
        steps.append("Add an alert channel (SLACK_BOT_TOKEN and SLACK_USER_ID, or PUSHOVER_APP_TOKEN and "
                     "PUSHOVER_USER_KEY), then run: eft doctor alerts")
    if not watches:
        steps.append("Add something to track: eft add --text \"...\", --image PATH or --url URL")
    drafts = [str(watch["id"]) for watch in watches if watch["status"] == "draft"]
    if drafts:
        steps.append("Draft watches waiting to be activated: %s" % ", ".join(drafts))
    active = any(watch["status"] == "active" for watch in watches)
    if active and rt.platform == "darwin" and not _plist_path().exists():
        steps.append("Start the background poller: eft agent install")
    return steps


def cmd_status(rt: Runtime, args: argparse.Namespace) -> Output:
    settings, store = rt.settings, rt.store
    watches = [_watch_view(store, watch) for watch in store.list_watches()]
    config = _config_view(settings)
    used, cap = store.ledger_total("ebay"), settings.ebay_daily_cap
    data = {
        "version": __version__,
        "config": config,
        "watches": watches,
        "last_tick": store.kv_get(LAST_TICK_KEY),
        "calls_today": {
            "ebay": {"used": used, "cap": cap, "by_purpose": store.ledger_breakdown("ebay")},
            "crusoe": {"used": store.ledger_total("crusoe"), "by_role": store.ledger_breakdown("crusoe")},
        },
        "waiting": _waiting_steps(rt, watches),
    }
    configured = [name for name, feature in config["features"].items() if feature["configured"]]
    last = data["last_tick"]
    lines = [
        "eft %s" % __version__,
        "configured: %s" % (", ".join(configured) or "nothing yet"),
        "eBay calls today: %d of %d" % (used, cap),
        "last poll: %s" % ("%s, %d problem(s)" % (last["at"], len(last["problems"])) if last else "never"),
    ]
    lines += _block("problems in the last poll:", last["problems"] if last else [])
    lines += _block("watches:", [_watch_line(view) for view in watches])
    lines += _block("still to do:", data["waiting"])
    return Output(data, "\n".join(lines))


def cmd_doctor(rt: Runtime, args: argparse.Namespace) -> Output:
    """Run the checks area by area, so that `env` still reports when the store cannot be opened."""
    rows: List[Dict[str, Any]] = []
    for area in doctor.AREAS if args.area == "all" else (args.area,):
        if area == "env":
            rows += doctor.run_checks("env", rt.settings, None)
            continue
        try:
            store = rt.store
        except Problem as problem:
            rows.append({"area": area, "name": "state", "ok": False, "detail": str(problem)})
            continue
        # Only the client an area needs is built, so `doctor ebay` works without a Crusoe key.
        client = {"alerts": "notifier"}.get(area, area)
        rows += doctor.run_checks(area, rt.settings, store, **{client: getattr(rt, client)})
    marks = {True: "ok  ", False: "FAIL", None: "note"}
    text = "\n".join("%s  %-7s %s: %s" % (marks[row["ok"]], row["area"], row["name"], row["detail"]) for row in rows)
    failed = any(row["ok"] is False for row in rows)
    return Output({"checks": rows, "ok": not failed}, text, code=1 if failed else 0)


# ---- add -----------------------------------------------------------------------

@dataclass
class Draft:
    """A spec in the making: the fields a model read, and what the user should be told or asked."""
    fields: Dict[str, Any]
    questions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def read_draft(rt: Runtime, role: str, prompt: prompts.Prompt, texts: Sequence[str],
               from_listing: bool = False) -> Draft:
    """Ask a model for a draft spec, then keep only the style codes that were really read.

    `texts` is what the user or a page actually said. A code the model offers is kept when it
    appears there or, for a photo, in the text the model reports reading off it. Any other
    code may have come from the model's memory, so it is left out and reported.

    The model's own questions for the user are passed on, except when it was reading a
    seller's listing (`from_listing`): free text written under the influence of seller text
    is not shown to the user.
    """
    system, parts, required = prompt
    data = rt.crusoe.chat_json(role, system, parts, required=required, validator=prompts.validate_draft).data
    read = list(texts)
    if role == "extract":
        read += [str(text) for text in data.get("visible_text") or []]
    offered = [str(code).strip() for code in data.get("style_codes") or [] if str(code).strip()]
    kept = prompts.stated_codes(offered, read)
    fields = {name: data.get(name) for name in ("brand", "model", "colorway", "size", "department", "kids_tier",
                                                  "visual_signature")}
    fields["style_codes"] = kept
    notes = ["style code %s was left out: it does not appear in the text that was read" % code
             for code in offered if code not in kept]
    asked = [] if from_listing else data.get("questions") or []
    questions = [model_words(question) for question in asked if isinstance(question, str) and question.strip()]
    return Draft(fields, questions, notes)


def _draft_from_image(rt: Runtime, path: str, note: str) -> Draft:
    photo = rt.crusoe.image_part(path)
    draft = read_draft(rt, "extract", prompts.extract(photo, note), [note])
    draft.fields["reference_images"] = [_keep_reference(rt.settings, path)]
    draft.fields["provenance"] = {"source": "photo"}
    return draft


def _keep_reference(settings: Settings, path: str) -> str:
    """Copy the user's photo into the state directory, so the watch does not depend on where it was."""
    source = Path(path).expanduser()
    target = settings.refs_dir / (uuid.uuid4().hex[:12] + source.suffix.lower())
    try:
        settings.ensure_state_dir()
        shutil.copyfile(str(source), str(target))
    except OSError as problem:
        raise Problem("cannot copy the image %s (%s)" % (source, problem.__class__.__name__))
    return str(target)


def _draft_from_text(rt: Runtime, text: str) -> Draft:
    draft = read_draft(rt, "parse", prompts.parse(text), [text])
    draft.fields["provenance"] = {"source": "text"}
    return draft


def _draft_from_url(rt: Runtime, url: str) -> Draft:
    """Turn an eBay link into a draft. The link is parsed as text; what it points at comes from the API."""
    link = links.parse_link(url)
    if link.kind == "not_ebay":
        raise Problem("that is not an eBay link. Read the retailer's page yourself and pass what it says with: "
                      "eft add --text \"brand, model, style code, color\"")
    if link.kind == "short":
        raise Problem("an eBay short link cannot be read without opening it, and this tool never opens eBay "
                      "pages. Ask for the full URL (ebay.com/itm/...) and run eft add --url with that.")
    if link.kind == "item":
        draft = _draft_from_item(rt, link)
    elif link.kind == "product":
        draft = _draft_from_product(rt, link)
    elif link.kind == "search":
        draft = _draft_from_search(rt, link)
    else:
        raise Problem("that eBay link is not an item, product or search link. Use the listing's own URL "
                      "(ebay.com/itm/...), or describe the item with: eft add --text \"...\"")
    if link.marketplace and link.marketplace != rt.settings.ebay_marketplace:
        draft.notes.append("the link is from %s but searches run on %s (EBAY_MARKETPLACE)"
                           % (link.marketplace, rt.settings.ebay_marketplace))
    return draft


def _draft_from_item(rt: Runtime, link: links.EbayLink) -> Draft:
    multi_size = False
    try:
        detail = rt.ebay.get_item_by_legacy_id(
            link.legacy_item_id, legacy_variation_id=link.variation_id, purpose="add")
    except EbayGroupRequired as parent:
        group_id = parent.item_group_id
        variations = rt.ebay.get_items_by_item_group(group_id, purpose="add") if group_id else []
        if not variations:
            raise Problem("that listing has several variations and eBay did not return them. Describe the item "
                          "with: eft add --text \"...\"")
        detail, multi_size = variations[0], True
    text = prompts.listing_text(detail)
    draft = read_draft(rt, "parse", prompts.parse(text), [text], from_listing=True)
    if multi_size:
        # The first variation's size is one of several, not the size the user wants.
        draft.fields["size"] = None
        draft.questions.append("The listing comes in several sizes. Which size do you want?")
    draft.fields["provenance"] = {"source": "ebay item"}
    return draft


def _draft_from_product(rt: Runtime, link: links.EbayLink) -> Draft:
    found = rt.ebay.search(epid=link.epid, limit=PRODUCT_TITLES, purpose="add")
    titles = [listing["title"] for listing in found["items"] if listing.get("title")]
    if not titles:
        raise Problem("eBay returned no listings for that product link. Describe the item with: "
                      "eft add --text \"...\"")
    text = "Titles of eBay listings for one catalog product:\n" + "\n".join("- " + title for title in titles)
    draft = read_draft(rt, "parse", prompts.parse(text), [text], from_listing=True)
    draft.fields["provenance"] = {"source": "ebay product"}
    return draft


def _draft_from_search(rt: Runtime, link: links.EbayLink) -> Draft:
    keywords = link.search.get("q")
    if not keywords:
        raise Problem("that search link has no keywords. Describe the item with: eft add --text \"...\"")
    draft = read_draft(rt, "parse", prompts.parse(keywords), [keywords])
    if link.search.get("buying_options"):
        draft.fields["buying_options"] = link.search["buying_options"]
    if link.search.get("price_max"):
        draft.fields["price_ceiling"] = link.search["price_max"]
        draft.questions.append("The search had a maximum price of %g. Is that your price ceiling?"
                               % link.search["price_max"])
    draft.fields["provenance"] = {"source": "ebay search"}
    return draft


def _open_questions(spec: Mapping[str, Any]) -> List[str]:
    """What always has to be confirmed with the user before a watch goes active."""
    questions = []
    if not spec.get("size"):
        questions.append("Which size, and in which size system (US, UK, EU, alpha)?")
    if not spec.get("department"):
        questions.append("Men's, women's, unisex or kids'?")
    if not spec.get("price_ceiling"):
        questions.append("What is the most you would pay? (optional price ceiling)")
    return questions


def cmd_add(rt: Runtime, args: argparse.Namespace) -> Output:
    if args.image:
        draft = _draft_from_image(rt, args.image, args.note or "")
    elif args.text:
        draft = _draft_from_text(rt, args.text)
    else:
        draft = _draft_from_url(rt, args.url)
    spec = normalize_spec(draft.fields)
    name = args.name or " ".join(part for part in (spec["brand"], spec["model"]) if part) or "Untitled watch"
    watch_id = rt.store.create_watch(name, spec)
    questions = _open_questions(spec) + [question for question in draft.questions if question]
    data = {
        "watch": {"id": watch_id, "name": name, "status": "draft"},
        "summary": spec_summary(spec), "spec": spec, "questions": questions,
        "problems": validate_spec(spec), "notes": draft.notes,
        "next": ["eft watch set %d KEY=VALUE ..." % watch_id, "eft dryrun %d" % watch_id,
                 "eft watch activate %d" % watch_id],
    }
    lines = ["draft watch %d: %s" % (watch_id, name), data["summary"]]
    lines += _block("to confirm:", questions) + _block("problems:", data["problems"]) + _block("notes:", draft.notes)
    lines += _block("next:", data["next"])
    return Output(data, "\n".join(lines))


# ---- watch ---------------------------------------------------------------------

def _comma_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ceiling(value: str) -> Optional[Dict[str, Any]]:
    """'180', '180 GBP' or '180 incl' (shipping included) -> the ceiling the spec stores."""
    if not value:
        return None
    ceiling: Dict[str, Any] = {"incl_shipping": False}
    for word in value.replace("$", " ").split():
        if word.lower().startswith("incl"):
            ceiling["incl_shipping"] = True
        elif word.isalpha() and len(word) == 3:
            ceiling["currency"] = word.upper()
        else:
            ceiling["amount"] = word
    try:
        if float(str(ceiling.get("amount", "")).replace(",", "")) <= 0:
            raise ValueError(value)
    except ValueError:
        raise UsageError("price_ceiling needs an amount above zero, such as 180, \"180 GBP\" or \"180 incl\"")
    return ceiling


def apply_settings(spec: Mapping[str, Any], pairs: Sequence[str]) -> Dict[str, Any]:
    """Apply `KEY=VALUE` settings from `eft watch set` and return the new normalized spec.

    An empty value clears a field. `require.<field>=required|optional|na` sets how a field
    counts toward a match.
    """
    raw = copy.deepcopy(dict(spec))
    explicit: Set[str] = set()
    for pair in pairs:
        key, equals, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if not equals or not key:
            raise UsageError("expected KEY=VALUE, got %r" % pair)
        if key.startswith("require."):
            name = key[len("require."):]
            if name not in MATCH_FIELDS or value.lower() not in REQUIREMENTS:
                raise UsageError("require.<field> takes one of %s for one of %s"
                                 % (", ".join(REQUIREMENTS), ", ".join(MATCH_FIELDS)))
            raw["requirements"] = dict(raw.get("requirements") or {}, **{name: value.lower()})
            explicit.add(name)
        elif key in LIST_SETTINGS:
            raw[key] = _comma_list(value)
        elif key == "price_ceiling":
            raw[key] = _ceiling(value)
        elif key == "colorway":
            raw[key] = dict(raw.get("colorway") or {}, official=value or None)
        elif key == "strictness":
            if value.lower() not in STRICTNESS:
                raise UsageError("strictness is %s" % " or ".join(STRICTNESS))
            raw[key] = value.lower()
        elif key in SETTABLE:
            raw[key] = value or None
        else:
            raise UsageError("unknown setting %r; settable: %s, require.<field>" % (key, ", ".join(SETTABLE)))
    return settle_requirements(spec, normalize_spec(raw), explicit)


def cmd_watch_list(rt: Runtime, args: argparse.Namespace) -> Output:
    views = [_watch_view(rt.store, watch) for watch in rt.store.list_watches()]
    return Output({"watches": views}, "\n".join(_watch_line(view) for view in views) or "no watches yet")


def cmd_watch_show(rt: Runtime, args: argparse.Namespace) -> Output:
    watch = _find_watch(rt.store, args.id)
    view = _watch_view(rt.store, watch)
    spec = normalize_spec(watch["spec"])
    data = dict(view, spec=spec, plan=watch["plan"], window_end=watch["window_end"],
                last_sweep_at=watch["last_sweep_at"])
    requirements = ", ".join("%s %s" % (name, spec["requirements"][name]) for name in MATCH_FIELDS)
    lines = [_watch_line(view), "requirements: " + requirements, "strictness: " + spec["strictness"]]
    for label, terms in (("must not", spec["must_not"]), ("must have", spec["must_have"])):
        if terms:
            lines.append("%s: %s" % (label, ", ".join(terms)))
    if watch["plan"]:
        lines.append("search: %s (category %s)" % (watch["plan"]["q"], watch["plan"].get("category_id") or "any"))
    lines += _block("problems:", view["problems"])
    return Output(data, "\n".join(lines))


def cmd_watch_set(rt: Runtime, args: argparse.Namespace) -> Output:
    watch = _find_watch(rt.store, args.id)
    spec = apply_settings(normalize_spec(watch["spec"]), args.settings)
    problems = validate_spec(spec)
    if problems and watch["status"] == "active":
        raise Problem("watch %d is active and that change would leave it unusable (%s). Pause it first, or fix "
                      "the setting." % (watch["id"], "; ".join(problems)))
    rt.store.update_watch(watch["id"], spec=spec, plan=_replan(watch, spec))
    data = {"watch": {"id": watch["id"], "name": watch["name"], "status": watch["status"]},
            "summary": spec_summary(spec), "spec": spec, "problems": problems}
    lines = ["watch %d: %s" % (watch["id"], data["summary"])] + _block("problems:", problems)
    return Output(data, "\n".join(lines))


def cmd_watch_activate(rt: Runtime, args: argparse.Namespace) -> Output:
    store = rt.store
    watch = _find_watch(store, args.id)
    spec = normalize_spec(watch["spec"])
    problems = validate_spec(spec)
    if problems:
        raise Problem("watch %d cannot be activated yet: %s" % (watch["id"], "; ".join(problems)))
    rt.settings.require("ebay")
    plan = build_plan(spec, rt.ebay)
    store.update_watch(watch["id"], status="active", spec=spec, plan=plan, window_end=store.now_iso())
    active = store.list_watches(status="active")
    hot = sum(1 for row in active if row["tier"] == "hot")
    cadence = cadence_minutes(hot, len(active) - hot, rt.settings.ebay_daily_cap)
    notes = []
    if not (rt.settings.has("slack") or rt.settings.has("pushover")):
        notes.append("no alert channel is configured yet, so alerts will wait in the outbox")
    data = {"watch": {"id": watch["id"], "name": watch["name"], "status": "active"}, "plan": plan,
            "search_every_minutes": cadence, "notes": notes}
    lines = ["watch %d is active: searching for \"%s\" every %d minutes"
             % (watch["id"], plan["q"], cadence["hot" if watch["tier"] == "hot" else "normal"])]
    return Output(data, "\n".join(lines + notes))


def _set_watch(rt: Runtime, ref: str, **fields: Any) -> Output:
    watch = _find_watch(rt.store, ref)
    rt.store.update_watch(watch["id"], **fields)
    watch = dict(watch, **fields)
    data = {"watch": {"id": watch["id"], "name": watch["name"], "status": watch["status"], "tier": watch["tier"]}}
    return Output(data, "watch %d (%s): %s, %s" % (watch["id"], watch["name"], watch["status"], watch["tier"]))


def cmd_watch_pause(rt: Runtime, args: argparse.Namespace) -> Output:
    return _set_watch(rt, args.id, status="paused")


def cmd_watch_hot(rt: Runtime, args: argparse.Namespace) -> Output:
    return _set_watch(rt, args.id, tier="hot")


def cmd_watch_normal(rt: Runtime, args: argparse.Namespace) -> Output:
    return _set_watch(rt, args.id, tier="normal")


def cmd_watch_remove(rt: Runtime, args: argparse.Namespace) -> Output:
    """Delete a watch and everything stored for it.

    Messages already sent are rewritten first, because deleting the rows that say where they
    are would leave their listing data on show for good. If one cannot be rewritten the watch
    is paused and emptied instead, and removing it is finished by running the command again.
    """
    store = rt.store
    watch = _find_watch(store, args.id)
    candidates = store.list_candidates(watch["id"])
    left = strip_cards(rt, candidates)
    if left:
        with store.transaction():
            store.update_watch(watch["id"], status="paused")
            for candidate in candidates:
                if candidate["tracked"]:
                    untrack(store, candidate, "ended", "watch_removed")
        raise Problem("watch %d is paused and its listing data is deleted, but %d sent message(s) could not be "
                      "rewritten just now. The poller strips them within six hours; run this command again "
                      "after that to finish removing the watch." % (watch["id"], left))
    _delete_reference_images(rt.settings, watch["spec"])
    with store.transaction():
        store.kv_delete(OVERFLOW_KEY % watch["id"])
        for candidate in candidates:
            store.kv_delete(RETRY_KEY % candidate["id"])
        store.delete_watch(watch["id"])
    return Output({"removed": {"id": watch["id"], "name": watch["name"]}},
                  "removed watch %d (%s) and everything stored for it" % (watch["id"], watch["name"]))


def _delete_reference_images(settings: Settings, spec: Mapping[str, Any]) -> None:
    """Delete a watch's reference photos, but only the copies this tool made in its own folder."""
    refs = settings.refs_dir.resolve()
    for path in spec.get("reference_images") or []:
        target = Path(path).expanduser().resolve()
        if refs in target.parents:
            try:
                target.unlink()
            except OSError:
                pass


# ---- dryrun, tick, inbox -------------------------------------------------------

def _result_view(result: FunnelResult, as_of: str, now: datetime) -> Dict[str, Any]:
    view = {"decision": result.decision, "reason": result.reason, "proof": result.proof, "unproven": result.unproven}
    view.update(listing_view(result.detail or result.summary, as_of, now))
    return view


def cmd_dryrun(rt: Runtime, args: argparse.Namespace) -> Output:
    """One search and the funnel over its newest listings, with nothing kept.

    The funnel records into a scratch store in memory that is thrown away, so no candidate,
    verdict or alert survives. The API clients still count their calls in the real ledger.
    """
    if not 1 <= args.limit <= DRYRUN_MAX:
        raise UsageError("--limit takes 1 to %d" % DRYRUN_MAX)
    store = rt.store
    watch = _find_watch(store, args.id)
    spec = normalize_spec(watch["spec"])
    plan = watch["plan"] or build_plan(spec)
    if not plan["q"]:
        raise Problem("watch %d has nothing to search for yet: set a brand, a model or a style code" % watch["id"])
    found = rt.ebay.search(plan["q"], category_id=plan.get("category_id"), buying_options=plan.get("buying_options"),
                           sort="newlyListed", limit=args.limit, purpose="dryrun")
    scratch = Store(":memory:", clock=store.clock)
    try:
        trial = scratch.get_watch(scratch.create_watch(watch["name"], spec, tier=watch["tier"]))
        results = rt.funnel(scratch).run(trial, found["items"])
    finally:
        scratch.close()
    now = store.clock()
    views = [_result_view(result, iso(now), now) for result in results]
    counts: Dict[str, int] = {}
    for view in views:
        counts[view["decision"]] = counts.get(view["decision"], 0) + 1
    data = {"watch": {"id": watch["id"], "name": watch["name"]}, "query": plan["q"], "total_found": found["total"],
            "judged": len(views), "decisions": counts, "results": views}
    lines = ["dry run for watch %d: \"%s\" found %d listings, judged the newest %d"
             % (watch["id"], plan["q"], found["total"], len(views))]
    for view in views:
        why = view["proof"] if view["decision"] == "alert" else view["reason"]
        lines.append("  %-7s %s (%s)" % (view["decision"], _listing_text(view), why))
    lines.append("nothing was kept and no alert was sent")
    return Output(data, "\n".join(lines))


@contextmanager
def tick_lock(settings: Settings) -> Iterator[bool]:
    """Hold the poller's lock for the length of one tick. Yields False when another tick has it.

    The agent and a hand-run `eft tick` must not overlap: both would deliver the same outbox.
    """
    settings.ensure_state_dir()
    with open(str(settings.state_dir / LOCK_FILE), "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True


def cmd_tick(rt: Runtime, args: argparse.Namespace) -> Output:
    with tick_lock(rt.settings) as free:
        if not free:
            return Output({"skipped": "another eft tick is running"}, "another eft tick is running; nothing done")
        report = run_tick(rt.settings, rt.store, rt.ebay, rt.crusoe, rt.notifier, transport=rt.transport)
    decisions = ", ".join("%d %s" % (count, name) for name, count in sorted(report["decisions"].items()))
    delivery = report["delivery"]
    lines = [
        "searched %d watch(es), refreshed %d listing(s), %d ended"
        % (report["searched"], report["refreshed"], report["ended"]),
        "judged: %s" % (decisions or "nothing new"),
        "alerts: %d queued, %d sent, %d held" % (report["queued"], delivery["sent"], delivery["held"]),
    ]
    lines += _block("problems:", report["problems"])
    return Output(report, "\n".join(lines), code=1 if report["problems"] else 0)


def _alert_view(store: Store, alert: Mapping[str, Any], now: datetime) -> Dict[str, Any]:
    """One alert for the inbox. Listing facts come from the tracked snapshot, which the poller
    keeps fresh, not from the alert as it was sent."""
    payload = alert["payload"]
    view = {"code": payload.get("code"), "watch": payload.get("watch_name"), "kind": alert["kind"],
            "tier": alert["tier"], "status": alert["status"], "sent_at": alert["sent_at"]}
    candidate = store.get_candidate_by_id(alert["candidate_id"]) if alert["candidate_id"] else None
    if candidate is not None:
        view["state"], view["starred"] = candidate["state"], bool(candidate["starred"])
        view.update(listing_view(candidate["listing"], candidate["fetched_at"], now))
        if payload.get("unconfirmed"):
            view["unconfirmed"] = payload["unconfirmed"]
    return view


def _alert_line(view: Mapping[str, Any]) -> str:
    head = "%s  %s (%s)  %s" % (view["code"], view["kind"], view.get("state") or view["status"], view["watch"])
    rest = _listing_text(view)
    return head + (" · " + rest if rest else "")


def cmd_inbox(rt: Runtime, args: argparse.Namespace) -> Output:
    store = rt.store
    now = store.clock()
    since = store.kv_get(INBOX_KEY) or iso(now - FIRST_INBOX_LOOKBACK)
    recent = [alert for alert in store.alerts_by_status("sent") if (alert["sent_at"] or "") > since]
    sent = [_alert_view(store, alert, now) for alert in recent if alert["kind"] != "system"]
    notices = [alert["payload"].get("text") for alert in recent if alert["kind"] == "system"]
    waiting = [_alert_view(store, alert, now) for alert in store.alerts_by_status("pending", "held")
               if alert["kind"] != "system"]
    store.kv_set(INBOX_KEY, iso(now))
    data = {"since": since, "sent": sent, "waiting": waiting, "notices": notices}
    lines = _block("sent since %s:" % since, [_alert_line(view) for view in sent])
    lines += _block("waiting to go out:", [_alert_line(view) for view in waiting])
    lines += _block("notices:", [str(text) for text in notices])
    return Output(data, "\n".join(lines) or "nothing new since %s" % since)


# ---- star, feedback, explain ---------------------------------------------------

def _star(rt: Runtime, code: str, starred: bool) -> Output:
    candidate = _find_candidate(rt.store, code)
    if not candidate["tracked"]:
        raise Problem("%s is not tracked (it is %s), so there is nothing to star"
                      % (candidate["code"], candidate["state"]))
    rt.store.update_candidate(candidate["id"], starred=starred)
    said = "%s is %s" % (candidate["code"], "starred: an auction gets reminders 60 and 10 minutes before it ends"
                         if starred else "no longer starred")
    return Output({"code": candidate["code"], "starred": starred}, said)


def cmd_star(rt: Runtime, args: argparse.Namespace) -> Output:
    return _star(rt, args.code, True)


def cmd_unstar(rt: Runtime, args: argparse.Namespace) -> Output:
    return _star(rt, args.code, False)


def replay(store: Store, watch: Mapping[str, Any],
           spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Which tracked listings a changed spec would drop, and which it would keep.

    Judged from what is stored (titles and per-field verdicts) with the tool's own rules: no
    model call and no eBay call. A change to what is being looked for is not replayed against
    old verdicts; it applies to listings judged from then on.
    """
    removed, kept = [], []
    for candidate in store.list_candidates(watch["id"], tracked=True):
        title = (candidate["listing"] or {}).get("title") or ""
        verdicts, _ = stored_evidence(store, candidate["id"])
        gone = rules.title_conflict(spec, title) or rules.decide(spec["requirements"], verdicts).decision == "reject"
        (removed if gone else kept).append(candidate)
    return removed, kept


def _patch_changes(patch: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: patch[key] for key in PATCH_KEYS if patch.get(key)}


def _patched_spec(old: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    return settle_requirements(old, apply_patch(old, patch), set(patch.get("set_requirement") or {}))


def _feedback_new(rt: Runtime, code: str, reason: str) -> Output:
    """Record "not a match", then ask a model what change to the spec would have avoided it."""
    store = rt.store
    candidate = _find_candidate(store, code)
    watch = store.get_watch(candidate["watch_id"])
    if watch is None:
        raise Problem("the watch for %s no longer exists" % candidate["code"])
    spec = normalize_spec(watch["spec"])
    title = (candidate["listing"] or {}).get("title") or ""
    untrack(store, candidate, "rejected", "user_feedback")
    system, parts, required = prompts.feedback(spec, reason, title)
    try:
        answer = rt.crusoe.chat_json("parse", system, parts, required=required, validator=prompts.validate_patch)
    except (CrusoeError, ConfigError) as problem:
        feedback_id = store.add_feedback("not_a_match", watch["id"], candidate["code"], reason)
        raise Problem("%s is marked as not a match (feedback %d), but no change to the watch could be proposed: %s"
                      % (candidate["code"], feedback_id, problem))
    patch = dict(_patch_changes(answer.data), explanation=model_words(answer.data["explanation"]))
    feedback_id = store.add_feedback("not_a_match", watch["id"], candidate["code"], reason, spec_patch=patch)
    changes = _patch_changes(patch)
    removed, kept = replay(store, watch, _patched_spec(spec, patch)) if changes else ([], [])
    data = {
        "feedback_id": feedback_id, "code": candidate["code"], "marked": "rejected",
        "watch": {"id": watch["id"], "name": watch["name"]},
        "patch": changes, "explanation": patch["explanation"],
        "would_remove": [row["code"] for row in removed], "would_keep": [row["code"] for row in kept],
        "apply_with": "eft feedback --apply %d" % feedback_id if changes else None,
    }
    lines = ["%s is marked as not a match and is no longer tracked" % candidate["code"]]
    if changes:
        lines += ["proposed change to watch %d: %s" % (watch["id"], json.dumps(changes, sort_keys=True)),
                  patch["explanation"],
                  "it would remove %d tracked listing(s) and keep %d" % (len(removed), len(kept)),
                  "apply it with: " + data["apply_with"]]
    else:
        lines += ["no change to the watch is proposed", patch["explanation"]]
    return Output(data, "\n".join(lines))


def _feedback_apply(rt: Runtime, feedback_id: int) -> Output:
    """Apply a change that `eft feedback CODE --reason` proposed earlier, once the user has agreed to it."""
    store = rt.store
    rows = {row["id"]: row for row in store.list_feedback()}
    row = rows.get(feedback_id)
    if row is None or row["label"] != "not_a_match" or not _patch_changes(row["spec_patch"] or {}):
        raise Problem("feedback %d has no proposed change to apply" % feedback_id)
    applied = "feedback %d" % feedback_id
    if any(other["label"] == "patch_applied" and other["reason"] == applied for other in rows.values()):
        raise Problem("the change from feedback %d has already been applied" % feedback_id)
    watch = store.get_watch(row["watch_id"]) if row["watch_id"] else None
    if watch is None:
        raise Problem("the watch that feedback %d was about no longer exists" % feedback_id)
    patch = row["spec_patch"]
    complaint = prompts.validate_patch(patch)
    if complaint:
        raise Problem("the stored change of feedback %d cannot be applied: %s" % (feedback_id, complaint))
    spec = _patched_spec(normalize_spec(watch["spec"]), patch)
    problems = validate_spec(spec)
    if problems:
        raise Problem("that change would leave watch %d unusable: %s" % (watch["id"], "; ".join(problems)))
    removed, kept = replay(store, watch, spec)
    with store.transaction():
        store.update_watch(watch["id"], spec=spec, plan=_replan(watch, spec))
        for candidate in removed:
            untrack(store, candidate, "rejected", "spec_changed")
        # The feedback table is a log: the approval is its own row, which also stops a second apply.
        store.add_feedback("patch_applied", watch["id"], row["candidate_code"], applied, spec_patch=patch,
                           approved=True)
    data = {"applied": feedback_id, "watch": {"id": watch["id"], "name": watch["name"]},
            "summary": spec_summary(spec), "spec": spec,
            "removed": [candidate["code"] for candidate in removed], "kept": [candidate["code"] for candidate in kept]}
    lines = ["applied the change from feedback %d to watch %d" % (feedback_id, watch["id"]), data["summary"],
             "removed %d tracked listing(s), kept %d" % (len(removed), len(kept))]
    return Output(data, "\n".join(lines))


def cmd_feedback(rt: Runtime, args: argparse.Namespace) -> Output:
    if args.apply is not None:
        if args.code or args.reason:
            raise UsageError("--apply takes a feedback id and nothing else")
        return _feedback_apply(rt, args.apply)
    if not args.code or not args.reason:
        raise UsageError("feedback needs a listing code and --reason TEXT, or --apply FEEDBACK_ID")
    return _feedback_new(rt, args.code, args.reason)


def _field_text(fields: Any) -> str:
    """One trace row's fields on a line: 'size: match (item specifics) "US Shoe Size: 10"'."""
    if not isinstance(fields, dict):
        return ""
    parts = []
    for name, value in fields.items():
        if isinstance(value, dict) and "verdict" in value and name in MATCH_FIELDS:
            said = "%s: %s" % (name, value["verdict"])
            if value.get("source"):
                said += " (%s)" % value["source"]
            if value.get("evidence"):
                said += " \"%s\"" % _short(value["evidence"])
            parts.append(said)
        else:
            parts.append("%s: %s" % (name, json.dumps(value) if isinstance(value, (dict, list)) else value))
    return ", ".join(parts)


def cmd_explain(rt: Runtime, args: argparse.Namespace) -> Output:
    """Print the stored funnel trace: each stage, the model that answered, and what it found."""
    store = rt.store
    candidate = _find_candidate(store, args.code)
    watch = store.get_watch(candidate["watch_id"]) or {}
    trace = [{"stage": row["stage"], "model": row["model_served"], "rule": row["rule_fired"], "fields": row["fields"],
              "latency_ms": row["latency_ms"], "tokens": row["tokens"], "at": row["created_at"]}
             for row in store.verdicts_for(candidate["id"])]
    data = {"code": candidate["code"], "watch": {"id": candidate["watch_id"], "name": watch.get("name")},
            "state": candidate["state"], "reason": candidate["reason"], "attempts": candidate["attempts"],
            "starred": bool(candidate["starred"]), "trace": trace,
            "listing": listing_view(candidate["listing"], candidate["fetched_at"], store.clock())}
    lines = ["%s for watch %s: %s (%s)" % (candidate["code"], watch.get("name") or candidate["watch_id"],
                                           candidate["state"], candidate["reason"] or "no reason recorded")]
    if data["listing"]:
        lines.append(_listing_text(data["listing"]))
    for step in trace:
        notes = [note for note in (step["model"], "rule: %s" % step["rule"] if step["rule"] else "",
                                   _field_text(step["fields"])) if note]
        lines.append("  %-15s %s" % (step["stage"], " · ".join(notes)))
    if not trace:
        lines.append("  no trace is stored for it")
    return Output(data, "\n".join(lines))


# ---- the background agent ------------------------------------------------------

def render_plist(settings: Settings, interval_minutes: int) -> str:
    """Fill in the launchd template with absolute paths and check that the result is a valid plist.

    launchd starts the job with an almost empty environment, so the Python that is running
    now, the launcher, the state directory and the .env file are all written in full.
    """
    values = {
        "LABEL": AGENT_LABEL,
        "PYTHON": os.path.abspath(sys.executable),
        "EFT": str(LAUNCHER),
        "INTERVAL_SECONDS": str(interval_minutes * 60),
        "STATE_DIR": os.path.abspath(str(settings.state_dir)),
        "ENV_FILE": os.path.abspath(str(settings.env_file)),
        "LOG": os.path.join(os.path.abspath(str(settings.state_dir)), LOG_FILE),
    }
    text = PLIST_TEMPLATE.read_text(encoding="utf-8")
    for name, value in values.items():
        text = text.replace("{{%s}}" % name, xml_escape(value))
    plistlib.loads(text.encode("utf-8"))
    return text


def _cron_line(settings: Settings, interval_minutes: int) -> str:
    """The crontab entry that runs the poller where there is no launchd."""
    every = "*/%d * * * *" % interval_minutes if interval_minutes < 60 else "0 */%d * * *" % (interval_minutes // 60)
    state_dir = os.path.abspath(str(settings.state_dir))
    return "%s EFT_STATE_DIR=%s EFT_ENV_FILE=%s %s %s tick >/dev/null 2>>%s" % (
        every, shlex.quote(state_dir), shlex.quote(os.path.abspath(str(settings.env_file))),
        shlex.quote(os.path.abspath(sys.executable)), shlex.quote(str(LAUNCHER)),
        shlex.quote(os.path.join(state_dir, LOG_FILE)))


def _launchd_domain() -> str:
    return "gui/%d" % os.getuid()


def cmd_agent_install(rt: Runtime, args: argparse.Namespace) -> Output:
    if args.interval < 1:
        raise UsageError("--interval takes a number of minutes, 1 or more")
    if rt.platform != "darwin":
        line = _cron_line(rt.settings, args.interval)
        note = "launchd is macOS only. Nothing was installed: add this line with `crontab -e`, or use a systemd timer."
        return Output({"installed": False, "cron": line, "note": note}, note + "\n" + line)
    path = _plist_path()
    text = render_plist(rt.settings, args.interval)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as problem:
        raise Problem("cannot write %s (%s)" % (path, problem.__class__.__name__))
    # A copy that is already loaded has to go first. This fails, harmlessly, when there is none.
    rt.run([LAUNCHCTL, "bootout", "%s/%s" % (_launchd_domain(), AGENT_LABEL)])
    code, said = rt.run([LAUNCHCTL, "bootstrap", _launchd_domain(), str(path)])
    if code != 0:
        raise Problem("wrote %s but launchctl did not load it (%s). Load it from your own terminal with: "
                      "launchctl bootstrap %s %s"
                      % (path, said or "exit %d" % code, _launchd_domain(), shlex.quote(str(path))))
    note = ("The agent reads keys from %s. Variables that are only exported in your shell are not visible to it."
            % rt.settings.env_file)
    if rt.settings.env_file_problem == "missing":
        note += " That file does not exist yet, so the agent has no keys until you create it."
    data = {"installed": True, "plist": str(path), "interval_minutes": args.interval, "note": note}
    return Output(data, "the poller runs `eft tick` every %d minutes (%s)\n%s" % (args.interval, path, note))


def _remove_agent(rt: Runtime) -> bool:
    """Unload the agent and delete its plist. True when there was one."""
    path = _plist_path()
    if rt.platform != "darwin" or not path.exists():
        return False
    rt.run([LAUNCHCTL, "bootout", "%s/%s" % (_launchd_domain(), AGENT_LABEL)])
    try:
        path.unlink()
    except OSError as problem:
        raise Problem("cannot delete %s (%s)" % (path, problem.__class__.__name__))
    return True


def cmd_agent_uninstall(rt: Runtime, args: argparse.Namespace) -> Output:
    removed = _remove_agent(rt)
    said = "the poller is unloaded and its plist deleted" if removed else "no agent was installed"
    if rt.platform != "darwin":
        said = "launchd is macOS only; remove the `eft tick` line from your crontab or systemd timer"
    return Output({"removed": removed}, said)


def cmd_agent_status(rt: Runtime, args: argparse.Namespace) -> Output:
    if rt.platform != "darwin":
        note = "launchd is macOS only; `eft status` shows when the poller last ran"
        return Output({"installed": None, "loaded": None, "note": note}, note)
    path = _plist_path()
    code, _ = rt.run([LAUNCHCTL, "print", "%s/%s" % (_launchd_domain(), AGENT_LABEL)])
    data = {"installed": path.exists(), "loaded": code == 0, "plist": str(path)}
    said = "plist %s, agent %s" % ("installed" if data["installed"] else "not installed",
                                   "loaded" if data["loaded"] else "not loaded")
    return Output(data, said)


# ---- uninstall -----------------------------------------------------------------

def _remove_references(refs_dir: Path) -> None:
    try:
        entries = list(refs_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        if REFERENCE_NAME.fullmatch(entry.name) and (entry.is_symlink() or not entry.is_dir()):
            with suppress(OSError):
                entry.unlink()
    with suppress(OSError):
        refs_dir.rmdir()


def _remove_state(settings: Settings) -> bool:
    """Delete the files this tool keeps in its state directory. True when the directory is gone too.

    Only its own files are deleted, by name, so an EFT_STATE_DIR that points somewhere
    shared loses nothing else. The directory itself goes only once it is empty.
    """
    state_dir = settings.state_dir
    for name in ("tracker.db", "tracker.db-wal", "tracker.db-shm", "tracker.db-journal", LOCK_FILE, LOG_FILE):
        try:
            (state_dir / name).unlink()
        except FileNotFoundError:
            pass
    _remove_references(settings.refs_dir)
    try:
        state_dir.rmdir()
    except OSError:
        return not state_dir.exists()
    return True


def cmd_uninstall(rt: Runtime, args: argparse.Namespace) -> Output:
    """Stop the poller and destroy everything stored, which is what eBay's license asks for on stopping."""
    settings = rt.settings
    if not args.yes:
        raise Problem("this stops the background poller and deletes every watch and all stored data in %s. "
                      "Run it again with --yes to go ahead." % settings.state_dir)
    agent_removed = _remove_agent(rt)
    store = rt.store
    left = strip_cards(rt, store.list_candidates())
    store.wipe()
    rt.discard_store()
    directory_removed = _remove_state(settings)
    manual = ["revoke the eBay keyset and the Crusoe key if you no longer need them",
              "delete the deletion-notice Worker (npx wrangler delete) and the Slack app"]
    if left:
        manual.insert(0, "delete %d Slack message(s) that could not be rewritten and still show listing details" % left)
    data = {"agent_removed": agent_removed, "store_wiped": True, "state_dir": str(settings.state_dir),
            "state_dir_removed": directory_removed, "messages_left": left, "manual_steps": manual}
    lines = ["all stored data is deleted" + (" and %s is removed" % settings.state_dir if directory_removed else
                                             "; %s holds other files and was left in place" % settings.state_dir)]
    return Output(data, "\n".join(lines + _block("left for you to do:", manual)))


# ---- the parser ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # --json is accepted before the command and after it. The second form must not reset the first.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="print one JSON document instead of text")
    parser = argparse.ArgumentParser(
        prog="eft", description="Track fashion listings on eBay through its official API. This tool reports "
                                "prices and bids; it never bids, makes offers or buys.")
    parser.add_argument("--json", action="store_true", help="print one JSON document instead of text")
    parser.add_argument("--version", action="version", version="eft " + __version__)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True

    def command(group: Any, name: str, run: Callable[[Runtime, argparse.Namespace], Output],
                summary: str) -> argparse.ArgumentParser:
        sub = group.add_parser(name, parents=[shared], help=summary, description=summary)
        sub.set_defaults(run=run)
        return sub

    command(commands, "status", cmd_status, "configuration, watches, the last poll and what is still to set up")
    command(commands, "doctor", cmd_doctor, "check the setup").add_argument(
        "area", nargs="?", default="all", choices=doctor.AREAS + ("all",))

    add = command(commands, "add", cmd_add, "draft a watch from a photo, a description or an eBay link")
    source = add.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", metavar="PATH", help="a photo of the item (yours or a retailer's, never eBay's)")
    source.add_argument("--text", help="a description, or text copied from a retailer's page")
    source.add_argument("--url", help="an eBay item, product or search link; it is parsed, never opened")
    add.add_argument("--note", help="what the user said about the photo")
    add.add_argument("--name", help="a name for the watch")

    watch = commands.add_parser("watch", help="list, change, activate, pause or remove watches")
    actions = watch.add_subparsers(dest="action", metavar="ACTION")
    actions.required = True
    command(actions, "list", cmd_watch_list, "all watches")
    for name, run, summary in (
        ("show", cmd_watch_show, "one watch in full"),
        ("activate", cmd_watch_activate, "start searching for a watch"),
        ("pause", cmd_watch_pause, "stop searching for a watch"),
        ("remove", cmd_watch_remove, "delete a watch and everything stored for it"),
        ("hot", cmd_watch_hot, "search a watch as often as the call budget allows"),
        ("normal", cmd_watch_normal, "search a watch at the normal cadence"),
    ):
        command(actions, name, run, summary).add_argument("id", metavar="ID", help="watch id or name")
    setter = command(actions, "set", cmd_watch_set, "change a watch's spec")
    setter.add_argument("id", metavar="ID", help="watch id or name")
    setter.add_argument("settings", nargs="+", metavar="KEY=VALUE",
                        help="one of %s, or require.<field>=required|optional|na" % ", ".join(SETTABLE))

    dryrun = command(commands, "dryrun", cmd_dryrun, "preview: one search and its decisions, nothing kept")
    dryrun.add_argument("id", metavar="ID", help="watch id or name")
    dryrun.add_argument("--limit", type=int, default=DRYRUN_LIMIT, metavar="N", help="newest N listings (default 10)")

    command(commands, "tick", cmd_tick, "run one poll now")
    command(commands, "inbox", cmd_inbox, "alerts sent since the last look, and any still waiting")
    for name, run, summary in (
        ("star", cmd_star, "single a listing out: reminders before its auction ends"),
        ("unstar", cmd_unstar, "undo star"),
        ("explain", cmd_explain, "why a listing was matched or dropped"),
    ):
        command(commands, name, run, summary).add_argument("code", metavar="CODE")

    feedback = command(commands, "feedback", cmd_feedback, "say a listing is not a match and get a proposed fix")
    feedback.add_argument("code", nargs="?", metavar="CODE")
    feedback.add_argument("--reason", help="why it is not a match, in the user's words")
    feedback.add_argument("--apply", type=int, metavar="FEEDBACK_ID", help="apply a change proposed earlier")

    agent = commands.add_parser("agent", help="the background job that runs `eft tick`")
    agent_actions = agent.add_subparsers(dest="action", metavar="ACTION")
    agent_actions.required = True
    install = command(agent_actions, "install", cmd_agent_install, "install and load the launchd user agent")
    install.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_MINUTES, metavar="MINUTES")
    command(agent_actions, "uninstall", cmd_agent_uninstall, "unload and delete the agent")
    command(agent_actions, "status", cmd_agent_status, "whether the agent is installed and loaded")

    command(commands, "uninstall", cmd_uninstall, "stop the poller and delete all stored data").add_argument(
        "--yes", action="store_true", help="really delete everything")
    return parser


HANDLED = (Problem, ConfigError, EbayError, CrusoeError, alerts.DeliveryError, BlockedUrl, TransportError)


def _fail(message: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"ok": False, "error": message}, indent=2))
    print("eft: %s" % message, file=sys.stderr)
    return 1


def main(argv: Optional[Sequence[str]] = None, runtime: Optional[Runtime] = None) -> int:
    """Run one command and return its exit code. `runtime` lets tests supply fake clients."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:      # argparse has printed its usage message or the help text
        return stop.code if isinstance(stop.code, int) else 2
    as_json = bool(getattr(args, "json", False))
    rt = runtime or Runtime()
    try:
        output = args.run(rt, args)
    except UsageError as problem:
        print("eft: error: %s" % problem, file=sys.stderr)
        return 2
    except HANDLED as problem:
        return _fail(str(problem), as_json)
    finally:
        if runtime is None:
            rt.close()
    print(json.dumps(output.data, indent=2, sort_keys=True) if as_json else output.text)
    return output.code
