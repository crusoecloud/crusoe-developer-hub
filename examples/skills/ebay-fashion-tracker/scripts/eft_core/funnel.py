"""The matching funnel: is this search result the item a watch is looking for?

Each new listing passes through stages, cheapest first. Model calls are plentiful and eBay calls
are scarce, so a model screens titles before an eBay detail call is spent on any of them.

  rules           the user's must-not terms and a clear department conflict in the title
  title_triage    text model, ten titles per call; only a reasoned "no" drops a listing
  photo_precheck  an unclear title and a reference photo to go by: compare the two photos
  detail_fetch    the full listing, or every variation of a multi-size listing
  spec_check      size, style code, department and price ceiling, decided by code
  detail_verify   text model: match, mismatch or unknown per field, each with a verbatim quote
  tag_read        a required size or style code still unproven: read tags in the other photos
  decision        rules.decide

The models supply evidence and `rules.decide` makes the call. Evidence is merged strongest
first: what code can check (item specifics, a variation, the title), then quoted listing text,
then the photo comparison, then a tag. A later source only fills in what is still unknown.

A run is recorded in one transaction after its last stage. If the run raises (a rejected key,
a bug), nothing has been written, so the next search finds every listing again and no verdict
is left in the store without the result that reports it.

What is stored follows eBay's license terms: a tracked candidate (alert, maybe, parked) keeps a
small snapshot for its card, and everything else is an id, a state and a reason code. Photos are
held in memory and never stored. The funnel neither enqueues alerts nor schedules refreshes:
the tick does both.
"""
from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import prompts, rules
from .errors import (
    ChatResult, CrusoeAuthError, CrusoeError, EbayAuthError, EbayBudgetExceeded, EbayError,
    EbayGroupRequired, EbayNotFound, EbayUnavailable,
)
from .http import BlockedUrl, TransportError, UrllibTransport
from .rules import MATCH, MISMATCH, UNKNOWN
from .spec import MATCH_FIELDS, normalize_spec
from .store import iso, parse_iso, strip_evidence

TRIAGE_BATCH = 10
MAX_TAG_PHOTOS = 4
PHOTO_HOST = "ebayimg.com"
MAX_PHOTO_BYTES = 10 * 1024 * 1024
# Failed attempts (a model with no valid answer, a listing eBay cannot return) before giving up.
MAX_ATTEMPTS = 3
# How long an untracked row (rejected, pending, unavailable) is kept when the listing has no end date.
UNTRACKED_KEEP_DAYS = 14

# Stage names as they appear in the stored trace, in the order they run.
STAGES = ("rules", "title_triage", "photo_precheck", "detail_fetch", "spec_check", "detail_verify",
          "tag_read", "decision")

REASON_MODEL_ERROR = "model_error"
REASON_QUARANTINED = "quarantined"
REASON_BUDGET = "ebay_budget_exceeded"

TRACKED_DECISIONS = ("alert", "maybe", "parked")
# Candidate states that mean "no verdict yet".
UNFINISHED_STATES = ("pending", "unavailable")
_STATE_FOR = {"alert": "alerted", "maybe": "maybe", "reject": "rejected", "parked": "parked", "pending": "pending"}
# What rules.title_conflict returns when a title carries a kids' size marker the watch does not want.
_TITLE_DEPARTMENT_CONFLICT = "mismatch:department"

# The Summary/Detail keys an alert card needs. Never the description or the item specifics.
# For a multi-size listing `item_id` and `web_url` are those of the variation that was judged,
# which can differ from the id the search returned (the candidate row's own item_id).
SNAPSHOT_KEYS = (
    "item_id", "legacy_item_id", "item_group_id", "title", "web_url", "image_url", "price", "shipping",
    "current_bid", "min_next_bid", "bid_count", "reserve_met", "buying_options", "end_at", "condition",
    "availability", "relisted",
)


@dataclass
class FunnelResult:
    """What the funnel concluded about one listing.

    decision is alert, maybe, reject, parked or pending. "pending" means there is no verdict yet:
    a model call failed (reason "model_error"), eBay could not return the listing (the stored
    row's state is "unavailable"), or the eBay budget ran out before the listing was reached
    (reason "ebay_budget_exceeded"; nothing was stored, so `candidate` is None).

    `summary` and `detail` are the API's own dicts, handed back for the caller to use in memory.
    Only `candidate` is what the store holds.
    """
    item_id: str
    decision: str
    reason: str
    fields: Dict[str, str] = field(default_factory=dict)       # {"size": "match", ...}
    sources: Dict[str, str] = field(default_factory=dict)      # {"size": "item specifics", ...}
    unproven: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    proof: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    detail: Optional[Dict[str, Any]] = None
    candidate: Optional[Dict[str, Any]] = None                 # the stored row


class _ModelFailed(Exception):
    """A model stage produced no usable answer. Carries the stage name for the trace."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _entry(verdict: str, source: Optional[str] = None, evidence: Optional[str] = None) -> Dict[str, str]:
    """One field verdict in the form the trace stores: verdict, where it came from, what was read."""
    entry = {"verdict": verdict}
    if verdict != UNKNOWN and source:
        entry["source"] = source
    if verdict != UNKNOWN and evidence:
        entry["evidence"] = evidence
    return entry


@dataclass
class _Case:
    """One listing on its way through the funnel. Lives in memory until the run is recorded."""
    summary: Dict[str, Any]
    row: Optional[Dict[str, Any]]            # the candidate stored by an earlier run, if any
    attempts: int
    fetched_at: str
    triage: str = "maybe"
    photo: Dict[str, str] = field(default_factory=dict)
    detail: Optional[Dict[str, Any]] = None
    multi_size: bool = False
    size_out_of_stock: bool = False
    over_ceiling: bool = False
    evidence: Dict[str, Dict[str, str]] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    unproven: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    decision: Optional[str] = None           # None while the listing is still being judged
    reason: str = ""
    state: str = ""

    @property
    def open(self) -> bool:
        return self.decision is None

    def close(self, decision: str, reason: str, state: Optional[str] = None) -> None:
        self.decision, self.reason = decision, reason
        self.state = state or _STATE_FOR[decision]

    def note(self, stage: str, answer: Optional[ChatResult] = None, fields: Optional[Dict[str, Any]] = None,
             rule: Optional[str] = None, share: int = 1) -> None:
        """Add one row to the trace. `share` splits a batched call's tokens between its listings."""
        step: Dict[str, Any] = {"stage": stage, "fields": fields, "rule_fired": rule}
        if answer is not None:
            step["model_served"] = answer.model_served
            step["latency_ms"] = answer.latency_ms
            step["tokens"] = (answer.prompt_tokens + answer.completion_tokens) // share
        self.trace.append(step)

    def verdict(self, name: str) -> str:
        return self.evidence.get(name, {}).get("verdict", UNKNOWN)

    def verdicts(self) -> Dict[str, str]:
        return {name: self.verdict(name) for name in MATCH_FIELDS}

    def settle(self, name: str, verdict: str, source: str, evidence: Optional[str] = None) -> None:
        """Record a field verdict, unless a stronger source settled the field earlier."""
        if verdict != UNKNOWN and self.verdict(name) == UNKNOWN:
            self.evidence[name] = _entry(verdict, source, evidence)


# ---- helpers that need no clients ----------------------------------------------

def is_photo_host(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host == PHOTO_HOST or host.endswith("." + PHOTO_HOST)


def fetch_image(url: str, transport: Any = None) -> bytes:
    """Download one listing photo into memory, from eBay's image host only."""
    if not is_photo_host(url):
        raise BlockedUrl("listing photos are only fetched from %s" % PHOTO_HOST)
    transport = transport if transport is not None else UrllibTransport()
    response = transport.request("GET", url, headers={"Accept": "image/*"}, timeout=20.0)
    if not response.ok:
        raise TransportError("image request returned HTTP %d" % response.status)
    if not response.headers.get("content-type", "").lower().startswith("image/"):
        raise TransportError("image request did not return an image")
    if len(response.body) > MAX_PHOTO_BYTES:
        raise TransportError("image is larger than %d bytes" % MAX_PHOTO_BYTES)
    return response.body


def snapshot(listing: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a listing that is stored while a candidate is tracked."""
    return {key: listing.get(key) for key in SNAPSHOT_KEYS}


def _unfinished(row: Mapping[str, Any]) -> bool:
    """True for a stored row that never got a verdict and is still worth another attempt.

    A tracked row in state "unavailable" belongs to the tick's refresh logic, not to the funnel.
    """
    return row["state"] in UNFINISHED_STATES and not row["tracked"] and row["attempts"] < MAX_ATTEMPTS


def _urgency(case: _Case) -> Tuple[bool, float]:
    """Soonest-ending first, so a run cut short by the eBay budget has done the urgent work."""
    end = parse_iso(case.summary.get("end_at"))
    return (end is None, end.timestamp() if end else 0.0)


def asking_price(listing: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The amount the ceiling is compared with: the current bid on an auction, else the price."""
    if "AUCTION" in (listing.get("buying_options") or []):
        return listing.get("current_bid") or listing.get("price")
    return listing.get("price")


def _size_label(size: Mapping[str, Any]) -> str:
    system = size.get("system")
    return "%s %s" % (system, size["value"]) if system in ("US", "UK", "EU", "JP", "CM") else str(size["value"])


def _stated_sizes(detail: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    """The sizes a listing's item specifics state in the watch's own size system: 'US 10'."""
    system = (spec.get("size") or {}).get("system")
    stated = rules.sizes_from_aspects(detail.get("aspects") or {})
    return ", ".join(_size_label(size) for size in stated if size["system"] == system)


def _pick_variation(
    spec: Mapping[str, Any], summary: Mapping[str, Any], variations: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, bool]:
    """Choose the variation of a multi-size listing to judge: (detail, size verdict, sold out).

    The variation whose item specifics state the wanted size is the listing, as far as this
    watch is concerned. If that size exists but is out of stock, the listing is parked. Size is
    a mismatch only when every variation states a different size in the watch's size system.
    """
    verdicts = [
        rules.check_size(spec.get("size"), rules.sizes_from_aspects(variation.get("aspects") or {}))
        for variation in variations
    ]
    in_size = [variation for variation, verdict in zip(variations, verdicts) if verdict == MATCH]
    if in_size:
        in_stock = [variation for variation in in_size if variation.get("availability") != "OUT_OF_STOCK"]
        return (in_stock or in_size)[0], MATCH, not in_stock
    shown = [variation for variation in variations if variation.get("item_id") == summary.get("item_id")]
    not_offered = all(verdict == MISMATCH for verdict in verdicts)
    return (shown or list(variations))[0], MISMATCH if not_offered else UNKNOWN, False


def _settled_by(check: Callable[..., str], spec: Mapping[str, Any], aspects: Mapping[str, Any],
                title: str) -> Tuple[str, str]:
    """Run a rule that reads item specifics and the title, and say which of the two settled it."""
    verdict = check(spec, aspects, title)
    return verdict, "item specifics" if check(spec, aspects) == verdict else "title"


def _backed_by_quote(verdict: str, quote_states_wanted: bool) -> str:
    """For a field with an exact wanted value, the quote has to agree with the verdict it supports."""
    if verdict == MATCH:
        return MATCH if quote_states_wanted else UNKNOWN
    if verdict == MISMATCH:
        return UNKNOWN if quote_states_wanted else MISMATCH
    return UNKNOWN


def _accepted(spec: Mapping[str, Any], case: _Case, name: str, answer: Mapping[str, str]) -> str:
    """The text model's verdict for one field, after the conditions code can check for itself.

    Size and style code have an exact wanted value, so a "match" must quote that value and a
    "mismatch" must not. A size "match" also needs normal strictness: on a strict watch only
    item specifics, a variation or a tag prove size. In a multi-size listing the text of one
    variation says nothing about size, so the model is not asked to settle it at all.
    """
    verdict, quote = answer["verdict"], answer["quote"]
    if name == "size":
        if case.multi_size or not spec.get("size"):
            return UNKNOWN
        verdict = _backed_by_quote(verdict, prompts.states_size([quote], spec["size"]))
        return UNKNOWN if verdict == MATCH and spec["strictness"] == "strict" else verdict
    if name == "style_code":
        return _backed_by_quote(verdict, rules.check_style_code(spec, {}, quote) == MATCH)
    return verdict


def _tag_proof(spec: Mapping[str, Any], tag: Mapping[str, Any]) -> Dict[str, str]:
    """What one tag photo proves: field name -> the text read off the tag, for fields it matches."""
    proof: Dict[str, str] = {}
    if tag["size"] and rules.check_size(spec.get("size"), [tag["size"]]) == MATCH:
        proof["size"] = _size_label(tag["size"])
    if rules.check_style_code(spec, {"Style Code": [tag["style_code"] or ""]}) == MATCH:
        proof["style_code"] = tag["style_code"]
    return proof


def _apply_rule(spec: Mapping[str, Any], case: _Case) -> rules.Decision:
    return rules.decide(spec["requirements"], case.verdicts(), case.flags,
                        over_ceiling=case.over_ceiling, size_out_of_stock=case.size_out_of_stock)


def _conclude(case: _Case, decision: rules.Decision) -> None:
    """Close a case with the decision rule's answer and add it, with the merged fields, to the trace."""
    case.unproven = list(decision.unproven)
    case.note("decision", fields={name: case.evidence.get(name, _entry(UNKNOWN)) for name in MATCH_FIELDS},
              rule=decision.reason)
    case.close(decision.decision, decision.reason)


def _totals(answers: Sequence[ChatResult]) -> ChatResult:
    """Add up a stage that made several calls, so it still writes a single trace row."""
    last = answers[-1]
    return ChatResult(
        data={}, model_requested=last.model_requested, model_served=last.model_served,
        swapped=any(answer.swapped for answer in answers),
        latency_ms=sum(answer.latency_ms for answer in answers),
        prompt_tokens=sum(answer.prompt_tokens for answer in answers),
        completion_tokens=sum(answer.completion_tokens for answer in answers),
    )


# ---- the funnel ----------------------------------------------------------------

class Funnel:
    """Judges search results for one watch. `ebay` and `crusoe` are the two API clients.

    `image_fetcher(url) -> bytes` downloads a listing photo. Only URLs the eBay API returned are
    passed to it, and the bytes go straight to a model call.
    """

    def __init__(self, ebay: Any, crusoe: Any, store: Any,
                 image_fetcher: Optional[Callable[[str], bytes]] = None) -> None:
        self.ebay = ebay
        self.crusoe = crusoe
        self.store = store
        self.image_fetcher = image_fetcher or fetch_image

    def run(self, watch: Mapping[str, Any], summaries: Sequence[Dict[str, Any]],
            recheck: bool = False) -> List[FunnelResult]:
        """Judge search results for a watch and record the outcome. One result per listing judged.

        Listings this watch has already judged are skipped unless `recheck` is set. A stored row
        without a verdict (a model call or the detail fetch failed last time) is not "judged":
        it is taken up again whenever a search returns the listing, up to MAX_ATTEMPTS times.
        A re-evaluation that fails leaves the earlier verdict and its snapshot as they were.

        CrusoeAuthError and EbayAuthError propagate, and then nothing from this run is stored.
        When the eBay budget runs out the run stops early: what was decided is stored, and the
        listings not reached come back as pending results without a candidate row.
        """
        spec = normalize_spec(watch["spec"])
        cases = self._open_cases(watch["id"], summaries, recheck)
        for case in cases:
            self._check_title(spec, case)
        self._triage(spec, [case for case in cases if case.open])
        shortlist = sorted((case for case in cases if case.open), key=_urgency)
        reference = self._reference_photo(spec) if any(case.triage == "maybe" for case in shortlist) else None
        try:
            for case in shortlist:
                self._evaluate(spec, case, reference)
        except EbayBudgetExceeded:
            # The case that hit the cap and every case after it stay open. Nothing is written
            # for them, so the next search finds them again.
            pass
        with self.store.transaction():
            return [self._result(spec, case, None if case.open else self._store(watch["id"], case))
                    for case in cases]

    def _open_cases(self, watch_id: int, summaries: Sequence[Dict[str, Any]], recheck: bool) -> List[_Case]:
        """One case per listing this run should judge, in the order the search returned them."""
        seen = self.store.seen_item_ids(watch_id)
        started = self.store.now_iso()
        cases: List[_Case] = []
        taken = set()
        for summary in summaries:
            item_id = summary["item_id"]
            if item_id in taken:
                continue
            taken.add(item_id)
            row = self.store.get_candidate(watch_id, item_id) if item_id in seen else None
            if row is not None and not recheck and not _unfinished(row):
                continue
            cases.append(_Case(summary=summary, row=row, attempts=row["attempts"] if row else 0,
                               fetched_at=started))
        return cases

    # ---- stage 0: rules ----------------------------------------------------------

    @staticmethod
    def _check_title(spec: Mapping[str, Any], case: _Case) -> None:
        """Drop a listing whose title breaks one of the user's own rules. No model, no eBay call.

        A price over the ceiling is only noted here: the listing still has to be judged, so that
        it can be parked and brought back when its price comes down.
        """
        summary = case.summary
        conflict = rules.title_conflict(spec, summary.get("title") or "")
        if conflict == _TITLE_DEPARTMENT_CONFLICT and summary.get("item_group_id"):
            # A size in the title never rejects a multi-size listing; its variations decide.
            conflict = None
        over = rules.over_ceiling(spec, asking_price(summary), summary.get("shipping"))
        case.note("rules", fields={"over_ceiling": over}, rule=conflict)
        if conflict:
            case.close("reject", conflict)

    # ---- stage 1: title triage ---------------------------------------------------

    def _triage(self, spec: Mapping[str, Any], cases: Sequence[_Case]) -> None:
        """Screen titles in batches. A failed batch leaves its listings pending, not dropped."""
        for start in range(0, len(cases), TRIAGE_BATCH):
            batch = cases[start:start + TRIAGE_BATCH]
            titles = [case.summary.get("title") or "" for case in batch]
            try:
                answer = self._ask("title_triage", "triage", prompts.triage(spec, titles), prompts.validate_triage)
            except _ModelFailed as failure:
                for case in batch:
                    self._model_failed(spec, case, failure.stage)
                continue
            for case, (verdict, reason) in zip(batch, prompts.read_triage(answer.data, len(batch))):
                case.note("title_triage", answer, fields={"verdict": verdict, "reason": reason}, share=len(batch))
                if verdict == "no":
                    case.close("reject", "triage:" + reason)
                else:
                    case.triage = verdict

    # ---- stages 2 to 6, one listing at a time ------------------------------------

    def _evaluate(self, spec: Mapping[str, Any], case: _Case, reference: Optional[Dict[str, Any]]) -> None:
        """Take one listing that passed triage through the remaining stages and the decision rule."""
        try:
            if case.triage == "maybe" and reference is not None:
                self._photo_precheck(spec, case, reference)
            if case.open:
                self._fetch_detail(spec, case)
            if case.open:
                self._check_spec(spec, case)
            if case.open:
                self._verify(spec, case)
                self._decide(spec, case)
        except _ModelFailed as failure:
            self._model_failed(spec, case, failure.stage)

    def _photo_precheck(self, spec: Mapping[str, Any], case: _Case, reference: Dict[str, Any]) -> None:
        """Stage 2: the title was unclear, so compare the search-result photo with the user's own.

        A clear "no" saves the eBay detail call. A "yes" is kept as evidence of model and
        colorway (never of authenticity) for fields the listing text leaves open.
        """
        photo = self._photo(case.summary.get("image_url"))
        if photo is None:
            return
        answer = self._ask("photo_precheck", "compare", prompts.compare(spec, reference, photo),
                           prompts.validate_compare)
        seen = {key: answer.data[key] for key in ("same_model", "same_colorway")}
        case.note("photo_precheck", answer, fields=seen)
        if seen["same_model"] == "no":
            case.close("reject", "photo:different_model")
        elif seen["same_colorway"] == "no" and spec["requirements"].get("colorway") != "na":
            case.close("reject", "photo:different_colorway")
        else:
            case.photo = seen

    def _fetch_detail(self, spec: Mapping[str, Any], case: _Case) -> None:
        """Stage 3: spend one eBay call on the listing, or on all variations of a multi-size one."""
        try:
            listings, case.multi_size = self._load_listing(case.summary)
            if not listings:
                raise EbayNotFound("the item group has no variations")
        except (EbayAuthError, EbayBudgetExceeded):
            raise
        except EbayError as problem:
            reason = ("ebay_not_found" if isinstance(problem, EbayNotFound)
                      else "ebay_unavailable" if isinstance(problem, EbayUnavailable) else "ebay_error")
            case.note("detail_fetch", rule=reason)
            case.attempts += 1
            case.close("pending", reason, state="unavailable")
            return
        case.fetched_at = self.store.now_iso()
        if case.multi_size:
            case.detail, size, case.size_out_of_stock = _pick_variation(spec, case.summary, listings)
            case.settle("size", size, "variation", _stated_sizes(case.detail, spec))
        else:
            case.detail = listings[0]
            case.size_out_of_stock = case.detail.get("availability") == "OUT_OF_STOCK"
        case.note("detail_fetch", fields={"multi_size": case.multi_size},
                  rule="size_out_of_stock" if case.size_out_of_stock else None)

    def _load_listing(self, summary: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
        """Return (details, multi_size): one detail, or every variation of an item group."""
        group_id = summary.get("item_group_id")
        if not group_id:
            try:
                return [self.ebay.get_item(summary["item_id"], purpose="detail")], False
            except EbayGroupRequired as parent:
                if not parent.item_group_id:
                    raise
                group_id = parent.item_group_id
        return list(self.ebay.get_items_by_item_group(group_id, purpose="detail")), True

    def _check_spec(self, spec: Mapping[str, Any], case: _Case) -> None:
        """Stage 4: settle what item specifics, the title and the price can settle without a model.

        A mismatch found here rejects the listing before a model call is spent on it.
        """
        detail = case.detail or {}
        aspects = detail.get("aspects") or {}
        title = detail.get("title") or ""
        if not case.multi_size:
            size = rules.check_size(spec.get("size"), rules.sizes_from_aspects(aspects))
            case.settle("size", size, "item specifics", _stated_sizes(detail, spec))
        case.settle("style_code", *_settled_by(rules.check_style_code, spec, aspects, title))
        # In a multi-size listing a kids' size in the title is one size of several, not the department.
        case.settle("department", *_settled_by(rules.check_department, spec, aspects,
                                                "" if case.multi_size else title))
        case.over_ceiling = rules.over_ceiling(spec, asking_price(detail), detail.get("shipping"))
        case.note("spec_check", fields=dict(case.evidence), rule="over_ceiling" if case.over_ceiling else None)
        early = _apply_rule(spec, case)
        if early.decision == "reject":
            _conclude(case, early)

    def _verify(self, spec: Mapping[str, Any], case: _Case) -> None:
        """Stage 5: a text model reads the whole listing and answers per field, quoting the listing."""
        text = prompts.listing_text(case.detail or {})
        answer = self._ask("detail_verify", "verify", prompts.verify(spec, text), prompts.validate_verify)
        answers, case.flags = prompts.read_verify(answer.data, text)
        accepted = {}
        for name in MATCH_FIELDS:
            verdict = _accepted(spec, case, name, answers[name])
            accepted[name] = _entry(verdict, "listing text", answers[name]["quote"])
            case.settle(name, verdict, "listing text", answers[name]["quote"])
        for name in ("model", "colorway"):
            if case.photo.get("same_" + name) == "yes":
                case.settle(name, MATCH, "photo")
        case.note("detail_verify", answer, fields=accepted,
                  rule="flags:" + ",".join(case.flags) if case.flags else None)

    def _decide(self, spec: Mapping[str, Any], case: _Case) -> None:
        """Apply the decision rule, reading tags first when a required size or style code is unproven.

        Tags are not read for a listing that is rejected or parked anyway, and not for size in a
        multi-size listing, whose photos are shared by every size.
        """
        decision = _apply_rule(spec, case)
        readable = [name for name in ("size", "style_code")
                    if name in decision.unproven and not (name == "size" and case.multi_size)]
        if decision.decision == "maybe" and readable:
            self._read_tags(spec, case, readable)
            decision = _apply_rule(spec, case)
        _conclude(case, decision)

    def _read_tags(self, spec: Mapping[str, Any], case: _Case, wanted: Sequence[str]) -> None:
        """Stage 6: look for a size tag or box label in the listing's other photos.

        One photo per call, MAX_TAG_PHOTOS at most. A tag can prove a field but never reject a
        listing: a photo may show another item's label, and a misread must not cost a real match.
        """
        answers: List[ChatResult] = []
        for url in ((case.detail or {}).get("additional_image_urls") or [])[:MAX_TAG_PHOTOS]:
            photo = self._photo(url)
            if photo is None:
                continue
            answers.append(self._ask("tag_read", "tag_read", prompts.tag_read(spec, photo),
                                     prompts.validate_tag_read))
            proof = _tag_proof(spec, prompts.read_tag(answers[-1].data))
            for name in wanted:
                if name in proof:
                    case.settle(name, MATCH, "tag", proof[name])
            if all(case.verdict(name) == MATCH for name in wanted):
                break
        if answers:
            case.note("tag_read", _totals(answers),
                      fields={name: case.evidence.get(name, _entry(UNKNOWN)) for name in wanted})

    # ---- model calls and photos --------------------------------------------------

    def _ask(self, stage: str, role: str, prompt: prompts.Prompt,
             validator: Callable[[Mapping[str, Any]], Optional[str]]) -> ChatResult:
        """One model call. Anything short of a valid answer becomes _ModelFailed for the stage.

        The client validates answers too. Checking again here keeps "never act on malformed model
        output" true whichever client is plugged in.
        """
        system, parts, required = prompt
        try:
            answer = self.crusoe.chat_json(role, system, parts, required=required, validator=validator)
        except CrusoeAuthError:
            raise
        except CrusoeError as problem:
            raise _ModelFailed(stage) from problem
        data = answer.data
        if not isinstance(data, dict) or any(key not in data for key in required) or validator(data):
            raise _ModelFailed(stage)
        return answer

    @staticmethod
    def _model_failed(spec: Mapping[str, Any], case: _Case, stage: str) -> None:
        """No valid answer: leave the listing pending for a later run, and stop after MAX_ATTEMPTS.

        A listing that keeps failing is moved to "maybe" so a person sees it. It never alerts on
        an answer that was not valid, and it is never silently dropped.
        """
        case.note(stage, rule=REASON_MODEL_ERROR)
        case.attempts += 1
        if case.attempts < MAX_ATTEMPTS:
            case.close("pending", REASON_MODEL_ERROR)
        else:
            case.unproven = rules.decide(spec["requirements"], case.verdicts()).unproven
            case.close("maybe", REASON_QUARANTINED)

    def _image_part(self, load: Callable[[], bytes]) -> Optional[Dict[str, Any]]:
        """Load a photo and wrap it for a model call, or return None when it cannot be used.

        A photo that will not download or convert is skipped: it is a missing piece of evidence,
        not a failed judgment.
        """
        try:
            return self.crusoe.image_part(load())
        except CrusoeAuthError:
            raise
        except (TransportError, BlockedUrl, OSError, CrusoeError):
            return None

    def _photo(self, url: Optional[str]) -> Optional[Dict[str, Any]]:
        """A listing photo, from a URL the eBay API returned. Held in memory only."""
        if not url:
            return None
        return self._image_part(lambda: self.image_fetcher(url))

    def _reference_photo(self, spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """The user's own reference photo, or None when the spec has none that can be read."""
        paths = spec.get("reference_images") or []
        if not paths:
            return None
        return self._image_part(Path(paths[0]).expanduser().read_bytes)

    # ---- recording ---------------------------------------------------------------

    def _store(self, watch_id: int, case: _Case) -> Dict[str, Any]:
        """Write one candidate row and its trace, and return the stored row.

        Listing content (the snapshot, quoted evidence) is kept for tracked outcomes only.
        """
        item_id = case.summary["item_id"]
        tracked = case.decision in TRACKED_DECISIONS
        earlier = case.row
        if case.decision == "pending" and earlier is not None and earlier["state"] not in UNFINISHED_STATES:
            # A failed re-evaluation must not wipe the verdict, or the snapshot, of an earlier run.
            self.store.update_candidate(earlier["id"], attempts=case.attempts)
        elif tracked:
            listing = case.detail or case.summary
            end = parse_iso(listing.get("end_at"))
            self.store.upsert_candidate(
                watch_id, item_id, case.state, reason=case.reason, attempts=case.attempts, tracked=True,
                listing=snapshot(listing), legacy_item_id=listing.get("legacy_item_id"),
                end_at=iso(end) if end else None, fetched_at=case.fetched_at, purge_at=None,
            )
        else:
            self.store.upsert_candidate(
                watch_id, item_id, case.state, reason=case.reason, attempts=case.attempts,
                end_at=None, purge_at=self._purge_at(case.summary),
            )
            if earlier is not None:
                self.store.purge_listing_data(earlier["id"])
        row = self.store.get_candidate(watch_id, item_id)
        for step in case.trace:
            fields = step["fields"] if tracked else strip_evidence(step["fields"])
            self.store.add_verdict(row["id"], **dict(step, fields=fields))
        return row

    def _purge_at(self, summary: Mapping[str, Any]) -> str:
        """When an untracked row is forgotten: when the listing ends, or after two weeks."""
        end = parse_iso(summary.get("end_at"))
        return iso(end or self.store.clock() + timedelta(days=UNTRACKED_KEEP_DAYS))

    @staticmethod
    def _result(spec: Mapping[str, Any], case: _Case, row: Optional[Dict[str, Any]]) -> FunnelResult:
        item_id = case.summary["item_id"]
        if case.open:
            return FunnelResult(item_id=item_id, decision="pending", reason=REASON_BUDGET, summary=case.summary)
        fields = case.verdicts() if case.detail is not None else {}
        sources = {name: entry["source"] for name, entry in case.evidence.items() if "source" in entry}
        return FunnelResult(
            item_id=item_id, decision=str(case.decision), reason=case.reason, fields=fields,
            sources=sources, unproven=list(case.unproven), flags=list(case.flags),
            proof=rules.proof_line(spec, fields, sources), summary=case.summary, detail=case.detail,
            candidate=row,
        )
