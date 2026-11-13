"""Client for Crusoe's OpenAI-compatible inference API: model catalog, roles, JSON answers.

Three properties of the service shape this module:

- Model ids are case-sensitive and change over time, so none is hardcoded. Each role
  ("triage", "compare", ...) is resolved against the live catalog by name pattern.
- A retired id is silently redirected to another model, so every answer records which
  model actually served it.
- An unknown request parameter can be answered with 403, so the request body is limited
  to a short allowlist.

The models return evidence as JSON. Nothing here decides whether a listing matches.
The API key travels only in the Authorization header and is removed from error messages.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

from . import __version__
from .config import ConfigError, Settings
from .errors import ChatResult, CrusoeAuthError, CrusoeError, CrusoeUnavailable
from .http import BlockedUrl, Response, TransportError, UrllibTransport, check_url
from .store import Store, parse_iso

ROLE_FAMILY = {
    "extract": "vision", "compare": "vision", "tag_read": "vision",
    "parse": "text", "triage": "text", "verify": "text",
}
# Ordered, case-insensitive name patterns per family. For each pattern the first catalog id
# that contains it is a candidate; candidates are tried in pattern order.
ROLE_PATTERNS = {
    "vision": ["gemma-4", "qwen3.8", "nemotron-3.5-super-vl", "glm-5.3-flash"],
    "text": ["gpt-oss-120b", "deepseek-v4.1-flash", "nemotron-3.5-lightning", "qwen3.8", "glm-5.3"],
}

CATALOG_KEY = "crusoe:models"
CATALOG_TTL = timedelta(hours=24)
# 429 is this account's own limit and 503 is the shared endpoint scaling up; both pass.
BACKOFF_SECONDS = (1.0, 2.0, 4.0)
RATE_WINDOW_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 1024
# gpt-oss always reasons, and reasoning tokens are spent from max_tokens.
GPT_OSS_MAX_TOKENS = 2048
TEMPERATURE = 0
CATALOG_TIMEOUT = 30.0
CHAT_TIMEOUT = 120.0
REPAIR_QUOTE_CHARS = 2000

SIPS = "/usr/bin/sips"
SIPS_TIMEOUT = 30.0
MAX_IMAGE_EDGE = 1024
MAX_PASSTHROUGH_BYTES = 4 * 1024 * 1024

_AUTH_HINTS = {
    401: "the API key is missing, wrong or expired; check CRUSOE_API_KEY",
    403: "the key was refused for this request, or a request parameter was not accepted",
}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.DOTALL)
_PIXELS_RE = re.compile(r"pixel(?:Width|Height):\s*(\d+)")


@dataclass
class _Reply:
    """What one chat completion response said, reduced to the parts this client uses."""
    content: str
    finish_reason: str
    model_served: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class CrusoeClient:
    """Asks Crusoe-hosted models for JSON. The network, sleeping and time are injectable."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.store = store
        self.transport = transport if transport is not None else UrllibTransport()
        self.sleep = sleep
        self.clock = clock
        # Send times of recent chat requests, for the rate limit. Kept in memory, so the
        # limit holds per process: the poller and a CLI command each count their own.
        self._sent_at: Deque[float] = deque()

    # ---- catalog -----------------------------------------------------------

    def list_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Return the live catalog as normalized rows, fetching it at most once a day.

        The copy in the store is shared by every process. If a routine refresh fails because
        the service is unavailable, the previous copy is used: a day-old catalog is still
        right about almost every id, and a redirected id is caught by the swap check.
        `refresh=True` always asks the service and never falls back.
        """
        cached, fresh = self._cached_catalog()
        if cached is not None and fresh and not refresh:
            return cached
        try:
            models = self._fetch_catalog()
        except CrusoeUnavailable:
            if cached is None or refresh:
                raise
            return cached
        self.store.kv_set(CATALOG_KEY, {"fetched_at": self.store.now_iso(), "models": models})
        return models

    def _cached_catalog(self) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        """Return (rows, still fresh) from the store, or (None, False) when there is no copy."""
        cached = self.store.kv_get(CATALOG_KEY)
        if not isinstance(cached, dict) or not isinstance(cached.get("models"), list):
            return None, False
        stamp = cached.get("fetched_at")
        fetched = parse_iso(stamp) if isinstance(stamp, str) else None
        fresh = fetched is not None and timedelta(0) <= self.store.clock() - fetched < CATALOG_TTL
        return cached["models"], fresh

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        response, _ = self._call("GET", "/models")
        try:
            payload = response.json()
        except ValueError:
            raise CrusoeError("the Crusoe model catalog was not JSON")
        rows = payload.get("data") if isinstance(payload, dict) else payload
        normalized = [normalize_model(row) for row in rows] if isinstance(rows, list) else []
        models = [row for row in normalized if row is not None]
        if not models:
            raise CrusoeError("the Crusoe model catalog listed no models")
        return models

    # ---- roles -------------------------------------------------------------

    def resolve_role(self, role: str) -> List[str]:
        """Model ids to try for a role, best first. Raises CrusoeError when there is none."""
        return [row["id"] for row in self._candidates(role)]

    def _candidates(self, role: str) -> List[Dict[str, Any]]:
        """Catalog rows for a role: the user's override first, then one match per pattern."""
        if role not in ROLE_FAMILY:
            raise ValueError("unknown role: %s" % role)
        family = ROLE_FAMILY[role]
        catalog = self.list_models()
        chosen: List[Dict[str, Any]] = []
        override = self.settings.model_overrides.get(family)
        if override:
            # Used exactly as typed, whether or not the catalog lists it. An unlisted id gets
            # a bare row, so no optional request parameter is sent to it.
            listed = [row for row in catalog if row["id"] == override]
            chosen.append(listed[0] if listed else normalize_model({"id": override}))
        usable = [row for row in catalog if family != "vision" or row["modality"] == "multimodal"]
        for pattern in ROLE_PATTERNS[family]:
            match = next((row for row in usable if pattern.lower() in row["id"].lower()), None)
            if match is not None and all(match["id"] != row["id"] for row in chosen):
                chosen.append(match)
        if not chosen:
            raise CrusoeError(
                "no model in the Crusoe catalog fits the %s role; set EFT_MODEL_%s to an exact model id"
                % (role, family.upper())
            )
        return chosen

    # ---- chat --------------------------------------------------------------

    def chat_json(
        self,
        role: str,
        system: str,
        parts: Sequence[Dict[str, Any]],
        required: Sequence[str] = (),
        validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """Ask the role's models, in order, until one returns a valid JSON object.

        `parts` is the user turn: text parts and image parts (see `image_part`). An answer is
        valid when it is a JSON object with every `required` key and `validator(data)` returns
        None; the validator returns a sentence describing the problem otherwise.

        A wrong key stops everything (CrusoeAuthError). Any other failure moves on to the next
        model. If every model was rate limited or unavailable the error is CrusoeUnavailable,
        otherwise CrusoeError.
        """
        problems: List[str] = []
        only_unavailable = True
        for row in self._candidates(role):
            try:
                return self._ask(role, row, system, list(parts), required, validator, max_tokens)
            except CrusoeAuthError:
                raise
            except CrusoeUnavailable as exc:
                problems.append("%s: %s" % (row["id"], exc))
            except CrusoeError as exc:
                only_unavailable = False
                problems.append("%s: %s" % (row["id"], exc))
        failure = CrusoeUnavailable if only_unavailable else CrusoeError
        raise failure("no usable answer for the %s role (%s)" % (role, "; ".join(problems)))

    def _ask(
        self,
        role: str,
        row: Dict[str, Any],
        system: str,
        parts: List[Dict[str, Any]],
        required: Sequence[str],
        validator: Optional[Callable[[Dict[str, Any]], Optional[str]]],
        max_tokens: Optional[int],
    ) -> ChatResult:
        """Get a valid answer from one model, or raise CrusoeError.

        The model gets two kinds of second chance, each at most once. A reply that ran out of
        tokens before writing anything (reasoning used them up) is asked for again with twice
        the room. A reply that cannot be used is sent back with the problem quoted.
        """
        budget = int(max_tokens) if max_tokens else default_max_tokens(row["id"])
        doubled = repaired = False
        turn = parts
        while True:
            reply = self._complete(role, row, system, turn, budget)
            data, problem = _judge(reply, required, validator)
            if data is not None:
                return ChatResult(
                    data=data,
                    model_requested=row["id"],
                    model_served=reply.model_served,
                    swapped=reply.model_served != row["id"],
                    latency_ms=reply.latency_ms,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                )
            if not reply.content.strip() and reply.finish_reason == "length" and not doubled:
                doubled, budget = True, budget * 2
                continue
            if repaired:
                raise CrusoeError("unusable after one repair attempt: %s" % self._safe(problem))
            repaired = True
            turn = parts + [_repair_part(str(problem), reply.content, required)]

    def _complete(
        self, role: str, row: Dict[str, Any], system: str, parts: List[Dict[str, Any]], max_tokens: int
    ) -> _Reply:
        """Send one chat completion. Only allowlisted top-level keys go into the body."""
        body: Dict[str, Any] = {
            "model": row["id"],
            "messages": _messages(system, parts),
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }
        if "response_format" in row["supported_parameters"]:
            body["response_format"] = {"type": "json_object"}
        response, latency_ms = self._call(
            "POST", "/chat/completions", payload=body, role=role, timeout=CHAT_TIMEOUT
        )
        try:
            answer = response.json()
        except ValueError:
            raise CrusoeError("the response was not JSON")
        return _read_reply(answer, row["id"], latency_ms)

    # ---- transport ---------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        timeout: float = CATALOG_TIMEOUT,
    ) -> Tuple[Response, int]:
        """Send a request; return the 2xx response and how long it took in milliseconds.

        429, 5xx and network failures are retried after 1, 2 and 4 seconds, then raise
        CrusoeUnavailable. 401 and 403 raise CrusoeAuthError at once: retrying cannot fix a
        refused key. A `role` marks a chat completion, which is the kind of request that is
        rate limited and counted in the ledger, once per attempt.
        """
        self.settings.require("crusoe")
        url = self._url(path)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = self._headers(has_body=body is not None)
        waits = list(BACKOFF_SECONDS)
        while True:
            if role is not None:
                self._wait_for_slot()
                self.store.ledger_add("crusoe", role)
            started = self.clock()
            try:
                response = self.transport.request(method, url, headers=headers, body=body, timeout=timeout)
            except TransportError as exc:
                problem = "network failure: %s" % self._safe(exc)
            else:
                if response.ok:
                    return response, int(round((self.clock() - started) * 1000))
                problem = self._retryable_problem(response)
            if not waits:
                raise CrusoeUnavailable(problem)
            self.sleep(waits.pop(0))

    def _url(self, path: str) -> str:
        """Build the endpoint URL, turning a refused base URL into a readable setting error."""
        url = self.settings.crusoe_api_base + path
        try:
            check_url(url)
        except BlockedUrl as exc:
            raise ConfigError("CRUSOE_API_BASE cannot be used: %s" % exc)
        return url

    def _headers(self, has_body: bool) -> Dict[str, str]:
        headers = {
            "Authorization": "Bearer " + self.settings.crusoe_api_key,
            "Accept": "application/json",
            "User-Agent": "ebay-fashion-tracker/" + __version__,
        }
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _retryable_problem(self, response: Response) -> str:
        """Describe a failed response that is worth retrying; raise for one that is not."""
        detail = "HTTP %d" % response.status
        said = self._safe(_error_text(response))
        if said:
            detail += ": " + said
        if response.status in _AUTH_HINTS:
            raise CrusoeAuthError("Crusoe refused the request (%s); %s" % (detail, _AUTH_HINTS[response.status]))
        if response.status != 429 and response.status < 500:
            raise CrusoeError("Crusoe rejected the request (%s)" % detail)
        return detail

    def _safe(self, text: Any, limit: int = 200) -> str:
        """Make outside text fit for an error message: no API key, one line, bounded length."""
        flat = " ".join(str(text).split())
        key = self.settings.crusoe_api_key
        if key:
            flat = flat.replace(key, "[redacted]")
        return flat[:limit]

    def _wait_for_slot(self) -> None:
        """Hold a chat request until fewer than crusoe_rpm were sent in the last 60 seconds."""
        now = self.clock()
        self._forget_sent_before(now - RATE_WINDOW_SECONDS)
        if len(self._sent_at) >= self.settings.crusoe_rpm:
            opens_at = self._sent_at[0] + RATE_WINDOW_SECONDS
            self.sleep(opens_at - now)
            now = max(self.clock(), opens_at)
            self._forget_sent_before(now - RATE_WINDOW_SECONDS)
        self._sent_at.append(now)

    def _forget_sent_before(self, cutoff: float) -> None:
        while self._sent_at and self._sent_at[0] <= cutoff:
            self._sent_at.popleft()

    # ---- images ------------------------------------------------------------

    def image_part(self, source: Union[str, Path, bytes]) -> Dict[str, Any]:
        """Turn a photo (a file path or raw bytes) into an image part for `chat_json`.

        On macOS the photo is converted with sips to a JPEG of at most 1024 px on its long
        edge, which keeps requests small and accepts HEIC and WebP. Where sips is missing or
        cannot run (inside a sandboxed shell it is not allowed to write its scratch file),
        a JPEG or PNG of up to 4 MB is sent as it is and anything else raises CrusoeError.
        """
        data = _image_bytes(source)
        if _can_use_sips():
            converted = _sips_to_jpeg(data)
            if converted:
                return {"type": "image", "data": converted, "mime": "image/jpeg"}
        mime = _sniff_mime(data)
        if mime is None:
            raise CrusoeError("the image is not a JPEG or PNG and could not be converted to one")
        if len(data) > MAX_PASSTHROUGH_BYTES:
            raise CrusoeError(
                "the image is %.1f MB and could not be downscaled; the limit is 4 MB" % (len(data) / 1048576.0)
            )
        return {"type": "image", "data": data, "mime": mime}


# ---- catalog rows --------------------------------------------------------------

def normalize_model(row: Any) -> Optional[Dict[str, Any]]:
    """Reduce a catalog row to the fields this tool uses; None for a row without an id."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
        return None
    architecture = row.get("architecture")
    modality = architecture.get("modality") if isinstance(architecture, dict) else None
    parameters = row.get("supported_parameters")
    context_length = row.get("context_length")
    return {
        "id": row["id"],
        "modality": modality.lower() if isinstance(modality, str) and modality else "text",
        "context_length": context_length if isinstance(context_length, int) else None,
        "supported_parameters": [p for p in parameters if isinstance(p, str)] if isinstance(parameters, list) else [],
        "shutdown_date": row.get("shutdown_date"),
    }


def default_max_tokens(model_id: str) -> int:
    return GPT_OSS_MAX_TOKENS if "gpt-oss" in model_id.lower() else DEFAULT_MAX_TOKENS


# ---- request and response shapes -----------------------------------------------

def _messages(system: str, parts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the system and user messages.

    A text-only turn is sent as a plain string, the form every chat template accepts; the
    list-of-parts form is used only when there is an image.
    """
    if all(part.get("type") == "text" for part in parts):
        content: Any = "\n\n".join(part["text"] for part in parts)
    else:
        content = [_wire_part(part) for part in parts]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _wire_part(part: Dict[str, Any]) -> Dict[str, Any]:
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": part["text"]}
    if kind == "image":
        encoded = base64.b64encode(part["data"]).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": "data:%s;base64,%s" % (part.get("mime") or "image/jpeg", encoded)},
        }
    raise ValueError("unknown part type: %r" % (kind,))


def _read_reply(answer: Any, requested: str, latency_ms: int) -> _Reply:
    """Pick the answer text and bookkeeping out of a chat completion response.

    Only `message.content` is read. Reasoning text, which arrives in `reasoning` or
    `reasoning_content` depending on the model, is ignored.
    """
    choices = answer.get("choices") if isinstance(answer, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(first, dict):
        raise CrusoeError("the response had no choices")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    usage = answer.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    served = answer.get("model")
    return _Reply(
        content=content if isinstance(content, str) else "",
        finish_reason=str(first.get("finish_reason") or ""),
        model_served=served if isinstance(served, str) and served else requested,
        latency_ms=latency_ms,
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
    )


def _count(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _error_text(response: Response) -> str:
    """Best-effort reason from an error body, which is not always the OpenAI envelope."""
    try:
        payload = response.json()
    except ValueError:
        return response.text()
    if isinstance(payload, dict):
        error = payload.get("error")
        nested = error.get("message") if isinstance(error, dict) else error
        for candidate in (nested, payload.get("message"), payload.get("detail")):
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return response.text()


# ---- JSON answers --------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Return the JSON value in a model reply, or raise ValueError.

    Open models often put the answer in a code fence, add a sentence around it, or leave a
    <think> block in front. The whole reply and the fenced block are tried before searching
    the prose, so a reply that is a list is reported as a list, not mined for its first item.
    """
    cleaned = _THINK_RE.sub("", text).strip()
    fenced = _FENCE_RE.search(cleaned)
    for candidate in [cleaned] + ([fenced.group(1)] if fenced else []):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return _first_object(cleaned)


def _first_object(text: str) -> Dict[str, Any]:
    """Decode the first complete JSON object that starts at any "{" in the text."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        return value
    raise ValueError("no JSON object in the reply")


def _judge(
    reply: _Reply,
    required: Sequence[str],
    validator: Optional[Callable[[Dict[str, Any]], Optional[str]]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (data, None) for a usable reply, or (None, what is wrong with it).

    The problem text never quotes the reply, which may repeat listing content.
    """
    if not reply.content.strip():
        return None, "the reply was empty"
    try:
        data = extract_json(reply.content)
    except ValueError:
        if reply.finish_reason == "length":
            return None, "the reply was cut off before the JSON was complete; answer more briefly"
        return None, "the reply was not valid JSON"
    if not isinstance(data, dict):
        return None, "the reply must be one JSON object"
    missing = [key for key in required if key not in data]
    if missing:
        return None, "the JSON object is missing: " + ", ".join(missing)
    complaint = validator(data) if validator is not None else None
    if complaint:
        return None, str(complaint)
    return data, None


def _repair_part(problem: str, previous: str, required: Sequence[str]) -> Dict[str, Any]:
    """Text added to the user turn for the repair attempt: the problem and what is expected.

    It extends the user turn instead of adding assistant and user messages, because some
    chat templates reject an empty assistant message or two user messages in a row.
    """
    lines = [
        "Your previous reply could not be used: %s." % problem.rstrip("."),
        "Reply again with one JSON object and nothing else: no code fence, no commentary.",
    ]
    if required:
        lines.append("It must have these keys: %s." % ", ".join(required))
    if previous.strip():
        lines += ["Your previous reply was:", "<<<", previous.strip()[:REPAIR_QUOTE_CHARS], ">>>"]
    return {"type": "text", "text": "\n".join(lines)}


# ---- images --------------------------------------------------------------------

def _image_bytes(source: Union[str, Path, bytes]) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        path = Path(source).expanduser()
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CrusoeError("cannot read the image file %s (%s)" % (path.name, exc.__class__.__name__))
    if not data:
        raise CrusoeError("the image is empty")
    return data


def _sniff_mime(data: bytes) -> Optional[str]:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


def _can_use_sips() -> bool:
    return sys.platform == "darwin" and os.path.exists(SIPS)


def _run_sips(arguments: List[str]) -> Optional[str]:
    """Run sips and return what it printed, or None when it failed or could not start."""
    try:
        done = subprocess.run(
            [SIPS] + arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=SIPS_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", errors="replace")


def _sips_to_jpeg(data: bytes) -> Optional[bytes]:
    """Convert image bytes to a JPEG of at most MAX_IMAGE_EDGE px, or None if sips cannot.

    sips works on files, so the bytes pass through a private temporary directory that is
    deleted before this returns. `sips -Z` enlarges small images as well as shrinking large
    ones, so the size is read first and a resize is requested only for an oversized image.
    """
    with tempfile.TemporaryDirectory(prefix="eft-image-") as folder:
        source = os.path.join(folder, "source")
        target = os.path.join(folder, "converted.jpg")
        with open(source, "wb") as handle:
            handle.write(data)
        measured = _run_sips(["-g", "pixelWidth", "-g", "pixelHeight", source])
        edges = [int(number) for number in _PIXELS_RE.findall(measured or "")]
        if len(edges) != 2:
            return None
        resize = ["-Z", str(MAX_IMAGE_EDGE)] if max(edges) > MAX_IMAGE_EDGE else []
        if _run_sips(["-s", "format", "jpeg"] + resize + [source, "--out", target]) is None:
            return None
        try:
            with open(target, "rb") as handle:
                return handle.read() or None
        except OSError:
            return None
