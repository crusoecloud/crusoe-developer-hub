"""A small HTTP layer with an injectable transport and a hard rule: never fetch eBay pages.

eBay's User Agreement prohibits automated access to its sites. This tool reads eBay
only through the official API hosts. `check_url` enforces that for every request,
including redirects, so a bug elsewhere cannot turn into scraping. A redirect also
never carries an API key to a host other than the one the request was made to.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_BODY_BYTES = 20 * 1024 * 1024

# Official API hosts. Listing images are served from ebayimg.com, a different domain.
EBAY_API_HOSTS = frozenset({"api.ebay.com", "api.sandbox.ebay.com"})
# eBay site and short-link hosts: ebay.com, www.ebay.co.uk, ebay.us, ebay.to, ...
_EBAY_SITE_RE = re.compile(r"(^|\.)ebay\.[a-z]{2,3}(\.[a-z]{2})?$")


class BlockedUrl(Exception):
    """The request was refused before it left the machine."""


class TransportError(Exception):
    """The request could not be completed (DNS, TLS, timeout, connection reset, a reply cut short)."""


def host_of(url: str) -> str:
    """The host a URL names, lower-cased and without the trailing dot DNS allows.

    "www.ebay.com." reaches the same server as "www.ebay.com", so every check on a host
    reads it through here.
    """
    return (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")


def is_ebay_site(host: str) -> bool:
    """True for a host on any eBay site or short-link domain, including the API hosts."""
    return bool(_EBAY_SITE_RE.search(host.lower().rstrip(".")))


def check_url(url: str) -> None:
    """Raise BlockedUrl unless the URL is https and not an eBay web page."""
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme != "https":
        raise BlockedUrl("only https URLs are requested: %s" % scheme)
    host = host_of(url)
    if not host:
        raise BlockedUrl("URL has no host")
    if is_ebay_site(host) and host not in EBAY_API_HOSTS:
        raise BlockedUrl(
            "eBay pages are never fetched (%s); use the Browse API on api.ebay.com" % host
        )


@dataclass
class Response:
    status: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))


def _origin(url: str) -> Tuple[str, int]:
    """The server an https URL is sent to: its host and port."""
    return host_of(url), urllib.parse.urlsplit(url).port or 443


def read_capped(stream: Any, limit: int) -> bytes:
    body = stream.read(limit + 1) or b""
    if len(body) > limit:
        raise TransportError("the response is larger than %d bytes" % limit)
    return body


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to the same host and port, and never to an eBay page."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        check_url(newurl)
        if _origin(newurl) != _origin(req.full_url):
            # urllib copies every header, Authorization included, to the target of a redirect.
            # A key is for the host it was issued for, so a redirect elsewhere is not followed.
            raise TransportError(
                "%s redirected to another host (%s), which is not followed"
                % (host_of(req.full_url) or "?", host_of(newurl) or "?")
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibTransport:
    """The real transport. HTTP error statuses are returned, not raised."""

    def __init__(self, max_body_bytes: int = MAX_BODY_BYTES) -> None:
        self._opener = urllib.request.build_opener(_GuardedRedirect())
        self._max_body_bytes = max_body_bytes

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Response:
        check_url(url)
        req = urllib.request.Request(url, data=body, method=method.upper(), headers=headers or {})
        try:
            return self._exchange(req, timeout)
        except BlockedUrl:
            raise
        except (urllib.error.URLError, HTTPException, OSError, ValueError) as exc:
            raise TransportError("%s %s failed: %s" % (method.upper(), host_of(url) or "?", _reason(exc)))

    def _exchange(self, req: urllib.request.Request, timeout: float) -> Response:
        """Send one request and read the whole reply. An HTTP error status is a reply like any other."""
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return Response(
                    status=resp.status,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body=read_capped(resp, self._max_body_bytes),
                )
        except urllib.error.HTTPError as exc:
            return Response(
                status=exc.code,
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=read_capped(exc, self._max_body_bytes),
            )


def _reason(exc: Exception) -> str:
    """Why a request failed, in words that are safe to log.

    An http.client error (a reply cut off part-way, a malformed status line or URL) can quote
    the request path, and a path can be a secret webhook address, so only its name is given.
    """
    return exc.__class__.__name__ if isinstance(exc, HTTPException) else str(exc)


def json_response(obj: Any, status: int = 200, headers: Optional[Dict[str, str]] = None) -> Response:
    out = {"content-type": "application/json"}
    out.update(headers or {})
    return Response(status=status, headers=out, body=json.dumps(obj).encode("utf-8"))


Handler = Callable[[str, str, Dict[str, str], Optional[bytes]], Response]


class FakeTransport:
    """Test double. Give it a handler, or queue responses with add().

    Every call is recorded in `calls` as a dict with method, url, headers, body.
    The eBay-page rule is enforced here too, so tests catch violations.
    """

    def __init__(self, handler: Optional[Handler] = None) -> None:
        self.handler = handler
        self.queue: List[Response] = []
        self.calls: List[Dict[str, Any]] = []

    def add(self, response: Response) -> "FakeTransport":
        self.queue.append(response)
        return self

    def add_json(self, obj: Any, status: int = 200, headers: Optional[Dict[str, str]] = None) -> "FakeTransport":
        return self.add(json_response(obj, status, headers))

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Response:
        check_url(url)
        self.calls.append({"method": method.upper(), "url": url, "headers": dict(headers or {}), "body": body})
        if self.handler is not None:
            return self.handler(method.upper(), url, dict(headers or {}), body)
        if not self.queue:
            raise AssertionError("FakeTransport: no response queued for %s %s" % (method, url))
        return self.queue.pop(0)
