"""A small HTTP layer with an injectable transport and a hard rule: never fetch eBay pages.

eBay's User Agreement prohibits automated access to its sites. This tool reads eBay
only through the official API hosts. `check_url` enforces that for every request,
including redirects, so a bug elsewhere cannot turn into scraping.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Official API hosts. Listing images are served from ebayimg.com, a different domain.
EBAY_API_HOSTS = frozenset({"api.ebay.com", "api.sandbox.ebay.com"})
# eBay site and short-link hosts: ebay.com, www.ebay.co.uk, ebay.us, ebay.to, ...
_EBAY_SITE_RE = re.compile(r"(^|\.)ebay\.[a-z]{2,3}(\.[a-z]{2})?$")


class BlockedUrl(Exception):
    """The request was refused before it left the machine."""


class TransportError(Exception):
    """The request could not be completed (DNS, TLS, timeout, connection reset)."""


def check_url(url: str) -> None:
    """Raise BlockedUrl unless the URL is https and not an eBay web page."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise BlockedUrl("only https URLs are requested: %s" % parts.scheme)
    host = (parts.hostname or "").lower()
    if not host:
        raise BlockedUrl("URL has no host")
    if _EBAY_SITE_RE.search(host) and host not in EBAY_API_HOSTS:
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


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibTransport:
    """The real transport. HTTP error statuses are returned, not raised."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_GuardedRedirect())

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
            with self._opener.open(req, timeout=timeout) as resp:
                return Response(
                    status=resp.status,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body=resp.read(),
                )
        except urllib.error.HTTPError as exc:
            return Response(
                status=exc.code,
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=exc.read() or b"",
            )
        except BlockedUrl:
            raise
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise TransportError("%s %s failed: %s" % (method.upper(), _host(url), exc))


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or "?"


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
