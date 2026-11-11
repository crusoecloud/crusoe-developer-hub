"""Log in to the workshop with the passcode on your desk card. Run it through ``./login.sh``.

    ./login.sh <passcode>                   download the bundle, unlock your desk's kubeconfig
    ./login.sh --bundle <file> <passcode>   same, from a copy of the bundle (e.g. AirDropped)
    ./login.sh --file <kubeconfig>          a kubeconfig handed to you directly

Every path checks the kubeconfig against the cluster, asks for your details (stored in the
``participant-info`` ConfigMap in your namespace), saves the kubeconfig to
``~/.sglang-workshop/kubeconfig`` and lets ``login.sh`` run ``check.sh``. Running it again is safe:
it says what changed.

What stays on disk: only ``~/.sglang-workshop/kubeconfig`` (and a backup of a different one it
replaced). The downloaded bundle is held in memory and never written; ``--bundle`` only reads the
user's file. The unlocked kubeconfig is written to a 0600 temp file for the access check and deleted
afterwards, on errors and Ctrl-C too.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import bundle
from .k8s import WORKSHOP_KUBECONFIG

#: Lines in this file are bundle URLs, tried in order (the first, then mirrors); ``#`` starts a comment.
BUNDLE_URL_FILE = Path(__file__).resolve().parents[1] / "bundle-url.txt"
#: Overrides the file (space-separated URLs); for instructors testing a local copy.
BUNDLE_URL_ENV = "WORKSHOP_BUNDLE_URL"

PRIVACY_POLICY_URL = "https://legal.crusoe.ai/#website-privacy-notice"
EMAIL_PROMPT = f"Email (used to send you the workshop materials; Crusoe privacy notice: {PRIVACY_POLICY_URL})"
CONFIGMAP = "participant-info"
LABEL = "workshop/participant-info"
MAX_NAME = 60
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
NAMESPACE_RE = re.compile(r"^ws-(\d{2,3})$")

DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_S = 15
KUBECTL_ATTEMPTS = 3
HELP_DESK = "ask at the help desk for the bundle file, then run ./login.sh --bundle <file> <passcode>"

#: Exit codes (login.sh passes them on).
EXIT_WRONG_PASSCODE = 2
EXIT_DOWNLOAD = 3
EXIT_NETWORK = 4
EXIT_REVOKED = 5
EXIT_OTHER = 6
EXIT_NOT_OPEN = 8

#: What the gist serves until the instructors upload the real bundle on the event day.
NOT_OPEN_PLACEHOLDER = b"WORKSHOP_NOT_OPEN_YET"
NOT_OPEN_MESSAGE = "the workshop login opens at the start of the session; try again then"

#: kubectl errors that mean the cluster could not be reached (not that the credential is wrong).
NETWORK_ERRORS = (
    "Unable to connect to the server",
    "i/o timeout",
    "no such host",
    "connection refused",
    "network is unreachable",
    "TLS handshake timeout",
    "context deadline exceeded",
    "Client.Timeout",
    "dial tcp",
)
#: kubectl's client-side validation first downloads the API server's whole OpenAPI schema (about
#: 10 MB): on a busy venue network that alone can exceed the 10 s request timeout. The ConfigMap is
#: generated here, and the API server still validates what it receives.
NO_CLIENT_VALIDATION = ("--validate=false",)
UNAUTHORIZED = ("Unauthorized", "must be logged in", "token has expired")


class LoginError(Exception):
    def __init__(self, message: str, code: int = EXIT_OTHER):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Details:
    email: str
    display_name: str
    show_name: bool
    marketing_opt_in: bool

    def as_data(self, logged_in_at: str) -> dict[str, str]:
        return {
            "email": self.email,
            "display_name": self.display_name,
            "show_name": str(self.show_name).lower(),
            "marketing_opt_in": str(self.marketing_opt_in).lower(),
            "logged_in_at": logged_in_at,
        }

    @classmethod
    def from_data(cls, data: dict[str, str] | None) -> Details | None:
        if not data or not data.get("email"):
            return None
        return cls(
            email=data.get("email", ""),
            display_name=data.get("display_name", ""),
            show_name=data.get("show_name") == "true",
            marketing_opt_in=data.get("marketing_opt_in") == "true",
        )


# --- getting the bundle -----------------------------------------------------------------------


def bundle_urls(
    environ: dict[str, str] | os._Environ = os.environ, path: Path = BUNDLE_URL_FILE
) -> list[str]:
    """The bundle URLs: ``WORKSHOP_BUNDLE_URL`` if set, else the non-comment lines of bundle-url.txt."""
    env = environ.get(BUNDLE_URL_ENV, "").split()
    if env:
        return env
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "sglang-workshop-login/1"},
    )
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r:
        return r.read()


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc or url


def download(urls: list[str], fetch: Callable[[str], bytes] = _fetch,
             sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print) -> dict[str, Any]:  # fmt: skip
    """The parsed bundle from the first URL that serves one. Retries network errors and 5xx answers;
    a 4xx answer, the not-open placeholder or a file that is not a bundle moves on to the next URL
    (a backup uploader's mirror can serve the bundle while the main gist still has the placeholder).
    If no URL serves a bundle and any served the placeholder, the login is not open yet."""
    if not urls:
        raise LoginError(
            f"no bundle URL is configured yet (workshop/bundle-url.txt); {HELP_DESK}", EXIT_DOWNLOAD
        )
    network_only, not_open, problems = True, False, []
    for url in urls:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                data = fetch(url)
                if data.strip().lstrip(b"\xef\xbb\xbf") == NOT_OPEN_PLACEHOLDER:
                    not_open = True
                    break  # no retry; try the next URL
                return bundle.loads(data)
            except urllib.error.HTTPError as exc:
                network_only = False
                problems.append(f"{_host(url)} answered HTTP {exc.code}")
                if exc.code < 500:
                    break
            except bundle.BundleError as exc:
                network_only = False
                problems.append(f"{_host(url)} did not serve the bundle ({exc})")
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                problems.append(f"cannot reach {_host(url)} ({reason})")
            if attempt < DOWNLOAD_ATTEMPTS:
                say(f"  download failed ({problems[-1]}), retrying ({attempt + 1}/{DOWNLOAD_ATTEMPTS})…")
                sleep(2.0 * attempt)
    if not_open:
        raise LoginError(NOT_OPEN_MESSAGE, EXIT_NOT_OPEN)
    if network_only:
        raise LoginError(
            f"network unreachable: {problems[-1]}. Check your Wi-Fi and run ./login.sh again; if it keeps "
            f"failing, {HELP_DESK}",
            EXIT_NETWORK,
        )
    raise LoginError(f"download failed: {problems[-1]}. {HELP_DESK[0].upper()}{HELP_DESK[1:]}", EXIT_DOWNLOAD)


def read_bundle_file(path: str) -> dict[str, Any]:
    try:
        data = Path(path).expanduser().read_bytes()
    except OSError as exc:
        raise LoginError(f"cannot read the bundle file {path}: {exc.strerror or exc}") from None
    try:
        return bundle.loads(data)
    except bundle.BundleError as exc:
        raise LoginError(f"{path}: {exc}") from None


def unlock(doc: dict[str, Any], passcode: str) -> bundle.Entry:
    try:
        return bundle.unlock(doc, passcode)
    except bundle.PasscodeFormatError as exc:
        raise LoginError(f"wrong passcode: {exc}", EXIT_WRONG_PASSCODE) from None
    except bundle.WrongPasscode:
        raise LoginError(
            "wrong passcode: it matches no desk. Check it against your card and type it again "
            "(case, dashes and spaces do not matter)",
            EXIT_WRONG_PASSCODE,
        ) from None
    except bundle.BundleError as exc:
        raise LoginError(f"{exc}; {HELP_DESK}", EXIT_DOWNLOAD) from None


# --- the kubeconfig ---------------------------------------------------------------------------


def kubeconfig_namespace(text: str) -> str:
    """The namespace of the kubeconfig's current context; :class:`LoginError` if it is not ws-NN."""
    try:
        doc = yaml.safe_load(text)
        cur = doc["current-context"]
        ctx = next(c["context"] for c in doc["contexts"] if c["name"] == cur)
        ns = ctx.get("namespace", "")
    except (yaml.YAMLError, KeyError, TypeError, StopIteration, AttributeError):
        raise LoginError("this is not a workshop kubeconfig (no current context)") from None
    if not NAMESPACE_RE.match(ns or ""):
        raise LoginError(
            f"this kubeconfig's namespace is {ns or 'empty'!r}, not a workshop namespace (ws-NN)"
        )
    return ns


def token_expiry(text: str) -> datetime | None:
    """The ``exp`` claim of the kubeconfig's token, if it is a JWT (read locally, not verified)."""
    try:
        token = yaml.safe_load(text)["users"][0]["user"]["token"]
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return datetime.fromtimestamp(int(claims["exp"]), timezone.utc)
    except (yaml.YAMLError, KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


Run = Callable[[list[str], str | None], subprocess.CompletedProcess]


def _run(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=60)


class Cluster:
    """kubectl with one explicit kubeconfig (never the environment's)."""

    def __init__(
        self, kubeconfig: Path, namespace: str, run: Run = _run, sleep: Callable[[float], None] = time.sleep
    ):
        self.kubeconfig, self.namespace, self.run, self.sleep = kubeconfig, namespace, run, sleep

    def kubectl(
        self, *args: str, stdin: str | None = None, retry: bool = True
    ) -> subprocess.CompletedProcess:
        cmd = [
            "kubectl",
            "--kubeconfig",
            str(self.kubeconfig),
            "--request-timeout=10s",
            "-n",
            self.namespace,
            *args,
        ]
        for attempt in range(1, KUBECTL_ATTEMPTS + 1):
            try:
                p = self.run(cmd, stdin)
            except FileNotFoundError:
                raise LoginError(
                    "kubectl not found: install it (https://kubernetes.io/docs/tasks/tools/) and run ./login.sh again"
                ) from None
            except subprocess.TimeoutExpired:
                p = subprocess.CompletedProcess(cmd, 1, "", "Unable to connect to the server: timeout")
            err = p.stderr or ""
            if p.returncode == 0 or not retry or not any(m in err for m in NETWORK_ERRORS):
                return p
            if attempt < KUBECTL_ATTEMPTS:
                self.sleep(2.0 * attempt)
        return p

    def check_access(self) -> None:
        p = self.kubectl("get", "pods", "-o", "name")
        if p.returncode == 0:
            return
        err = (p.stderr or "").strip()
        last = err.splitlines()[-1] if err else f"exit {p.returncode}"
        if any(m in err for m in UNAUTHORIZED):
            raise LoginError(
                "this desk's access has been revoked or has expired: ask an instructor for a spare card",
                EXIT_REVOKED,
            )
        if any(m in err for m in NETWORK_ERRORS):
            raise LoginError(
                f"network unreachable: cannot reach the workshop cluster ({last}). Check your Wi-Fi "
                "(a VPN or firewall can block it too) and run ./login.sh again",
                EXIT_NETWORK,
            )
        raise LoginError(
            f"your kubeconfig was accepted but cannot list pods in {self.namespace}: {last}; ask an instructor"
        )

    def read_info(self) -> dict[str, str] | None:
        p = self.kubectl("get", "configmap", CONFIGMAP, "-o", "json")
        if p.returncode != 0:
            if "NotFound" in (p.stderr or "") or "not found" in (p.stderr or ""):
                return None
            raise LoginError(f"cannot read your details: {(p.stderr or '').strip().splitlines()[-1:]}")
        return json.loads(p.stdout).get("data") or {}

    def write_info(self, data: dict[str, str], exists: bool) -> None:
        """Create the ConfigMap on the first login, replace it after that (never ``apply``).

        A re-run goes straight to ``replace``: a ``create`` refused with AlreadyExists is still
        charged to the namespace's ConfigMap quota until the quota controller recounts, so repeated
        logins could fill the quota (6) and block the leaderboard result.
        """
        doc = json.dumps({
            "apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": CONFIGMAP, "namespace": self.namespace, "labels": {LABEL: "true"}},
            "data": data,
        })  # fmt: skip
        if exists:
            p = self.kubectl("replace", *NO_CLIENT_VALIDATION, "-f", "-", stdin=doc)
            if p.returncode != 0 and "NotFound" in (p.stderr or ""):  # deleted since we read it
                p = self.kubectl("create", *NO_CLIENT_VALIDATION, "-f", "-", stdin=doc, retry=False)
        else:
            p = self.kubectl("create", *NO_CLIENT_VALIDATION, "-f", "-", stdin=doc, retry=False)
            if p.returncode != 0 and "AlreadyExists" in (p.stderr or ""):  # created since we read it
                p = self.kubectl("replace", *NO_CLIENT_VALIDATION, "-f", "-", stdin=doc)
        if p.returncode != 0:
            err = (p.stderr or "").strip()
            hint = (
                " (the namespace's ConfigMap quota is full; ask an instructor)"
                if "exceeded quota" in err
                else ""
            )
            raise LoginError(
                f"cannot save your details{hint}: {err.splitlines()[-1] if err else p.returncode}"
            )


def save_kubeconfig(text: str, path: Path = WORKSHOP_KUBECONFIG, now: datetime | None = None) -> str:
    """Write the kubeconfig (0600, directory 0700), backing up a different existing one.
    Returns what happened, for the summary."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    status = f"saved to {_tilde(path)}"
    if path.exists():
        if path.read_text() == text:
            os.chmod(path, 0o600)
            return f"unchanged ({_tilde(path)})"
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        os.replace(path, backup)
        os.chmod(backup, 0o600)
        status = f"replaced {_tilde(path)} (previous one kept as {_tilde(backup)})"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".kubeconfig-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return status


def _tilde(p: Path) -> str:
    home = str(Path.home())
    s = str(p)
    return "~" + s[len(home) :] if s.startswith(home + os.sep) else s


# --- your details -----------------------------------------------------------------------------

Ask = Callable[[str], str]


def _input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise LoginError("no answer (input closed); run ./login.sh again in a terminal") from None


def valid_email(s: str) -> bool:
    return len(s) <= 254 and bool(EMAIL_RE.match(s))


def ask_details(prev: Details | None, ask: Ask = _input, say: Callable[[str], None] = print) -> Details:
    say("")
    while True:
        keep = f" [{prev.email}]" if prev else ""
        email = ask(f"{EMAIL_PROMPT}{keep}: ").strip() or (prev.email if prev else "")
        if valid_email(email):
            break
        say(
            "  please enter a valid email address, like name@example.com"
            if email
            else "  an email is required"
        )
    while True:
        keep = f" [{prev.display_name}]" if prev else ""
        name = " ".join(ask(f"Display name (required, max {MAX_NAME} characters){keep}: ").split())
        name = name or (prev.display_name if prev else "")
        if not name:
            say("  a display name is required")
        elif len(name) > MAX_NAME:
            say(f"  that is {len(name)} characters; please use at most {MAX_NAME}")
        elif any(ord(c) < 32 for c in name):
            say("  please use printable characters only")
        else:
            break
    show = _yes_no(ask, say, "Show your name on the results screen?", prev.show_name if prev else False)
    mkt = _yes_no(
        ask, say, "Can Crusoe contact you about Crusoe Cloud?", prev.marketing_opt_in if prev else False
    )
    return Details(email, name, show, mkt)


def _yes_no(ask: Ask, say: Callable[[str], None], question: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        a = ask(f"{question} {hint} ").strip().lower()
        if not a:
            return default
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        say("  please answer y or n")


def describe_changes(prev: Details | None, new: Details) -> str:
    if prev is None:
        return "details saved"
    changed = []
    for f, label in (("email", "email"), ("display_name", "display name"), ("show_name", "show name"),
                     ("marketing_opt_in", "contact about Crusoe Cloud")):  # fmt: skip
        a, b = getattr(prev, f), getattr(new, f)
        if a != b:
            changed.append(f"{label}: {_yn(a)} -> {_yn(b)}" if isinstance(a, bool) else f"{label} changed")
    return "details updated (" + "; ".join(changed) + ")" if changed else "details unchanged"


def _yn(v: bool) -> str:
    return "yes" if v else "no"


# --- main -------------------------------------------------------------------------------------


def login(args: argparse.Namespace, *, ask: Ask = _input, say: Callable[[str], None] = print, run: Run = _run,
          fetch: Callable[[str], bytes] = _fetch, target: Path = WORKSHOP_KUBECONFIG,
          now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> str:  # fmt: skip
    """The whole login; returns the namespace. Raises :class:`LoginError`."""
    if args.file:
        try:
            text = Path(args.file).expanduser().read_text()
        except OSError as exc:
            raise LoginError(f"cannot read {args.file}: {exc.strerror or exc}") from None
        ns = kubeconfig_namespace(text)
        desk = None
    else:
        code = " ".join(args.passcode).strip() or ask("Passcode (from your desk card): ")
        try:
            bundle.normalize_passcode(code)  # a typo in the format fails before any download
        except bundle.PasscodeFormatError as exc:
            raise LoginError(f"wrong passcode: {exc}", EXIT_WRONG_PASSCODE) from None
        if args.bundle:
            doc = read_bundle_file(args.bundle)
        else:
            say("Downloading the login bundle…")
            doc = download(bundle_urls(), fetch=fetch, say=say)
        entry = unlock(doc, code)
        text, desk = entry.kubeconfig, entry.desk
        ns = kubeconfig_namespace(text)
        if ns != entry.namespace:
            raise LoginError(
                f"login bundle is inconsistent (desk {desk}: {entry.namespace} vs {ns}); {HELP_DESK}"
            )
    who = f"desk {desk} ({ns})" if desk is not None else ns

    exp = token_expiry(text)
    if exp is not None and exp <= now():
        raise LoginError(
            f"this desk's access has been revoked or has expired (expired {exp.astimezone():%b %d, %-I:%M %p %Z}): "
            "ask an instructor for a spare card",
            EXIT_REVOKED,
        )

    fd, tmp = tempfile.mkstemp(prefix="sglang-workshop-", suffix=".kubeconfig")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        cluster = Cluster(Path(tmp), ns, run=run)
        say(f"Checking access for {who}…")
        cluster.check_access()
        say(f"  ok: {who}")
        info = cluster.read_info()
        prev = Details.from_data(info)
        if prev:
            say("You have logged in before; press Enter to keep an answer.")
        details = ask_details(prev, ask=ask, say=say)
        cluster.write_info(details.as_data(now().strftime("%Y-%m-%dT%H:%M:%SZ")), exists=info is not None)
    finally:
        os.unlink(tmp)
    saved = save_kubeconfig(text, target)
    say("")
    say(f"Logged in: {who}.")
    say(f"  kubeconfig: {saved}")
    say(f"  {describe_changes(prev, details)}")
    if exp is not None:
        say(f"  access valid until {exp.astimezone():%a %b %d, %-I:%M %p %Z}")
    return ns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="./login.sh",
        description="Log in to the workshop with the passcode on your desk card.",
        epilog="Case, dashes and spaces in the passcode do not matter.",
    )
    p.add_argument("passcode", nargs="*", help="the passcode on your desk card, e.g. K7F2-XQ9M-HC4T-W8PA")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--bundle", metavar="FILE", help="use a local copy of the login bundle instead of downloading it"
    )
    src.add_argument("--file", metavar="KUBECONFIG", help="use a kubeconfig you were given (no passcode)")
    a = p.parse_args(argv)
    if a.file and a.passcode:
        p.error("--file takes no passcode")
    return a


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        login(args)
    except LoginError as exc:
        print(f"login failed: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("\nlogin cancelled; nothing was saved on this laptop", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
