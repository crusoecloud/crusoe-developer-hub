"""The encrypted login bundle: one file that holds every desk's kubeconfig, each locked by its own passcode.

The instructors build it before the session; ``login.sh`` downloads it and
unlocks the one entry that matches the passcode on the participant's desk card.

Format (JSON, version 1)::

    {"format": "sglang-workshop-bundle", "version": 1, "bundle_id": <hex>,
     "created_at": <RFC 3339>, "kdf": {"name": "scrypt", "n": ..., "r": ..., "p": ..., "salt": <b64>},
     "entries": {<lookup id hex>: {"nonce": <b64>, "ct": <b64>}, ...}}

Per passcode, scrypt (one salt for the whole bundle, so unlocking costs one derivation however many
desks there are) gives 64 bytes: the first 32 are the AES-256-GCM key, the rest name the entry
(lookup id), so a wrong passcode is detected without trying every entry. Each entry is encrypted
with a random nonce and authenticated together with the bundle id and its lookup id. The plaintext
is ``{"desk": N, "namespace": "ws-NN", "kubeconfig": "..."}``. Nothing about the desks is in the clear.

Passcodes: 16 characters from a 31-character alphabet without 0, O, 1, I or L (about 79 bits),
printed in groups of four (``K7F2-XQ9M-HC4T-W8PA``). Case, dashes and spaces are ignored.
"""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

FORMAT = "sglang-workshop-bundle"
VERSION = 1

#: Uppercase letters and digits without the look-alikes 0/O, 1/I/L.
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
PASSCODE_LEN = 16
GROUP = 4

#: scrypt cost: 2**17 * 8 * 128 bytes = 128 MiB and a fraction of a second per derivation.
SCRYPT_N = 2**17
SCRYPT_R = 8
SCRYPT_P = 1

LOOKUP_BYTES = 16


class BundleError(Exception):
    """The bundle file is not a readable login bundle."""


class PasscodeFormatError(ValueError):
    """The text cannot be a passcode (wrong length or characters that are never on a card)."""


class WrongPasscode(Exception):
    """No entry in the bundle matches the passcode."""


@dataclass(frozen=True)
class Entry:
    desk: int
    namespace: str
    kubeconfig: str


# --- passcodes --------------------------------------------------------------------------------


def new_passcode() -> str:
    """A fresh random passcode, grouped for printing."""
    return format_passcode("".join(secrets.choice(ALPHABET) for _ in range(PASSCODE_LEN)))


def format_passcode(code: str) -> str:
    c = normalize_passcode(code)
    return "-".join(c[i : i + GROUP] for i in range(0, len(c), GROUP))


def normalize_passcode(text: str) -> str:
    """Upper-case, drop dashes and whitespace, and check the result could be on a card."""
    c = "".join(ch for ch in text.upper() if ch not in "- \t\r\n")
    bad = sorted({ch for ch in c if ch not in ALPHABET})
    if bad:
        shown = " ".join(repr(ch) for ch in bad)
        raise PasscodeFormatError(
            f"the passcode contains {shown}, which is never on a card "
            "(cards use no 0, O, 1, I or L; check the card again)"
        )
    if len(c) != PASSCODE_LEN:
        raise PasscodeFormatError(
            f"the passcode has {len(c)} characters; a card's has {PASSCODE_LEN} (four groups of four)"
        )
    return c


# --- crypto -----------------------------------------------------------------------------------


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode(), validate=True)


def _derive(code: str, salt: bytes, n: int, r: int, p: int) -> tuple[bytes, str]:
    """``(aes key, lookup id hex)`` for a normalized passcode."""
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    out = Scrypt(salt=salt, length=32 + 32, n=n, r=r, p=p).derive(code.encode())
    return out[:32], out[32 : 32 + LOOKUP_BYTES].hex()


def _aad(bundle_id: str, lookup: str) -> bytes:
    return f"{FORMAT}/v{VERSION}|{bundle_id}|{lookup}".encode()


def build(entries: list[tuple[str, Entry]], *, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P,
          now: datetime | None = None) -> dict[str, Any]:  # fmt: skip
    """The bundle document for ``[(passcode, entry), ...]``. Passcodes must be distinct."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt, bundle_id = secrets.token_bytes(16), secrets.token_hex(8)
    out: dict[str, dict[str, str]] = {}
    for code, e in entries:
        key, lookup = _derive(normalize_passcode(code), salt, n, r, p)
        if lookup in out:
            raise ValueError("two desks have the same passcode")
        nonce = secrets.token_bytes(12)
        plain = json.dumps({"desk": e.desk, "namespace": e.namespace, "kubeconfig": e.kubeconfig}).encode()
        out[lookup] = {
            "nonce": _b64(nonce),
            "ct": _b64(AESGCM(key).encrypt(nonce, plain, _aad(bundle_id, lookup))),
        }
    created = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "format": FORMAT,
        "version": VERSION,
        "bundle_id": bundle_id,
        "created_at": created,
        "kdf": {"name": "scrypt", "n": n, "r": r, "p": p, "salt": _b64(salt)},
        "entries": dict(sorted(out.items())),  # sorted by lookup id: no desk order leaks
    }


def dumps(doc: dict[str, Any]) -> bytes:
    return (json.dumps(doc, indent=1) + "\n").encode()


def loads(data: bytes) -> dict[str, Any]:
    """Parse and sanity-check a bundle; :class:`BundleError` if it is not one."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        head = data[:60].decode("utf-8", "replace").strip()
        raise BundleError(f"not a login bundle (starts with {head!r})") from None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise BundleError("not a login bundle (no 'format' field)")
    if doc.get("version") != VERSION:
        raise BundleError(
            f"bundle version {doc.get('version')} is not supported; update the repository (git pull)"
        )
    kdf = doc.get("kdf") or {}
    if kdf.get("name") != "scrypt" or not isinstance(doc.get("entries"), dict):
        raise BundleError("login bundle is damaged (kdf or entries missing)")
    return doc


def unlock(doc: dict[str, Any], passcode: str) -> Entry:
    """The entry for ``passcode``. Raises :class:`PasscodeFormatError`, :class:`WrongPasscode` or
    :class:`BundleError` (entry present but does not decrypt: the file was damaged)."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    code = normalize_passcode(passcode)
    kdf = doc["kdf"]
    try:
        key, lookup = _derive(code, _unb64(kdf["salt"]), int(kdf["n"]), int(kdf["r"]), int(kdf["p"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise BundleError(f"login bundle is damaged ({exc})") from None
    item = doc["entries"].get(lookup)
    if item is None:
        raise WrongPasscode()
    try:
        plain = AESGCM(key).decrypt(_unb64(item["nonce"]), _unb64(item["ct"]), _aad(doc["bundle_id"], lookup))
        d = json.loads(plain)
        return Entry(desk=int(d["desk"]), namespace=str(d["namespace"]), kubeconfig=str(d["kubeconfig"]))
    except (InvalidTag, KeyError, ValueError, TypeError):
        raise BundleError("login bundle is damaged (your entry does not decrypt); get a fresh copy") from None
