"""Deterministic checks and the decision rule.

The models supply evidence per field; this module makes the call. Nothing here asks
a model anything, and no model-reported confidence number is used anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .spec import MATCH_FIELDS

MATCH, MISMATCH, UNKNOWN = "match", "mismatch", "unknown"
VERDICTS = (MATCH, MISMATCH, UNKNOWN)
# Flags a model may raise after reading a listing in context. Any of them rejects.
BLOCKING_FLAGS = ("replica", "custom", "lot", "box_only", "parts")

_ALPHA = {
    "XXS": "XXS", "2XS": "XXS", "XS": "XS", "EXTRA SMALL": "XS", "X-SMALL": "XS",
    "S": "S", "SMALL": "S", "M": "M", "MEDIUM": "M", "MED": "M",
    "L": "L", "LARGE": "L", "XL": "XL", "EXTRA LARGE": "XL", "X-LARGE": "XL",
    "XXL": "XXL", "2XL": "XXL", "XXXL": "XXXL", "3XL": "XXXL",
}
ALPHA_SIZES = frozenset(_ALPHA.values())


@dataclass
class Decision:
    decision: str                      # alert | maybe | reject | parked
    reason: str
    unproven: List[str] = field(default_factory=list)


def decide(
    requirements: Mapping[str, str],
    fields: Mapping[str, str],
    flags: Iterable[str] = (),
    over_ceiling: bool = False,
    size_out_of_stock: bool = False,
) -> Decision:
    """Apply the match rule.

    - A blocking flag or a mismatch on any applicable field rejects.
    - A listing that only fails on something that can change (price, stock) is parked.
    - Every required field proven: alert. Otherwise: maybe, naming what is unproven.
    """
    raised = [f for f in flags if f in BLOCKING_FLAGS]
    if raised:
        return Decision("reject", "flag:" + raised[0])
    for name in MATCH_FIELDS:
        if requirements.get(name, "na") == "na":
            continue
        if fields.get(name, UNKNOWN) == MISMATCH:
            return Decision("reject", "mismatch:" + name)
    unproven = [
        name for name in MATCH_FIELDS
        if requirements.get(name) == "required" and fields.get(name, UNKNOWN) != MATCH
    ]
    if size_out_of_stock:
        return Decision("parked", "size_out_of_stock", unproven)
    if over_ceiling:
        return Decision("parked", "over_ceiling", unproven)
    if not unproven:
        return Decision("alert", "all_required_fields_proven")
    return Decision("maybe", "unproven:" + ",".join(unproven), unproven)


# ---- size ------------------------------------------------------------------

def normalize_size_value(value: Any) -> Optional[str]:
    """'10.0' -> '10', '10 1/2' -> '10.5', 'Medium' -> 'M', '7y' -> '7Y'."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    if not text:
        return None
    if text in _ALPHA:
        return _ALPHA[text]
    half = re.fullmatch(r"(\d+)\s+1/2\s*([YCK])?", text)
    if half:
        return "%g%s" % (int(half.group(1)) + 0.5, half.group(2) or "")
    number = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*([YCK])?", text)
    if number:
        return "%g%s" % (float(number.group(1).replace(",", ".")), number.group(2) or "")
    return text


def _aspect_size_system(name: str) -> Optional[str]:
    low = name.lower()
    if re.search(r"\bus\b", low):
        return "US"
    if re.search(r"\buk\b", low):
        return "UK"
    if re.search(r"\b(eu|eur|euro|european)\b", low):
        return "EU"
    if re.search(r"\b(jp|japan)\b", low):
        return "JP"
    if re.search(r"\bcm\b", low):
        return "CM"
    return None


def sizes_from_aspects(aspects: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """Pull stated sizes out of eBay item specifics such as 'US Shoe Size': ['10']."""
    out: List[Dict[str, Any]] = []
    for name, values in (aspects or {}).items():
        low = name.strip().lower()
        if "size" not in low or low in ("size type",):
            continue
        system = _aspect_size_system(name)
        if isinstance(values, str):
            values = [values]
        for value in values:
            normalized = normalize_size_value(value)
            if normalized is None:
                continue
            row_system = system
            if row_system is None and normalized in ALPHA_SIZES:
                row_system = "alpha"
            out.append({"value": normalized, "system": row_system, "source": "item specifics"})
    return out


def check_size(spec_size: Optional[Mapping[str, Any]], stated: Sequence[Mapping[str, Any]]) -> str:
    """Size is proven only by a size the listing states in the watch's own size system.

    A size in another system is never converted: it stays unknown.
    """
    if not spec_size or not spec_size.get("value") or not spec_size.get("system"):
        return UNKNOWN
    wanted = normalize_size_value(spec_size["value"])
    same_system = [s for s in stated if s.get("system") == spec_size["system"]]
    if not same_system:
        return UNKNOWN
    return MATCH if any(s.get("value") == wanted for s in same_system) else MISMATCH


# ---- department ------------------------------------------------------------

# Only unambiguous markers count here. Words such as "kids", "boys" or "youth" also
# appear in brand and band names (Kids See Ghosts, Beastie Boys, Sonic Youth), so
# they are left for the model to read in context.
_KIDS_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(GS|PS|TD)(?![A-Za-z0-9])")
_KIDS_WORD_RE = re.compile(
    r"\b(grade[ -]school|pre[ -]?school|toddlers?)\b"
    r"|(?<![A-Za-z0-9.])\d{1,2}(?:\.5)?Y(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def has_kids_markers(title: str) -> bool:
    """True when a title states a kids' size outright: GS, PS, TD, Grade School, 7Y."""
    return bool(_KIDS_TOKEN_RE.search(title or "") or _KIDS_WORD_RE.search(title or ""))


def _aspect_department(aspects: Mapping[str, Sequence[str]]) -> Optional[str]:
    for name, values in (aspects or {}).items():
        if name.strip().lower() not in ("department", "gender"):
            continue
        if isinstance(values, str):
            values = [values]
        text = " ".join(values).lower()
        if any(word in text for word in ("kid", "boy", "girl", "baby", "toddler", "youth")):
            return "kids"
        if "unisex" in text:
            return "unisex"
        if "women" in text:
            return "women"
        if "men" in text:
            return "men"
    return None


def check_department(spec: Mapping[str, Any], aspects: Mapping[str, Sequence[str]], title: str = "") -> str:
    wanted = spec.get("department")
    if not wanted:
        return UNKNOWN
    listed = _aspect_department(aspects)
    kids_in_title = has_kids_markers(title)
    if wanted == "kids":
        if listed == "kids" or (listed is None and kids_in_title):
            return MATCH
        return MISMATCH if listed in ("men", "women", "unisex") else UNKNOWN
    if listed == "kids" or (listed is None and kids_in_title):
        return MISMATCH
    if listed is None:
        return UNKNOWN
    if wanted == "unisex" or listed == "unisex" or listed == wanted:
        return MATCH
    return MISMATCH


# ---- style code ------------------------------------------------------------

_CODE_ASPECTS = ("style code", "mpn", "manufacturer part number", "sku", "model number", "style number")
_PLACEHOLDERS = {"DOESNOTAPPLY", "NA", "NONE", "UNKNOWN", "SEEDESCRIPTION", "NOTAPPLICABLE", "UNBRANDED"}


def normalize_code(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def check_style_code(spec: Mapping[str, Any], aspects: Mapping[str, Sequence[str]], title: str = "") -> str:
    wanted = [normalize_code(code) for code in (spec.get("style_codes") or [])]
    wanted = [code for code in wanted if len(code) >= 5]
    if not wanted:
        return UNKNOWN
    listed: List[str] = []
    for name, values in (aspects or {}).items():
        if name.strip().lower() not in _CODE_ASPECTS:
            continue
        if isinstance(values, str):
            values = [values]
        for value in values:
            code = normalize_code(value)
            if len(code) >= 5 and code not in _PLACEHOLDERS and re.search(r"\d", code):
                listed.append(code)
    if any(code in listed for code in wanted):
        return MATCH
    squeezed_title = normalize_code(title)
    if any(code in squeezed_title for code in wanted):
        return MATCH
    return MISMATCH if listed else UNKNOWN


# ---- price and title rules -------------------------------------------------

def _amount(money: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not money:
        return None
    try:
        return float(money.get("value"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def over_ceiling(
    spec: Mapping[str, Any],
    price: Optional[Mapping[str, Any]],
    shipping: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True only when the price is known, in the ceiling's currency, and above it.

    This compares against a number the user typed. It is not a judgment about price.
    """
    ceiling = spec.get("price_ceiling")
    amount = _amount(price)
    if not ceiling or amount is None:
        return False
    if str((price or {}).get("currency", "")).upper() != ceiling["currency"]:
        return False
    if ceiling.get("incl_shipping"):
        amount += _amount(shipping) or 0.0
    return amount > float(ceiling["amount"])


def _own_words(spec: Mapping[str, Any]) -> str:
    """Brand and model text, so a term that is part of the item's own name is not a trap."""
    return " ".join(str(spec.get(k) or "") for k in ("brand", "model")).lower()


def title_conflict(spec: Mapping[str, Any], title: str) -> Optional[str]:
    """Stage 0: a reason to drop a listing on its title alone, or None.

    Only the user's own must-not terms are applied here. Words like 'replica' or
    'lot' are left to the model, which reads them in context.
    """
    low = (title or "").lower()
    own = _own_words(spec)
    for term in spec.get("must_not") or []:
        needle = term.strip().lower()
        if not needle or needle in own:
            continue
        if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(needle), low):
            return "must_not:" + needle
    wanted = spec.get("department")
    if wanted in ("men", "women", "unisex") and has_kids_markers(title):
        return "mismatch:department"
    return None


def proof_line(spec: Mapping[str, Any], fields: Mapping[str, str], sources: Mapping[str, str]) -> str:
    """'Size US 10 (item specifics) · DZ5485-612 (title) · colorway (photo)'."""
    parts: List[str] = []
    for name in MATCH_FIELDS:
        if fields.get(name) != MATCH:
            continue
        source = sources.get(name)
        if name == "size" and spec.get("size"):
            size = spec["size"]
            label = "Size %s%s" % ((size.get("system") + " ") if size.get("system") not in (None, "alpha") else "",
                                   size.get("value"))
        elif name == "style_code" and spec.get("style_codes"):
            label = spec["style_codes"][0]
        elif name == "colorway":
            colorway = spec.get("colorway") or {}
            label = "Colorway %s" % (colorway.get("official") or "/".join(colorway.get("nicknames") or []) or "")
        else:
            label = "%s %s" % (name.replace("_", " ").capitalize(), spec.get(name) or "")
        parts.append("%s (%s)" % (label.strip(), source) if source else label.strip())
    return " · ".join(parts)
