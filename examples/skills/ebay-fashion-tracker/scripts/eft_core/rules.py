"""Deterministic checks and the decision rule.

The models supply evidence per field; this module makes the call. Nothing here asks
a model anything, and no model-reported confidence number is used anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .spec import MATCH_FIELDS, parse_size_text

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

# A waist and an inseam in inches, the way jeans are sized: "32x34", "32 X 34", "32/34",
# "W32 L34", "32W x 34L". The pair must stand on its own: "132x34", "32x34.5" and "32x34-36" are not one.
_WAIST_INSEAM_RE = re.compile(
    r"(?<![\w.])(?:"
    r"W?\s*(\d{2})\s*[X/]\s*L?\s*(\d{2})"
    r"|W\s*(\d{2})\s*L\s*(\d{2})"
    r"|(\d{2})\s*W\s*[X/]?\s*(\d{2})\s*L"
    r")(?!\w|[.-]\d)"
)
_WAIST_INSEAM_VALUE_RE = re.compile(r"\d{2}X\d{2}")
_INCHES_RE = re.compile(r"(\d{2})(?:\.0)?\s*(?:IN\.?|INCH|INCHES|\")?")


def _waist_by_inseam(match: re.Match[str]) -> str:
    """The normalized form of a waist/inseam size: waist 32, inseam 34 is '32X34'."""
    return "X".join(number for number in match.groups() if number)


def waist_inseam_sizes(text: Any) -> List[str]:
    """Every waist/inseam size written in a text, normalized: 'W32 L34, also 34x34' -> ['32X34', '34X34']."""
    return [_waist_by_inseam(match) for match in _WAIST_INSEAM_RE.finditer(str(text or "").upper())]


def normalize_size_value(value: Any, system: Optional[str] = None) -> Optional[str]:
    """'10.0' -> '10', '10 1/2' -> '10.5', 'Medium' -> 'M', '7y' -> '7Y', 'W32 L34' -> '32X34'.

    `system` is the size system the value is known to belong to, if any. A waist/inseam pair
    is read as one only where it can be one: under 'US Shoe Size', '11/12' is two shoe sizes.
    """
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    if not text:
        return None
    if text in _ALPHA:
        return _ALPHA[text]
    pair = _WAIST_INSEAM_RE.fullmatch(text) if system in (None, "waist_inseam") else None
    if pair:
        return _waist_by_inseam(pair)
    half = re.fullmatch(r"(\d+)\s+1/2\s*([YCK])?", text)
    if half:
        return "%g%s" % (int(half.group(1)) + 0.5, half.group(2) or "")
    number = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*([YCK])?", text)
    if number:
        return "%g%s" % (float(number.group(1).replace(",", ".")), number.group(2) or "")
    return text


# A normalized value that is one size and nothing else: 10, 10.5, 7Y, 44 2/3.
_PLAIN_SIZE_RE = re.compile(r"\d+(?:\.\d+)?(?: \d/\d)?[YCK]?")


def _is_waist_inseam(value: Any) -> bool:
    return bool(_WAIST_INSEAM_VALUE_RE.fullmatch(str(value or "")))


def _is_plain_size(value: Any, system: Optional[str] = None) -> bool:
    """True for a normalized value code can compare: '10', '7Y', '44 2/3', 'M'. Not "MEN'S 10" or '10 M'.

    For a waist/inseam watch the one comparable form is the pair, '32X34'. Under any other
    system a pair such as '11/12' is two sizes, which is not something code can compare.
    """
    if system == "waist_inseam":
        return _is_waist_inseam(value)
    return value in ALPHA_SIZES or bool(_PLAIN_SIZE_RE.fullmatch(str(value or "")))


def _own_system(value: Any) -> Optional[str]:
    """The size system a normalized value shows by its form, for an aspect that names none ('Size')."""
    if value in ALPHA_SIZES:
        return "alpha"
    return "waist_inseam" if _is_waist_inseam(value) else None


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


def _without_own_system(value: Any, system: Optional[str]) -> Any:
    """'US 10' or '10 US' under a US size aspect is the size 10: the seller repeated the system."""
    named = parse_size_text(value) if system else None
    return named["value"] if named and named["system"] == system else value


_WAIST_ASPECTS = ("waist size", "waist")
_INSEAM_ASPECTS = ("inseam",)


def _inches(aspects: Mapping[str, Sequence[str]], names: Sequence[str]) -> Optional[str]:
    """The whole number of inches the named item specific gives, when it gives exactly one."""
    found: List[str] = []
    for name, values in aspects.items():
        if name.strip().lower() not in names:
            continue
        for value in [values] if isinstance(values, str) else values:
            inches = _INCHES_RE.fullmatch(str(value).strip().upper())
            found.append(inches.group(1) if inches else "")
    return found[0] if len(found) == 1 and found[0] else None


def _measured_waist_inseam(aspects: Mapping[str, Sequence[str]]) -> Optional[str]:
    """'Waist Size': ['32 in'] beside 'Inseam': ['34 in'] states the jeans size '32X34'."""
    waist, inseam = _inches(aspects, _WAIST_ASPECTS), _inches(aspects, _INSEAM_ASPECTS)
    return "%sX%s" % (waist, inseam) if waist and inseam else None


def sizes_from_aspects(aspects: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """Pull stated sizes out of eBay item specifics such as 'US Shoe Size': ['10'].

    Sellers can type their own values, so a row's value is not always a plain size.
    `check_size` decides what such a value can prove. A jeans size given as two measurements
    ('Waist Size' and 'Inseam') is added as one row, marked "measured".
    """
    out: List[Dict[str, Any]] = []
    for name, values in (aspects or {}).items():
        low = name.strip().lower()
        if "size" not in low or low in ("size type",):
            continue
        system = _aspect_size_system(name)
        if isinstance(values, str):
            values = [values]
        for value in values:
            normalized = normalize_size_value(_without_own_system(value, system), system)
            if normalized is None:
                continue
            out.append({"value": normalized, "system": system or _own_system(normalized),
                        "source": "item specifics"})
    measured = _measured_waist_inseam(aspects or {})
    if measured:
        out.append({"value": measured, "system": "waist_inseam", "source": "item specifics", "measured": True})
    return out


def check_size(spec_size: Optional[Mapping[str, Any]], stated: Sequence[Mapping[str, Any]]) -> str:
    """Size is proven only by a size the listing states in the watch's own size system.

    A size in another system is never converted: it stays unknown. So does a value that is
    not one plain size ("Men's 10", "10 M", "10.5 / UK 9.5"): it can match the wanted size
    word for word, but it cannot contradict it. What the seller meant is left for the model
    to read in context, so a real match is not rejected over how a value was typed.

    A waist and an inseam given as two measurements can prove a jeans size and cannot
    contradict it either: a seller may have measured a worn pair whose label says otherwise.
    """
    if not spec_size or not spec_size.get("value") or not spec_size.get("system"):
        return UNKNOWN
    wanted = normalize_size_value(spec_size["value"], spec_size["system"])
    same_system = [s for s in stated if s.get("system") == spec_size["system"]]
    if any(s.get("value") == wanted for s in same_system):
        return MATCH
    labeled = [s.get("value") for s in same_system if not s.get("measured")]
    plain = all(_is_plain_size(value, spec_size["system"]) for value in [wanted] + labeled)
    return MISMATCH if labeled and plain else UNKNOWN


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


def _kids_markers(text: str) -> List[str]:
    """Every outright kids' size marker in a text, upper-cased: ['GS', '6Y']."""
    text = text or ""
    markers = _KIDS_TOKEN_RE.findall(text) + [hit.group(0) for hit in _KIDS_WORD_RE.finditer(text)]
    return [marker.upper() for marker in markers]


def has_kids_markers(title: str) -> bool:
    """True when a title states a kids' size outright: GS, PS, TD, Grade School, 7Y."""
    return bool(_kids_markers(title))


def _own_words(spec: Mapping[str, Any]) -> str:
    """Brand and model text, so a term that is part of the item's own name is not a trap."""
    return " ".join(str(spec.get(k) or "") for k in ("brand", "model")).lower()


def _kids_size_in_title(spec: Mapping[str, Any], title: str) -> bool:
    """A kids' size marker in the title that is not part of the watch's own brand or model.

    The PS in "PS Paul Smith" is the brand, not Preschool. A marker the item's own name
    carries cannot count against an adult watch; what the title means by it is left for the
    model and the item specifics.
    """
    own = set(_kids_markers(_own_words(spec).upper()))
    return any(marker not in own for marker in _kids_markers(title))


_KIDS_DEPARTMENT_WORDS = ("kid", "boy", "girl", "baby", "toddler", "youth")


def _departments_named(value: str) -> Set[str]:
    """The departments one item-specific value names: 'Unisex Kids' is kids, 'Men, Women' is both."""
    low = value.lower()
    if any(word in low for word in _KIDS_DEPARTMENT_WORDS):
        return {"kids"}
    if "unisex" in low:
        return {"unisex"}
    named: Set[str] = set()
    if "women" in low:
        named.add("women")
    if re.search(r"(?<!wo)men", low):
        named.add("men")
    return named


def _aspect_departments(aspects: Mapping[str, Sequence[str]]) -> Set[str]:
    """Every department the Department and Gender item specifics name. Empty when they say nothing."""
    named: Set[str] = set()
    for name, values in (aspects or {}).items():
        if name.strip().lower() not in ("department", "gender"):
            continue
        if isinstance(values, str):
            values = [values]
        for value in values:
            named |= _departments_named(str(value))
    return named


def check_department(spec: Mapping[str, Any], aspects: Mapping[str, Sequence[str]], title: str = "") -> str:
    """Compare the wanted department with the one the listing states.

    Item specifics decide when they name a department. A listing can name several ("Men" and
    "Women"), and it is a mismatch only when the wanted one is not among them. When they say
    nothing, an outright kids' size in the title is the only thing code will read.
    """
    wanted = spec.get("department")
    if not wanted:
        return UNKNOWN
    listed = _aspect_departments(aspects)
    if wanted == "kids":
        if listed:
            return MATCH if "kids" in listed else MISMATCH
        return MATCH if has_kids_markers(title) else UNKNOWN
    if not listed:
        return MISMATCH if _kids_size_in_title(spec, title) else UNKNOWN
    adults = listed - {"kids"}
    if wanted == "unisex":
        return MATCH if adults else MISMATCH
    return MATCH if adults & {wanted, "unisex"} else MISMATCH


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


def title_conflict(spec: Mapping[str, Any], title: str) -> Optional[str]:
    """Stage 0: a reason to drop a listing on its title alone, or None.

    Only the user's own must-not terms and an outright kids' size on an adult watch are
    applied here, and neither when the word is part of the watch's own brand or model.
    Words like 'replica' or 'lot' are left to the model, which reads them in context.
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
    # A watch whose department is marked "not applicable" is not judged on it, here or later.
    applies = (spec.get("requirements") or {}).get("department", "na") != "na"
    if applies and wanted in ("men", "women", "unisex") and _kids_size_in_title(spec, title):
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
            # A letter size or a waist/inseam pair is not prefixed: "Size waist_inseam 32x34" says nothing.
            named = size.get("system") not in (None, "alpha", "waist_inseam")
            label = "Size %s%s" % ((size.get("system") + " ") if named else "", size.get("value"))
        elif name == "style_code" and spec.get("style_codes"):
            label = spec["style_codes"][0]
        elif name == "colorway":
            colorway = spec.get("colorway") or {}
            label = "Colorway %s" % (colorway.get("official") or "/".join(colorway.get("nicknames") or []) or "")
        else:
            label = "%s %s" % (name.replace("_", " ").capitalize(), spec.get(name) or "")
        parts.append("%s (%s)" % (label.strip(), source) if source else label.strip())
    return " · ".join(parts)
