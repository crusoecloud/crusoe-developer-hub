"""The item spec: what a watch is looking for.

A spec is a plain dict so it round-trips through JSON and model output. `normalize_spec`
turns whatever a model or a person supplied into the canonical shape documented in
references/spec-schema.md, and `validate_spec` says what is still wrong with it.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Mapping, Optional

# The fields a listing is judged on, in the order they appear on a card's proof line.
MATCH_FIELDS = ("brand", "model", "style_code", "colorway", "size", "department")
REQUIREMENTS = ("required", "optional", "na")
SIZE_SYSTEMS = ("US", "UK", "EU", "JP", "CM", "alpha", "waist_inseam")
DEPARTMENTS = ("men", "women", "unisex", "kids")
KIDS_TIERS = ("GS", "PS", "TD")
BUYING_OPTIONS = ("FIXED_PRICE", "AUCTION", "BEST_OFFER")
STRICTNESS = ("normal", "strict")

_ALPHA_SIZES = {"XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"}


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: List[str] = []
    for item in value:
        text = _clean(item)
        if text and text not in out:
            out.append(text)
    return out


def parse_size_text(text: Any) -> Optional[Dict[str, Any]]:
    """'US 10', 'UK 9.5', 'EU 44', 'M', '32x34', '10' -> {'value', 'system', 'width'}."""
    raw = _clean(text)
    if not raw:
        return None
    upper = raw.upper()
    waist = re.fullmatch(r"W?\s*(\d{2})\s*[X/]\s*L?\s*(\d{2})", upper)
    if waist:
        return {"value": "%sx%s" % (waist.group(1), waist.group(2)), "system": "waist_inseam", "width": None}
    system = None
    prefixed = re.match(r"^(US|UK|EUR|EU|JP|CM)\s*[:.\-]?\s*(\d.*)$", upper)
    if prefixed:
        system = "EU" if prefixed.group(1) == "EUR" else prefixed.group(1)
        upper = prefixed.group(2).strip()
    else:
        suffixed = re.match(r"^(.+?)\s*(US|UK|EU|EUR|JP|CM)$", upper)
        if suffixed and re.search(r"\d", suffixed.group(1)):
            system = "EU" if suffixed.group(2) == "EUR" else suffixed.group(2)
            upper = suffixed.group(1).strip()
    if system is None and upper in _ALPHA_SIZES:
        system = "alpha"
    return {"value": upper, "system": system, "width": None}


def _normalize_size(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        size_value = _clean(value.get("value"))
        if not size_value:
            return None
        system = _clean(value.get("system"))
        if system:
            canonical = {s.lower(): s for s in SIZE_SYSTEMS}
            system = canonical.get(system.lower(), system)
        if system is None:
            parsed = parse_size_text(size_value)
            if parsed:
                size_value, system = parsed["value"], parsed["system"]
        return {"value": size_value.upper(), "system": system, "width": _clean(value.get("width"))}
    return parse_size_text(value)


def _normalize_colorway(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        out = {
            "official": _clean(value.get("official")),
            "nicknames": _str_list(value.get("nicknames")),
            "colors": [c.lower() for c in _str_list(value.get("colors") or value.get("colours"))],
        }
    else:
        out = {"official": _clean(value), "nicknames": [], "colors": []}
    if not out["official"] and not out["nicknames"] and not out["colors"]:
        return None
    return out


def _normalize_ceiling(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        amount, currency = value.get("amount"), _clean(value.get("currency")) or "USD"
        incl_shipping = bool(value.get("incl_shipping", False))
    else:
        amount, currency, incl_shipping = value, "USD", False
    try:
        amount_f = float(str(amount).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if amount_f <= 0:
        return None
    return {"amount": amount_f, "currency": currency.upper(), "incl_shipping": incl_shipping}


def default_requirements(spec: Mapping[str, Any]) -> Dict[str, str]:
    """A field the spec has a value for is required; one it lacks does not apply.

    Style code is the exception: sellers often leave it out, so it is optional. A
    mismatching code still rejects a listing; a missing one does not hold up an alert.
    """
    present = {
        "brand": bool(spec.get("brand")),
        "model": bool(spec.get("model")),
        "style_code": bool(spec.get("style_codes")),
        "colorway": bool(spec.get("colorway")),
        "size": bool(spec.get("size")),
        "department": bool(spec.get("department")),
    }
    out = {name: ("required" if has else "na") for name, has in present.items()}
    if present["style_code"]:
        out["style_code"] = "optional"
    return out


def normalize_spec(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce loose input (model output, user edits) into the canonical spec shape."""
    raw = dict(raw or {})
    department = _clean(raw.get("department"))
    kids_tier = _clean(raw.get("kids_tier"))
    if department:
        low = department.lower()
        if low in ("mens", "men's", "male", "man"):
            department = "men"
        elif low in ("womens", "women's", "female", "woman", "ladies"):
            department = "women"
        elif low in ("gs", "ps", "td"):
            department, kids_tier = "kids", low.upper()
        elif low in ("kid", "youth", "boys", "girls", "children", "child"):
            department = "kids"
        else:
            department = low
    if kids_tier:
        kids_tier = kids_tier.upper()

    style_codes = _str_list(raw.get("style_codes") or raw.get("style_code"))
    buying = [b.upper() for b in _str_list(raw.get("buying_options"))] or list(BUYING_OPTIONS)
    strictness = (_clean(raw.get("strictness")) or "normal").lower()

    spec: Dict[str, Any] = {
        "brand": _clean(raw.get("brand")),
        "model": _clean(raw.get("model")),
        "style_codes": [code.upper() for code in style_codes],
        "colorway": _normalize_colorway(raw.get("colorway") or raw.get("colourway")),
        "size": _normalize_size(raw.get("size")),
        "department": department,
        "kids_tier": kids_tier,
        "conditions": [c.upper() for c in _str_list(raw.get("conditions"))],
        "price_ceiling": _normalize_ceiling(raw.get("price_ceiling")),
        "buying_options": [b for b in buying if b in BUYING_OPTIONS] or list(BUYING_OPTIONS),
        "must_have": _str_list(raw.get("must_have")),
        "must_not": _str_list(raw.get("must_not")),
        "visual_signature": _str_list(raw.get("visual_signature")),
        "reference_images": _str_list(raw.get("reference_images")),
        "strictness": strictness if strictness in STRICTNESS else "normal",
        "provenance": dict(raw.get("provenance") or {}),
    }
    requirements = default_requirements(spec)
    supplied = raw.get("requirements") or {}
    if isinstance(supplied, Mapping):
        for name, value in supplied.items():
            if name in MATCH_FIELDS and str(value).lower() in REQUIREMENTS:
                requirements[name] = str(value).lower()
    spec["requirements"] = requirements
    return spec


def validate_spec(spec: Mapping[str, Any]) -> List[str]:
    """Return human-readable problems. An empty list means the watch can be activated."""
    problems: List[str] = []
    if not (spec.get("brand") or spec.get("model") or spec.get("style_codes")):
        problems.append("needs at least a brand, a model or a style code to search for")
    size = spec.get("size")
    requirements = spec.get("requirements") or {}
    if size:
        if size.get("system") is None:
            problems.append("size %s has no size system (US, UK, EU, alpha...)" % size.get("value"))
        elif size["system"] not in SIZE_SYSTEMS:
            problems.append("unknown size system: %s" % size["system"])
    if requirements.get("size") == "required" and not size:
        problems.append("size is required but not set")
    department = spec.get("department")
    if department and department not in DEPARTMENTS:
        problems.append("unknown department: %s" % department)
    kids_tier = spec.get("kids_tier")
    if kids_tier and kids_tier not in KIDS_TIERS:
        problems.append("unknown kids tier: %s" % kids_tier)
    if kids_tier and department != "kids":
        problems.append("kids tier is set but department is not kids")
    for name, value in requirements.items():
        if name not in MATCH_FIELDS or value not in REQUIREMENTS:
            problems.append("bad requirement: %s=%s" % (name, value))
    if all(requirements.get(name, "na") != "required" for name in MATCH_FIELDS):
        problems.append("no field is required, so nothing could ever be proven a match")
    return problems


def spec_summary(spec: Mapping[str, Any]) -> str:
    """One line a person can check at a glance."""
    parts: List[str] = []
    name = " ".join(p for p in (spec.get("brand"), spec.get("model")) if p)
    if name:
        parts.append(name)
    colorway = spec.get("colorway") or {}
    color_bits = [b for b in ([colorway.get("official")] + list(colorway.get("nicknames") or [])) if b]
    if color_bits:
        parts.append(" / ".join(color_bits))
    size = spec.get("size")
    if size:
        system = size.get("system")
        label = size["value"] if system in (None, "alpha", "waist_inseam") else "%s %s" % (system, size["value"])
        if system is None:
            label += " (size system not set)"
        parts.append("size " + label)
    if spec.get("department"):
        parts.append(spec["department"] + (" " + spec["kids_tier"] if spec.get("kids_tier") else ""))
    if spec.get("style_codes"):
        parts.append(", ".join(spec["style_codes"]))
    ceiling = spec.get("price_ceiling")
    if ceiling:
        parts.append("up to %s %g%s" % (ceiling["currency"], ceiling["amount"],
                                        " incl. shipping" if ceiling.get("incl_shipping") else ""))
    return " · ".join(parts) if parts else "(empty spec)"


def apply_patch(spec: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply a feedback patch and return a new normalized spec.

    Patch keys: set {field: value}, add_must_not [..], add_must_have [..],
    add_nicknames [..], set_requirement {field: required|optional|na}.
    """
    out = copy.deepcopy(dict(spec))
    for name, value in (patch.get("set") or {}).items():
        out[name] = value
    out["must_not"] = _str_list(list(out.get("must_not") or []) + _str_list(patch.get("add_must_not")))
    out["must_have"] = _str_list(list(out.get("must_have") or []) + _str_list(patch.get("add_must_have")))
    nicknames = _str_list(patch.get("add_nicknames"))
    if nicknames:
        colorway = dict(out.get("colorway") or {"official": None, "nicknames": [], "colors": []})
        colorway["nicknames"] = _str_list(list(colorway.get("nicknames") or []) + nicknames)
        out["colorway"] = colorway
    requirements = dict(out.get("requirements") or {})
    for name, value in (patch.get("set_requirement") or {}).items():
        requirements[name] = value
    out["requirements"] = requirements
    return normalize_spec(out)
