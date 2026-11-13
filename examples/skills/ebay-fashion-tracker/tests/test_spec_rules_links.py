import unittest

import support  # noqa: F401
from eft_core import rules, spec
from eft_core.links import canonical_item_url, parse_link
from eft_core.rules import MATCH, MISMATCH, UNKNOWN


def dunk_spec(**overrides):
    base = {
        "brand": "Nike", "model": "Dunk Low", "style_code": "dd1391-100",
        "colorway": {"official": "White/Black", "nicknames": ["Panda"]},
        "size": "US 10", "department": "Mens", "price_ceiling": 180,
    }
    base.update(overrides)
    return spec.normalize_spec(base)


class SpecNormalization(unittest.TestCase):
    def test_loose_input_becomes_canonical(self):
        s = dunk_spec()
        self.assertEqual(s["style_codes"], ["DD1391-100"])
        self.assertEqual(s["size"], {"value": "10", "system": "US", "width": None})
        self.assertEqual(s["department"], "men")
        self.assertEqual(s["colorway"]["nicknames"], ["Panda"])
        self.assertEqual(s["price_ceiling"], {"amount": 180.0, "currency": "USD", "incl_shipping": False})
        self.assertEqual(s["buying_options"], ["FIXED_PRICE", "AUCTION", "BEST_OFFER"])
        self.assertEqual(spec.validate_spec(s), [])

    def test_default_requirements(self):
        s = dunk_spec()
        self.assertEqual(s["requirements"]["brand"], "required")
        self.assertEqual(s["requirements"]["size"], "required")
        self.assertEqual(s["requirements"]["style_code"], "optional")
        vintage = spec.normalize_spec({"model": "chore jacket", "size": "L", "colorway": "faded indigo"})
        self.assertEqual(vintage["requirements"]["brand"], "na")
        self.assertEqual(vintage["requirements"]["style_code"], "na")
        self.assertEqual(vintage["size"]["system"], "alpha")
        self.assertEqual(spec.validate_spec(vintage), [])

    def test_size_text_forms(self):
        cases = {
            "US 10": ("10", "US"), "UK9.5": ("9.5", "UK"), "EUR 44": ("44", "EU"), "44 EU": ("44", "EU"),
            "m": ("M", "alpha"), "32x34": ("32x34", "waist_inseam"), "10": ("10", None), "27cm": ("27", "CM"),
        }
        for text, (value, system) in cases.items():
            parsed = spec.parse_size_text(text)
            self.assertEqual((parsed["value"], parsed["system"]), (value, system), text)

    def test_validation_reports_problems(self):
        self.assertIn("size 10 has no size system", " ".join(spec.validate_spec(dunk_spec(size="10"))))
        empty = spec.normalize_spec({})
        problems = " ".join(spec.validate_spec(empty))
        self.assertIn("needs at least a brand", problems)
        self.assertIn("no field is required", problems)
        kids = spec.normalize_spec({"brand": "Nike", "model": "Dunk", "department": "GS"})
        self.assertEqual((kids["department"], kids["kids_tier"]), ("kids", "GS"))
        self.assertEqual(spec.validate_spec(kids), [])

    def test_summary_and_patch(self):
        s = dunk_spec()
        summary = spec.spec_summary(s)
        for piece in ("Nike Dunk Low", "Panda", "size US 10", "men", "DD1391-100", "up to USD 180"):
            self.assertIn(piece, summary)
        patched = spec.apply_patch(s, {
            "add_must_not": ["GS", "gs"], "add_nicknames": ["Oreo"],
            "set_requirement": {"style_code": "required"}, "set": {"size": "US 10.5"},
        })
        self.assertEqual(patched["must_not"], ["GS", "gs"])
        self.assertEqual(patched["colorway"]["nicknames"], ["Panda", "Oreo"])
        self.assertEqual(patched["requirements"]["style_code"], "required")
        self.assertEqual(patched["size"]["value"], "10.5")
        self.assertEqual(s["size"]["value"], "10", "the original spec is not mutated")


class DecisionRule(unittest.TestCase):
    def setUp(self):
        self.req = dunk_spec()["requirements"]
        self.all_match = {name: MATCH for name in spec.MATCH_FIELDS}

    def test_alert_when_every_required_field_is_proven(self):
        fields = dict(self.all_match, style_code=UNKNOWN)   # optional field may stay unknown
        self.assertEqual(rules.decide(self.req, fields).decision, "alert")

    def test_any_mismatch_rejects_even_on_an_optional_field(self):
        decision = rules.decide(self.req, dict(self.all_match, style_code=MISMATCH))
        self.assertEqual((decision.decision, decision.reason), ("reject", "mismatch:style_code"))

    def test_unproven_required_field_is_a_maybe_that_names_it(self):
        decision = rules.decide(self.req, dict(self.all_match, size=UNKNOWN, colorway=UNKNOWN))
        self.assertEqual(decision.decision, "maybe")
        self.assertEqual(decision.unproven, ["colorway", "size"])

    def test_not_applicable_fields_are_ignored(self):
        req = dict(self.req, brand="na", style_code="na")
        fields = dict(self.all_match, brand=MISMATCH, style_code=MISMATCH)
        self.assertEqual(rules.decide(req, fields).decision, "alert")

    def test_blocking_flags_reject_and_unknown_flags_do_not(self):
        self.assertEqual(rules.decide(self.req, self.all_match, flags=["replica"]).reason, "flag:replica")
        self.assertEqual(rules.decide(self.req, self.all_match, flags=["stock_photo"]).decision, "alert")

    def test_changeable_failures_park_instead_of_rejecting(self):
        self.assertEqual(rules.decide(self.req, self.all_match, over_ceiling=True).decision, "parked")
        parked = rules.decide(self.req, self.all_match, size_out_of_stock=True)
        self.assertEqual((parked.decision, parked.reason), ("parked", "size_out_of_stock"))
        # A real mismatch still wins over parking.
        self.assertEqual(rules.decide(self.req, dict(self.all_match, model=MISMATCH), over_ceiling=True).decision,
                         "reject")


class SizeRules(unittest.TestCase):
    def test_normalize_size_value(self):
        cases = {"10.0": "10", "10 1/2": "10.5", "Medium": "M", "2xl": "XXL", "7y": "7Y", "44,5": "44.5", " ": None}
        for raw, expected in cases.items():
            self.assertEqual(rules.normalize_size_value(raw), expected, raw)

    def test_sizes_from_aspects(self):
        aspects = {"US Shoe Size": ["10"], "UK Shoe Size": ["9"], "EU Shoe Size": ["44"],
                   "Size Type": ["Regular"], "Brand": ["Nike"], "Size": ["M"]}
        stated = rules.sizes_from_aspects(aspects)
        self.assertEqual({(s["system"], s["value"]) for s in stated},
                         {("US", "10"), ("UK", "9"), ("EU", "44"), ("alpha", "M")})

    def test_check_size_uses_only_the_watchs_own_system(self):
        want = {"value": "10", "system": "US"}
        self.assertEqual(rules.check_size(want, [{"value": "10", "system": "US"}]), MATCH)
        self.assertEqual(rules.check_size(want, [{"value": "10.5", "system": "US"}]), MISMATCH)
        # A UK or EU size is never converted: it proves nothing either way.
        self.assertEqual(rules.check_size(want, [{"value": "9", "system": "UK"}]), UNKNOWN)
        self.assertEqual(rules.check_size(want, [{"value": "10", "system": None}]), UNKNOWN)
        self.assertEqual(rules.check_size(want, []), UNKNOWN)
        # A kids' 7Y is not a men's 7.
        self.assertEqual(rules.check_size({"value": "7", "system": "US"}, [{"value": "7Y", "system": "US"}]), MISMATCH)
        self.assertEqual(rules.check_size({"value": "10", "system": None}, [{"value": "10", "system": "US"}]), UNKNOWN)


class DepartmentAndStyleCode(unittest.TestCase):
    def test_kids_markers_are_strict(self):
        for title in ("Nike Dunk Low GS Panda", "Jordan 1 (PS) Bred", "Dunk Low Grade School", "Nike Dunk sz 7Y"):
            self.assertTrue(rules.has_kids_markers(title), title)
        for title in ("Nike Dunk Low Panda Men's 10", "PS5 bundle", "Kids See Ghosts hoodie L",
                      "Beastie Boys tour tee", "Sonic Youth Goo shirt", "TDE hoodie"):
            self.assertFalse(rules.has_kids_markers(title), title)

    def test_check_department(self):
        men = dunk_spec()
        self.assertEqual(rules.check_department(men, {"Department": ["Men"]}), MATCH)
        self.assertEqual(rules.check_department(men, {"Department": ["Unisex Adults"]}), MATCH)
        self.assertEqual(rules.check_department(men, {"Gender": ["Men"]}), MATCH)
        self.assertEqual(rules.check_department(men, {"Gender": ["Women"]}), MISMATCH)
        self.assertEqual(rules.check_department(men, {"Department": ["Women"]}), MISMATCH)
        self.assertEqual(rules.check_department(men, {"Department": ["Unisex Kids"]}), MISMATCH)
        self.assertEqual(rules.check_department(men, {}, "Nike Dunk Low GS"), MISMATCH)
        self.assertEqual(rules.check_department(men, {}, "Nike Dunk Low"), UNKNOWN)
        kids = dunk_spec(department="kids")
        self.assertEqual(rules.check_department(kids, {}, "Nike Dunk Low GS"), MATCH)
        self.assertEqual(rules.check_department(kids, {"Department": ["Men"]}), MISMATCH)
        self.assertEqual(rules.check_department(kids, {}, "Nike Dunk Low"), UNKNOWN)

    def test_check_style_code(self):
        s = dunk_spec()
        self.assertEqual(rules.check_style_code(s, {"Style Code": ["DD1391 100"]}), MATCH)
        self.assertEqual(rules.check_style_code(s, {}, "Nike Dunk Low Panda dd1391-100 sz 10"), MATCH)
        self.assertEqual(rules.check_style_code(s, {"Style Code": ["CW1590-100"]}), MISMATCH)
        self.assertEqual(rules.check_style_code(s, {"MPN": ["Does not apply"]}), UNKNOWN)
        self.assertEqual(rules.check_style_code(s, {}, "Nike Dunk Low Panda"), UNKNOWN)
        self.assertEqual(rules.check_style_code(dunk_spec(style_code=None), {"Style Code": ["X1234-1"]}), UNKNOWN)


class PriceAndTitleRules(unittest.TestCase):
    def test_over_ceiling(self):
        s = dunk_spec()
        self.assertTrue(rules.over_ceiling(s, {"value": "180.01", "currency": "USD"}))
        self.assertFalse(rules.over_ceiling(s, {"value": "180.00", "currency": "USD"}))
        self.assertFalse(rules.over_ceiling(s, {"value": "500", "currency": "GBP"}), "other currency is not judged")
        self.assertFalse(rules.over_ceiling(s, None))
        with_shipping = dunk_spec(price_ceiling={"amount": 180, "incl_shipping": True})
        self.assertTrue(rules.over_ceiling(with_shipping, {"value": "175", "currency": "USD"},
                                           {"value": "10", "currency": "USD"}))

    def test_title_conflict_applies_user_terms_but_not_the_items_own_name(self):
        s = dunk_spec(must_not=["Disrupt", "low"])
        self.assertEqual(rules.title_conflict(s, "Nike Dunk Low Disrupt 2 Panda"), "must_not:disrupt")
        self.assertIsNone(rules.title_conflict(s, "Nike Dunk Low Panda"), "'low' is part of the model name")
        self.assertEqual(rules.title_conflict(s, "Nike Dunk Low Panda GS 6Y"), "mismatch:department")
        margiela = spec.normalize_spec({"brand": "Maison Margiela", "model": "Replica", "must_not": ["replica"]})
        self.assertIsNone(rules.title_conflict(margiela, "Maison Margiela Replica GAT sneakers 43"))

    def test_trap_words_are_not_banned_at_stage_zero(self):
        s = dunk_spec()
        for title in ("Nike Dunk Low Panda replica box", "Lot of 2 Nike Dunk Low", "Nike Dunk Low custom"):
            self.assertIsNone(rules.title_conflict(s, title), title)

    def test_proof_line(self):
        s = dunk_spec()
        fields = {"size": MATCH, "style_code": MATCH, "colorway": MATCH, "brand": UNKNOWN}
        line = rules.proof_line(s, fields, {"size": "item specifics", "style_code": "title", "colorway": "photo"})
        self.assertEqual(line, "DD1391-100 (title) · Colorway White/Black (photo) · Size US 10 (item specifics)")


class Links(unittest.TestCase):
    def test_item_links(self):
        for url in ("https://www.ebay.com/itm/256012345678",
                    "https://www.ebay.com/itm/nike-dunk-low-panda/256012345678?hash=item3b9&var=555666777",
                    "ebay.com/itm/256012345678?_trkparms=abc"):
            link = parse_link(url)
            self.assertEqual((link.kind, link.legacy_item_id, link.marketplace), ("item", "256012345678", "EBAY_US"), url)
        self.assertEqual(parse_link("https://www.ebay.com/itm/x/256012345678?var=555666777").variation_id, "555666777")
        self.assertIsNone(parse_link("https://www.ebay.com/itm/256012345678?var=0").variation_id)
        self.assertEqual(parse_link("https://www.ebay.co.uk/itm/256012345678").marketplace, "EBAY_GB")
        self.assertEqual(parse_link("https://www.ebay.com.au/itm/256012345678").marketplace, "EBAY_AU")

    def test_product_and_search_links(self):
        product = parse_link("https://www.ebay.com/p/1234567890?iid=1")
        self.assertEqual((product.kind, product.epid), ("product", "1234567890"))
        search = parse_link("https://www.ebay.com/sch/i.html?_nkw=nike+dunk+panda&_sacat=15709&LH_Auction=1"
                            "&_udlo=50&_udhi=180&LH_ItemCondition=1000%7C3000")
        self.assertEqual(search.kind, "search")
        self.assertEqual(search.search, {
            "q": "nike dunk panda", "category_id": "15709", "buying_options": ["AUCTION"],
            "price_min": 50.0, "price_max": 180.0, "condition_ids": ["1000", "3000"],
        })

    def test_short_links_and_other_sites(self):
        self.assertEqual(parse_link("https://ebay.us/m/abc123").kind, "short")
        self.assertEqual(parse_link("https://ebay.to/3xYz").kind, "short")
        self.assertEqual(parse_link("https://www.nike.com/t/dunk-low/DD1391-100").kind, "not_ebay")
        self.assertEqual(parse_link("https://www.ebay.com/usr/someone").kind, "other_ebay")
        self.assertEqual(parse_link("").kind, "not_ebay")
        self.assertFalse(parse_link("https://notebay.com/itm/256012345678").is_ebay)

    def test_canonical_item_url(self):
        self.assertEqual(canonical_item_url("256012345678"), "https://www.ebay.com/itm/256012345678")
        self.assertEqual(canonical_item_url("1", "EBAY_GB"), "https://www.ebay.co.uk/itm/1")


if __name__ == "__main__":
    unittest.main()
