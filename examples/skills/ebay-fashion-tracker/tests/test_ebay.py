import base64
import json
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone

import support
from eft_core import rules
from eft_core.config import ConfigError
from eft_core.ebay import EbayClient, detail_from, strip_html, summarize
from eft_core.errors import (
    EbayAuthError,
    EbayBudgetExceeded,
    EbayError,
    EbayGroupRequired,
    EbayNotFound,
    EbayUnavailable,
)
from eft_core.http import FakeTransport, Response, TransportError, json_response

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE = "https://api.ebay.com/buy/browse/v1"
ITEM_ID = "v1|256012345678|0"
GROUP_ID = "256099999999"
GROUP_HREF = BROWSE + "/item/get_items_by_item_group?item_group_id=" + GROUP_ID
SELLER = {"username": "sneaker_seller_99", "feedbackPercentage": "99.8", "feedbackScore": 1543}
USD = "USD"

SUMMARY_KEYS = {
    "item_id", "legacy_item_id", "title", "price", "current_bid", "bid_count", "buying_options",
    "shipping", "end_at", "created_at", "origin_at", "relisted", "image_url", "additional_image_urls",
    "condition", "condition_id", "item_group_id", "web_url", "category_ids",
}
DETAIL_KEYS = SUMMARY_KEYS | {
    "aspects", "description", "availability", "min_next_bid", "unique_bidders", "reserve_met",
}


def money(value):
    return {"value": value, "currency": USD}


# One search result as the Browse API returns it: an auction with bids.
AUCTION_RAW = {
    "itemId": ITEM_ID,
    "legacyItemId": "256012345678",
    "title": "Nike Dunk Low Retro Panda White Black DD1391-100 Men's Size 10",
    "leafCategoryIds": ["15709"],
    "categories": [
        {"categoryId": "15709", "categoryName": "Athletic Shoes"},
        {"categoryId": "93427", "categoryName": "Men's Shoes"},
    ],
    "image": {"imageUrl": "https://i.ebayimg.com/images/g/AAA/s-l225.jpg"},
    "additionalImages": [
        {"imageUrl": "https://i.ebayimg.com/images/g/BBB/s-l225.jpg"},
        {"imageUrl": "https://i.ebayimg.com/images/g/CCC/s-l225.jpg"},
    ],
    "thumbnailImages": [{"imageUrl": "https://i.ebayimg.com/images/g/AAA/s-l64.jpg"}],
    "price": money("61.00"),
    "currentBidPrice": money("61.00"),
    "bidCount": 7,
    "buyingOptions": ["AUCTION"],
    "itemHref": BROWSE + "/item/v1%7C256012345678%7C0",
    "itemWebUrl": "https://www.ebay.com/itm/256012345678",
    "seller": SELLER,
    "condition": "Pre-owned",
    "conditionId": "3000",
    "shippingOptions": [
        {"shippingCostType": "FIXED", "shippingCost": money("12.50")},
        {"shippingCostType": "FIXED", "shippingCost": money("30.00")},
    ],
    "itemLocation": {"postalCode": "941**", "country": "US"},
    "itemCreationDate": "2026-11-12T07:15:00.000Z",
    "itemOriginDate": "2026-11-12T07:15:00.000Z",
    "itemEndDate": "2026-11-19T07:15:00.000Z",
    "listingMarketplaceId": "EBAY_US",
    "adultOnly": False,
}

# A relisted fixed-price listing with one entry per size behind an item group.
RELISTED_GROUP_RAW = {
    "itemId": "v1|256099999999|556000000002",
    "legacyItemId": GROUP_ID,
    "title": "Nike Dunk Low Panda - Pick Your Size",
    "leafCategoryIds": ["15709"],
    "image": {"imageUrl": "https://i.ebayimg.com/images/g/DDD/s-l225.jpg"},
    "price": money("139.99"),
    "buyingOptions": ["FIXED_PRICE", "BEST_OFFER"],
    "itemGroupHref": GROUP_HREF,
    "itemGroupType": "SELLER_DEFINED_VARIATIONS",
    "itemWebUrl": "https://www.ebay.com/itm/256099999999",
    "seller": SELLER,
    "condition": "New with box",
    "conditionId": "1000",
    "shippingOptions": [{"shippingCostType": "CALCULATED"}],
    "itemCreationDate": "2026-11-12T09:00:00.000Z",
    "itemOriginDate": "2026-10-01T09:00:00.000Z",
}

DESCRIPTION_HTML = (
    "<div><style>p{margin:0}</style><h2>Nike Dunk Low Panda</h2>"
    "<p>Worn <b>twice</b>.&nbsp;Size tag reads US&nbsp;10.</p>"
    "<ul><li>Style DD1391<span>-100</span></li><li>Original box &amp; laces</li></ul>"
    "<script>track('view')</script><!-- relisted from last week --></div>"
)

# The same auction as `item/{item_id}` returns it.
ITEM_RAW = {
    "itemId": ITEM_ID,
    "legacyItemId": "256012345678",
    "title": "Nike Dunk Low Retro Panda White Black DD1391-100 Men's Size 10",
    "shortDescription": "Worn twice, original box.",
    "description": DESCRIPTION_HTML,
    "price": money("61.00"),
    "currentBidPrice": money("61.00"),
    "minimumPriceToBid": money("62.00"),
    "bidCount": 7,
    "uniqueBidderCount": 4,
    "reservePriceMet": True,
    "buyingOptions": ["AUCTION"],
    "categoryPath": "Clothing, Shoes & Accessories|Men|Men's Shoes|Athletic Shoes",
    "categoryIdPath": "11450|260010|93427|15709",
    "categoryId": "15709",
    "condition": "Pre-owned",
    "conditionId": "3000",
    "image": {"imageUrl": "https://i.ebayimg.com/images/g/AAA/s-l1600.jpg"},
    "additionalImages": [{"imageUrl": "https://i.ebayimg.com/images/g/BBB/s-l1600.jpg"}],
    "brand": "Nike",
    "color": "White",
    "size": "10",
    "gender": "Men",
    "mpn": "DD1391-100",
    "itemCreationDate": "2026-11-12T07:15:00.000Z",
    "itemEndDate": "2026-11-19T07:15:00.000Z",
    "seller": SELLER,
    "estimatedAvailabilities": [{
        "deliveryOptions": ["SHIP_TO_HOME"],
        "estimatedAvailabilityStatus": "IN_STOCK",
        "estimatedAvailableQuantity": 1,
        "estimatedSoldQuantity": 0,
    }],
    "shippingOptions": [{"shippingServiceCode": "USPS Ground Advantage", "shippingCost": money("12.50")}],
    "localizedAspects": [
        {"type": "STRING", "name": "US Shoe Size", "value": "10"},
        {"type": "STRING", "name": "Department", "value": "Men"},
        {"type": "STRING", "name": "Theme", "value": "Classic"},
        {"type": "STRING", "name": "Theme", "value": "Retro"},
    ],
    "itemWebUrl": "https://www.ebay.com/itm/256012345678",
}

# What `fieldgroups=COMPACT` returns: prices, bids and stock, but no title, options or end date.
COMPACT_RAW = {
    "itemId": ITEM_ID,
    "legacyItemId": "256012345678",
    "price": money("75.00"),
    "currentBidPrice": money("75.00"),
    "minimumPriceToBid": money("76.00"),
    "bidCount": 9,
    "uniqueBidderCount": 5,
    "reservePriceMet": False,
    "estimatedAvailabilities": [{"estimatedAvailabilityStatus": "LIMITED_STOCK"}],
    "itemWebUrl": "https://www.ebay.com/itm/256012345678",
}

ITEM_GROUP_RAW = {
    "items": [
        {
            "itemId": "v1|256099999999|556000000001",
            "title": "Nike Dunk Low Panda - Pick Your Size",
            "price": money("139.99"),
            "buyingOptions": ["FIXED_PRICE"],
            "localizedAspects": [{"type": "STRING", "name": "US Shoe Size", "value": "9.5"}],
            "estimatedAvailabilities": [{"estimatedAvailabilityStatus": "OUT_OF_STOCK"}],
            "primaryItemGroup": {"itemGroupId": GROUP_ID, "itemGroupHref": GROUP_HREF},
            "seller": SELLER,
        },
        {
            "itemId": "v1|256099999999|556000000002",
            "title": "Nike Dunk Low Panda - Pick Your Size",
            "price": money("139.99"),
            "buyingOptions": ["FIXED_PRICE"],
            "localizedAspects": [{"type": "STRING", "name": "US Shoe Size", "value": "10"}],
            "estimatedAvailabilities": [{"estimatedAvailabilityStatus": "IN_STOCK"}],
            "seller": SELLER,
        },
    ],
    "commonDescriptions": [{
        "description": "<p>Brand new. Every size ships in the original box.</p>",
        "itemIds": ["v1|256099999999|556000000001", "v1|256099999999|556000000002"],
    }],
}


def error_body(error_id, message="The request could not be completed.", parameters=None):
    row = {"errorId": error_id, "domain": "API_BROWSE", "category": "REQUEST", "message": message}
    if parameters:
        row["parameters"] = parameters
    return {"errors": [row]}


class FakeEbayApi:
    """Stands in for api.ebay.com: mints numbered tokens and replays queued Browse answers."""

    def __init__(self):
        self.transport = FakeTransport(self._handle)
        self.answers = []
        self.token_answer = None      # a Response or an exception that replaces the default mint
        self.minted = 0

    def _handle(self, method, url, headers, body):
        if url.endswith("/identity/v1/oauth2/token"):
            return self._take(self.token_answer or self._mint())
        if not self.answers:
            raise AssertionError("no Browse answer queued for %s" % url)
        return self._take(self.answers.pop(0))

    def _mint(self):
        self.minted += 1
        return json_response({
            "access_token": "token-%d" % self.minted, "expires_in": 7200, "token_type": "Application Access Token",
        })

    @staticmethod
    def _take(answer):
        if isinstance(answer, Exception):
            raise answer
        return answer

    def reply(self, obj, status=200):
        self.answers.append(json_response(obj, status))

    def fail(self, answer):
        """Queue a raw Response or an exception for the next Browse call."""
        self.answers.append(answer)

    def browse_calls(self):
        return [call for call in self.transport.calls if "/buy/browse/v1/" in call["url"]]

    def token_calls(self):
        return [call for call in self.transport.calls if call["url"].endswith("/oauth2/token")]


class EbayTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = support.Clock()
        self.store = support.make_store(self.clock)
        self.api = FakeEbayApi()
        self.client = self.make_client()

    def make_client(self, **env):
        return EbayClient(support.make_settings(**env), self.store, transport=self.api.transport)

    def last_url(self):
        return self.api.browse_calls()[-1]["url"]

    def last_query(self):
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.last_url()).query)

    def search(self, *args, **kwargs):
        """Run a search against an empty result and return the query string eBay was sent."""
        self.api.reply({"total": 0})
        self.client.search(*args, **kwargs)
        return self.last_query()


class Token(EbayTestCase):
    def test_request_is_a_client_credentials_grant_for_the_one_public_scope(self):
        self.assertEqual(self.client.token(), "token-1")
        call = self.api.transport.calls[0]
        basic = base64.b64encode(b"test-client-id:test-client-secret").decode("ascii")
        self.assertEqual((call["method"], call["url"]), ("POST", TOKEN_URL))
        self.assertEqual(call["headers"]["Authorization"], "Basic " + basic)
        self.assertEqual(call["headers"]["Content-Type"], "application/x-www-form-urlencoded")
        self.assertEqual(
            call["body"], b"grant_type=client_credentials&scope=https%3A%2F%2Fapi.ebay.com%2Foauth%2Fapi_scope"
        )

    def test_sandbox_keys_go_to_the_sandbox_host(self):
        self.make_client(EBAY_ENV="sandbox").token()
        self.assertEqual(self.api.transport.calls[0]["url"], "https://api.sandbox.ebay.com/identity/v1/oauth2/token")

    def test_token_is_reused_until_a_minute_before_it_expires(self):
        self.assertEqual(self.client.token(), "token-1")
        self.clock.advance(seconds=7200 - 61)
        self.assertEqual(self.client.token(), "token-1")
        self.assertEqual(self.api.minted, 1)
        self.clock.advance(seconds=1)
        self.assertEqual(self.client.token(), "token-2")
        self.assertEqual(self.api.minted, 2)

    def test_cache_lives_in_the_store_so_a_new_process_reuses_it(self):
        self.client.token()
        self.assertEqual(self.store.kv_get("ebay:token")["access_token"], "token-1")
        self.assertEqual(self.make_client().token(), "token-1")
        self.assertEqual(self.api.minted, 1)

    def test_token_minted_for_another_keyset_or_environment_is_not_reused(self):
        self.client.token()
        self.assertEqual(self.make_client(EBAY_CLIENT_ID="another-client-id").token(), "token-2")
        self.assertEqual(self.make_client(EBAY_ENV="sandbox").token(), "token-3")

    def test_missing_credentials_are_a_config_error_and_send_nothing(self):
        client = self.make_client(EBAY_CLIENT_SECRET="")
        with self.assertRaises(ConfigError) as caught:
            client.token()
        self.assertIn("EBAY_CLIENT_SECRET", str(caught.exception))
        self.assertEqual(self.api.transport.calls, [])

    def test_rejected_keyset_is_an_auth_error_and_nothing_is_cached(self):
        self.api.token_answer = json_response(
            {"error": "invalid_client", "error_description": "client authentication failed"}, status=401
        )
        with self.assertRaises(EbayAuthError) as caught:
            self.client.token()
        self.assertEqual(caught.exception.status, 401)
        self.assertIn("invalid_client: client authentication failed", str(caught.exception))
        self.assertIsNone(self.store.kv_get("ebay:token"))

    def test_other_token_failures(self):
        for answer, expected in (
            (json_response({"error": "server_error"}, status=500), EbayUnavailable),
            (TransportError("POST api.ebay.com failed: timed out"), EbayUnavailable),
            (Response(status=200, body=b"<html>sign in to the wifi</html>"), EbayUnavailable),
            (json_response({"token_type": "Application Access Token"}), EbayError),
        ):
            self.api.token_answer = answer
            with self.assertRaises(EbayError) as caught:
                self.client.token()
            self.assertIs(type(caught.exception), expected)

    def test_mints_are_ledgered_apart_from_browse_calls(self):
        self.client.token()
        self.api.token_answer = json_response({"error": "server_error"}, status=500)
        self.clock.advance(hours=3)
        with self.assertRaises(EbayUnavailable):
            self.client.token()
        self.assertEqual(self.store.ledger_breakdown("ebay_token"), {"mint": 2})
        self.assertEqual(self.client.calls_today(), 0)


class SearchRequest(EbayTestCase):
    def test_buying_options_filter_is_always_sent(self):
        query = self.search("nike dunk low panda")
        self.assertTrue(self.last_url().startswith(BROWSE + "/item_summary/search?"))
        self.assertEqual(self.api.browse_calls()[-1]["method"], "GET")
        self.assertEqual(query["q"], ["nike dunk low panda"])
        self.assertEqual(query["filter"], ["buyingOptions:{FIXED_PRICE|AUCTION|BEST_OFFER}"])
        self.assertEqual((query["limit"], query["offset"]), (["200"], ["0"]))

    def test_buying_options_filter_cannot_be_dropped_or_replaced(self):
        everything = ["buyingOptions:{FIXED_PRICE|AUCTION|BEST_OFFER}"]
        self.assertEqual(self.search("dunk", buying_options=[])["filter"], everything)
        self.assertEqual(self.search("dunk", extra_filters={"buyingOptions": "{CLASSIFIED_AD}"})["filter"], everything)
        self.assertEqual(self.search("dunk", buying_options=["auction"])["filter"], ["buyingOptions:{AUCTION}"])
        self.assertEqual(self.search(category_id="15709")["filter"], everything)

    def test_unknown_buying_option_is_refused_before_any_request(self):
        with self.assertRaises(ValueError):
            self.client.search("dunk", buying_options=["AUCTION},price:[..1]"])
        self.assertEqual(self.api.transport.calls, [])

    def test_start_window_is_written_in_utc_from_aware_datetimes(self):
        pacific = timezone(timedelta(hours=-8))
        query = self.search(
            "dunk",
            start_from=datetime(2026, 11, 12, 7, 0, 0, 250000, tzinfo=timezone.utc),
            start_to=datetime(2026, 11, 12, 0, 0, tzinfo=pacific),
        )
        self.assertEqual(
            query["filter"],
            ["buyingOptions:{FIXED_PRICE|AUCTION|BEST_OFFER},"
             "itemStartDate:[2026-11-12T07:00:00Z..2026-11-12T08:00:00Z]"],
        )

    def test_window_may_be_open_at_either_end(self):
        moment = datetime(2026, 11, 12, 7, 0, tzinfo=timezone.utc)
        self.assertIn("itemStartDate:[2026-11-12T07:00:00Z..]", self.search("dunk", start_from=moment)["filter"][0])
        self.assertIn("itemStartDate:[..2026-11-12T07:00:00Z]", self.search("dunk", start_to=moment)["filter"][0])

    def test_naive_datetime_is_refused_before_any_request(self):
        with self.assertRaises(ValueError):
            self.client.search("dunk", start_from=datetime(2026, 11, 12, 7, 0))
        self.assertEqual(self.api.transport.calls, [])

    def test_filters_are_joined_with_commas_and_the_whole_value_is_url_encoded(self):
        self.search(
            "nike dunk",
            buying_options=["FIXED_PRICE", "AUCTION"],
            start_from=datetime(2026, 11, 12, 7, 0, tzinfo=timezone.utc),
            start_to=datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc),
            extra_filters={"conditionIds": "{1000|3000}"},
        )
        url = self.last_url()
        self.assertIn("q=nike%20dunk&", url)
        self.assertIn(
            "filter=buyingOptions%3A%7BFIXED_PRICE%7CAUCTION%7D"
            "%2CitemStartDate%3A%5B2026-11-12T07%3A00%3A00Z..2026-11-12T08%3A00%3A00Z%5D"
            "%2CconditionIds%3A%7B1000%7C3000%7D&",
            url,
        )
        self.assertEqual(
            self.last_query()["filter"],
            ["buyingOptions:{FIXED_PRICE|AUCTION},"
             "itemStartDate:[2026-11-12T07:00:00Z..2026-11-12T08:00:00Z],"
             "conditionIds:{1000|3000}"],
        )

    def test_q_is_cut_to_100_characters_at_a_word_boundary(self):
        self.api.reply({"total": 0})
        result = self.client.search(" ".join(["panda"] * 30))
        sent = self.last_query()["q"][0]
        self.assertEqual(sent, " ".join(["panda"] * 16))     # 95 characters; a 17th word would make 101
        self.assertEqual(result["warnings"], ["q was cut to 95 characters"])
        self.assertEqual(self.search("x" * 150)["q"], ["x" * 100])

    def test_q_of_exactly_100_characters_is_sent_whole(self):
        self.api.reply({"total": 0})
        result = self.client.search("y" * 100)
        self.assertEqual(self.last_query()["q"], ["y" * 100])
        self.assertEqual(result["warnings"], [])

    def test_search_needs_something_to_search_for(self):
        for q in (None, "", "   "):
            with self.assertRaises(ValueError):
                self.client.search(q)
        self.assertEqual(self.api.transport.calls, [])
        query = self.search(category_id="15709")
        self.assertEqual(query["category_ids"], ["15709"])
        self.assertNotIn("q", query)
        self.assertEqual(self.search(epid="2146478390")["epid"], ["2146478390"])
        self.assertEqual(self.search(gtin="0195866123456")["gtin"], ["0195866123456"])

    def test_limit_and_offset_are_brought_into_what_ebay_accepts(self):
        query = self.search("dunk", limit=500, offset=200)
        self.assertEqual((query["limit"], query["offset"]), (["200"], ["200"]))
        query = self.search("dunk", limit=0)
        self.assertEqual(query["limit"], ["1"])
        query = self.search("dunk", limit=100, offset=250)
        self.assertEqual((query["limit"], query["offset"]), (["100"], ["200"]))

    def test_sort_and_fieldgroups_are_passed_through(self):
        query = self.search("dunk", sort="newlyListed", fieldgroups="ASPECT_REFINEMENTS")
        self.assertEqual((query["sort"], query["fieldgroups"]), (["newlyListed"], ["ASPECT_REFINEMENTS"]))
        plain = self.search("dunk")
        self.assertNotIn("sort", plain)
        self.assertNotIn("fieldgroups", plain)


class SearchResult(EbayTestCase):
    def page(self, total, **kwargs):
        self.api.reply({"total": total, "itemSummaries": [AUCTION_RAW, RELISTED_GROUP_RAW]})
        return self.client.search("nike dunk low", **kwargs)

    def test_result_shape(self):
        self.api.reply({
            "href": BROWSE + "/item_summary/search?q=nike%20dunk%20low",
            "total": 450,
            "limit": 200,
            "offset": 0,
            "itemSummaries": [AUCTION_RAW, RELISTED_GROUP_RAW],
            "refinement": {"dominantCategoryId": "15709", "aspectDistributions": []},
            "warnings": [{"errorId": 12024, "category": "REQUEST", "message": "The sort value is not supported."}],
        })
        result = self.client.search("nike dunk low", fieldgroups="ASPECT_REFINEMENTS,MATCHING_ITEMS")
        self.assertEqual(set(result), {"total", "items", "next_offset", "dominant_category_id", "warnings"})
        self.assertEqual(result["total"], 450)
        self.assertEqual(result["items"], [summarize(AUCTION_RAW), summarize(RELISTED_GROUP_RAW)])
        self.assertEqual(result["next_offset"], 200)
        self.assertEqual(result["dominant_category_id"], "15709")
        self.assertEqual(result["warnings"], ["The sort value is not supported."])

    def test_next_offset_walks_the_pages_and_stops_at_the_last(self):
        self.assertEqual(self.page(450)["next_offset"], 200)
        self.assertEqual(self.page(450, offset=200)["next_offset"], 400)
        self.assertIsNone(self.page(450, offset=400)["next_offset"])
        self.assertIsNone(self.page(200)["next_offset"])
        self.assertEqual(self.page(120, limit=50, offset=50)["next_offset"], 100)

    def test_next_offset_stops_at_the_10000_results_ebay_will_serve(self):
        self.assertEqual(self.page(50000, offset=9600)["next_offset"], 9800)
        self.assertIsNone(self.page(50000, offset=9800)["next_offset"])

    def test_empty_result(self):
        self.api.reply({"href": BROWSE + "/item_summary/search?q=zzz", "total": 0, "limit": 200, "offset": 0})
        self.assertEqual(
            self.client.search("zzz"),
            {"total": 0, "items": [], "next_offset": None, "dominant_category_id": None, "warnings": []},
        )


class Summarize(unittest.TestCase):
    def test_auction(self):
        self.assertEqual(summarize(AUCTION_RAW), {
            "item_id": ITEM_ID,
            "legacy_item_id": "256012345678",
            "title": "Nike Dunk Low Retro Panda White Black DD1391-100 Men's Size 10",
            "price": {"value": "61.00", "currency": "USD"},
            "current_bid": {"value": "61.00", "currency": "USD"},
            "bid_count": 7,
            "buying_options": ["AUCTION"],
            "shipping": {"value": "12.50", "currency": "USD"},
            "end_at": "2026-11-19T07:15:00Z",
            "created_at": "2026-11-12T07:15:00Z",
            "origin_at": "2026-11-12T07:15:00Z",
            "relisted": False,
            "image_url": "https://i.ebayimg.com/images/g/AAA/s-l225.jpg",
            "additional_image_urls": [
                "https://i.ebayimg.com/images/g/BBB/s-l225.jpg",
                "https://i.ebayimg.com/images/g/CCC/s-l225.jpg",
            ],
            "condition": "Pre-owned",
            "condition_id": "3000",
            "item_group_id": None,
            "web_url": "https://www.ebay.com/itm/256012345678",
            "category_ids": ["15709", "93427"],
        })

    def test_relisted_item_with_an_item_group(self):
        summary = summarize(RELISTED_GROUP_RAW)
        self.assertTrue(summary["relisted"])
        self.assertEqual(summary["origin_at"], "2026-10-01T09:00:00Z")
        self.assertEqual(summary["created_at"], "2026-11-12T09:00:00Z")
        self.assertEqual(summary["item_group_id"], GROUP_ID)
        self.assertEqual(summary["buying_options"], ["FIXED_PRICE", "BEST_OFFER"])
        self.assertIsNone(summary["current_bid"])
        self.assertIsNone(summary["bid_count"])
        self.assertIsNone(summary["end_at"])
        self.assertIsNone(summary["shipping"])        # calculated shipping carries no figure

    def test_relisted_needs_more_than_a_minute_between_origin_and_creation(self):
        def relisted(origin, created):
            return summarize({"itemOriginDate": origin, "itemCreationDate": created})["relisted"]

        self.assertFalse(relisted("2026-11-12T07:14:00.000Z", "2026-11-12T07:15:00.000Z"))
        self.assertTrue(relisted("2026-11-12T07:13:59.000Z", "2026-11-12T07:15:00.000Z"))
        self.assertFalse(relisted(None, "2026-11-12T07:15:00.000Z"))
        self.assertFalse(relisted("2026-11-12T07:15:00.000Z", None))

    def test_no_seller_field_survives(self):
        for raw in (AUCTION_RAW, RELISTED_GROUP_RAW):
            self.assertIn("seller", raw)
            summary = summarize(raw)
            self.assertEqual(set(summary), SUMMARY_KEYS)
            stored = json.dumps(summary)
            for value in ("sneaker_seller_99", "seller", "feedback", "99.8", "1543"):
                self.assertNotIn(value, stored)

    def test_sparse_item_gets_empty_values_not_errors(self):
        summary = summarize({"itemId": "v1|1|0", "price": {"value": "5.00"}, "buyingOptions": "AUCTION"})
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(summary["title"], "")
        self.assertIsNone(summary["price"])            # an amount without a currency is not a price
        self.assertEqual(summary["buying_options"], [])
        self.assertEqual(summary["additional_image_urls"], [])
        self.assertEqual(summary["category_ids"], [])
        self.assertFalse(summary["relisted"])


class DetailFrom(unittest.TestCase):
    def test_every_summary_key_plus_the_detail_keys(self):
        detail = detail_from(ITEM_RAW)
        self.assertEqual(set(detail), DETAIL_KEYS)
        self.assertEqual(detail["item_id"], ITEM_ID)
        self.assertEqual(detail["current_bid"], {"value": "61.00", "currency": "USD"})
        self.assertEqual(detail["end_at"], "2026-11-19T07:15:00Z")
        self.assertEqual(detail["category_ids"], ["15709", "11450", "260010", "93427"])

    def test_aspects_with_top_level_fields_folded_in(self):
        self.assertEqual(detail_from(ITEM_RAW)["aspects"], {
            "US Shoe Size": ["10"],
            "Department": ["Men"],
            "Theme": ["Classic", "Retro"],
            "Brand": ["Nike"],
            "Color": ["White"],
            "Size": ["10"],
            "Gender": ["Men"],
            "MPN": ["DD1391-100"],
        })

    def test_an_item_specific_is_never_replaced_by_a_top_level_field(self):
        raw = dict(ITEM_RAW, brand="NIKE INC", mpn="does not apply", localizedAspects=[
            {"name": "Brand", "value": "Nike"},
            {"name": "mpn", "value": "DD1391-100"},
        ])
        aspects = detail_from(raw)["aspects"]
        self.assertEqual(aspects["Brand"], ["Nike"])
        self.assertEqual(aspects["mpn"], ["DD1391-100"])
        self.assertNotIn("MPN", aspects)
        self.assertEqual(aspects["Color"], ["White"])

    def test_folded_fields_feed_the_deterministic_checks(self):
        aspects = detail_from(ITEM_RAW)["aspects"]
        stated_sizes = rules.sizes_from_aspects(aspects)
        self.assertEqual(rules.check_style_code({"style_codes": ["DD1391-100"]}, aspects), rules.MATCH)
        self.assertEqual(rules.check_size({"value": "10", "system": "US"}, stated_sizes), rules.MATCH)

    def test_top_level_gender_reaches_the_department_check(self):
        raw = dict(ITEM_RAW, localizedAspects=[{"name": "US Shoe Size", "value": "10"}])
        aspects = detail_from(raw)["aspects"]
        self.assertEqual(aspects["Department"], ["Men"])
        self.assertEqual(rules.check_department({"department": "men"}, aspects), rules.MATCH)
        self.assertEqual(rules.check_department({"department": "women"}, aspects), rules.MISMATCH)

    def test_description_is_plain_text(self):
        self.assertEqual(
            detail_from(ITEM_RAW)["description"],
            "Nike Dunk Low Panda Worn twice. Size tag reads US 10. Style DD1391-100 Original box & laces",
        )

    def test_description_is_capped_at_4000_characters(self):
        long_html = "<div>" + "<p>" + "x" * 5000 + "</p>" + "</div>"
        self.assertEqual(detail_from(dict(ITEM_RAW, description=long_html))["description"], "x" * 4000)
        # "size 10 " repeats every 8 characters, so the cut lands on a space, which is trimmed.
        wordy = detail_from(dict(ITEM_RAW, description="<p>size 10</p>" * 1000))["description"]
        self.assertEqual(len(wordy), 3999)
        self.assertTrue(wordy.endswith("size 10"))

    def test_short_description_is_used_when_there_is_no_description(self):
        raw = {key: value for key, value in ITEM_RAW.items() if key != "description"}
        self.assertEqual(detail_from(raw)["description"], "Worn twice, original box.")

    def test_bid_state_and_availability(self):
        detail = detail_from(ITEM_RAW)
        self.assertEqual(detail["availability"], "IN_STOCK")
        self.assertEqual(detail["min_next_bid"], {"value": "62.00", "currency": "USD"})
        self.assertEqual(detail["unique_bidders"], 4)
        self.assertIs(detail["reserve_met"], True)
        self.assertEqual(detail["bid_count"], 7)

    def test_fixed_price_item_has_no_bid_state(self):
        raw = {"itemId": "v1|2|0", "price": money("139.99"), "buyingOptions": ["FIXED_PRICE"],
               "estimatedAvailabilities": [{"estimatedAvailabilityStatus": "SOMETHING_NEW"}]}
        detail = detail_from(raw)
        self.assertIsNone(detail["min_next_bid"])
        self.assertIsNone(detail["unique_bidders"])
        self.assertIsNone(detail["reserve_met"])
        self.assertIsNone(detail["availability"])     # only the three documented states are passed on
        self.assertEqual(detail["aspects"], {})
        self.assertEqual(detail["description"], "")

    def test_no_seller_field_survives(self):
        detail = detail_from(ITEM_RAW)
        stored = json.dumps(detail)
        for value in ("sneaker_seller_99", "seller", "feedback"):
            self.assertNotIn(value, stored)


class StripHtml(unittest.TestCase):
    def test_tags_entities_and_whitespace(self):
        self.assertEqual(strip_html("<p>Size <b>US 10</b></p>\n\n<p>No&nbsp;box &amp; no laces</p>"),
                         "Size US 10 No box & no laces")
        self.assertEqual(strip_html("Line one<br>Line two<BR/>Line three"), "Line one Line two Line three")
        self.assertEqual(strip_html("<table><tr><td>Brand</td><td>Nike</td></tr></table>"), "Brand Nike")

    def test_inline_tags_do_not_split_a_word(self):
        self.assertEqual(strip_html('DD1391<span class="x">-100</span>'), "DD1391-100")

    def test_scripts_styles_and_comments_are_dropped(self):
        markup = (
            "<STYLE>p{color:red}</STYLE>Visible"
            "<script type='x'>if (a < b) { hide() }</script><!-- hidden > note --> text"
        )
        self.assertEqual(strip_html(markup), "Visible text")
        self.assertEqual(strip_html("Kept<script>never closed"), "Kept")

    def test_text_that_only_looks_like_markup_is_kept(self):
        self.assertEqual(strip_html("waist < 32 and inseam > 30"), "waist < 32 and inseam > 30")
        self.assertEqual(strip_html("fits sizes <M and up, see <b>chart</b>"), "fits sizes <M and up, see chart")
        self.assertEqual(strip_html("&lt;b&gt;not bold&lt;/b&gt;"), "<b>not bold</b>")

    def test_malformed_and_empty_input_never_raise(self):
        self.assertEqual(strip_html(None), "")
        self.assertEqual(strip_html(""), "")
        self.assertEqual(strip_html("<![weird]>ok<![endif]>"), "ok")
        self.assertEqual(strip_html("<p>cut off <a href="), "cut off <a href=")
        self.assertEqual(strip_html(12), "12")


class ItemEndpoints(EbayTestCase):
    def test_get_item_url_encodes_the_pipes(self):
        self.api.reply(ITEM_RAW)
        detail = self.client.get_item(ITEM_ID)
        self.assertEqual(self.last_url(), BROWSE + "/item/v1%7C256012345678%7C0")
        self.assertEqual(detail, detail_from(ITEM_RAW))

    def test_get_item_compact_asks_for_the_compact_fieldgroup(self):
        self.api.reply(COMPACT_RAW)
        detail = self.client.get_item(ITEM_ID, compact=True, purpose="refresh")
        self.assertEqual(self.last_url(), BROWSE + "/item/v1%7C256012345678%7C0?fieldgroups=COMPACT")
        self.assertEqual(set(detail), DETAIL_KEYS)
        self.assertEqual(detail["current_bid"], {"value": "75.00", "currency": "USD"})
        self.assertEqual(detail["min_next_bid"], {"value": "76.00", "currency": "USD"})
        self.assertEqual((detail["bid_count"], detail["unique_bidders"]), (9, 5))
        self.assertIs(detail["reserve_met"], False)
        self.assertEqual(detail["availability"], "LIMITED_STOCK")
        # What COMPACT leaves out comes back empty.
        self.assertEqual((detail["title"], detail["buying_options"], detail["end_at"], detail["image_url"]),
                         ("", [], None, None))

    def test_get_item_by_legacy_id(self):
        self.api.reply(ITEM_RAW)
        detail = self.client.get_item_by_legacy_id("256012345678")
        self.assertEqual(self.last_url(), BROWSE + "/item/get_item_by_legacy_id?legacy_item_id=256012345678")
        self.assertEqual(detail["item_id"], ITEM_ID)

    def test_get_item_by_legacy_id_with_a_variation(self):
        self.api.reply(ITEM_RAW)
        self.client.get_item_by_legacy_id(GROUP_ID, legacy_variation_id="556000000002")
        self.assertEqual(
            self.last_url(),
            BROWSE + "/item/get_item_by_legacy_id?legacy_item_id=256099999999&legacy_variation_id=556000000002",
        )

    def test_get_items_by_item_group(self):
        self.api.reply(ITEM_GROUP_RAW)
        details = self.client.get_items_by_item_group(GROUP_ID)
        self.assertEqual(self.last_url(), BROWSE + "/item/get_items_by_item_group?item_group_id=256099999999")
        self.assertEqual([d["item_id"] for d in details],
                         ["v1|256099999999|556000000001", "v1|256099999999|556000000002"])
        self.assertEqual([d["aspects"]["US Shoe Size"] for d in details], [["9.5"], ["10"]])
        self.assertEqual([d["availability"] for d in details], ["OUT_OF_STOCK", "IN_STOCK"])
        for detail in details:
            self.assertEqual(set(detail), DETAIL_KEYS)
            self.assertEqual(detail["item_group_id"], GROUP_ID)
            self.assertEqual(detail["description"], "Brand new. Every size ships in the original box.")
            self.assertNotIn("sneaker_seller_99", json.dumps(detail))

    def test_item_group_keeps_a_variation_s_own_description(self):
        raw = json.loads(json.dumps(ITEM_GROUP_RAW))
        raw["items"][0]["description"] = "<p>This size has a damaged box.</p>"
        self.api.reply(raw)
        details = self.client.get_items_by_item_group(GROUP_ID)
        self.assertEqual(details[0]["description"], "This size has a damaged box.")
        self.assertEqual(details[1]["description"], "Brand new. Every size ships in the original box.")

    def test_empty_item_group(self):
        self.api.reply({"items": []})
        self.assertEqual(self.client.get_items_by_item_group(GROUP_ID), [])


class Errors(EbayTestCase):
    def failure(self, call=None):
        with self.assertRaises(EbayError) as caught:
            (call or (lambda: self.client.get_item(ITEM_ID)))()
        return caught.exception

    def test_each_documented_error_id_and_status_maps_to_its_type(self):
        cases = (
            (404, 11001, EbayNotFound),         # item id not found
            (404, 11003, EbayNotFound),         # legacy id not found
            (400, 11001, EbayNotFound),         # the id decides, whatever the status
            (404, 99999, EbayNotFound),         # the status decides when the id says nothing
            (400, 11004, EbayUnavailable),      # not available for purchase
            (429, 2001, EbayUnavailable),
            (500, 2003, EbayUnavailable),
            (503, 99999, EbayUnavailable),
            (401, 1001, EbayAuthError),
            (403, 1100, EbayAuthError),
            (400, 12001, EbayError),
            (409, 99999, EbayError),
        )
        for status, error_id, expected in cases:
            self.api.reply(error_body(error_id, "The request could not be completed."), status)
            error = self.failure()
            self.assertIs(type(error), expected, (status, error_id))
            self.assertEqual((error.status, error.error_ids), (status, [error_id]))
            self.assertEqual(
                str(error),
                "eBay item lookup failed: HTTP %d, error %d: The request could not be completed." % (status, error_id),
            )

    def test_status_alone_is_enough_when_the_body_is_not_json(self):
        for status, expected in ((401, EbayAuthError), (403, EbayAuthError), (404, EbayNotFound),
                                 (429, EbayUnavailable), (502, EbayUnavailable), (400, EbayError)):
            self.api.fail(Response(status=status, body=b"<html>Service Unavailable</html>"))
            error = self.failure()
            self.assertIs(type(error), expected, status)
            self.assertEqual((error.status, error.error_ids), (status, []))
            self.assertEqual(str(error), "eBay item lookup failed: HTTP %d" % status)

    def test_group_required_carries_the_item_group_id_from_the_message(self):
        message = ("The legacy Id is invalid. Use 'get_items_by_item_group' resource to retrieve the item group "
                   "details with the URL '%s'" % GROUP_HREF)
        self.api.reply(error_body(11006, message), 400)
        error = self.failure(lambda: self.client.get_item_by_legacy_id(GROUP_ID))
        self.assertIs(type(error), EbayGroupRequired)
        self.assertEqual(error.item_group_id, GROUP_ID)
        self.assertEqual((error.status, error.error_ids), (400, [11006]))

    def test_group_required_carries_the_item_group_id_from_a_parameter(self):
        parameters = [{"name": "legacyItemId", "value": GROUP_ID}, {"name": "itemGroupHref", "value": GROUP_HREF}]
        self.api.reply(error_body(11006, "The legacy Id is invalid.", parameters), 400)
        error = self.failure(lambda: self.client.get_item_by_legacy_id(GROUP_ID))
        self.assertIs(type(error), EbayGroupRequired)
        self.assertEqual(error.item_group_id, GROUP_ID)

    def test_group_required_without_a_parsable_id(self):
        self.api.reply(error_body(11006, "The legacy Id is invalid."), 400)
        error = self.failure(lambda: self.client.get_item_by_legacy_id(GROUP_ID))
        self.assertIs(type(error), EbayGroupRequired)
        self.assertIsNone(error.item_group_id)

    def test_network_failure_is_unavailable(self):
        self.api.fail(TransportError("GET api.ebay.com failed: timed out"))
        error = self.failure()
        self.assertIs(type(error), EbayUnavailable)
        self.assertIsNone(error.status)
        self.assertEqual(str(error), "eBay item lookup failed: GET api.ebay.com failed: timed out")

    def test_success_status_without_json_is_unavailable(self):
        self.api.fail(Response(status=200, body=b"<html>sign in to the wifi</html>"))
        self.assertIs(type(self.failure()), EbayUnavailable)
        self.api.fail(Response(status=200, body=b"[]"))
        self.assertIs(type(self.failure()), EbayUnavailable)

    def test_search_errors_are_mapped_the_same_way(self):
        self.api.reply(error_body(12001, "The call must have a valid 'q' parameter."), 400)
        error = self.failure(lambda: self.client.search("dunk"))
        self.assertIs(type(error), EbayError)
        self.assertTrue(str(error).startswith("eBay search failed: HTTP 400, error 12001"))

    def test_a_token_ebay_rejects_is_forgotten_so_the_next_call_mints_a_new_one(self):
        self.api.reply(error_body(1001, "Invalid access token"), 401)
        self.assertIs(type(self.failure()), EbayAuthError)
        self.assertIsNone(self.store.kv_get("ebay:token"))
        self.api.reply(ITEM_RAW)
        self.client.get_item(ITEM_ID)
        self.assertEqual(self.api.minted, 2)
        self.assertEqual(self.api.browse_calls()[-1]["headers"]["Authorization"], "Bearer token-2")

    def test_access_denied_keeps_the_token(self):
        self.api.reply(error_body(1100, "Access denied"), 403)
        self.assertIs(type(self.failure()), EbayAuthError)
        self.assertEqual(self.store.kv_get("ebay:token")["access_token"], "token-1")


class BudgetAndLedger(EbayTestCase):
    def test_cap_reached_raises_without_sending_anything(self):
        client = self.make_client(EFT_EBAY_DAILY_CAP="3")
        self.store.ledger_add("ebay", "search", calls=3)
        for call in (
            lambda: client.search("dunk"),
            lambda: client.get_item(ITEM_ID),
            lambda: client.get_item_by_legacy_id("256012345678"),
            lambda: client.get_items_by_item_group(GROUP_ID),
        ):
            with self.assertRaises(EbayBudgetExceeded) as caught:
                call()
            self.assertIn("3 of 3", str(caught.exception))
        self.assertEqual(self.api.transport.calls, [])     # not even a token request
        self.assertEqual(client.calls_today(), 3)          # and a refused call is not counted
        self.assertEqual(client.remaining_today(), 0)

    def test_cap_is_per_utc_day(self):
        client = self.make_client(EFT_EBAY_DAILY_CAP="2")
        self.api.reply(ITEM_RAW)
        self.api.reply(ITEM_RAW)
        client.get_item(ITEM_ID)
        self.assertEqual(client.remaining_today(), 1)
        client.get_item(ITEM_ID)
        with self.assertRaises(EbayBudgetExceeded):
            client.get_item(ITEM_ID)
        self.assertEqual(len(self.api.browse_calls()), 2)
        self.clock.advance(days=1)
        self.api.reply(ITEM_RAW)
        client.get_item(ITEM_ID)
        self.assertEqual((client.calls_today(), client.remaining_today()), (1, 1))

    def test_ledger_rows_by_purpose_with_token_mints_kept_apart(self):
        self.api.reply({"total": 0})
        self.client.search("dunk")
        self.api.reply({"total": 0})
        self.client.search("dunk", purpose="sweep")
        self.api.reply({"total": 0})
        self.client.search("dunk", limit=1, fieldgroups="ASPECT_REFINEMENTS", purpose="plan")
        self.api.reply(ITEM_RAW)
        self.client.get_item(ITEM_ID)
        self.api.reply(COMPACT_RAW)
        self.client.get_item(ITEM_ID, compact=True, purpose="refresh")
        self.api.reply(ITEM_RAW)
        self.client.get_item_by_legacy_id("256012345678")
        self.api.reply(ITEM_GROUP_RAW)
        self.client.get_items_by_item_group(GROUP_ID)
        self.assertEqual(
            self.store.ledger_breakdown("ebay"), {"detail": 3, "plan": 1, "refresh": 1, "search": 1, "sweep": 1}
        )
        self.assertEqual(self.store.ledger_breakdown("ebay_token"), {"mint": 1})
        self.assertEqual(self.client.calls_today(), 7)
        self.assertEqual(self.client.remaining_today(), 4800 - 7)

    def test_failed_calls_are_counted_too(self):
        self.api.reply(error_body(11001, "The specified item Id was not found."), 404)
        with self.assertRaises(EbayNotFound):
            self.client.get_item(ITEM_ID)
        self.api.fail(TransportError("GET api.ebay.com failed: connection reset"))
        with self.assertRaises(EbayUnavailable):
            self.client.get_item(ITEM_ID, compact=True, purpose="refresh")
        self.assertEqual(self.store.ledger_breakdown("ebay"), {"detail": 1, "refresh": 1})

    def test_a_failed_token_request_is_not_a_browse_call(self):
        self.api.token_answer = json_response({"error": "invalid_client"}, status=401)
        with self.assertRaises(EbayAuthError):
            self.client.search("dunk")
        self.assertEqual(self.api.browse_calls(), [])
        self.assertEqual(self.client.calls_today(), 0)
        self.assertEqual(self.store.ledger_breakdown("ebay_token"), {"mint": 1})


class Headers(EbayTestCase):
    def browse_headers(self, client):
        self.api.reply({"total": 0})
        client.search("dunk")
        return self.api.browse_calls()[-1]["headers"]

    def test_bearer_token_and_marketplace(self):
        headers = self.browse_headers(self.client)
        self.assertEqual(headers["Authorization"], "Bearer token-1")
        self.assertEqual(headers["X-EBAY-C-MARKETPLACE-ID"], "EBAY_US")
        self.assertNotIn("X-EBAY-C-ENDUSERCTX", headers)

    def test_zip_adds_the_end_user_context_on_the_us_marketplace(self):
        headers = self.browse_headers(self.make_client(EBAY_ZIP="94107"))
        self.assertEqual(headers["X-EBAY-C-ENDUSERCTX"], "contextualLocation=country%3DUS%2Czip%3D94107")
        self.assertEqual(headers["X-EBAY-C-MARKETPLACE-ID"], "EBAY_US")

    def test_zip_is_not_sent_to_another_marketplace(self):
        headers = self.browse_headers(self.make_client(EBAY_ZIP="94107", EBAY_MARKETPLACE="EBAY_GB"))
        self.assertEqual(headers["X-EBAY-C-MARKETPLACE-ID"], "EBAY_GB")
        self.assertNotIn("X-EBAY-C-ENDUSERCTX", headers)

    def test_item_calls_carry_the_same_headers(self):
        client = self.make_client(EBAY_ZIP="94107")
        self.api.reply(ITEM_RAW)
        client.get_item(ITEM_ID)
        headers = self.api.browse_calls()[-1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer token-1")
        self.assertEqual(headers["X-EBAY-C-MARKETPLACE-ID"], "EBAY_US")
        self.assertEqual(headers["X-EBAY-C-ENDUSERCTX"], "contextualLocation=country%3DUS%2Czip%3D94107")


class SecretsStayOutOfMessages(EbayTestCase):
    SECRET = "test-client-secret"
    BASIC = base64.b64encode(b"test-client-id:test-client-secret").decode("ascii")
    TOKEN = "v^1.1#i^1#f^0#p^1#r^0#I^3#t^H4sIAAAAAAAAAOVYa2wUVRTe"

    KEYSET_ECHO = "Authorization 'Basic %s' for a client with secret %s" % (BASIC, SECRET)
    ECHO = "Authorization 'Bearer %s'. %s" % (TOKEN, KEYSET_ECHO)

    def use_real_looking_token(self):
        self.api.token_answer = json_response({"access_token": self.TOKEN, "expires_in": 7200})

    def assert_safe(self, error):
        for text in (str(error), repr(error)):
            for secret in (self.SECRET, self.BASIC, self.TOKEN):
                self.assertNotIn(secret, text)
            self.assertIn("[redacted]", text)

    def raised_by(self, call):
        with self.assertRaises(EbayError) as caught:
            call()
        return caught.exception

    def test_browse_error_that_echoes_the_credentials(self):
        self.use_real_looking_token()
        self.api.reply(error_body(1001, "Invalid access token. " + self.ECHO), 401)
        error = self.raised_by(lambda: self.client.get_item(ITEM_ID))
        self.assert_safe(error)
        self.assertIn("HTTP 401, error 1001: Invalid access token.", str(error))

    def test_long_error_text_is_not_cut_in_the_middle_of_a_credential(self):
        self.use_real_looking_token()
        self.api.reply(error_body(1001, "x" * 280 + " " + self.TOKEN), 401)
        error = self.raised_by(lambda: self.client.get_item(ITEM_ID))
        self.assertNotIn(self.TOKEN[:8], str(error))
        self.assertLess(len(str(error)), 400)

    def test_network_failure_that_echoes_the_credentials(self):
        self.use_real_looking_token()
        self.api.fail(TransportError("GET api.ebay.com failed: " + self.ECHO))
        self.assert_safe(self.raised_by(lambda: self.client.search("dunk")))

    def test_token_error_that_echoes_the_credentials(self):
        self.api.token_answer = json_response(
            {"error": "invalid_client", "error_description": self.KEYSET_ECHO}, status=401
        )
        self.assert_safe(self.raised_by(self.client.token))

    def test_token_network_failure_that_echoes_the_credentials(self):
        self.api.token_answer = TransportError("POST api.ebay.com failed: " + self.KEYSET_ECHO)
        self.assert_safe(self.raised_by(self.client.token))

    def test_budget_message_carries_only_counts(self):
        client = self.make_client(EFT_EBAY_DAILY_CAP="0")
        error = self.raised_by(lambda: client.search("dunk"))
        self.assertEqual(str(error), "eBay daily call cap reached (0 of 0); no request was sent")


if __name__ == "__main__":
    unittest.main()
