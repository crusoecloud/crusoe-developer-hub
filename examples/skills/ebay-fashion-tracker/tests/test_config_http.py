import io
import os
import tempfile
import unittest
import urllib.request

import support  # noqa: F401  (sets up sys.path)
from eft_core import config
from eft_core.http import (
    BlockedUrl, FakeTransport, TransportError, _GuardedRedirect, check_url, json_response, read_capped,
)


class EnvParsing(unittest.TestCase):
    def test_parses_plain_quoted_export_and_comments(self):
        text = "\n".join([
            "# a comment",
            "",
            "CRUSOE_API_KEY=abc123",
            'EBAY_CLIENT_ID="quoted value"',
            "export SLACK_USER_ID=U123   # trailing comment",
            "EMPTY=",
            "not a pair",
            "9BAD=nope",
        ])
        values = config.parse_env_text(text)
        self.assertEqual(values["CRUSOE_API_KEY"], "abc123")
        self.assertEqual(values["EBAY_CLIENT_ID"], "quoted value")
        self.assertEqual(values["SLACK_USER_ID"], "U123")
        self.assertEqual(values["EMPTY"], "")
        self.assertNotIn("9BAD", values)

    def test_missing_file_is_reported_not_raised(self):
        values, problem = config.read_env_file(config.Path("/nonexistent/dir/.env"))
        self.assertEqual(values, {})
        self.assertEqual(problem, "missing")


class SettingsLoading(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="eft-cfg-")

    def _write_env(self, text, name="secrets.txt"):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_file_values_are_used_and_environment_wins(self):
        env_path = self._write_env("CRUSOE_API_KEY=from-file\nEBAY_CLIENT_ID=id-from-file\n")
        settings = config.load_settings(
            environ={"HOME": self.root, "EFT_ENV_FILE": env_path, "CRUSOE_API_KEY": "from-env"},
            skill_root=self.root,
        )
        self.assertEqual(settings.crusoe_api_key, "from-env")
        self.assertEqual(settings.ebay_client_id, "id-from-file")
        self.assertIsNone(settings.env_file_problem)

    def test_empty_environment_value_falls_back_to_file(self):
        env_path = self._write_env("CRUSOE_API_KEY=from-file\n")
        settings = config.load_settings(
            environ={"HOME": self.root, "EFT_ENV_FILE": env_path, "CRUSOE_API_KEY": ""}, skill_root=self.root
        )
        self.assertEqual(settings.crusoe_api_key, "from-file")

    def test_require_names_variables_and_never_values(self):
        settings = config.load_settings(
            environ={"HOME": self.root, "EBAY_CLIENT_ID": "super-secret-id"}, skill_root=self.root
        )
        self.assertTrue(settings.has("crusoe") is False)
        with self.assertRaises(config.ConfigError) as caught:
            settings.require("ebay")
        message = str(caught.exception)
        self.assertIn("EBAY_CLIENT_SECRET", message)
        self.assertNotIn("EBAY_CLIENT_ID,", message)
        self.assertNotIn("super-secret-id", message)

    def test_defaults(self):
        settings = config.load_settings(environ={"HOME": self.root}, skill_root=self.root)
        self.assertEqual(settings.crusoe_api_base, config.DEFAULT_CRUSOE_BASE)
        self.assertEqual(settings.ebay_marketplace, "EBAY_US")
        self.assertEqual(settings.ebay_daily_cap, 4800)
        self.assertEqual(settings.quiet_hours, (22 * 60, 8 * 60))
        self.assertEqual(settings.ebay_api_base, "https://api.ebay.com")
        self.assertEqual(settings.env_file_problem, "missing")

    def test_state_dir_per_platform(self):
        mac = config.default_state_dir({"HOME": "/Users/x"}, platform="darwin")
        self.assertEqual(str(mac), "/Users/x/Library/Application Support/ebay-fashion-tracker")
        linux = config.default_state_dir({"HOME": "/home/x"}, platform="linux")
        self.assertEqual(str(linux), "/home/x/.local/share/ebay-fashion-tracker")
        override = config.default_state_dir({"HOME": "/home/x", "EFT_STATE_DIR": "/tmp/eft"}, platform="linux")
        self.assertEqual(str(override), "/tmp/eft")

    def test_bad_values_raise_config_error(self):
        with self.assertRaises(config.ConfigError):
            config.parse_quiet_hours("late-early")
        with self.assertRaises(config.ConfigError):
            config.load_settings(environ={"HOME": self.root, "EBAY_ENV": "staging"}, skill_root=self.root)

    def test_model_overrides(self):
        settings = config.load_settings(
            environ={"HOME": self.root, "EFT_MODEL_VISION": "google/gemma-4-31b-it"}, skill_root=self.root
        )
        self.assertEqual(settings.model_overrides, {"vision": "google/gemma-4-31b-it"})


class UrlGuard(unittest.TestCase):
    def test_api_and_other_hosts_are_allowed(self):
        for url in (
            "https://api.ebay.com/buy/browse/v1/item_summary/search?q=x",
            "https://api.sandbox.ebay.com/identity/v1/oauth2/token",
            "https://i.ebayimg.com/images/g/abc/s-l1600.jpg",
            "https://api.inference.crusoecloud.com/v1/models",
            "https://slack.com/api/chat.postMessage",
        ):
            check_url(url)

    def test_ebay_pages_and_short_links_are_blocked(self):
        for url in (
            "https://www.ebay.com/itm/123456789012",
            "https://ebay.com/sch/i.html?_nkw=dunk",
            "https://www.ebay.co.uk/itm/123456789012",
            "https://www.ebay.com.au/itm/1",
            "https://ebay.us/abc123",
            "https://ebay.to/xyz",
            "https://auth.ebay.com/oauth2/authorize",
        ):
            with self.assertRaises(BlockedUrl, msg=url):
                check_url(url)

    def test_plain_http_is_blocked(self):
        with self.assertRaises(BlockedUrl):
            check_url("http://api.ebay.com/buy/browse/v1/item/1")

    def test_fake_transport_records_calls_and_enforces_the_guard(self):
        transport = FakeTransport().add_json({"ok": True})
        response = transport.request("get", "https://api.ebay.com/x", headers={"A": "b"})
        self.assertTrue(response.ok)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(transport.calls[0]["method"], "GET")
        with self.assertRaises(BlockedUrl):
            transport.request("GET", "https://www.ebay.com/itm/1")

    def test_fake_transport_handler(self):
        transport = FakeTransport(handler=lambda m, u, h, b: json_response({"url": u}, status=201))
        response = transport.request("POST", "https://example.com/a", body=b"{}")
        self.assertEqual(response.status, 201)
        self.assertEqual(response.json()["url"], "https://example.com/a")


class RedirectGuard(unittest.TestCase):
    def follow(self, target):
        request = urllib.request.Request("https://api.ebay.com/a", headers={"Authorization": "Bearer token"})
        return _GuardedRedirect().redirect_request(request, None, 302, "Found", {}, target)

    def test_a_redirect_on_the_same_host_is_followed(self):
        self.assertEqual(self.follow("https://api.ebay.com/b").full_url, "https://api.ebay.com/b")

    def test_a_redirect_to_another_host_is_refused_so_credentials_stay_where_they_were_sent(self):
        for target in ("https://example.com/b", "https://api.ebay.com:8443/b", "https://api.ebay.com.example.com/b"):
            with self.assertRaises(TransportError, msg=target):
                self.follow(target)

    def test_a_redirect_to_an_ebay_page_or_to_plain_http_is_blocked(self):
        for target in ("https://www.ebay.com/itm/1", "http://api.ebay.com/b"):
            with self.assertRaises(BlockedUrl, msg=target):
                self.follow(target)


class BodyCap(unittest.TestCase):
    def test_a_body_within_the_limit_is_returned_and_a_larger_one_is_refused(self):
        self.assertEqual(read_capped(io.BytesIO(b"12345"), 5), b"12345")
        with self.assertRaises(TransportError):
            read_capped(io.BytesIO(b"123456"), 5)


if __name__ == "__main__":
    unittest.main()
