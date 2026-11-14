import email.message
import http.client
import io
import os
import stat
import tempfile
import unittest
import urllib.request
import urllib.response

import support  # noqa: F401  (sets up sys.path)
from eft_core import config
from eft_core.http import (
    BlockedUrl, FakeTransport, TransportError, UrllibTransport, _GuardedRedirect, check_url, json_response,
    read_capped,
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
        relative = config.default_state_dir({"HOME": "/home/x", "EFT_STATE_DIR": "state"}, platform="linux",
                                            root=config.Path("/opt/skill"))
        self.assertEqual(str(relative), "/opt/skill/state")
        xdg = config.default_state_dir({"HOME": "/home/x", "XDG_DATA_HOME": "/data"}, platform="linux")
        self.assertEqual(str(xdg), "/data/ebay-fashion-tracker")
        relative_xdg = config.default_state_dir({"HOME": "/home/x", "XDG_DATA_HOME": "data"}, platform="linux")
        self.assertEqual(str(relative_xdg), "/home/x/.local/share/ebay-fashion-tracker")

    def test_relative_paths_do_not_depend_on_the_current_directory(self):
        self._write_env("EFT_STATE_DIR=state\nCRUSOE_API_KEY=from-file\n")
        self.addCleanup(os.chdir, os.getcwd())
        for folder in (self.root, tempfile.mkdtemp(prefix="eft-cwd-")):
            os.chdir(folder)
            settings = config.load_settings(
                environ={"HOME": self.root, "EFT_ENV_FILE": "secrets.txt"}, skill_root=self.root)
            self.assertEqual(str(settings.env_file), os.path.join(self.root, "secrets.txt"), folder)
            self.assertEqual(settings.crusoe_api_key, "from-file", folder)
            self.assertEqual(str(settings.state_dir), os.path.join(self.root, "state"), folder)
            self.assertEqual(str(settings.refs_dir), os.path.join(self.root, "state", "refs"), folder)
        from_environment = config.load_settings(
            environ={"HOME": self.root, "EFT_STATE_DIR": "elsewhere/state"}, skill_root=self.root)
        self.assertEqual(str(from_environment.state_dir), os.path.join(self.root, "elsewhere", "state"))

    def test_the_state_folder_is_created_for_its_owner_only(self):
        self.addCleanup(os.umask, os.umask(0o022))    # the usual default, whatever this shell has

        def mode(path):
            return stat.S_IMODE(os.stat(str(path)).st_mode)

        settings = config.load_settings(environ={"HOME": self.root, "EFT_STATE_DIR": "state"}, skill_root=self.root)
        settings.ensure_state_dir()
        self.assertEqual((mode(settings.state_dir), mode(settings.refs_dir)), (0o700, 0o700))
        # A folder that was already there, perhaps one the user shares, keeps its own permissions.
        shared = os.path.join(self.root, "shared")
        os.mkdir(shared, 0o755)
        settings = config.load_settings(environ={"HOME": self.root, "EFT_STATE_DIR": shared}, skill_root=self.root)
        settings.ensure_state_dir()
        self.assertEqual((mode(shared), mode(settings.refs_dir)), (0o755, 0o700))

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
            "https://www.ebay.com.sg/itm/1",
            "https://www.ebay.com./itm/1",
            "https://www.ebay.co.uk./itm/1",
            "https://ebay.us./abc123",
        ):
            with self.assertRaises(BlockedUrl, msg=url):
                check_url(url)
        check_url("https://api.ebay.com./buy/browse/v1/item/1")    # the API host, written with its root dot

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


class CannedHttps(urllib.request.BaseHandler):
    """Answers https requests from memory, so that urllib's own redirect and error handling runs.

    `replies` maps a URL to (status, headers, body). A body that is an exception is raised
    when the reply is read, as a connection that drops part-way would.
    """
    handler_order = 100     # ahead of urllib's HTTPSHandler, which therefore never opens a socket

    def __init__(self, replies):
        self.replies = replies
        self.seen = []

    def https_open(self, req):
        self.seen.append((req.full_url, dict(req.header_items())))
        status, headers, body = self.replies[req.full_url]
        message = email.message.Message()
        for name, value in headers.items():
            message[name] = value
        reply = urllib.response.addinfourl(io.BytesIO(b"" if isinstance(body, Exception) else body),
                                           message, req.full_url, status)
        reply.msg = "canned"
        if isinstance(body, Exception):
            reply.read = lambda *args: (_ for _ in ()).throw(body)
        return reply


CRUSOE_MODELS = "https://api.inference.crusoecloud.com/v1/models"
KEY = {"Authorization": "Bearer crusoe-key"}


class RealTransport(unittest.TestCase):
    def transport(self, replies):
        transport = UrllibTransport()
        self.https = CannedHttps(replies)
        # No proxy from the environment: the requests seen must be the ones the transport made.
        transport._opener = urllib.request.build_opener(
            _GuardedRedirect(), self.https, urllib.request.ProxyHandler({}))
        return transport

    def urls_requested(self):
        return [url for url, _ in self.https.seen]

    def test_an_error_status_is_returned_not_raised(self):
        transport = self.transport({CRUSOE_MODELS: (404, {"Content-Type": "application/json"}, b'{"error": "nope"}')})
        response = transport.request("GET", CRUSOE_MODELS, headers=KEY)
        self.assertEqual((response.status, response.json()), (404, {"error": "nope"}))
        self.assertEqual(response.headers["content-type"], "application/json")

    def test_a_reply_cut_off_part_way_is_a_transport_error(self):
        slack = "https://slack.com/api/chat.postMessage"
        cut_off = http.client.IncompleteRead(b'{"ok": tr')
        for status in (200, 502):
            transport = self.transport({slack: (status, {}, cut_off)})
            with self.assertRaises(TransportError, msg=status) as caught:
                transport.request("POST", slack, headers={"Authorization": "Bearer fake-slack-secret"}, body=b"{}")
            self.assertIn("POST slack.com failed", str(caught.exception))
            self.assertNotIn("fake-slack-secret", str(caught.exception))

    def test_an_http_client_error_is_named_but_never_quoted(self):
        hook = "https://hooks.example/services/T0/SECRET"
        refused = http.client.InvalidURL("URL can't contain control characters. '/services/T0/SECRET' (found ' ')")
        transport = self.transport({hook: (200, {}, refused)})
        with self.assertRaises(TransportError) as caught:
            transport.request("GET", hook)
        self.assertEqual(str(caught.exception), "GET hooks.example failed: InvalidURL")

    def test_a_redirect_to_another_host_is_refused_and_the_key_is_not_sent_on(self):
        elsewhere = "https://login.other-host.example/start"
        # Another port on the same host is another server as far as a key is concerned.
        other_port = "https://api.inference.crusoecloud.com:8443/v1/models"
        for target in (elsewhere, other_port):
            transport = self.transport({CRUSOE_MODELS: (302, {"Location": target}, b""), target: (200, {}, b"hi")})
            with self.assertRaises(TransportError, msg=target):
                transport.request("GET", CRUSOE_MODELS, headers=KEY)
            self.assertEqual(self.urls_requested(), [CRUSOE_MODELS])

    def test_a_redirect_on_the_same_host_keeps_the_key(self):
        moved = CRUSOE_MODELS + "/"
        transport = self.transport({CRUSOE_MODELS: (301, {"Location": moved}, b""), moved: (200, {}, b"[]")})
        self.assertEqual(transport.request("GET", CRUSOE_MODELS, headers=KEY).json(), [])
        url, headers = self.https.seen[1]
        self.assertEqual((url, headers.get("Authorization")), (moved, KEY["Authorization"]))

    def test_a_request_without_a_key_does_not_follow_a_redirect_to_another_host_either(self):
        photo, moved = "https://i.ebayimg.com/images/g/abc/s-l1600.jpg", "https://cdn.example/abc.jpg"
        transport = self.transport({photo: (302, {"Location": moved}, b""), moved: (200, {}, b"jpeg")})
        with self.assertRaises(TransportError):
            transport.request("GET", photo, headers={"Accept": "image/*"})
        self.assertEqual(self.urls_requested(), [photo])

    def test_a_redirect_to_an_ebay_page_is_refused(self):
        photo = "https://i.ebayimg.com/images/g/abc/s-l1600.jpg"
        for page in ("https://www.ebay.com/itm/123456789012", "https://www.ebay.com./itm/123456789012"):
            transport = self.transport({photo: (302, {"Location": page}, b"")})
            with self.assertRaises(BlockedUrl, msg=page):
                transport.request("GET", photo)
            self.assertEqual(self.urls_requested(), [photo])


if __name__ == "__main__":
    unittest.main()
