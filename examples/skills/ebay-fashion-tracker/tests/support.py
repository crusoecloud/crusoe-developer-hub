"""Shared test setup: put scripts/ on the import path and build throwaway settings."""
import atexit
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

# Every temporary file or directory a test creates lands under one root that is removed
# when the test run exits, so a run leaves nothing behind.
_SESSION_TMP = tempfile.mkdtemp(prefix="eft-tests-")
tempfile.tempdir = _SESSION_TMP
atexit.register(shutil.rmtree, _SESSION_TMP, ignore_errors=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from eft_core.config import load_settings  # noqa: E402
from eft_core.store import Store  # noqa: E402

FIXED_NOW = datetime(2026, 11, 12, 18, 0, 0, tzinfo=timezone.utc)


class Clock:
    """A clock tests can move."""

    def __init__(self, start=FIXED_NOW):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        from datetime import timedelta
        self.now = self.now + timedelta(**kwargs)
        return self.now


def make_settings(**env):
    """Settings built from an explicit environment and an empty skill root (no .env)."""
    tmp = tempfile.mkdtemp(prefix="eft-test-")
    base = {
        "HOME": tmp,
        "EFT_STATE_DIR": os.path.join(tmp, "state"),
        "CRUSOE_API_KEY": "test-crusoe-key",
        "EBAY_CLIENT_ID": "test-client-id",
        "EBAY_CLIENT_SECRET": "test-client-secret",
    }
    base.update(env)
    return load_settings(environ=base, skill_root=tmp)


def make_store(clock=None):
    return Store(":memory:", clock=clock or Clock())
