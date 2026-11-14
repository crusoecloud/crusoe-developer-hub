"""Settings and .env loading. Standard library only; Python 3.9+.

Secrets are read from the process environment first and from a .env file second.
Nothing in this module prints or returns a secret value in an error message.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

APP_NAME = "ebay-fashion-tracker"
SKILL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRUSOE_BASE = "https://api.inference.crusoecloud.com/v1"
# eBay allows 5,000 Browse calls a day per application; stop short of it.
DEFAULT_EBAY_DAILY_CAP = 4800
DEFAULT_QUIET_HOURS = "22:00-08:00"

# Which variables each feature needs. Used for "is it configured" and for error messages.
FEATURE_VARS: Dict[str, List[str]] = {
    "crusoe": ["CRUSOE_API_KEY"],
    "ebay": ["EBAY_CLIENT_ID", "EBAY_CLIENT_SECRET"],
    "slack": ["SLACK_BOT_TOKEN", "SLACK_USER_ID"],
    "pushover": ["PUSHOVER_APP_TOKEN", "PUSHOVER_USER_KEY"],
    "heartbeat": ["EFT_HEARTBEAT_URL"],
}


class ConfigError(Exception):
    """A setting is missing or malformed. The message never contains secret values."""


def parse_env_text(text: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines. Supports comments, blank lines, `export`, and quotes."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key[0].isdigit() or not all(c.isalnum() or c == "_" for c in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            comment_at = value.find(" #")
            if comment_at != -1:
                value = value[:comment_at].rstrip()
        out[key] = value
    return out


def read_env_file(path: Path) -> Tuple[Dict[str, str], Optional[str]]:
    """Return (values, problem). A missing or unreadable file is reported, not raised.

    Claude Code's sandbox commonly denies reads of .env files; in that case the
    variables have to come from the process environment instead.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, "missing"
    except OSError as exc:
        return {}, "unreadable (%s)" % exc.__class__.__name__
    return parse_env_text(text), None


def default_state_dir(environ: Mapping[str, str], platform: Optional[str] = None) -> Path:
    override = environ.get("EFT_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path(environ.get("HOME") or Path.home())
    if (platform or sys.platform) == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    data_home = environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home) if data_home else home / ".local" / "share"
    return base / APP_NAME


def parse_quiet_hours(value: str) -> Tuple[int, int]:
    """'22:00-08:00' -> (1320, 480): minutes after local midnight for start and end."""
    try:
        start_text, end_text = value.strip().split("-")
        out = []
        for part in (start_text, end_text):
            hours, minutes = part.strip().split(":")
            hours_i, minutes_i = int(hours), int(minutes)
            if not (0 <= hours_i <= 23 and 0 <= minutes_i <= 59):
                raise ValueError(part)
            out.append(hours_i * 60 + minutes_i)
        return out[0], out[1]
    except (ValueError, AttributeError):
        raise ConfigError("EFT_QUIET_HOURS must look like 22:00-08:00")


@dataclass(frozen=True)
class Settings:
    skill_root: Path
    state_dir: Path
    env_file: Path
    env_file_problem: Optional[str]
    crusoe_api_key: str = ""
    crusoe_api_base: str = DEFAULT_CRUSOE_BASE
    ebay_client_id: str = ""
    ebay_client_secret: str = ""
    ebay_marketplace: str = "EBAY_US"
    ebay_zip: str = ""
    ebay_env: str = "production"
    ebay_daily_cap: int = DEFAULT_EBAY_DAILY_CAP
    # Client-side request limit per minute. 20 fits Crusoe's 30 RPM tier for accounts
    # without a payment method; accounts with one are allowed 600.
    crusoe_rpm: int = 20
    slack_bot_token: str = ""
    slack_user_id: str = ""
    pushover_app_token: str = ""
    pushover_user_key: str = ""
    heartbeat_url: str = ""
    quiet_hours: Tuple[int, int] = (22 * 60, 8 * 60)
    # 'vision' / 'text' -> exact model id that overrides catalog-based selection.
    model_overrides: Dict[str, str] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.state_dir / "tracker.db"

    @property
    def refs_dir(self) -> Path:
        """Where the user's own reference photos are kept. eBay images are never stored."""
        return self.state_dir / "refs"

    @property
    def ebay_api_base(self) -> str:
        host = "api.sandbox.ebay.com" if self.ebay_env == "sandbox" else "api.ebay.com"
        return "https://" + host

    def _values(self) -> Dict[str, str]:
        return {
            "CRUSOE_API_KEY": self.crusoe_api_key,
            "EBAY_CLIENT_ID": self.ebay_client_id,
            "EBAY_CLIENT_SECRET": self.ebay_client_secret,
            "SLACK_BOT_TOKEN": self.slack_bot_token,
            "SLACK_USER_ID": self.slack_user_id,
            "PUSHOVER_APP_TOKEN": self.pushover_app_token,
            "PUSHOVER_USER_KEY": self.pushover_user_key,
            "EFT_HEARTBEAT_URL": self.heartbeat_url,
        }

    def missing(self, feature: str) -> List[str]:
        """Names of the variables a feature still needs."""
        values = self._values()
        return [name for name in FEATURE_VARS[feature] if not values.get(name)]

    def has(self, feature: str) -> bool:
        return not self.missing(feature)

    def require(self, feature: str) -> None:
        names = self.missing(feature)
        if not names:
            return
        where = str(self.env_file)
        if self.env_file_problem and self.env_file_problem != "missing":
            where += " (%s from this shell; export the variables instead)" % self.env_file_problem
        raise ConfigError(
            "%s is not configured: set %s in the environment or in %s"
            % (feature, ", ".join(names), where)
        )

    def ensure_state_dir(self) -> None:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.refs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)


def load_settings(
    environ: Optional[Mapping[str, str]] = None,
    skill_root: Optional[Path] = None,
) -> Settings:
    """Build Settings from the environment, falling back to the .env file."""
    environ = dict(os.environ if environ is None else environ)
    root = Path(skill_root) if skill_root else SKILL_ROOT
    env_file = Path(environ.get("EFT_ENV_FILE") or root / ".env").expanduser()
    file_values, problem = read_env_file(env_file)

    def get(name: str, default: str = "") -> str:
        value = environ.get(name)
        if value is None or value == "":
            value = file_values.get(name, "")
        value = value.strip()
        return value if value else default

    merged_for_paths = dict(file_values)
    merged_for_paths.update({k: v for k, v in environ.items() if v})

    cap_text = get("EFT_EBAY_DAILY_CAP", str(DEFAULT_EBAY_DAILY_CAP))
    try:
        cap = int(cap_text)
    except ValueError:
        raise ConfigError("EFT_EBAY_DAILY_CAP must be a whole number")
    try:
        crusoe_rpm = max(1, int(get("EFT_CRUSOE_RPM", "20")))
    except ValueError:
        raise ConfigError("EFT_CRUSOE_RPM must be a whole number")

    ebay_env = get("EBAY_ENV", "production").lower()
    if ebay_env not in ("production", "sandbox"):
        raise ConfigError("EBAY_ENV must be production or sandbox")

    overrides = {}
    for family in ("vision", "text"):
        value = get("EFT_MODEL_" + family.upper())
        if value:
            overrides[family] = value

    return Settings(
        skill_root=root,
        state_dir=default_state_dir(merged_for_paths),
        env_file=env_file,
        env_file_problem=problem,
        crusoe_api_key=get("CRUSOE_API_KEY"),
        crusoe_api_base=get("CRUSOE_API_BASE", DEFAULT_CRUSOE_BASE).rstrip("/"),
        ebay_client_id=get("EBAY_CLIENT_ID"),
        ebay_client_secret=get("EBAY_CLIENT_SECRET"),
        ebay_marketplace=get("EBAY_MARKETPLACE", "EBAY_US"),
        ebay_zip=get("EBAY_ZIP"),
        ebay_env=ebay_env,
        ebay_daily_cap=cap,
        crusoe_rpm=crusoe_rpm,
        slack_bot_token=get("SLACK_BOT_TOKEN"),
        slack_user_id=get("SLACK_USER_ID"),
        pushover_app_token=get("PUSHOVER_APP_TOKEN"),
        pushover_user_key=get("PUSHOVER_USER_KEY"),
        heartbeat_url=get("EFT_HEARTBEAT_URL"),
        quiet_hours=parse_quiet_hours(get("EFT_QUIET_HOURS", DEFAULT_QUIET_HOURS)),
        model_overrides=overrides,
    )
