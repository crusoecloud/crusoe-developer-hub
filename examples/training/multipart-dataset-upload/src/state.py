"""Resumable run state persisted to a JSON file. `load()` reads it (pure);
`resume_or_start()` runs the interactive resume/reset decision; afterwards
`session.completed` signals an already-finished prior run and `resuming` an
in-progress one being continued.

The persisted data is composed of three component-owned sub-configs -- the local
file, the upload parameters, and the server session -- so each logical piece owns
its own fields and (via dataclasses.asdict) its own serialization."""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import constants, runtime
from .format import human_size

if TYPE_CHECKING:
    from .client import UploadsClient


_DEFAULT_STATE_FILE = ".crusoe-upload-state.json"


@dataclass(frozen=True)
class SessionSnapshot:
    """Result of a live session check: the fetched Upload (or None), and whether the server was reachable."""

    upload: dict | None
    reachable: bool


@dataclass(frozen=True)
class FileConfig:
    """The local file: its path and the facts captured at selection (frozen; md5 arrives later via with_md5)."""

    path: str | None = None
    filename: str | None = None
    size: int | None = None
    mtime: float | None = None
    md5: str | None = None
    mime_type: str | None = None

    def with_md5(self, md5: str) -> FileConfig:
        """A copy carrying the computed whole-file checksum."""
        return replace(self, md5=md5)


@dataclass(frozen=True)
class UploadParams:
    """The chosen upload settings and derived part count (frozen; built by parameters.choose_params)."""

    purpose: str = constants.DEFAULT_PURPOSE
    part_size: int = constants.DEFAULT_PART_BYTES
    workers: int = constants.DEFAULT_WORKERS
    num_parts: int | None = None

    def __post_init__(self) -> None:
        """Reject values that would break later arithmetic/threading (also guards a corrupt file)."""
        if self.purpose not in constants.PURPOSES:
            raise ValueError(f"purpose must be one of {constants.PURPOSES}, got {self.purpose!r}")
        if not 1 <= self.part_size <= constants.MAX_PART_BYTES:
            raise ValueError(f"part_size {self.part_size} out of range 1..{constants.MAX_PART_BYTES}")
        if not 1 <= self.workers <= constants.MAX_WORKERS:
            raise ValueError(f"workers {self.workers} out of range 1..{constants.MAX_WORKERS}")
        if self.num_parts is not None and self.num_parts < 1:
            raise ValueError(f"num_parts must be >= 1, got {self.num_parts}")


@dataclass(frozen=True)
class Session:
    """The server-side session (`parts` maps str(index) -> part id); frozen, so mutate via its helpers below."""

    upload_id: str | None = None
    expires_at: int | None = None
    parts: dict[str, str] = field(default_factory=dict)
    completed: bool = False
    file_id: str | None = None

    def with_part(self, index: int, part_id: str) -> Session:
        """A copy with one more landed part recorded."""
        return replace(self, parts={**self.parts, str(index): part_id})

    def without_parts(self, indexes: list[int]) -> Session:
        """A copy with the given part indices dropped (the server no longer holds them)."""
        dropped = {str(i) for i in indexes}
        return replace(self, parts={k: v for k, v in self.parts.items() if k not in dropped})

    def opened(self, upload_id: str, expires_at: int | None) -> Session:
        """A copy recording the server-opened session (its id and expiry)."""
        return replace(self, upload_id=upload_id, expires_at=expires_at)

    def completed_with(self, file_id: str) -> Session:
        """A copy marking the upload complete with the assembled file id."""
        return replace(self, completed=True, file_id=file_id)


@dataclass
class State:
    """One upload run: runtime handles plus the three persisted sub-configs."""

    # Runtime only (never persisted). The API key stays in the environment.
    api_key: str = ""
    state_file_path: str = _DEFAULT_STATE_FILE
    resuming: bool = False
    _saved: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # Persisted, each owned by its logical component.
    file: FileConfig = field(default_factory=FileConfig)
    params: UploadParams = field(default_factory=UploadParams)
    session: Session = field(default_factory=Session)

    created_at: str | None = None
    updated_at: str | None = None

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def load(cls, api_key: str, state_file_path: str = _DEFAULT_STATE_FILE) -> State:
        """Read the saved file into memory, parsed into sub-configs. No prompts, no network."""
        state = cls(api_key=api_key, state_file_path=state_file_path)
        state._saved = state._read_saved()
        return state

    def resume_or_start(self, client: UploadsClient, confirm: Callable[..., bool] = runtime.confirm) -> None:
        """Decide whether to keep the saved state, then apply it once: adopt it (resume or re-show)
        or discard it and start fresh. The caller then inspects `session`/`resuming`."""
        saved = self._saved
        if not saved or not (saved["file"].path or saved["session"].upload_id):
            keep = False  # empty/skeletal leftover: nothing to resume
        else:
            # One live check; a session that finished while we were away is folded in so the
            # "already completed" decision below happens in a single place.
            snapshot = self._refresh_session(client, saved["session"].upload_id)
            server_file_id = self._server_completed_file_id(snapshot)
            if server_file_id:
                saved["session"] = saved["session"].completed_with(server_file_id)

            if saved["session"].completed and saved["session"].file_id:
                keep = self._keep_completed(confirm)
            else:
                keep = self._keep_in_progress(snapshot, confirm)

        if keep:
            self._load_into_self()
        else:
            self._start_fresh()

    def _keep_completed(self, confirm: Callable[..., bool]) -> bool:
        """A prior upload finished (locally recorded or just discovered on the server). True to
        re-show that result (the caller loads it and exits), False to start a new one (default)."""
        saved = self._saved
        print("The previous upload already completed:")
        print(f"  File:    {saved['file'].path}")
        print(f"  File id: {saved['session'].file_id}")
        return not confirm("Start a new upload? (no = re-show that result and exit)", default=True)

    def _keep_in_progress(self, snapshot: SessionSnapshot, confirm: Callable[..., bool]) -> bool:
        """A saved upload that isn't completed. True to resume it, False to start fresh."""
        self._print_saved_summary(snapshot)

        error_reason = self._validate_resume(snapshot)
        if error_reason:
            print(f"  Cannot resume: {error_reason}")
            print("  Starting fresh and clearing the saved state.")
            return False
        return confirm("Resume from this state? (no = start fresh and clear saved state)", default=True)

    def _print_saved_summary(self, snapshot: SessionSnapshot) -> None:
        """Print the saved upload's key fields and the live session status."""
        saved = self._saved
        session = saved["session"]
        done = len(session.parts)
        total = saved["params"].num_parts or "?"
        print(f"Found saved state file: {self.state_file_path}")
        print(f"  Created:   {self._fmt_ts(saved['created_at'])}")
        print(f"  Updated:   {self._fmt_ts(saved['updated_at'])}")
        print(f"  File:      {saved['file'].path} ({human_size(saved['file'].size or 0)})")
        print(f"  Upload:    {session.upload_id or 'not yet created'}")
        print(f"  Status:    {self._session_status_text(session.upload_id, snapshot)}")
        expiry = self._expiry_text(session.expires_at)
        if expiry:
            print(f"  {expiry}")
        print(f"  Parts:     {done}/{total} uploaded")

    @staticmethod
    def _refresh_session(client: UploadsClient, upload_id: str | None) -> SessionSnapshot:
        """Fetch the session. reachable=False means the check failed, not that it's gone."""
        if not upload_id:
            return SessionSnapshot(None, True)
        try:
            return SessionSnapshot(client.get(upload_id), True)
        except Exception as e:  # noqa: BLE001 - a transient error must not discard resumable state
            print(f"  warning: could not reach the server to check the session ({e})")
            return SessionSnapshot(None, False)

    @staticmethod
    def _server_completed_file_id(snapshot: SessionSnapshot) -> str | None:
        """The assembled file id if the live session already reports completed, else None."""
        if snapshot.upload is None or snapshot.upload.get("status") != constants.COMPLETED_STATUS:
            return None
        return (snapshot.upload.get("file") or {}).get("id")

    @staticmethod
    def _session_status_text(upload_id: str | None, snapshot: SessionSnapshot) -> str:
        if not upload_id:
            return "not yet created"
        if not snapshot.reachable:
            return "unreachable (will try to resume)"
        if snapshot.upload is None:
            return "gone"
        return snapshot.upload.get("status", "unknown")

    def _start_fresh(self) -> None:
        """Discard any saved file and stamp a fresh start time (persisted on the first save)."""
        self._clear_state()
        self.created_at = datetime.now().isoformat()

    def _load_into_self(self) -> None:
        """Adopt the parsed saved sub-configs as this run's live state."""
        saved = self._saved
        self.resuming = True
        self.file = saved["file"]
        self.params = saved["params"]
        self.session = saved["session"]
        self.created_at = saved["created_at"]

    def clear(self) -> None:
        """Delete the saved state file (e.g. after cancelling the upload)."""
        self._clear_state()

    def _validate_resume(self, snapshot: SessionSnapshot) -> str | None:
        """Return why an in-progress upload can't be resumed, or None if it can."""
        file = self._saved["file"]
        path = file.path
        if not path or not Path(path).exists():
            return f"the source file {path!r} is no longer on disk"
        st = Path(path).stat()
        if file.size is not None and st.st_size != file.size:
            return "the source file's size changed since the upload started"
        if file.mtime is not None and abs(st.st_mtime - file.mtime) > constants.MTIME_TOLERANCE_SECONDS:
            return "the source file was modified since the upload started"

        session = self._saved["session"]
        if not session.upload_id:
            return None  # nothing created yet; the file selection is still reusable
        if not snapshot.reachable:
            return None  # couldn't verify the session; let the user try rather than discard state
        if snapshot.upload is None:
            return "the upload session no longer exists on the server"
        status = snapshot.upload.get("status")
        if status in {constants.EXPIRED_STATUS, constants.CANCELLED_STATUS}:
            return f"the upload session {status} (parts expire with the session)"
        if status == constants.FAILED_STATUS:
            error = snapshot.upload.get("error") or {}
            return f"the previous assembly failed ({error.get('code')}: {error.get('message')})"
        return None

    # --- step gates --------------------------------------------------------

    def should_pick_file(self) -> bool:
        return not (self.resuming and self.file.path)

    def should_choose_params(self) -> bool:
        return not (self.resuming and self.params.num_parts)

    def should_compute_md5(self) -> bool:
        return not (self.resuming and self.file.md5)

    def should_create_upload(self) -> bool:
        return not self.session.upload_id

    def ordered_part_ids(self) -> list[str]:
        return [self.session.parts[str(i)] for i in range(self.params.num_parts or 0)]

    def expected_num_parts(self) -> int:
        """Part count implied by the current file size and part size."""
        return math.ceil(self.file.size / self.params.part_size)

    def print_selected_params(self) -> None:
        """Print the parameters reused from a prior run (when the prompts are skipped)."""
        print(
            f"  using saved parameters: purpose={self.params.purpose}, "
            f"part_size={human_size(self.params.part_size)}, workers={self.params.workers}, "
            f"parts={self.params.num_parts}"
        )

    def print_summary(self) -> None:
        """Print the finished-upload result and how to use the file id."""
        rows = [
            ("File", self.file.path),
            ("Size", human_size(self.file.size or 0)),
            ("Purpose", self.params.purpose),
            ("Parts", f"{self.params.num_parts} x up to {human_size(self.params.part_size)}"),
            ("md5", self.file.md5),
            ("Upload id", self.session.upload_id),
            ("File id", self.session.file_id),
        ]
        width = max(len(label) for label, _ in rows)
        print("\nSummary\n")
        for label, value in rows:
            print(f"  {label:<{width}}: {value}")
        print(f"\n  Use file id {self.session.file_id} wherever you reference a file (e.g. a fine-tuning job).")
        print("  See it under Datasets in the Crusoe Console: https://console.crusoecloud.com/foundry/datasets")
        print(f"  State file {self.state_file_path} can be removed now that the upload is complete.")

    # --- persistence -------------------------------------------------------

    def record_part(self, index: int, part_id: str) -> None:
        self.session = self.session.with_part(index, part_id)
        self.save()

    def save(self) -> None:
        data = {
            "file": asdict(self.file),
            "params": asdict(self.params),
            "session": asdict(self.session),
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
        }
        # Write to a sibling temp file and atomically replace, so a crash mid-write (this
        # runs after every landed part) can't truncate the file and lose all resume progress.
        path = Path(self.state_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _read_saved(self) -> dict:
        """Parse the saved file into sub-configs; {} if missing, corrupt, or malformed."""
        try:
            with open(self.state_file_path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            print(f"  warning: state file {self.state_file_path} is corrupt; starting fresh.")
            return {}
        try:
            return {
                "file": FileConfig(**(raw.get("file") or {})),
                "params": UploadParams(**(raw.get("params") or {})),
                "session": Session(**(raw.get("session") or {})),
                "created_at": raw.get("created_at"),
                "updated_at": raw.get("updated_at"),
            }
        except (TypeError, ValueError):
            print(f"  warning: state file {self.state_file_path} has unexpected or invalid fields; starting fresh.")
            return {}

    def _clear_state(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.state_file_path)

    @staticmethod
    def _expiry_text(expires_at: int | None) -> str | None:
        """The session's expiry as an absolute time, marked once the window has already passed."""
        if not expires_at:
            return None
        when = datetime.fromtimestamp(expires_at)
        if datetime.now() < when:
            return f"Expires:   {when:%Y-%m-%d %H:%M:%S}"
        return f"Expired:   {when:%Y-%m-%d %H:%M:%S} (parts expired with the session; will start fresh)"

    @staticmethod
    def _fmt_ts(iso: str | None) -> str:
        if not iso:
            return "unknown"
        try:
            return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return iso
