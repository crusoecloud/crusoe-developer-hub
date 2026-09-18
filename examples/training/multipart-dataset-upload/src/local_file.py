"""The local training file end to end: select and validate it, read it through a held
descriptor (so an atomic replace can't swap bytes mid-upload), fingerprint it with md5,
and verify it hasn't changed on disk since selection.

`State.file` (a FileConfig) is the persisted record; a `LocalFile` is the live handle
built from it -- `choose()` on a fresh run, `reopen()` on a resume."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import threading
from collections.abc import Callable
from pathlib import Path

from . import constants, runtime
from .format import human_size
from .state import FileConfig, State

# Extensions the platform cares about that mimetypes doesn't know (or gets wrong).
_MIME_BY_EXT = {
    ".jsonl": "text/jsonl",
    ".ndjson": "application/x-ndjson",
    ".json": "application/json",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".parquet": "application/octet-stream",
}


class LocalFile:
    """A held descriptor on the chosen upload file plus the operations run on it:
    positional reads, a background whole-file md5, and the change-on-disk guard. It
    remembers the size/mtime from selection so verify_unchanged() checks against those."""

    def __init__(self, path: str, size: int, mtime: float):
        self.path = path
        self.size = size
        self._mtime = mtime
        self._lock = threading.Lock()
        self._fd = os.open(path, os.O_RDONLY)
        _try_shared_lock(self._fd)
        self._md5_thread: threading.Thread | None = None
        self._md5_digest: str | None = None
        self._md5_error: Exception | None = None

    @classmethod
    def choose(cls, state: State, ask: Callable[..., str] = runtime.path) -> LocalFile:
        """Prompt for a file (with Tab path autocomplete), validate it, record it on the
        state (path, size, mtime, inferred MIME type), and open it."""
        hint = " (Tab to autocomplete)" if runtime.autocomplete_available() else ""
        while True:
            raw = ask(f"File to upload{hint}", default=os.getcwd() + os.sep)
            if not raw.strip():
                continue
            path = Path(raw).expanduser()
            problem = _validate(path)
            if problem:
                print(f"  {problem}")
                continue
            st = path.stat()
            state.file = FileConfig(
                path=str(path),
                filename=path.name,
                size=st.st_size,
                mtime=st.st_mtime,
                mime_type=_infer_mime_type(path.name),
            )
            state.save()
            print(f"  selected: {path} ({human_size(st.st_size)})")
            return cls(str(path), st.st_size, st.st_mtime)

    @classmethod
    def reopen(cls, state: State) -> LocalFile:
        """Reopen the file recorded on a resumed state, guarding against the saved size/mtime."""
        print(f"  using saved file: {state.file.path} ({human_size(state.file.size or 0)})")
        return cls(state.file.path, state.file.size, state.file.mtime)

    def read(self, offset: int, length: int) -> bytes:
        """Read exactly `length` bytes from `offset` (fewer only at EOF)."""
        chunks: list[bytes] = []
        remaining = length
        position = offset
        while remaining > 0:
            block = self._pread(position, remaining)
            if not block:
                break  # EOF
            chunks.append(block)
            position += len(block)
            remaining -= len(block)
        return b"".join(chunks)

    def start_md5(self, state: State) -> None:
        """Start the whole-file md5 on a background thread (overlaps the upload); no-op if a saved md5 exists."""
        if not state.should_compute_md5():
            print(f"  using saved md5: {state.file.md5}")
            return
        print("  computing md5 in the background while the upload runs...")
        self._md5_thread = threading.Thread(target=self._compute_md5, daemon=True)
        self._md5_thread.start()

    def finish_md5(self, state: State) -> None:
        """Join the background md5 (if one was started) and persist it on the state."""
        if self._md5_thread is None:
            return
        if self._md5_thread.is_alive():
            print(f"  waiting for the md5 checksum to finish ({human_size(self.size)} hashed sequentially)...")
        self._md5_thread.join()
        if self._md5_error is not None:
            raise self._md5_error
        state.file = state.file.with_md5(self._md5_digest)
        state.save()
        print(f"  md5: {state.file.md5}")

    def verify_unchanged(self) -> None:
        """Abort if the file's size or mtime changed since it was selected."""
        st = Path(self.path).stat()
        changed_size = st.st_size != self.size
        changed_mtime = abs(st.st_mtime - self._mtime) > constants.MTIME_TOLERANCE_SECONDS
        if changed_size or changed_mtime:
            runtime.abort(
                "error: the source file changed on disk during the upload; "
                "nothing was completed. Re-run to upload the current version."
            )

    def close(self) -> None:
        os.close(self._fd)

    def _pread(self, offset: int, length: int) -> bytes:
        if hasattr(os, "pread"):
            return os.pread(self._fd, length, offset)  # thread-safe, no shared offset
        with self._lock:
            os.lseek(self._fd, offset, os.SEEK_SET)
            return os.read(self._fd, length)

    def _compute_md5(self) -> None:
        try:
            digest = hashlib.md5()
            offset = 0
            while offset < self.size:
                block = self.read(offset, min(constants.MD5_READ_BLOCK_BYTES, self.size - offset))
                if not block:
                    break
                digest.update(block)
                offset += len(block)
            self._md5_digest = digest.hexdigest()
        except Exception as e:  # noqa: BLE001 - surfaced to the caller via finish_md5()
            self._md5_error = e


def _infer_mime_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _validate(path: Path) -> str | None:
    """Return a human-readable reason the file is unusable, or None if it's fine."""
    if not path.exists():
        return f"no such file: {path}"
    if not path.is_file():
        return f"not a regular file: {path}"
    size = path.stat().st_size
    if size == 0:
        return f"file is empty: {path}"
    if size > constants.MAX_UPLOAD_BYTES:
        return (
            f"file is {human_size(size)}, above the {human_size(constants.MAX_UPLOAD_BYTES)} "
            "per-upload ceiling - split it first or contact Crusoe for a higher limit."
        )
    return None


def _try_shared_lock(fd: int) -> None:
    """Take a non-blocking shared advisory lock; silently skip if unavailable."""
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (ImportError, OSError):
        pass  # advisory only: not on this platform, or someone holds a write lock
