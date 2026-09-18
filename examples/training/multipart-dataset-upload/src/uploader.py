"""The upload engine: a smart wrapper around the Uploads API client. It creates the
session, splits the file into parts and uploads them concurrently with per-part retry,
then completes and polls until the file is assembled. An `Uploader` owns its client for
its lifetime; each landed part is saved so an interrupted run resumes where it left off."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple, TypeVar

import httpx
import openai

from . import constants, runtime
from .client import PartData, UploadsClient, UploadSpec
from .format import human_duration, human_size
from .local_file import LocalFile
from .state import State

_T = TypeVar("_T")


class PartUploadError(RuntimeError):
    """A single part could not be uploaded after exhausting its retries."""


class PartSegment(NamedTuple):
    """One part's position in the file: index, byte offset, and byte length."""

    index: int
    offset: int
    length: int


class _LandedPart(NamedTuple):
    """The result of uploading one part: its index, server part id, and byte length."""

    index: int
    part_id: str
    length: int


@dataclass(frozen=True)
class _PartUpload:
    """The inputs to one add-part call: the session id and the (filename, bytes)."""

    upload_id: str
    data: PartData


@dataclass(frozen=True)
class _Clock:
    """Injected time source so retry backoff and poll timing are deterministic in tests."""

    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic


# Real wall clock; a module singleton keeps it out of function defaults (ruff B008).
_DEFAULT_CLOCK = _Clock()


class Uploader:
    """Drives the multipart Uploads API over a client it owns for its lifetime."""

    def __init__(self, client: UploadsClient):
        self.client = client

    def close(self) -> None:
        self.client.close()

    # --- preflight -----------------------------------------------------------

    def check_connection(self) -> None:
        """Verify the key and network with one cheap list call, so a bad key or unreachable host
        fails with a plain message here instead of a raw traceback deep in the upload flow."""
        try:
            self.client.list_uploads(limit=1)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                runtime.abort(
                    "error: Crusoe rejected the API key. Check CRUSOE_API_KEY at "
                    "https://console.crusoecloud.com/security/inference-api-keys"
                )
            runtime.abort(f"error: connection check failed (HTTP {e.response.status_code}): {e}")
        except httpx.RequestError as e:
            runtime.abort(f"error: could not reach Crusoe ({e}); check your network and try again.")
        print("  connected to Crusoe; API key accepted.")

    # --- session -------------------------------------------------------------

    def create_upload(self, state: State) -> None:
        """Open the upload session, or reuse and reconcile an existing one on resume."""
        if not state.should_create_upload():
            print(f"  reusing upload session: {state.session.upload_id}")
            self._reconcile_parts(state)
            return
        print("  creating upload session...")
        spec = UploadSpec(
            filename=state.file.filename,
            purpose=state.params.purpose,
            size_bytes=state.file.size,
            mime_type=state.file.mime_type,
        )
        upload = self._retry(lambda: self.client.create(spec))
        state.session = state.session.opened(upload.id, upload.expires_at)
        state.save()
        print(f"  upload session: {state.session.upload_id} (status {upload.status})")
        if state.session.expires_at:
            print(
                f"  session expires: {datetime.fromtimestamp(state.session.expires_at):%Y-%m-%d %H:%M:%S} "
                "(all parts must land before then)"
            )

    def _reconcile_parts(self, state: State) -> None:
        """On resume, drop any recorded part ids the server no longer holds."""
        if not state.session.parts:
            return
        try:
            server_ids = self.client.list_part_ids(state.session.upload_id)
        except Exception as e:  # noqa: BLE001 - best-effort; fall back to trusting saved state
            print(f"  warning: could not list server parts ({e}); trusting saved state.")
            return
        if server_ids is None:
            # 404: the whole session is gone (expired/cancelled), not just some parts.
            state.clear()
            runtime.abort("  the upload session no longer exists; re-run to start a new upload.")
        stale = [int(index) for index, part_id in state.session.parts.items() if part_id not in server_ids]
        if stale:
            state.session = state.session.without_parts(stale)
            state.save()
            print(f"  {len(stale)} recorded part(s) missing on server; they will be re-uploaded.")

    # --- parts ---------------------------------------------------------------

    def upload_parts(self, state: State, local: LocalFile) -> None:
        """Upload every not-yet-landed part concurrently, saving progress as each lands."""
        _ensure_plan_consistent(state)  # before any part I/O, not just before complete()
        plan = _plan_parts(state)
        pending = [segment for segment in plan if str(segment.index) not in state.session.parts]
        total = state.params.num_parts or 0
        if not pending:
            print(f"  all {total} part(s) already uploaded")
            return

        workers = min(state.params.workers, len(pending))
        print(
            f"  uploading {len(pending)} of {total} part(s) with {workers} worker(s); "
            "Ctrl-C to pause or cancel — landed parts are saved; resume with make run before the session expires."
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._upload_one, state, local, segment) for segment in pending]
            self._drain(state, futures)
        print()
        print(f"  all {total} part(s) uploaded")

    def _upload_one(self, state: State, local: LocalFile, segment: PartSegment) -> _LandedPart:
        """Read and upload one part on a worker thread; returns the landed part."""
        data = (state.file.filename, local.read(segment.offset, segment.length))
        part = _PartUpload(state.session.upload_id, data)
        return _LandedPart(segment.index, self._add_part_with_retry(part), segment.length)

    def _drain(
        self,
        state: State,
        futures: list[Future],
        confirm: Callable[..., bool] = runtime.confirm,
        clock: _Clock = _DEFAULT_CLOCK,
    ) -> None:
        """Persist each part as it lands (main thread); on Ctrl-C or fatal error, cancel and pause/abort."""
        total = state.params.num_parts or 0
        size = state.file.size or 0
        uploaded = _uploaded_bytes(state, _plan_parts(state))  # bytes already on the server before this run
        start_bytes = uploaded
        start_time = clock.now()
        try:
            for future in as_completed(futures):
                landed = future.result()
                state.record_part(landed.index, landed.part_id)  # persists immediately
                uploaded += landed.length
                rate = _rate(uploaded - start_bytes, clock.now() - start_time)
                eta = (size - uploaded) / rate if rate else None
                _progress(f"  parts {len(state.session.parts)}/{total}", uploaded, size, rate, eta)
        except KeyboardInterrupt:
            _cancel_futures(futures)
            self._pause_or_cancel(state, confirm)
        except PartUploadError as e:
            _cancel_futures(futures)
            runtime.abort(f"\n  {e}\n  {len(state.session.parts)}/{total} part(s) saved; resume with: make run")

    def _pause_or_cancel(self, state: State, confirm: Callable[..., bool] = runtime.confirm) -> None:
        """On Ctrl-C mid-upload: keep the session for resume (default), or cancel it."""
        done = len(state.session.parts)
        total = state.params.num_parts or 0
        print()
        if confirm(f"Paused at {done}/{total} parts. Keep the upload for resume? (no = cancel it)", default=True):
            runtime.abort(f"  {done}/{total} part(s) saved; resume with: make run", status=0)
        self._cancel_upload(state)

    def _cancel_upload(self, state: State) -> None:
        """Cancel the server session (best-effort) and clear local state."""
        try:
            self.client.cancel(state.session.upload_id)
            print(f"  cancelled upload {state.session.upload_id}")
        except Exception as e:  # noqa: BLE001 - a terminal session may reject cancel; not fatal
            print(f"  warning: could not cancel {state.session.upload_id}: {e}")
        state.clear()
        runtime.abort("  upload cancelled.", status=0)

    def _retry(self, action: Callable[[], _T], clock: _Clock = _DEFAULT_CLOCK) -> _T:
        """Run an Uploads API call, retrying 429/5xx and connection errors with exponential
        backoff; other 4xx re-raise at once (the caller decides what they mean, e.g. 409)."""
        last: Exception = RuntimeError("no attempts made")
        for attempt in range(1, constants.RETRY_MAX_ATTEMPTS + 1):
            try:
                return action()
            except openai.APIStatusError as e:
                # 4xx other than 429 (bad key, expired/completed session, oversized part,
                # 409 conflict) cannot be fixed by retrying, so surface them immediately.
                if _is_non_retryable_status(e.status_code):
                    raise
                last = e
            except Exception as e:  # noqa: BLE001 - connection/5xx errors are retryable
                last = e
            if attempt < constants.RETRY_MAX_ATTEMPTS:
                clock.sleep(constants.RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        raise last

    def _add_part_with_retry(self, part: _PartUpload, clock: _Clock = _DEFAULT_CLOCK) -> str:
        """Upload one part via the shared retry; surface any failure as a PartUploadError."""
        try:
            return self._retry(lambda: self.client.add_part(part.upload_id, part.data).id, clock)
        except openai.APIStatusError as e:
            if _is_non_retryable_status(e.status_code):
                raise PartUploadError(f"part rejected ({e.status_code}): {e}") from e
            raise PartUploadError(f"part failed after {constants.RETRY_MAX_ATTEMPTS} attempts: {e}") from e
        except Exception as e:  # noqa: BLE001 - a connection error exhausted its retries
            raise PartUploadError(f"part failed after {constants.RETRY_MAX_ATTEMPTS} attempts: {e}") from e

    # --- complete + assemble -------------------------------------------------

    def complete_upload(self, state: State) -> None:
        """Complete the upload with ordered part ids + md5, then poll until the file id is ready."""
        if state.session.completed and state.session.file_id:
            print(f"  already completed: file {state.session.file_id}")
            return
        _ensure_plan_consistent(state)
        have = len(state.session.parts)
        if have != state.params.num_parts:
            runtime.abort(f"error: only {have}/{state.params.num_parts} part(s) present; cannot complete.")

        part_ids = state.ordered_part_ids()
        print(f"  completing upload with {len(part_ids)} ordered part(s){' + md5' if state.file.md5 else ''}...")
        try:
            self._retry(lambda: self.client.complete(state.session.upload_id, part_ids, state.file.md5))
        except openai.APIStatusError as e:
            # The server returns 409 whenever the session has left UPLOADING for any reason,
            # not just "already completed": a prior run may have completed it (then died before
            # saving), or someone/something may have cancelled or let it expire. Re-fetch the
            # live status to tell a still-running assembly apart from a terminal session.
            if e.status_code != constants.HTTP_CONFLICT:
                raise
            upload = self._retry(lambda: self.client.get(state.session.upload_id))
            status = (upload or {}).get("status")
            if status not in (constants.PENDING_STATUS, constants.COMPLETED_STATUS):
                state.clear()
                runtime.abort(
                    f"error: upload {state.session.upload_id} is no longer active "
                    f"(status={status!r}); re-run to start over."
                )
            print("  already finalized by an earlier run; resuming the assembly poll.")

        file_id = self._poll_assembly(state)
        state.session = state.session.completed_with(file_id)
        state.save()
        print(f"\n  file ready: {state.session.file_id}")

    def _poll_assembly(self, state: State, clock: _Clock = _DEFAULT_CLOCK) -> str:
        """Poll GET /uploads/{id} until the file id is ready; abort on any failure."""
        interval = constants.ASSEMBLY_POLL_INTERVAL_SECONDS
        timeout = human_duration(constants.ASSEMBLY_TIMEOUT_SECONDS)
        print(
            f"  assembling server-side; polling every {interval}s, up to {timeout} "
            "(Ctrl-C to stop, assembly continues and survives restarts)."
        )
        start = clock.now()
        deadline = start + constants.ASSEMBLY_TIMEOUT_SECONDS
        try:
            while True:
                now = clock.now()
                if now >= deadline:
                    break
                try:
                    upload = self.client.get(state.session.upload_id)
                except Exception as e:  # noqa: BLE001 - a transient poll error retries on the next interval
                    print(f"\n  warning: status poll failed ({e}); retrying")
                    clock.sleep(interval)
                    continue
                if upload is None:
                    runtime.abort("\nerror: upload session disappeared during assembly")
                file_id = _assembled_file_id_or_abort(upload)
                if file_id:
                    return file_id
                print(
                    f"\r  assembling... {human_duration(now - start)} elapsed (timeout {timeout})   ",
                    end="",
                    flush=True,
                )
                clock.sleep(interval)
        except KeyboardInterrupt:
            runtime.abort(
                "\n  stopped polling; assembly continues server-side.\n  resume with: make run",
                status=0,
            )
        runtime.abort(f"\nerror: assembly did not finish within {constants.ASSEMBLY_TIMEOUT_SECONDS}s")


# --- pure helpers (no client) ------------------------------------------------


def _plan_parts(state: State) -> list[PartSegment]:
    """Return the (index, offset, length) of every part, in file order."""
    size = state.file.size or 0
    part_size = state.params.part_size
    count = state.params.num_parts or 0
    return [PartSegment(i, i * part_size, min(part_size, size - i * part_size)) for i in range(count)]


def _ensure_plan_consistent(state: State) -> None:
    """Abort if a corrupted/hand-edited state file's num_parts no longer matches the file, which
    would make _plan_parts tile it wrong (empty/negative parts) or assemble the wrong bytes."""
    expected = state.expected_num_parts()
    if expected != state.params.num_parts:
        runtime.abort(
            f"error: state file inconsistent (num_parts={state.params.num_parts} but "
            f"file_size/part_size implies {expected}); refusing to continue. "
            f"Remove {state.state_file_path} and re-run."
        )


def _cancel_futures(futures: list[Future]) -> None:
    for future in futures:
        future.cancel()


def _uploaded_bytes(state: State, plan: list[PartSegment]) -> int:
    """Bytes already on the server before this run (sum of the parts already landed)."""
    done = {int(index) for index in state.session.parts}
    return sum(segment.length for segment in plan if segment.index in done)


def _is_non_retryable_status(status_code: int) -> bool:
    """A 4xx other than 429 cannot be fixed by retrying the same request."""
    return status_code < 500 and status_code != constants.HTTP_TOO_MANY_REQUESTS


def _assembled_file_id_or_abort(upload: dict) -> str | None:
    """Return the assembled file id, None while pending; abort on any failure."""
    status = upload.get("status")
    if status == constants.PENDING_STATUS:
        return None
    if status == constants.COMPLETED_STATUS:
        file_id = (upload.get("file") or {}).get("id")
        if file_id:
            return file_id
        # the server can report completed before the file id is attached; treat that as failure
        runtime.abort("\nerror: assembly finished but produced no file (assembly failed)")
    if status == constants.FAILED_STATUS:
        error = upload.get("error") or {}
        runtime.abort(f"\nerror: assembly failed [{error.get('code')}]: {error.get('message')}")
    if status in (constants.CANCELLED_STATUS, constants.EXPIRED_STATUS):
        runtime.abort(f"\nerror: upload was {status} before assembly finished")
    runtime.abort(f"\nerror: upload ended with status {status!r}")


def _rate(bytes_this_run: int, elapsed: float) -> float | None:
    """Upload throughput in bytes/sec for this run, or None until there's a measurable interval."""
    return bytes_this_run / elapsed if elapsed > 0 and bytes_this_run > 0 else None


def _progress(prefix: str, done: int, total: int, rate: float | None = None, eta: float | None = None) -> None:
    """Rewrite an in-place progress line: percent, bytes done/total, and (when known) rate + ETA."""
    pct = (done / total * 100) if total else 100.0
    speed = f"  {human_size(int(rate))}/s" if rate else ""
    remaining = f"  ETA {human_duration(eta)}" if eta is not None else ""
    print(f"\r{prefix}: {pct:5.1f}%  ({human_size(done)}/{human_size(total)}){speed}{remaining}   ", end="", flush=True)
