"""Unit tests for src/uploader.py, driven by injected in-memory fakes (see tests.mocks)
that verify part ordering and md5 on complete. No real API calls."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import Future

import httpx
import openai
import pytest

from src import constants, local_file, uploader
from src.client import UploadsClient
from src.state import FileConfig, Session, State, UploadParams
from tests.mocks import (
    AddPartStub,
    CancelClient,
    CompleteConflictClient,
    CompleteErrorClient,
    CompleteThenTerminalClient,
    GoneSessionClient,
    MockUploadsClient,
    PollClient,
    status_error,
)


def _noop_clock():
    """A clock whose sleep() does nothing, so retry/poll tests don't actually wait."""
    return uploader._Clock(sleep=lambda *_a, **_k: None)


def _make_state(path, payload: bytes, part_size: int = 1024 * 1024, workers: int = 4) -> State:
    file_path = path / "data.bin"
    file_path.write_bytes(payload)
    state = State(api_key="k", state_file_path=str(path / "state.json"))
    state.file = FileConfig(
        path=str(file_path),
        filename=file_path.name,
        size=len(payload),
        mtime=file_path.stat().st_mtime,
        mime_type="application/octet-stream",
    )
    state.params = UploadParams(part_size=part_size, workers=workers, num_parts=math.ceil(len(payload) / part_size))
    return state


# --- connection preflight (check_connection) ------------------------------


def _transport_client(status_code: int) -> UploadsClient:
    """A real UploadsClient whose list call returns `status_code` via a mock transport."""
    return UploadsClient(
        api_key="k",
        transport=httpx.MockTransport(lambda _request: httpx.Response(status_code, json={"data": []})),
    )


def test_check_connection_accepts_a_good_key():
    client = _transport_client(200)
    try:
        uploader.Uploader(client).check_connection()  # returns normally
    finally:
        client.close()


@pytest.mark.parametrize("status_code", [401, 403, 500])
def test_check_connection_aborts_on_error(status_code):
    client = _transport_client(status_code)
    try:
        with pytest.raises(SystemExit):
            uploader.Uploader(client).check_connection()
    finally:
        client.close()


def test_plan_parts_tiles_the_file_exactly(tmp_path):
    state = _make_state(tmp_path, b"x" * (2 * 1024 * 1024 + 500), part_size=1024 * 1024)
    plan = uploader._plan_parts(state)
    assert len(plan) == state.params.num_parts == 3
    assert [length for _i, _off, length in plan] == [1024 * 1024, 1024 * 1024, 500]
    assert sum(length for _i, _off, length in plan) == state.file.size
    # offsets are contiguous and start at 0
    assert [off for _i, off, _len in plan] == [0, 1024 * 1024, 2 * 1024 * 1024]


def test_end_to_end_upload_preserves_bytes_and_md5(tmp_path):
    payload = b"".join(bytes([i % 256]) * 4096 for i in range(700))  # ~2.7 MiB, deterministic
    state = _make_state(tmp_path, payload, part_size=1024 * 1024, workers=4)
    up = uploader.Uploader(MockUploadsClient(payload))

    local = local_file.LocalFile.reopen(state)
    try:
        local.start_md5(state)
        up.create_upload(state)
        up.upload_parts(state, local)
        local.finish_md5(state)
        local.verify_unchanged()
        up.complete_upload(state)
    finally:
        local.close()

    assert state.session.completed is True
    assert state.session.file_id == "file_fake"
    assert state.file.md5 == hashlib.md5(payload).hexdigest()
    assert len(state.session.parts) == state.params.num_parts


def test_resume_reconciles_a_lost_part(tmp_path):
    payload = bytes(range(256)) * (5 * 1024)  # ~1.25 MiB
    state = _make_state(tmp_path, payload, part_size=256 * 1024, workers=3)
    client = MockUploadsClient(payload)
    up = uploader.Uploader(client)

    local = local_file.LocalFile.reopen(state)
    try:
        up.create_upload(state)
        # Upload only the first two parts, then simulate the server losing part 1.
        for segment in uploader._plan_parts(state)[:2]:
            data = (state.file.filename, local.read(segment.offset, segment.length))
            part = uploader._PartUpload(state.session.upload_id, data)
            state.record_part(segment.index, up._add_part_with_retry(part))
        del client.store[state.session.parts["1"]]

        up.create_upload(state)  # reuse + reconcile drops the lost part
        assert "1" not in state.session.parts

        up.upload_parts(state, local)
        assert len(state.session.parts) == state.params.num_parts
        up.complete_upload(state)
    finally:
        local.close()

    assert state.session.completed and state.session.file_id == "file_fake"


@pytest.mark.parametrize(
    ("status_code", "expected_calls"),
    [
        pytest.param(400, 1, id="bad-request-fails-fast"),
        pytest.param(401, 1, id="auth-fails-fast"),
        pytest.param(429, constants.RETRY_MAX_ATTEMPTS, id="rate-limit-retries"),
        pytest.param(503, constants.RETRY_MAX_ATTEMPTS, id="server-error-retries"),
    ],
)
def test_add_part_retry_classifies_status(status_code, expected_calls):
    client = AddPartStub(status_code)  # fail_on=None -> raises every call
    part = uploader._PartUpload("upload_x", ("f", b"data"))
    with pytest.raises(uploader.PartUploadError):
        uploader.Uploader(client)._add_part_with_retry(part, clock=_noop_clock())
    assert client.calls == expected_calls


# --- shared retry helper (_retry), used by create/complete/add_part ---------


def _counting_action(fail_times: int, status_code: int):
    """An action that raises `status_code` its first `fail_times` calls, then returns 'ok'."""
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise status_error(status_code)
        return "ok"

    return action, calls


def test_retry_succeeds_after_transient_errors():
    action, calls = _counting_action(fail_times=2, status_code=503)
    assert uploader.Uploader(CancelClient())._retry(action, clock=_noop_clock()) == "ok"
    assert calls["n"] == 3  # two 503s retried, third succeeds


def test_retry_reraises_non_retryable_at_once():
    action, calls = _counting_action(fail_times=1, status_code=409)  # 409 = the complete() conflict
    with pytest.raises(openai.APIStatusError):
        uploader.Uploader(CancelClient())._retry(action, clock=_noop_clock())
    assert calls["n"] == 1  # not retried; the caller decides what 409 means


def test_retry_exhausts_and_raises_last_error():
    action, calls = _counting_action(fail_times=99, status_code=503)
    with pytest.raises(openai.APIStatusError):
        uploader.Uploader(CancelClient())._retry(action, clock=_noop_clock())
    assert calls["n"] == constants.RETRY_MAX_ATTEMPTS


def test_upload_parts_aborts_and_keeps_landed_parts_on_failure(tmp_path):
    # 3 parts, one worker (deterministic order); the 2nd add_part is rejected 4xx.
    state = _make_state(tmp_path, b"x" * (3 * 1024 * 1024), part_size=1024 * 1024, workers=1)
    state.session = Session(upload_id="up1")  # skip create_upload
    up = uploader.Uploader(AddPartStub(400, fail_on=2))

    local = local_file.LocalFile.reopen(state)
    try:
        with pytest.raises(SystemExit):
            up.upload_parts(state, local)
    finally:
        local.close()

    assert set(state.session.parts) == {"0"}  # only the part that landed before the failure


@pytest.mark.parametrize(
    ("keep", "expect_cancelled", "expect_file"),
    [
        pytest.param(False, "up1", False, id="cancel-cancels-session-and-clears"),
        pytest.param(True, None, True, id="keep-preserves-for-resume"),
    ],
)
def test_pause_or_cancel(tmp_path, keep, expect_cancelled, expect_file):
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))
    state.session = Session(upload_id="up1")
    state.file = FileConfig(path="/some/file.bin")
    state.params = UploadParams(num_parts=3)
    state.save()

    client = CancelClient()
    with pytest.raises(SystemExit):
        uploader.Uploader(client)._pause_or_cancel(state, confirm=lambda *a, **k: keep)
    assert client.cancelled == expect_cancelled
    assert (tmp_path / "state.json").exists() is expect_file


# --- assembly polling (_poll_assembly) ------------------------------------


def _poll_state(tmp_path):
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))
    state.session = Session(upload_id="up1")
    return state


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param(
            [{"status": "pending"}, {"status": "pending"}, {"status": "completed", "file": {"id": "fX"}}],
            id="pending-then-completed",
        ),
        pytest.param(
            [RuntimeError("network blip"), {"status": "completed", "file": {"id": "fX"}}],
            id="transient-error-recovers",
        ),
    ],
)
def test_poll_assembly_returns_file_id(tmp_path, responses):
    result = uploader.Uploader(PollClient(responses))._poll_assembly(_poll_state(tmp_path), clock=_noop_clock())
    assert result == "fX"


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param([None], id="session-disappeared"),
        pytest.param([KeyboardInterrupt()], id="ctrl-c-stops-polling"),
    ],
)
def test_poll_assembly_aborts(tmp_path, responses):
    with pytest.raises(SystemExit):
        uploader.Uploader(PollClient(responses))._poll_assembly(_poll_state(tmp_path), clock=_noop_clock())


def test_poll_assembly_times_out(tmp_path):
    # now() jumps past the deadline on the second check, so one poll runs then the loop exits.
    times = iter([0.0, 0.0, 1e9])
    clock = uploader._Clock(sleep=lambda *_a, **_k: None, now=lambda: next(times))
    responses = [{"status": "pending"}, {"status": "pending"}]
    with pytest.raises(SystemExit):
        uploader.Uploader(PollClient(responses))._poll_assembly(_poll_state(tmp_path), clock=clock)


# --- reconcile against a vanished session ---------------------------------


def test_reconcile_parts_aborts_when_session_gone(tmp_path):
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))
    state.session = Session(upload_id="up1", parts={"0": "p0"})
    state.save()

    with pytest.raises(SystemExit):
        uploader.Uploader(GoneSessionClient())._reconcile_parts(state)
    assert not (tmp_path / "state.json").exists()  # cleared, so a re-run starts fresh


# --- interrupt wiring during part fan-out ---------------------------------


def _drain_state(tmp_path):
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))
    state.session = Session(upload_id="up1")
    state.file = FileConfig(filename="f.bin", size=2 * constants.MiB)
    state.params = UploadParams(num_parts=2, part_size=constants.MiB)
    return state


def test_drain_pauses_and_cancels_pending_on_interrupt(tmp_path):
    # as_completed yields the finished future first; its result() raises KeyboardInterrupt,
    # so the still-running part must be cancelled and the session kept for resume.
    interrupted: Future = Future()
    interrupted.set_exception(KeyboardInterrupt())
    pending: Future = Future()  # never completes

    client = CancelClient()
    with pytest.raises(SystemExit):
        uploader.Uploader(client)._drain(_drain_state(tmp_path), [interrupted, pending], confirm=lambda *a, **k: True)

    assert pending.cancelled()  # remaining work was cancelled
    assert client.cancelled is None  # "keep" path preserved the session


def test_drain_cancels_session_when_user_declines_resume(tmp_path):
    interrupted: Future = Future()
    interrupted.set_exception(KeyboardInterrupt())

    client = CancelClient()
    state = _drain_state(tmp_path)
    with pytest.raises(SystemExit):
        uploader.Uploader(client)._drain(state, [interrupted], confirm=lambda *a, **k: False)

    assert client.cancelled == "up1"  # decline -> cancel the server session


# --- complete recovery when a prior run already completed -------------------


def _completable_state(tmp_path):
    state = _make_state(tmp_path, b"z" * 2048, part_size=1024)  # 2 parts
    state.session = Session(parts={"0": "p0", "1": "p1"})  # both already uploaded
    return state


def test_complete_upload_recovers_from_409_already_completed(tmp_path):
    # A prior run completed the session but died before saving; the retried complete()
    # gets a 409, and we adopt the already-assembled file instead of crashing.
    state = _completable_state(tmp_path)
    uploader.Uploader(CompleteConflictClient()).complete_upload(state)
    assert state.session.completed is True
    assert state.session.file_id == "file_done"


def test_complete_upload_reraises_non_conflict_error(tmp_path):
    state = _completable_state(tmp_path)
    with pytest.raises(openai.APIStatusError):
        uploader.Uploader(CompleteErrorClient()).complete_upload(state)


def test_complete_upload_aborts_when_409_session_is_terminal(tmp_path):
    # A 409 whose live status is terminal (cancelled/expired) must not be mistaken for
    # "already completed": clear local state and abort so a re-run starts fresh.
    state = _completable_state(tmp_path)
    state.save()
    with pytest.raises(SystemExit):
        uploader.Uploader(CompleteThenTerminalClient()).complete_upload(state)
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda s: setattr(s, "params", UploadParams(part_size=s.params.part_size, num_parts=5)),
            id="num-parts-inconsistent",
        ),
        pytest.param(lambda s: setattr(s, "session", s.session.without_parts([1])), id="part-missing"),
    ],
)
def test_complete_upload_aborts_on_bad_state(tmp_path, mutate):
    # The pre-complete guards must refuse to assemble a corrupt/incomplete part set.
    state = _completable_state(tmp_path)
    mutate(state)
    with pytest.raises(SystemExit):
        uploader.Uploader(CompleteConflictClient()).complete_upload(state)
