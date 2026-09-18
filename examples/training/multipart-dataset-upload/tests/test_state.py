"""Unit tests for src/state.py: load (pure) + resume_or_start (resume/reset decision)."""

from __future__ import annotations

import pytest

from src.state import FileConfig, Session, State, UploadParams
from tests.mocks import FlakyClient, PendingClient, StatusClient


def test_expiry_text_marks_future_vs_past():
    assert State._expiry_text(None) is None
    assert State._expiry_text(4102444800).startswith("Expires:")  # year 2100
    assert State._expiry_text(946684800).startswith("Expired:")  # year 2000


@pytest.mark.parametrize(
    ("start_new", "expect_complete"),
    [
        pytest.param(True, False, id="start-new-upload"),
        pytest.param(False, True, id="re-show-and-exit"),
    ],
)
def test_completed_state_resolves_to_fresh_or_show(tmp_path, start_new, expect_complete):
    state_path = tmp_path / "state.json"
    seed = State(api_key="k", state_file_path=str(state_path))
    seed.session = Session(completed=True, file_id="file_prev")
    seed.file = FileConfig(path="/old/train.jsonl")
    seed.save()

    state = State.load("k", str(state_path))
    # confirm is injected, not monkeypatched; completed path never touches the client.
    state.resume_or_start(client=None, confirm=lambda *a, **k: start_new)

    assert state.session.completed is expect_complete
    if start_new:
        assert state.session.file_id is None  # fresh state, previous result cleared
        # Starting fresh must not leave a skeletal state file behind, or the next
        # run would show its stale created/updated times as "saved state".
        assert not state_path.exists()
    else:
        assert state.session.file_id == "file_prev"


def _seed_in_progress(tmp_path):
    data_file = tmp_path / "in-progress.bin"
    data_file.write_bytes(b"x" * 4096)
    state_path = tmp_path / "state.json"
    seed = State(api_key="k", state_file_path=str(state_path))
    seed.file = FileConfig(path=str(data_file), size=data_file.stat().st_size, mtime=data_file.stat().st_mtime)
    seed.session = Session(upload_id="up1")
    seed.params = UploadParams(num_parts=1)
    seed.save()
    return state_path


def test_declining_resume_resolves_to_fresh_and_clears(tmp_path):
    # A real, unchanged file + a pending session make the resume genuinely valid,
    # so the decline path is reached without patching the unit's own _validate_resume().
    state_path = _seed_in_progress(tmp_path)

    state = State.load("k", str(state_path))
    state.resume_or_start(PendingClient(), confirm=lambda *a, **k: False)

    assert state.resuming is False  # declined -> fresh, not a resume
    assert not state_path.exists()  # declined -> nothing left, incl. old timestamps


def test_accepting_resume_resolves_to_resume(tmp_path):
    state_path = _seed_in_progress(tmp_path)

    state = State.load("k", str(state_path))
    state.resume_or_start(PendingClient(), confirm=lambda *a, **k: True)

    assert state.resuming is True
    assert state.session.completed is False
    assert state.session.upload_id == "up1"


def test_unreachable_session_is_not_treated_as_gone(tmp_path):
    # A transient failure to reach the server must not discard resumable state.
    state_path = _seed_in_progress(tmp_path)

    state = State.load("k", str(state_path))
    state.resume_or_start(FlakyClient(), confirm=lambda *a, **k: True)

    assert state.resuming is True
    assert state_path.exists()  # not discarded despite the failed check


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_terminal_session_resolves_to_fresh(tmp_path, status):
    # A terminal session (failed/expired/cancelled) can't be resumed: report it and
    # start fresh. Exercised through the public resume_or_start, not the private _validate_resume().
    state_path = _seed_in_progress(tmp_path)

    state = State.load("k", str(state_path))
    state.resume_or_start(StatusClient(status))

    assert state.resuming is False
    assert not state_path.exists()


def test_changed_source_file_resolves_to_fresh(tmp_path):
    # The source file changed since the upload started, so its parts are stale: the
    # in-progress session can't be resumed even though the server session is fine.
    state_path = _seed_in_progress(tmp_path)
    (tmp_path / "in-progress.bin").write_bytes(b"x" * 8192)  # was 4096

    state = State.load("k", str(state_path))
    state.resume_or_start(PendingClient())

    assert state.resuming is False
    assert not state_path.exists()


def test_server_completed_while_away_is_adopted(tmp_path):
    # The session finished assembling between runs; adopt the server's file id and
    # route to the completed flow instead of re-running or re-completing the upload.
    state_path = _seed_in_progress(tmp_path)

    state = State.load("k", str(state_path))
    state.resume_or_start(StatusClient("completed", file_id="file_live"), confirm=lambda *a, **k: False)

    assert state.session.completed is True
    assert state.session.file_id == "file_live"


def test_empty_state_file_is_cleared_silently(tmp_path):
    # A skeletal leftover (timestamps only, no file chosen, no session) has nothing
    # to resume; it must be cleared without prompting or a "Cannot resume" notice.
    state_path = tmp_path / "state.json"
    State(api_key="k", state_file_path=str(state_path)).save()
    assert state_path.exists()

    state = State.load("k", str(state_path))
    state.resume_or_start(client=None)

    assert state.session.completed is False
    assert not state_path.exists()
