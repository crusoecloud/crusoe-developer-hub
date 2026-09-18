"""Unit tests for src/local_file.py: file selection/validation, MIME inference, positional
reads at EOF, and the change-on-disk guard. No network; a real temp file backs each case."""

from __future__ import annotations

import pytest

from src import local_file
from src.state import FileConfig, State


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


def _empty(path):
    path.write_bytes(b"")
    return path


# --- selection + validation --------------------------------------------------


@pytest.mark.parametrize(
    ("make", "expect_ok"),
    [
        pytest.param(lambda p: p / "missing.bin", False, id="missing"),
        pytest.param(lambda p: _empty(p / "empty.bin"), False, id="empty"),
        pytest.param(lambda p: _write(p / "ok.bin", b"hello"), True, id="ok"),
    ],
)
def test_validate(tmp_path, make, expect_ok):
    path = make(tmp_path)
    problem = local_file._validate(path)
    assert (problem is None) is expect_ok


def test_choose_records_file_and_infers_mime(tmp_path):
    # First answer is rejected by _validate; the loop must re-prompt and accept the second.
    good = _write(tmp_path / "good.jsonl", b"hello")
    answers = iter([str(tmp_path / "missing.bin"), str(good)])
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))

    local = local_file.LocalFile.choose(state, ask=lambda *_a, **_k: next(answers))
    try:
        assert state.file.path == str(good)
        assert state.file.size == 5
        assert state.file.mime_type == "text/jsonl"  # inferred from the extension
        assert local.path == str(good)
    finally:
        local.close()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        pytest.param("train.jsonl", "text/jsonl", id="jsonl"),
        pytest.param("data.ndjson", "application/x-ndjson", id="ndjson"),
        pytest.param("cfg.json", "application/json", id="json"),
        pytest.param("blob.unknownext", "application/octet-stream", id="unknown"),
    ],
)
def test_infer_mime_type(filename, expected):
    assert local_file._infer_mime_type(filename) == expected


# --- reading -----------------------------------------------------------------


def test_reads_short_at_eof(tmp_path):
    # No API here: LocalFile is just a held fd read via os.pread on a local file.
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdefghij")  # 10 bytes
    local = local_file.LocalFile(str(path), path.stat().st_size, path.stat().st_mtime)
    try:
        assert local.read(0, 4) == b"abcd"  # full read within bounds
        assert local.read(6, 100) == b"ghij"  # request past EOF returns only the tail
        assert local.read(10, 4) == b""  # at EOF -> empty
    finally:
        local.close()


# --- change-on-disk guard ----------------------------------------------------


def _selected_state(tmp_path, payload: bytes) -> State:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(payload)
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))
    state.file = FileConfig(path=str(file_path), size=len(payload), mtime=file_path.stat().st_mtime)
    return state


def test_verify_unchanged_passes_then_aborts_on_change(tmp_path):
    state = _selected_state(tmp_path, b"a" * 1000)
    local = local_file.LocalFile.reopen(state)
    try:
        local.verify_unchanged()  # unchanged: no raise

        # Grow the file so size (and mtime) diverge from what was recorded at selection.
        with open(state.file.path, "wb") as f:
            f.write(b"b" * 2000)
        with pytest.raises(SystemExit):
            local.verify_unchanged()
    finally:
        local.close()
