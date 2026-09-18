"""Shared test doubles: in-memory fakes standing in for src.client.UploadsClient, plus
the small helpers they return/raise. Start here; if this grows unwieldy, split into a
tests/mocks/ package (one file per mock)."""

from __future__ import annotations

import hashlib
import types

import httpx
import openai

from src.client import PartData, UploadSpec


def obj(**kwargs):
    """A tiny attribute bag for fake SDK return values (`.id`, `.status`, ...)."""
    return types.SimpleNamespace(**kwargs)


def status_error(status_code: int) -> openai.APIStatusError:
    """An openai.APIStatusError carrying the given HTTP status, as the SDK would raise."""
    response = httpx.Response(status_code, request=httpx.Request("POST", "http://x"))
    return openai.APIStatusError("boom", response=response, body=None)


class MockUploadsClient:
    """In-memory stand-in for the whole client: records parts and verifies ordering + md5."""

    def __init__(self, expected_payload: bytes):
        self._expected = expected_payload
        self.store: dict[str, bytes] = {}
        self._counter = 0
        self.declared_bytes: int | None = None
        self.assembled_file_id: str | None = None

    def create(self, spec: UploadSpec):
        self.declared_bytes = spec.size_bytes
        return obj(id="upload_fake", status="pending", expires_at=1893456000)

    def add_part(self, upload_id: str, data: PartData):
        self._counter += 1
        part_id = f"part_{self._counter:04d}"
        self.store[part_id] = data[1]  # data is (filename, bytes)
        return obj(id=part_id)

    def complete(self, upload_id: str, part_ids: list[str], md5: str | None = None):
        assembled = b"".join(self.store[p] for p in part_ids)
        assert assembled == self._expected, "assembled bytes differ from source (ordering bug)"
        if md5 is not None:
            assert md5 == hashlib.md5(self._expected).hexdigest(), "md5 mismatch"
        self.assembled_file_id = "file_fake"
        return obj(id=upload_id, status="pending", file=None)

    def get(self, upload_id: str):
        if self.assembled_file_id:
            return {"status": "completed", "file": {"id": self.assembled_file_id}}
        return {"status": "pending", "file": None}

    def list_part_ids(self, upload_id: str) -> set[str]:
        return set(self.store)


class AddPartStub:
    """add_part that raises an APIStatusError on a chosen call number, or on every call
    when fail_on is None. Counts calls so retry behaviour can be asserted."""

    def __init__(self, status_code: int, fail_on: int | None = None):
        self._status_code = status_code
        self._fail_on = fail_on
        self.calls = 0

    def add_part(self, upload_id: str, data: PartData):
        self.calls += 1
        if self._fail_on is None or self.calls == self._fail_on:
            raise status_error(self._status_code)
        return obj(id=f"part_{self.calls}")


class CancelClient:
    """Records the upload id passed to cancel()."""

    def __init__(self):
        self.cancelled: str | None = None

    def cancel(self, upload_id: str):
        self.cancelled = upload_id


class PollClient:
    """get() replays a scripted sequence; an Exception item is raised, anything else returned."""

    def __init__(self, responses):
        self._it = iter(responses)

    def get(self, upload_id: str):
        item = next(self._it)
        if isinstance(item, BaseException):
            raise item
        return item


class GoneSessionClient:
    """list_part_ids returns None: the upload session itself is gone (404)."""

    def list_part_ids(self, upload_id: str):
        return None


class CompleteConflictClient:
    """complete() raises 409 (session already left UPLOADING); get() shows the finished file."""

    def complete(self, upload_id: str, part_ids: list[str], md5: str | None = None):
        raise status_error(409)

    def get(self, upload_id: str):
        return {"status": "completed", "file": {"id": "file_done"}}


class CompleteErrorClient:
    """complete() raises a non-409 error that must not be swallowed."""

    def complete(self, upload_id: str, part_ids: list[str], md5: str | None = None):
        raise status_error(400)


class CompleteThenTerminalClient:
    """complete() 409s, but the live session turns out terminal (cancelled), not assembling."""

    def complete(self, upload_id: str, part_ids: list[str], md5: str | None = None):
        raise status_error(409)

    def get(self, upload_id: str):
        return {"status": "cancelled"}


class PendingClient:
    """Session is still resumable (status pending)."""

    def get(self, upload_id: str) -> dict:
        return {"status": "pending"}


class FlakyClient:
    """Session check fails transiently (not a real 404)."""

    def get(self, upload_id: str) -> dict:
        raise RuntimeError("network down")


class StatusClient:
    """Session reports a given (terminal or completed) status."""

    def __init__(self, status: str, file_id: str | None = None):
        self._status = status
        self._file_id = file_id

    def get(self, upload_id: str) -> dict:
        upload: dict = {"status": self._status, "error": {"code": "x", "message": "y"}}
        if self._file_id is not None:
            upload["file"] = {"id": self._file_id}
        return upload
