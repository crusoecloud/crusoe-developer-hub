"""Wrapper over the Crusoe Uploads API: the OpenAI SDK for the canonical writes,
plus httpx for Crusoe's REST visibility endpoints that aren't in the SDK."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from openai import OpenAI
from openai.types import Upload
from openai.types.uploads import UploadPart

from . import constants


@dataclass(frozen=True)
class UploadSpec:
    """The immutable description of a file to upload, passed to `create`."""

    filename: str
    purpose: str
    size_bytes: int
    mime_type: str


# A part payload (filename, bytes); the filename is metadata only, not the declared name.
PartData = tuple[str, bytes]


class UploadsClient:
    """Thin client over the Uploads API: the SDK for writes, httpx for status reads."""

    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None):
        # max_retries=0: the uploader owns per-part retry, so the SDK must not double-retry.
        self.sdk = OpenAI(api_key=api_key, base_url=constants.BASE_URL, max_retries=0)
        # transport is injected only by tests (httpx.MockTransport); production leaves it None.
        self._http = httpx.Client(
            base_url=constants.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=constants.HTTP_TIMEOUT_SECONDS,
            transport=transport,
        )

    # --- canonical write path (OpenAI SDK) ---------------------------------

    def create(self, spec: UploadSpec) -> Upload:
        """Open an upload session for the file described by `spec`."""
        return self.sdk.uploads.create(
            filename=spec.filename,
            purpose=spec.purpose,
            bytes=spec.size_bytes,
            mime_type=spec.mime_type,
        )

    def add_part(self, upload_id: str, data: PartData) -> UploadPart:
        """Upload one chunk; safe to call from many threads on the shared client."""
        return self.sdk.uploads.parts.create(upload_id, data=data)

    def complete(self, upload_id: str, part_ids: list[str], md5: str | None = None) -> Upload:
        """Finalize the upload with the ordered part ids and an optional md5."""
        if md5 is None:  # omit md5 entirely when unknown; passing None would send a literal null
            return self.sdk.uploads.complete(upload_id, part_ids=part_ids)
        return self.sdk.uploads.complete(upload_id, part_ids=part_ids, md5=md5)

    def cancel(self, upload_id: str) -> Upload:
        """Cancel the upload session."""
        return self.sdk.uploads.cancel(upload_id)

    # --- Crusoe REST visibility extensions (httpx) -------------------------

    def get(self, upload_id: str) -> dict | None:
        """Retrieve one upload session, or None if it no longer exists (404)."""
        resp = self._http.get(f"/uploads/{upload_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def list_part_ids(self, upload_id: str) -> set[str] | None:
        """Part ids the server holds, or None if the session no longer exists (404)."""
        resp = self._http.get(f"/uploads/{upload_id}/parts")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return {p["id"] for p in resp.json().get("data", [])}

    def list_uploads(self, limit: int = 1) -> dict:
        """List the project's uploads. A cheap authenticated read, used as a startup preflight."""
        resp = self._http.get("/uploads", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()
