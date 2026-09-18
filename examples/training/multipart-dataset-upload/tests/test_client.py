"""Unit tests for src/client.py's Crusoe REST visibility endpoints (get, list_part_ids),
driven by an injected httpx.MockTransport so no real network calls are made. These verify
the 404 -> None contract that uploader/state resume logic depends on."""

from __future__ import annotations

import httpx
import pytest

from src.client import UploadsClient


def _client(handler) -> UploadsClient:
    """A UploadsClient whose httpx transport is a MockTransport running `handler`."""
    return UploadsClient(api_key="k", transport=httpx.MockTransport(handler))


def test_get_returns_parsed_dict_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/uploads/up1")  # BASE_URL carries a /v1 prefix
        return httpx.Response(200, json={"id": "up1", "status": "pending"})

    client = _client(handler)
    try:
        assert client.get("up1") == {"id": "up1", "status": "pending"}
    finally:
        client.close()


def test_get_returns_none_on_404():
    client = _client(lambda _request: httpx.Response(404, json={"error": "not found"}))
    try:
        assert client.get("gone") is None
    finally:
        client.close()


def test_get_raises_on_server_error():
    client = _client(lambda _request: httpx.Response(503))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.get("up1")
    finally:
        client.close()


def test_list_part_ids_returns_id_set_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/uploads/up1/parts")  # BASE_URL carries a /v1 prefix
        return httpx.Response(200, json={"data": [{"id": "p0"}, {"id": "p1"}]})

    client = _client(handler)
    try:
        assert client.list_part_ids("up1") == {"p0", "p1"}
    finally:
        client.close()


def test_list_part_ids_returns_none_on_404():
    # 404 here means the whole session is gone, distinct from an empty part set.
    client = _client(lambda _request: httpx.Response(404))
    try:
        assert client.list_part_ids("gone") is None
    finally:
        client.close()


def test_list_uploads_returns_parsed_dict_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/uploads")  # BASE_URL carries a /v1 prefix
        return httpx.Response(200, json={"object": "list", "data": [], "has_more": False})

    client = _client(handler)
    try:
        assert client.list_uploads() == {"object": "list", "data": [], "has_more": False}
    finally:
        client.close()


def test_list_uploads_raises_on_bad_key():
    # The startup preflight relies on a rejected key surfacing as an HTTPStatusError.
    client = _client(lambda _request: httpx.Response(401, json={"error": "unauthorized"}))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.list_uploads()
    finally:
        client.close()
