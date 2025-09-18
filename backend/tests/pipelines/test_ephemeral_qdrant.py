from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from open_webui.pipelines.user_ephemeral_rag.nodes.qdrant_ephemeral import (  # noqa: E402
    EphemeralQdrantClient,
)
from open_webui.pipelines.user_ephemeral_rag.nodes.text_chunk import Chunk  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, timeout: int) -> _FakeResponse:
        self.calls.append({"method": "GET", "url": url})
        return _FakeResponse(404)

    def put(self, url: str, json: Dict[str, Any], timeout: int) -> _FakeResponse:
        self.calls.append({"method": "PUT", "url": url, "json": json})
        return _FakeResponse(200, {"result": "ok"})

    def post(self, url: str, json: Dict[str, Any], timeout: int) -> _FakeResponse:
        self.calls.append({"method": "POST", "url": url, "json": json})
        if url.endswith("/search"):
            return _FakeResponse(200, {"result": []})
        return _FakeResponse(200, {"result": {"status": "ack"}})


@pytest.fixture
def chunks() -> List[Chunk]:
    return [
        Chunk(chunk_id="doc-1", text="hello world", page_number=1, chunk_index=0),
        Chunk(chunk_id="doc-2", text="second chunk", page_number=1, chunk_index=1),
    ]


@pytest.fixture
def vectors() -> List[List[float]]:
    return [[0.6, 0.8], [0.1, 0.99]]


def test_upsert_sets_ttl_and_payload(chunks: List[Chunk], vectors: List[List[float]]) -> None:
    session = _RecordingSession()
    client = EphemeralQdrantClient(session=session, default_ttl_hours=2)
    result = client.upsert(
        tenant_id="tenant",
        session_id="session",
        filename="file.pdf",
        chunks=chunks,
        vectors=vectors,
    )

    assert "expire_at" in result
    expire_at = datetime.fromisoformat(result["expire_at"])
    assert expire_at - datetime.now(timezone.utc) <= timedelta(hours=2, minutes=1)

    upsert_call = [call for call in session.calls if call["method"] == "PUT"][-1]
    payloads = upsert_call["json"]["points"]
    assert payloads[0]["payload"]["tenant_id"] == "tenant"
    assert payloads[0]["payload"]["session_id"] == "session"
    assert payloads[0]["payload"]["chunk_index"] == 0
    assert payloads[0]["payload"]["text"] == "hello world"


def test_search_builds_filters(chunks: List[Chunk], vectors: List[List[float]]) -> None:
    session = _RecordingSession()
    client = EphemeralQdrantClient(session=session)
    client.search(query_vector=[0.1, 0.2], tenant_id="tenant", session_id="abc")

    search_call = [call for call in session.calls if call["method"] == "POST"][-1]
    filters = search_call["json"]["filter"]["must"]
    keys = {item["key"] for item in filters}
    assert {"tenant_id", "expire_at", "session_id"}.issubset(keys)
