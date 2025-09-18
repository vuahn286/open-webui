"""Qdrant integration for ephemeral tenant scoped storage."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import requests

from .text_chunk import Chunk

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_EPHEMERAL", "user_ephemeral")
NAMESPACE_KEY = os.getenv("QDRANT_NAMESPACE_KEY", "tenant_id")
DEFAULT_TTL = int(os.getenv("TTL_HOURS", "48"))


@dataclass(slots=True)
class QdrantPoint:
    """Represents a Qdrant vector point."""

    id: str
    vector: List[float]
    payload: Dict[str, Any]


class EphemeralQdrantClient:
    """Wrapper around the Qdrant HTTP API for ephemeral workloads."""

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        namespace_key: str | None = None,
        default_ttl_hours: int | None = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.url = (url or QDRANT_URL).rstrip("/")
        self.collection = collection or QDRANT_COLLECTION
        self.namespace_key = namespace_key or NAMESPACE_KEY
        self.default_ttl_hours = default_ttl_hours or DEFAULT_TTL
        self.session = session or requests.Session()

    def _collection_url(self) -> str:
        return f"{self.url}/collections/{self.collection}"

    def ensure_collection(self, vector_size: int) -> None:
        response = self.session.get(self._collection_url(), timeout=30)
        if response.status_code == 200:
            return
        if response.status_code not in (404, 400):
            response.raise_for_status()
        payload = {
            "vectors": {"size": vector_size, "distance": "Cosine"},
            "optimizers_config": {"default_segment_number": 1},
        }
        create_response = self.session.put(self._collection_url(), json=payload, timeout=30)
        create_response.raise_for_status()

    def upsert(
        self,
        tenant_id: str,
        session_id: Optional[str],
        filename: str,
        chunks: Iterable[Chunk],
        vectors: Iterable[List[float]],
        ttl_hours: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        expire_at = now + timedelta(hours=ttl_hours or self.default_ttl_hours)
        expire_at_iso = expire_at.isoformat()

        chunks_list = list(chunks)
        vectors_list = list(vectors)
        if not chunks_list:
            return {"expire_at": expire_at_iso, "points": []}

        if collection:
            self.collection = collection
        self.ensure_collection(len(vectors_list[0]))

        points = []
        for chunk, vector in zip(chunks_list, vectors_list):
            point_id = str(uuid4())
            payload = {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "expire_at": expire_at_iso,
                "created_at": now.isoformat(),
                "text": chunk.text,
            }
            points.append(QdrantPoint(id=point_id, vector=list(map(float, vector)), payload=payload))

        body = {"points": [asdict(point) for point in points]}
        response = self.session.put(
            f"{self._collection_url()}/points?wait=true", json=body, timeout=30
        )
        response.raise_for_status()
        return {"expire_at": expire_at_iso, "points": [point.id for point in points]}

    def search(
        self,
        query_vector: List[float],
        tenant_id: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        if collection:
            self.collection = collection
        must_filters = [
            {"key": self.namespace_key, "match": {"value": tenant_id}},
            {"key": "expire_at", "range": {"gte": datetime.now(timezone.utc).isoformat()}},
        ]
        if session_id:
            must_filters.append({"key": "session_id", "match": {"value": session_id}})

        body = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
            "filter": {"must": must_filters},
        }
        response = self.session.post(
            f"{self._collection_url()}/points/search", json=body, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        return {"matches": payload.get("result", [])}

    def cleanup_expired(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        filters = [
            {"key": "expire_at", "range": {"lt": datetime.now(timezone.utc).isoformat()}},
        ]
        if tenant_id:
            filters.append({"key": self.namespace_key, "match": {"value": tenant_id}})

        body = {"filter": {"must": filters}}
        response = self.session.post(
            f"{self._collection_url()}/points/delete?wait=true", json=body, timeout=30
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        return {"status": result.get("status", "acknowledged"), "details": result}


def upsert_ephemeral(**kwargs: Any) -> Dict[str, Any]:
    client = EphemeralQdrantClient()
    return client.upsert(**kwargs)


def search_ephemeral(**kwargs: Any) -> Dict[str, Any]:
    client = EphemeralQdrantClient()
    return client.search(**kwargs)


def cleanup_ephemeral(**kwargs: Any) -> Dict[str, Any]:
    client = EphemeralQdrantClient()
    return client.cleanup_expired(**kwargs)


__all__ = [
    "EphemeralQdrantClient",
    "QdrantPoint",
    "cleanup_ephemeral",
    "search_ephemeral",
    "upsert_ephemeral",
]
