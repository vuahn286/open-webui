"""Qdrant integration for long-term knowledge storage."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import requests

from ...user_ephemeral_rag.nodes.text_chunk import Chunk

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_LONGTERM", "internal_documents")


@dataclass(slots=True)
class QdrantPoint:
    id: str
    vector: List[float]
    payload: Dict[str, Any]


class LongTermQdrantClient:
    """Lightweight wrapper for Qdrant HTTP APIs."""

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.url = (url or QDRANT_URL).rstrip("/")
        self.collection = collection or COLLECTION_NAME
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
            "on_disk_payload": True,
        }
        create_response = self.session.put(self._collection_url(), json=payload, timeout=30)
        create_response.raise_for_status()

    def upsert(
        self,
        doc_id: str,
        filename: str,
        source: str,
        tags: List[str],
        chunks: Iterable[Chunk],
        vectors: Iterable[List[float]],
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        chunks_list = list(chunks)
        vectors_list = list(vectors)
        if not chunks_list:
            return {"points": []}
        if collection:
            self.collection = collection
        self.ensure_collection(len(vectors_list[0]))

        now = datetime.now(timezone.utc).isoformat()
        points = []
        for chunk, vector in zip(chunks_list, vectors_list):
            points.append(
                QdrantPoint(
                    id=str(uuid4()),
                    vector=list(map(float, vector)),
                    payload={
                        "doc_id": doc_id,
                        "filename": filename,
                        "source": source,
                        "tags": tags,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                        "created_at": now,
                        "text": chunk.text,
                    },
                )
            )
        body = {"points": [asdict(point) for point in points]}
        response = self.session.put(
            f"{self._collection_url()}/points?wait=true", json=body, timeout=30
        )
        response.raise_for_status()
        return {"points": [point.id for point in points]}

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        if collection:
            self.collection = collection
        body = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        response = self.session.post(
            f"{self._collection_url()}/points/search", json=body, timeout=30
        )
        response.raise_for_status()
        return {"matches": response.json().get("result", [])}


def upsert_long_term(**kwargs: Any) -> Dict[str, Any]:
    client = LongTermQdrantClient()
    return client.upsert(**kwargs)


def search_long_term(**kwargs: Any) -> Dict[str, Any]:
    client = LongTermQdrantClient()
    return client.search(**kwargs)


__all__ = ["LongTermQdrantClient", "QdrantPoint", "search_long_term", "upsert_long_term"]
