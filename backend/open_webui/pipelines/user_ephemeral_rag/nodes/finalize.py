"""Finalize node for building consistent pipeline responses."""
from __future__ import annotations

from typing import Any, Dict, List


def build_ingest_response(
    pages: List[Any],
    chunks: List[Any],
    vectors: List[List[float]],
    expire_at: str,
    doc_id: str,
) -> Dict[str, Any]:
    """Return the canonical ingest response payload."""

    return {
        "pages": pages,
        "chunks": chunks,
        "vectors": vectors,
        "expire_at": expire_at,
        "doc_id": doc_id,
    }


def build_search_response(matches: List[Dict[str, Any]], reasons: List[str]) -> Dict[str, Any]:
    """Return the canonical search response payload."""

    return {"matches": matches, "reasons": reasons}


__all__ = ["build_ingest_response", "build_search_response"]
