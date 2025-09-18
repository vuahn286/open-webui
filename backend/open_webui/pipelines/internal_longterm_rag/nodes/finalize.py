"""Finalize responses for internal long-term pipeline."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def build_ingest_response(
    pages: List[Any],
    chunks: List[Any],
    vectors: List[List[float]],
    graph_stats: Dict[str, int],
    doc_id: str,
    ingested_at: datetime,
) -> Dict[str, Any]:
    """Return normalized ingest response."""

    return {
        "pages": pages,
        "chunks": chunks,
        "vectors": vectors,
        "graph_nodes": graph_stats.get("nodes", 0),
        "graph_relations": graph_stats.get("relations", 0),
        "doc_id": doc_id,
        "ingested_at": ingested_at.isoformat(),
    }


def build_search_response(matches: List[Dict[str, Any]], reasons: List[str]) -> Dict[str, Any]:
    """Return normalized search response."""

    return {"matches": matches, "reasons": reasons}


__all__ = ["build_ingest_response", "build_search_response"]
