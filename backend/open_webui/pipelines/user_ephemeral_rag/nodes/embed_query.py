"""Query embedding helper."""
from __future__ import annotations

from typing import List

from .embed import Embedder


def embed_query(query: str, model_name: str | None = None) -> List[float]:
    """Return a normalized embedding for the query."""

    embedder = Embedder(model_name=model_name)
    embeddings = embedder.embed_texts([query])
    return embeddings[0] if embeddings else []


__all__ = ["embed_query"]
