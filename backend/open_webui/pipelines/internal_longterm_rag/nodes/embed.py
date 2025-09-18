"""Embedding helper for long-term pipeline."""
from __future__ import annotations

import os
from typing import Iterable, List

from ...user_ephemeral_rag.nodes.embed import Embedder, EmbeddingResult
from ...user_ephemeral_rag.nodes.text_chunk import Chunk

DEFAULT_MODEL = os.getenv("INTERNAL_EMBED_MODEL", os.getenv("EMBED_MODEL"))


class LongTermEmbedder(Embedder):
    """Embedder that honors internal pipeline defaults."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name=model_name or DEFAULT_MODEL)

    def embed_chunks(self, chunks: Iterable[Chunk]) -> EmbeddingResult:
        return super().embed_chunks(chunks)


__all__ = ["LongTermEmbedder"]
