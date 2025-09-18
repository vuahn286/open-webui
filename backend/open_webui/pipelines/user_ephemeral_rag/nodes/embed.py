"""Embedding node with deterministic test mode."""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .text_chunk import Chunk

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEST_MODE = os.getenv("EMBED_TEST_MODE", "0").lower() in {"1", "true", "yes"}


def _hash_embedding(text: str, size: int = 384) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeats = (size + len(digest) - 1) // len(digest)
    raw = (digest * repeats)[:size]
    vector = [b / 255.0 for b in raw]
    return _normalize(vector)


def _normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


@dataclass(slots=True)
class EmbeddingResult:
    """Container for generated embeddings."""

    vectors: List[List[float]]
    model: str


class Embedder:
    """Generate embeddings for chunks or ad-hoc texts."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self._model = None

    def _load_model(self) -> None:  # pragma: no cover - heavy dependency
        if TEST_MODE:
            return
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed_chunks(self, chunks: Iterable[Chunk]) -> EmbeddingResult:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts)
        return EmbeddingResult(vectors=vectors, model=self.model_name)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if TEST_MODE:
            return [_hash_embedding(text) for text in texts]
        self._load_model()
        assert self._model is not None
        embeddings = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


__all__ = ["Embedder", "EmbeddingResult"]
