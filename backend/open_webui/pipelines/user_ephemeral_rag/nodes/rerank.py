"""Optional reranking support for the ephemeral pipeline."""
from __future__ import annotations

import os
from typing import Any, Dict, List

ENABLED = os.getenv("EPHEMERAL_RERANK", "false").lower() in {"1", "true", "yes"}
TEST_MODE = os.getenv("RERANK_TEST_MODE", "0").lower() in {"1", "true", "yes"}
DEFAULT_MODEL = os.getenv("EPHEMERAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


class EphemeralReranker:
    """Apply BGE reranking when enabled."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self._model = None

    def _load_model(self) -> None:  # pragma: no cover - heavy dependency
        if TEST_MODE or not ENABLED:
            return
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ENABLED or not matches:
            return matches
        if TEST_MODE:
            return sorted(matches, key=lambda item: len(item.get("payload", {})), reverse=True)
        self._load_model()
        assert self._model is not None
        pairs = [[query, match.get("payload", {}).get("text", "")] for match in matches]
        scores = self._model.predict(pairs)
        for match, score in zip(matches, scores):
            match["score"] = float(score)
        return sorted(matches, key=lambda item: item.get("score", 0), reverse=True)


__all__ = ["EphemeralReranker"]
