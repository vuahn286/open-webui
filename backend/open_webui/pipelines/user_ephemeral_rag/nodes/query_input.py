"""Query normalization for search mode."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Query:
    """Normalized search query."""

    text: str


def normalize_query(query: str) -> Query:
    """Trim whitespace and build query container."""

    return Query(text=query.strip())


__all__ = ["Query", "normalize_query"]
