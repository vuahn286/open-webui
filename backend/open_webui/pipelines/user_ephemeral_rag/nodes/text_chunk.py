"""Text chunking utilities shared across pipelines."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List

from .docling_ocr import Page

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


@dataclass(slots=True)
class Chunk:
    """Represents an extracted text chunk."""

    chunk_id: str
    text: str
    page_number: int
    chunk_index: int


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sliding_window(tokens: List[str], size: int, overlap: int) -> Iterator[List[str]]:
    step = max(size - overlap, 1)
    for start in range(0, len(tokens), step):
        yield tokens[start : start + size]


def chunk_pages(pages: Iterable[Page], prefix: str) -> List[Chunk]:
    """Token-aware chunking with configurable overlap."""

    chunks: List[Chunk] = []
    for page in pages:
        normalized = _normalize_whitespace(page.text)
        if not normalized:
            continue
        tokens = normalized.split(" ")
        for index, token_window in enumerate(_sliding_window(tokens, CHUNK_SIZE, CHUNK_OVERLAP)):
            text = " ".join(token_window).strip()
            if not text:
                continue
            chunk_id = f"{prefix}-p{page.page_number}-c{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    page_number=page.page_number,
                    chunk_index=index,
                )
            )
    return chunks


__all__ = ["Chunk", "chunk_pages", "CHUNK_SIZE", "CHUNK_OVERLAP"]
