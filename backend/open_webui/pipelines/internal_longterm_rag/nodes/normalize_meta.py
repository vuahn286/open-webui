"""Metadata normalization for internal documents."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List

from ...user_ephemeral_rag.nodes.docling_ocr import OCRResult
from .source_select import SourceDocument


@dataclass(slots=True)
class NormalizedDocument:
    """Document enriched with normalized metadata."""

    doc_id: str
    title: str
    authors: List[str]
    language: str
    tags: List[str]
    source: str
    pages: List
    normalized_at: datetime


def _detect_language(text: str) -> str:
    if not text:
        return "unknown"
    ascii_ratio = sum(char.isascii() for char in text) / len(text)
    if ascii_ratio > 0.9:
        return "en"
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵ]", text, flags=re.IGNORECASE):
        return "vi"
    return "multi"


def normalize_document(document: SourceDocument, ocr: OCRResult) -> NormalizedDocument:
    """Normalize metadata for a document using OCR results."""

    first_page_text = ocr.pages[0].text if ocr.pages else ""
    title_match = re.search(r"^(.{5,120})$", first_page_text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else document.file.filename
    authors = re.findall(r"(?m)^By ([^\n]+)", first_page_text)
    language = _detect_language("\n".join(page.text for page in ocr.pages))

    return NormalizedDocument(
        doc_id=document.doc_id,
        title=title,
        authors=authors,
        language=language,
        tags=document.tags,
        source=document.source,
        pages=ocr.pages,
        normalized_at=datetime.utcnow(),
    )


__all__ = ["NormalizedDocument", "normalize_document"]
