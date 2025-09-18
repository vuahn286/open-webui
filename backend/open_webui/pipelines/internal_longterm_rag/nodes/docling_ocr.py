"""Re-export Docling OCR for internal pipeline."""
from __future__ import annotations

from ...user_ephemeral_rag.nodes.docling_ocr import DoclingOCR, OCRResult, Page

__all__ = ["DoclingOCR", "OCRResult", "Page"]
