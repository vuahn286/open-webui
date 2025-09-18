"""OCR node leveraging Docling with Tesseract fallback."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

from .file_ingest import IngestedFile

try:  # pragma: no cover - optional dependency
    import pytesseract
except ModuleNotFoundError:  # pragma: no cover - fallback path
    pytesseract = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Page:
    """Represents a single text page extracted from a document."""

    page_number: int
    text: str
    source: str


@dataclass(slots=True)
class OCRResult:
    """Structured OCR response."""

    pages: List[Page]
    metadata: dict
    processed_at: datetime


class DoclingOCR:
    """Call Docling OCR service with graceful degradation."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120) -> None:
        self.base_url = base_url or os.getenv("DOCLING_URL", "http://docling:8000/ocr")
        self.timeout = timeout

    def __call__(self, file: IngestedFile) -> OCRResult:
        pages = self._call_docling(file)
        if not pages:
            LOGGER.warning("Docling returned no pages for %s; attempting Tesseract fallback.", file.filename)
            pages = self._tesseract_fallback(file)

        return OCRResult(pages=pages, metadata={}, processed_at=datetime.utcnow())

    def _call_docling(self, file: IngestedFile) -> List[Page]:
        try:
            response = requests.post(
                self.base_url,
                files={"file": (file.filename, file.data, file.content_type)},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                LOGGER.error("Docling OCR failed (%s): %s", response.status_code, response.text)
                return []
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover - network issues
            LOGGER.error("Docling OCR request error: %s", exc)
            return []
        except json.JSONDecodeError:  # pragma: no cover - invalid response
            LOGGER.error("Docling OCR returned invalid JSON")
            return []

        pages: List[Page] = []
        for index, page in enumerate(payload.get("pages", []), start=1):
            text = page.get("text", "").strip()
            if text:
                pages.append(Page(page_number=index, text=text, source="docling"))
        return pages

    def _tesseract_fallback(self, file: IngestedFile) -> List[Page]:
        if pytesseract is None:
            LOGGER.warning("pytesseract is not installed; fallback skipped.")
            return []

        try:
            with BytesIO(file.data) as buffer:
                image = Image.open(buffer)
                text = pytesseract.image_to_string(image)
        except Exception as exc:  # pragma: no cover - optical recognition errors
            LOGGER.error("Tesseract fallback failed for %s: %s", file.filename, exc)
            return []

        return [Page(page_number=1, text=text.strip(), source="tesseract")]


__all__ = ["DoclingOCR", "OCRResult", "Page"]
