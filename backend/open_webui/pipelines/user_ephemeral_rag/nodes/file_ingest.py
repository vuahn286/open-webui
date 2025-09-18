"""File ingestion node for the ephemeral RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import mimetypes


@dataclass(slots=True)
class IngestedFile:
    """In-memory representation of an uploaded file."""

    filename: str
    content_type: str
    size_bytes: int
    data: bytes
    tenant_id: Optional[str]
    session_id: Optional[str]


FileLike = Union[bytes, BinaryIO, Path, str]


def _read_file_contents(file: FileLike) -> Tuple[bytes, Optional[str]]:
    """Return the file bytes and detected filename from supported inputs."""

    if isinstance(file, bytes):
        return file, None

    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.read_bytes(), path.name

    data = file.read()
    guessed_name = getattr(file, "name", None)
    return data, Path(guessed_name).name if guessed_name else None


def ingest_file(
    file: FileLike,
    tenant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> IngestedFile:
    """Read input file-like object and return standardized representation."""

    data, detected_name = _read_file_contents(file)
    final_name = filename or detected_name or "uploaded_file"
    content_type = mimetypes.guess_type(final_name)[0] or "application/octet-stream"

    return IngestedFile(
        filename=final_name,
        content_type=content_type,
        size_bytes=len(data),
        data=data,
        tenant_id=tenant_id,
        session_id=session_id,
    )


__all__ = ["IngestedFile", "ingest_file"]
