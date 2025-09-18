"""Source selection utilities for the internal long-term pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from uuid import uuid4

try:  # pragma: no cover - optional dependency
    import boto3
except ModuleNotFoundError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

from ...user_ephemeral_rag.nodes.file_ingest import IngestedFile, ingest_file


@dataclass(slots=True)
class SourceDocument:
    """Represents a document prepared for ingestion."""

    doc_id: str
    file: IngestedFile
    source: str
    tags: List[str]


def _iter_local_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                yield file


def _load_s3_objects(uri: str) -> Iterable[IngestedFile]:
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 ingestion. Install boto3 and retry.")
    _, _, bucket, *key_parts = uri.split("/", 3)
    key_prefix = key_parts[0] if key_parts else ""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            yield ingest_file(body, filename=Path(key).name)


def select_sources(
    path: Optional[str] = None,
    file: Optional[os.PathLike[str] | str] = None,
    url: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[SourceDocument]:
    """Gather source documents from supported backends."""

    collected: List[SourceDocument] = []
    resolved_tags = tags or []

    if path:
        path_obj = Path(path)
        for candidate in _iter_local_files(path_obj):
            ingested = ingest_file(candidate)
            collected.append(
                SourceDocument(
                    doc_id=str(uuid4()),
                    file=ingested,
                    source=str(candidate.resolve()),
                    tags=resolved_tags,
                )
            )
    if file:
        ingested = ingest_file(file)
        collected.append(
            SourceDocument(
                doc_id=str(uuid4()),
                file=ingested,
                source="uploaded",
                tags=resolved_tags,
            )
        )
    if url and url.startswith("s3://"):
        for ingested in _load_s3_objects(url):
            collected.append(
                SourceDocument(
                    doc_id=str(uuid4()),
                    file=ingested,
                    source=url,
                    tags=resolved_tags,
                )
            )

    return collected


__all__ = ["SourceDocument", "select_sources"]
