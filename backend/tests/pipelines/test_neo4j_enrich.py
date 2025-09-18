from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from open_webui.pipelines.internal_longterm_rag.nodes.neo4j_enrich import (  # noqa: E402
    Neo4jEnricher,
)
from open_webui.pipelines.user_ephemeral_rag.nodes.text_chunk import Chunk  # noqa: E402


class _FakeTx:
    def __init__(self) -> None:
        self.queries: List[Dict[str, Any]] = []

    def run(self, query: str, **params: Any) -> None:
        self.queries.append({"query": query, "params": params})


class _FakeSession:
    def __init__(self, tx: _FakeTx) -> None:
        self.tx = tx

    def __enter__(self):  # pragma: no cover - context behaviour
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - context behaviour
        return False

    def execute_write(self, func, *args, **kwargs):
        return func(self.tx, *args, **kwargs)


class _FakeDriver:
    def __init__(self) -> None:
        self.tx = _FakeTx()
        self.session_obj = _FakeSession(self.tx)

    def session(self) -> _FakeSession:
        return self.session_obj


def test_enrichment_builds_entities() -> None:
    driver = _FakeDriver()
    enricher = Neo4jEnricher(driver=driver)
    chunk = Chunk(chunk_id="doc-1", text="Alice met Bob at https://example.com", page_number=1, chunk_index=0)
    stats = enricher.enrich(
        doc_id="doc",
        filename="file.pdf",
        source="source",
        chunks=[chunk],
    )

    assert stats.nodes >= 2
    assert stats.relations >= 3

    recorded_queries = driver.tx.queries
    merge_entity_queries = [query for query in recorded_queries if "MERGE (e:Entity" in query["query"]]
    assert merge_entity_queries
