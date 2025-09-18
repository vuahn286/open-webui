"""Neo4j enrichment node for internal documents."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ...user_ephemeral_rag.nodes.text_chunk import Chunk

try:  # pragma: no cover - optional dependency
    from neo4j import GraphDatabase
except ModuleNotFoundError:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment]

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "secret")


@dataclass(slots=True)
class GraphStats:
    """Holds mutation counts for graph enrichment."""

    nodes: int
    relations: int


class Neo4jEnricher:
    """Enriches ingested chunks into Neo4j knowledge graph."""

    INDEX_QUERIES = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY",
    ]

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        driver=None,
    ) -> None:
        if driver is not None:
            self._driver = driver
        else:
            if GraphDatabase is None:
                raise RuntimeError("neo4j driver is required for enrichment")
            self._driver = GraphDatabase.driver(uri or NEO4J_URI, auth=(user or NEO4J_USER, password or NEO4J_PASS))

    def close(self) -> None:  # pragma: no cover - resource cleanup
        if hasattr(self._driver, "close"):
            self._driver.close()

    @staticmethod
    def _extract_entities(text: str) -> List[Tuple[str, str]]:
        entities: List[Tuple[str, str]] = []
        for match in re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", text):
            entities.append((match.strip(), "MENTION"))
        for ref in re.findall(r"https?://[^\s]+", text):
            entities.append((ref.strip(), "REF"))
        return entities

    def _ensure_indexes(self) -> None:
        with self._driver.session() as session:
            for query in self.INDEX_QUERIES:
                session.execute_write(lambda tx, q=query: tx.run(q))

    def enrich(
        self,
        doc_id: str,
        filename: str,
        source: str,
        chunks: Iterable[Chunk],
    ) -> GraphStats:
        self._ensure_indexes()
        node_count = 0
        relation_count = 0

        with self._driver.session() as session:
            for chunk in chunks:
                summary = session.execute_write(
                    self._merge_chunk,
                    doc_id,
                    filename,
                    source,
                    chunk,
                )
                node_count += summary.get("nodes", 0)
                relation_count += summary.get("relations", 0)

        return GraphStats(nodes=node_count, relations=relation_count)

    @staticmethod
    def _merge_chunk(tx, doc_id: str, filename: str, source: str, chunk: Chunk) -> Dict[str, int]:
        tx.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "ON CREATE SET d.filename = $filename, d.source = $source",
            doc_id=doc_id,
            filename=filename,
            source=source,
        )
        tx.run(
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.idx = $idx, c.page = $page, c.text = $text",
            chunk_id=chunk.chunk_id,
            idx=chunk.chunk_index,
            page=chunk.page_number,
            text=chunk.text,
        )
        tx.run(
            "MATCH (d:Document {doc_id: $doc_id}), (c:Chunk {chunk_id: $chunk_id}) "
            "MERGE (d)-[:HAS_CHUNK]->(c)",
            doc_id=doc_id,
            chunk_id=chunk.chunk_id,
        )
        relation_count = 1
        node_count = 0

        entities = Neo4jEnricher._extract_entities(chunk.text)
        for name, entity_type in entities:
            tx.run(
                "MERGE (e:Entity {name: $name, type: $type})",
                name=name,
                type=entity_type,
            )
            tx.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}), (e:Entity {name: $name, type: $type}) "
                "MERGE (c)-[:%s]->(e)" % ("MENTIONS" if entity_type == "MENTION" else "REFS"),
                chunk_id=chunk.chunk_id,
                name=name,
                type=entity_type,
            )
            node_count += 1
            relation_count += 1

        return {"nodes": node_count, "relations": relation_count}


__all__ = ["GraphStats", "Neo4jEnricher"]
