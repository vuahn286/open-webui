# Open WebUI RAG Pipelines: Ephemeral & Internal Long-term

This directory contains two reference pipelines implemented with the Open WebUI
pipeline conventions. Both pipelines integrate **Docling OCR**, **Qdrant**
vector search and optional reranking. The internal long-term pipeline also
enriches metadata into **Neo4j**.

## Pipeline Overview

| Pipeline | Ingestion DAG | Search DAG | Storage |
| --- | --- | --- | --- |
| `user_ephemeral_rag` | `file_ingest → docling_ocr → text_chunk → embed → qdrant_upsert_ephemeral → finalize` | `query_input → embed_query → qdrant_search_ephemeral → rerank → finalize` | Qdrant collection scoped per tenant/session with TTL cleanup |
| `internal_longterm_rag` | `source_select → docling_ocr → normalize_meta → text_chunk → embed → qdrant_upsert_longterm → neo4j_enrich → finalize` | `query_input → embed_query → qdrant_search_longterm → rerank → finalize` | Qdrant long-term collection + Neo4j knowledge graph |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DOCLING_URL` | `http://docling:8000/ocr` | Docling OCR endpoint |
| `QDRANT_URL` | `http://qdrant:6333` | Base URL for Qdrant HTTP API |
| `QDRANT_COLLECTION_EPHEMERAL` | `user_ephemeral` | Ephemeral collection name |
| `QDRANT_COLLECTION_LONGTERM` | `internal_documents` | Long-term collection |
| `QDRANT_NAMESPACE_KEY` | `tenant_id` | Payload key used for tenant filtering |
| `TTL_HOURS` | `48` | Default TTL for ephemeral vectors |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Default embedding model |
| `INTERNAL_EMBED_MODEL` | – | Overrides `EMBED_MODEL` for the internal pipeline |
| `EPHEMERAL_RERANK` | `false` | Enable reranker for the ephemeral search lane |
| `EPHEMERAL_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Optional override |
| `INTERNAL_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model for internal search |
| `RERANK_TEST_MODE` | `0` | Deterministic scoring for tests or air-gapped setups |
| `EMBED_TEST_MODE` | `0` | Hash-based embeddings for tests |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j Bolt endpoint |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASS` | `secret` | Neo4j password |

## Docker Compose Snippets

Add the following service overrides to compose files as needed.

```yaml
services:
  docling:
    image: ghcr.io/docling-ai/docling:latest
    ports:
      - "8000:8000"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: "neo4j/secret"
    ports:
      - "7687:7687"
      - "7474:7474"
```

Use the environment variables from the table above in your `backend/.env` file
or container environment to configure per deployment.

## Testing

The included unit tests focus on deterministic behaviours (payload structure,
filters, TTL handling, Neo4j enrichment logic). Run them via:

```bash
cd backend
pytest tests/pipelines -q
```

Enable `EMBED_TEST_MODE=1` and `RERANK_TEST_MODE=1` when executing the pipelines
in environments without access to the heavy ML models.
