# RAG Service

FastAPI microservice for RAG (Retrieval-Augmented Generation) queries.

## Overview

The RAG service provides similarity search over the knowledge base embeddings stored in PostgreSQL. It:

1. **Starts immediately** (non-blocking startup) - service becomes available even if embeddings aren't ready
2. **Polls PostgreSQL** in background - checks every 5 seconds for embeddings (up to 10 minutes)
3. **Loads embeddings** when available - parses pgvector format and builds FAISS index in memory
4. **Exposes REST API** - provides query endpoints for knowledge base retrieval

### Key Features

- **Non-blocking startup**: Service starts immediately, loads index in background
- **Graceful degradation**: Service stays in "not ready" state until embeddings available
- **Automatic recovery**: Polls PostgreSQL until embeddings found
- **No circular dependencies**: Can start before init job completes

## API Endpoints

### `POST /rag/query`

Query the knowledge base for relevant error solutions.

**Request:**
```json
{
  "query": "error message or log summary",
  "top_k": 10,
  "top_n": 3,
  "similarity_threshold": 0.6
}
```

**Response:**
```json
{
  "query": "error message",
  "results": [
    {
      "error_id": "error_123",
      "error_title": "Error Title",
      "similarity_score": 0.85,
      "source_file": "file.pdf",
      "page": 5,
      "sections": {
        "description": "...",
        "symptoms": "...",
        "resolution": "...",
        "code": "...",
        "benefits": "..."
      }
    }
  ],
  "metadata": {
    "num_results": 3,
    "search_time_ms": 12.5,
    "top_k": 10,
    "top_n": 3,
    "similarity_threshold": 0.6
  }
}
```

### `GET /health`

Health check endpoint. Returns service status even if index is not loaded.

**Response:**
```json
{
  "status": "healthy",
  "index_size": 109
}
```

Or if index not loaded:
```json
{
  "status": "unhealthy",
  "reason": "Index not loaded"
}
```

### `GET /ready`

Readiness check - ensures index is loaded. Returns 503 if index not ready, 200 when ready.

**Response (ready):**
```json
{
  "status": "ready",
  "index_size": 109
}
```

**Response (not ready):**
- HTTP 503 with error detail

### `POST /rag/reload`

Reload the index from PostgreSQL without restarting the service.

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection URL (required)
- `EMBEDDINGS_LLM_URL` - URL of the embedding service (default: `http://alm-embedding:8080`)
- `RAG_MODEL_NAME` - Name of the embedding model (default: `nomic-ai/nomic-embed-text-v1.5`)
- `PORT` - Service port (default: `8002`)

## Startup Behavior

The service uses a **background task** to load the index, allowing it to start even if embeddings aren't available yet:

1. **Service starts** → FastAPI application becomes available
2. **Background task starts** → Begins polling PostgreSQL every 5 seconds
3. **If embeddings found** → Loads index, service becomes ready
4. **If embeddings not found** → Continues polling (up to 10 minutes)
5. **Service state**:
   - `/health` always returns 200 (service is running)
   - `/ready` returns 503 until index loaded, then 200

This design allows the RAG service to start independently of the init job, eliminating circular dependencies.

## Deployment

The service is deployed as a Kubernetes deployment via Helm chart.

**Prerequisites:**
- PostgreSQL with `pgvector` extension enabled
- `ragembedding` table (created automatically by init job)
- Embeddings populated in database (via init job)

**Startup Sequence:**
1. RAG service pod starts
2. Waits for PostgreSQL (initContainer)
3. Service starts, begins background polling
4. When embeddings available, loads index automatically
5. Service becomes ready for queries

## Dependencies

- **PostgreSQL** with `ragembedding` table populated (via init job)
- **pgvector extension** - for vector storage and queries
- **Embedding service (TEI)** - for generating query embeddings
- **FAISS** - for in-memory similarity search
- **FastAPI** - web framework
- **asyncpg** - async PostgreSQL driver

