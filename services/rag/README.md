# RAG Service

FastAPI microservice for RAG (Retrieval-Augmented Generation) queries.

## Overview

The RAG service provides similarity search over the knowledge base embeddings stored in MinIO. It:

1. **Waits for index** (initContainer) - ensures index is ready before main container starts
2. **Loads once at startup** - loads FAISS index and metadata from MinIO using temp files
3. **Exposes REST API** - provides query endpoints for knowledge base retrieval

### Key Features

- **Deterministic startup**: InitContainer ensures index is ready before service starts
- **Fast loading**: Loads pre-built FAISS index from MinIO (no rebuilding needed)
- **No polling**: One-time load at startup, no background polling
- **MinIO-based**: Artifacts stored in MinIO (index.faiss, metadata.pkl, LATEST.json)

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

Reload the index from MinIO without restarting the service.

## Environment Variables

- `MINIO_ENDPOINT` - MinIO endpoint (required)
- `MINIO_PORT` - MinIO port (default: `9000`)
- `MINIO_ACCESS_KEY` - MinIO access key (required)
- `MINIO_SECRET_KEY` - MinIO secret key (required)
- `RAG_BUCKET_NAME` - MinIO bucket name (default: `rag-index`)
- `EMBEDDINGS_LLM_URL` - URL of the embedding service (default: `http://alm-embedding:8080`)
- `RAG_MODEL_NAME` - Name of the embedding model (default: `nomic-ai/nomic-embed-text-v1.5`)
- `PORT` - Service port (default: `8002`)

## Startup Behavior

The service uses an **initContainer** to ensure the index is ready before the main container starts:

1. **InitContainer starts** → Waits for `LATEST.json` with `status: "READY"` in MinIO
2. **Main container starts** → Loads index from MinIO once at startup
3. **Service ready** → `/ready` endpoint returns 200
4. **Service state**:
   - `/health` returns service status
   - `/ready` returns 503 until index loaded, then 200

This design ensures deterministic startup - the service only starts when the index is ready.

## Deployment

The service is deployed as a Kubernetes deployment via Helm chart.

**Prerequisites:**
- MinIO/S3 accessible
- RAG index artifacts in MinIO bucket (created by init job):
  - `index.faiss` - FAISS index
  - `metadata.pkl` - Error metadata
  - `LATEST.json` - Status pointer (status: READY)

**Startup Sequence:**
1. RAG service pod starts
2. InitContainer waits for RAG index in MinIO (checks LATEST.json status=READY)
3. Main container starts and loads index from MinIO once
4. Service becomes ready for queries

## Dependencies

- **MinIO/S3** with RAG index artifacts (via init job)
  - `index.faiss` - FAISS index file
  - `metadata.pkl` - Error metadata
  - `LATEST.json` - Status pointer file
- **Embedding service (TEI)** - for generating query embeddings
- **FAISS** - for in-memory similarity search
- **FastAPI** - web framework
- **minio** - MinIO client library

