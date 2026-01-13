"""
RAG Service - FastAPI service for RAG queries.
"""

import os
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from index_loader import RAGIndexLoader
import time
import logging
import httpx
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Service", version="0.1.0")

# Global index loader (initialized on startup)
index_loader: Optional[RAGIndexLoader] = None

# Global HTTP client for embedding service (with connection pooling)
embedding_client: Optional[httpx.AsyncClient] = None


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str = Field(description="Query text to search for")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of top candidates to retrieve"
    )
    top_n: int = Field(
        default=3, ge=1, le=20, description="Number of final results to return"
    )
    similarity_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum similarity threshold (0-1)"
    )


class ErrorSection(BaseModel):
    """Error section data."""

    description: Optional[str] = None
    symptoms: Optional[str] = None
    resolution: Optional[str] = None
    code: Optional[str] = None
    benefits: Optional[str] = None


class ErrorResult(BaseModel):
    """Single error result."""

    error_id: str
    error_title: str
    similarity_score: float
    source_file: Optional[str] = None
    page: Optional[int] = None
    sections: ErrorSection


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    query: str
    results: List[ErrorResult]
    metadata: Dict[str, Any]


async def load_index():
    """Load index from MinIO. Returns True if successful, False otherwise."""
    global index_loader

    model_name = os.getenv("RAG_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
    bucket_name = os.getenv("RAG_BUCKET_NAME", "rag-index")

    # Initialize index loader if not already done
    if index_loader is None:
        print("Initializing RAG index loader...")
        index_loader = RAGIndexLoader(
            bucket_name=bucket_name,
            model_name=model_name,
        )

    try:
        await index_loader.load_index()
        print("✓ RAG index loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load RAG index: {e}")
        return False


async def poll_for_index():
    """Background task that polls for RAG index every 20 seconds."""
    global index_loader

    poll_interval = 20  # seconds

    while True:
        try:
            # Check if index is already loaded
            if index_loader is not None and index_loader.index is not None:
                # Index is loaded, check periodically if it needs reloading
                await asyncio.sleep(poll_interval)
                continue

            # Try to load the index
            logger.info("Polling for RAG index...")
            success = await load_index()

            if success:
                logger.info("RAG index loaded successfully via polling")
            else:
                logger.debug(
                    "RAG index not available yet, will retry in %d seconds",
                    poll_interval,
                )

        except Exception as e:
            logger.error("Error during index polling: %s", e, exc_info=True)

        # Wait before next poll
        await asyncio.sleep(poll_interval)


@app.on_event("startup")
async def startup_event():
    """Initialize service and start polling for index."""
    global embedding_client

    # Try to load index, but don't fail if it doesn't exist
    index_loaded = await load_index()
    if not index_loaded:
        logger.info(
            "RAG index not found at startup. Will poll for index in background..."
        )

    # Always start background polling task (it will sleep if index is already loaded)
    asyncio.create_task(poll_for_index())
    logger.info("Started background polling task for RAG index (every 20 seconds)")

    # Initialize persistent HTTP client for embedding service with connection pooling
    embedding_url = os.getenv("EMBEDDINGS_LLM_URL", "http://alm-embedding:8080")
    embedding_client = httpx.AsyncClient(
        base_url=embedding_url,
        timeout=30.0,
        limits=httpx.Limits(
            max_keepalive_connections=20,  # Keep up to 20 connections alive
            max_connections=100,  # Maximum total connections
            keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
        ),
    )
    logger.info("Initialized embedding service HTTP client with connection pooling")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global embedding_client

    if embedding_client is not None:
        await embedding_client.aclose()
        logger.info("Closed embedding service HTTP client")
        embedding_client = None


@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes probes."""
    if index_loader is None or index_loader.index is None:
        return {"status": "unhealthy", "reason": "Index not loaded"}
    return {
        "status": "healthy",
        "index_size": index_loader.index.ntotal if index_loader.index else 0,
    }


@app.get("/ready")
def readiness_check():
    """Readiness check - ensures index is loaded."""
    if index_loader is None or index_loader.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {"status": "ready", "index_size": index_loader.index.ntotal}


@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system for relevant error solutions.

    This endpoint:
    1. Generates embedding for the query
    2. Performs similarity search using FAISS
    3. Returns top-N most relevant errors
    """
    if index_loader is None or index_loader.index is None:
        raise HTTPException(
            status_code=503, detail="RAG index not loaded. Service is not ready."
        )

    start_time = time.time()

    # Log query start
    logger.info("=" * 70)
    logger.info("QUERYING RAG SYSTEM")
    logger.info("=" * 70)
    query_preview = (
        request.query[:100] + "..." if len(request.query) > 100 else request.query
    )
    logger.info("Query: %s", query_preview)
    logger.info(
        "Parameters: top_k=%d, top_n=%d, threshold=%.2f",
        request.top_k,
        request.top_n,
        request.similarity_threshold,
    )

    try:
        # Step 1: Generate query embedding
        # Use persistent HTTP client with connection pooling
        if embedding_client is None:
            raise HTTPException(
                status_code=503, detail="Embedding service client not initialized"
            )

        logger.info("Calling TEI at: %s/embeddings", embedding_client.base_url)
        logger.info("  Model: nomic-ai/nomic-embed-text-v1.5")
        logger.info("  Total texts: 1 (will be batched into chunks of 30)")

        # Prepare query text with task prefix (for nomic models)
        query_text = f"search_query: {request.query}"

        logger.info("  Processing batch 1/1 (1 texts)...")

        # Call embedding service using persistent client
        embedding_response = await embedding_client.post(
            "/embeddings",
            json={
                "input": [query_text],
                "model": "nomic-embed-text-v1.5",
            },
        )
        embedding_response.raise_for_status()

        logger.info("  Batch 1 completed (1 embeddings)")
        logger.info("All batches completed (1 total embeddings)")

        # Extract embedding
        embedding_data = embedding_response.json()
        if "data" in embedding_data and len(embedding_data["data"]) > 0:
            query_embedding = np.array(
                embedding_data["data"][0]["embedding"], dtype=np.float32
            )
        elif "embeddings" in embedding_data and len(embedding_data["embeddings"]) > 0:
            query_embedding = np.array(
                embedding_data["embeddings"][0], dtype=np.float32
            )
        else:
            raise ValueError("Unexpected embedding response format")

        # Normalize embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        logger.info("Generated embeddings: shape=(1, %d)", len(query_embedding))

        # Step 2: Similarity search in FAISS
        logger.info("Performing FAISS similarity search...")
        query_vector = query_embedding.reshape(1, -1)
        similarities, indices = index_loader.index.search(query_vector, request.top_k)

        # Flatten results
        similarities = similarities[0]
        indices = indices[0]

        # Count candidates (excluding -1 indices)
        num_candidates = len([idx for idx in indices if idx != -1])

        # Step 3: Filter by threshold and format results
        results = []
        for idx, similarity in zip(indices, similarities):
            if idx == -1:  # FAISS returns -1 when not enough results
                continue

            if similarity < request.similarity_threshold:
                continue

            error_id = index_loader.index_to_error_id[idx]
            error_data = index_loader.error_store[error_id]

            # Extract sections
            sections = error_data.get("sections", {})
            metadata = error_data.get("metadata", {})

            result = ErrorResult(
                error_id=error_id,
                error_title=error_data.get("error_title", error_id),
                similarity_score=float(similarity),
                source_file=metadata.get("source_file"),
                page=metadata.get("page"),
                sections=ErrorSection(
                    description=sections.get("description"),
                    symptoms=sections.get("symptoms"),
                    resolution=sections.get("resolution"),
                    code=sections.get("code"),
                    benefits=sections.get("benefits"),
                ),
            )
            results.append(result)

        # Step 4: Take top-N results
        num_filtered = len(results)
        results = results[: request.top_n]
        num_returned = len(results)

        search_time_ms = (time.time() - start_time) * 1000

        # Log query completion
        logger.info("Query complete in %.2fms", search_time_ms)
        logger.info("  Retrieved: %d candidates", num_candidates)
        logger.info("  Filtered: %d above threshold", num_filtered)
        logger.info("  Returned: %d results", num_returned)

        return QueryResponse(
            query=request.query,
            results=results,
            metadata={
                "num_results": num_returned,
                "search_time_ms": search_time_ms,
                "top_k": request.top_k,
                "top_n": request.top_n,
                "similarity_threshold": request.similarity_threshold,
            },
        )

    except Exception as e:
        logger.error("Error processing query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/rag/reload")
async def reload_index():
    """
    Reload the index from MinIO.

    Useful for updating the index without restarting the service.
    """
    if index_loader is None:
        raise HTTPException(status_code=503, detail="Index loader not initialized")

    try:
        await index_loader.reload_index()
        return {
            "status": "success",
            "message": "Index reloaded",
            "index_size": index_loader.index.ntotal if index_loader.index else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading index: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
