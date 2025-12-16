"""
RAG Service - FastAPI service for RAG queries.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from index_loader import RAGIndexLoader
import time

app = FastAPI(title="RAG Service", version="0.1.0")

# Global index loader (initialized on startup)
index_loader: Optional[RAGIndexLoader] = None


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


async def load_index_background():
    """Background task to load index from PostgreSQL (polls until available)."""
    global index_loader

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable is required")
        return

    model_name = os.getenv("RAG_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")

    print("Initializing RAG index loader...")
    index_loader = RAGIndexLoader(database_url=database_url, model_name=model_name)

    # Wait for embeddings to be available (poll PostgreSQL)
    # This allows the service to start before the init job completes
    max_wait_time = 600  # 10 minutes
    wait_interval = 5  # Check every 5 seconds
    elapsed = 0

    print("Waiting for embeddings to be available in PostgreSQL...")
    while elapsed < max_wait_time:
        try:
            await index_loader.load_index()
            print("✓ RAG index loaded successfully")
            return
        except ValueError as e:
            if "No embeddings found" in str(e):
                if elapsed == 0 or elapsed % 30 == 0:  # Print every 30 seconds
                    print(
                        f"Embeddings not yet available (waited {elapsed}s), retrying in {wait_interval}s..."
                    )
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
            else:
                print(f"✗ Failed to load RAG index: {e}")
                return  # Don't raise, just return - service will stay in "not ready" state
        except Exception as e:
            # Check if this is a "table doesn't exist" error - continue polling
            error_str = str(e).lower()
            if (
                "does not exist" in error_str
                or "undefinedtable" in error_str
                or "relation" in error_str
            ):
                # Table doesn't exist yet - init job is still creating it
                if elapsed == 0 or elapsed % 30 == 0:  # Print every 30 seconds
                    print(
                        f"Table not yet created (waited {elapsed}s), retrying in {wait_interval}s..."
                    )
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
            else:
                # Some other error - log and continue polling (might be transient)
                if elapsed == 0 or elapsed % 30 == 0:  # Print every 30 seconds
                    print(f"Error loading index (waited {elapsed}s): {e}")
                    print(f"  Retrying in {wait_interval}s...")
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval

    # If we get here, we've timed out
    print(f"⚠ WARNING: Failed to load RAG index after {max_wait_time} seconds")
    print("  Service will remain in 'not ready' state until embeddings are available")


@app.on_event("startup")
async def startup_event():
    """Start background task to load index."""
    # Start background task - don't block startup
    asyncio.create_task(load_index_background())


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

    try:
        # Step 1: Generate query embedding
        # For now, we'll need to call the embedding service
        # This should be the same TEI service used during indexing
        embedding_url = os.getenv("EMBEDDINGS_LLM_URL", "http://alm-embedding:8080")

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Prepare query text with task prefix (for nomic models)
            query_text = f"search_query: {request.query}"

            # Call embedding service
            embedding_response = await client.post(
                f"{embedding_url}/embeddings",
                json={
                    "input": [query_text],
                    "model": "nomic-embed-text-v1.5",
                },
            )
            embedding_response.raise_for_status()

            # Extract embedding
            embedding_data = embedding_response.json()
            if "data" in embedding_data and len(embedding_data["data"]) > 0:
                query_embedding = np.array(
                    embedding_data["data"][0]["embedding"], dtype=np.float32
                )
            elif (
                "embeddings" in embedding_data and len(embedding_data["embeddings"]) > 0
            ):
                query_embedding = np.array(
                    embedding_data["embeddings"][0], dtype=np.float32
                )
            else:
                raise ValueError("Unexpected embedding response format")

        # Normalize embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Step 2: Similarity search in FAISS
        query_vector = query_embedding.reshape(1, -1)
        similarities, indices = index_loader.index.search(query_vector, request.top_k)

        # Flatten results
        similarities = similarities[0]
        indices = indices[0]

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
        results = results[: request.top_n]

        search_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            query=request.query,
            results=results,
            metadata={
                "num_results": len(results),
                "search_time_ms": search_time_ms,
                "top_k": request.top_k,
                "top_n": request.top_n,
                "similarity_threshold": request.similarity_threshold,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/rag/reload")
async def reload_index():
    """
    Reload the index from PostgreSQL.

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
