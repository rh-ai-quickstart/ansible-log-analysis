#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ansible Error RAG System - Embedding and Indexing Module

This module implements:
- Groups chunks by error_id
- Creates composite embeddings (description + symptoms)
- Builds FAISS index for similarity search
- Persists index and metadata to disk

Uses TEI (text-embeddings-inference) service for embeddings.
Model is hardcoded to nomic-ai/nomic-embed-text-v1.5.
Service URL defaults to http://alm-embedding:8080 (can be overridden via EMBEDDINGS_LLM_URL).
"""

import os
import pickle
import numpy as np
import requests
import json
import io
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
import faiss

from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """
    Embedding client for text-embeddings-inference (TEI) service.

    Uses OpenAI-compatible API format. TEI doesn't require authentication
    for internal cluster deployments.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        # Use hardcoded defaults from config
        self.model_name = model_name or config.embeddings.model_name
        self.api_url = api_url or config.embeddings.api_url

        if not self.api_url:
            raise ValueError(
                "api_url is required. "
                "Please configure EMBEDDINGS_LLM_URL as an environment variable or in your .env file."
            )

        self._init_api_client()

    def _init_api_client(self):
        """Initialize TEI embedding client."""
        logger.debug("Initializing TEI embedding client: %s", self.api_url)
        logger.debug("  Model: %s", self.model_name)

        # Determine embedding dimension based on model
        # nomic-embed-text-v1.5 has 768 dimensions
        if "nomic" in self.model_name.lower():
            self.embedding_dim = 768
        else:
            self.embedding_dim = 768  # Default for nomic models

        logger.debug("TEI client initialized")
        logger.debug("  Embedding dimension: %d", self.embedding_dim)

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings via TEI (text-embeddings-inference) API.

        Args:
            texts: List of texts to embed (may include task prefixes like "search_document:")
            normalize_embeddings: Whether to L2-normalize embeddings
            show_progress_bar: Unused (kept for API compatibility)

        Returns:
            Numpy array of embeddings
        """
        embeddings = self._encode_tei_api(texts)

        embeddings = np.array(embeddings)

        # Normalize if requested (TEI may normalize, but we handle it here for consistency)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Protect against division by zero - use np.maximum to ensure minimum norm of 1e-8
            # This prevents inf/nan values if TEI returns a zero vector (unlikely but possible)
            embeddings = embeddings / np.maximum(norms, 1e-8)

        return embeddings

    def _encode_tei_api(self, texts: List[str]) -> List[List[float]]:
        """
        Encode using text-embeddings-inference (OpenAI-compatible API).

        TEI supports task prefixes for nomic models:
        - search_document: for documents (already added in create_composite_embeddings)
        - search_query: for queries (added in query_pipeline)

        Texts passed here may already have prefixes, so we don't add them again.

        Batches requests to respect TEI's MAX_CLIENT_BATCH_SIZE limit (default: 16).
        """
        headers = {
            "Content-Type": "application/json",
        }

        # Ensure URL ends with /embeddings for OpenAI format
        url = self.api_url
        if not url.endswith("/embeddings"):
            url = url.rstrip("/") + "/embeddings"

        # TEI batch size limit (TEI MAX_CLIENT_BATCH_SIZE is 32, we use 30 to be safe)
        BATCH_SIZE = 30

        logger.debug("Calling TEI at: %s", url)
        logger.debug("  Model: %s", self.model_name)
        logger.debug(
            "  Total texts: %d (will be batched into chunks of %d)",
            len(texts),
            BATCH_SIZE,
        )

        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.debug(
                "  Processing batch %d/%d (%d texts)...",
                batch_num,
                total_batches,
                len(batch),
            )

            payload = {
                "model": self.model_name,
                "input": batch,
            }

            try:
                response = requests.post(
                    url, json=payload, headers=headers, timeout=120
                )

                if response.status_code != 200:
                    logger.error("Response status: %d", response.status_code)
                    logger.error("Response body: %s", response.text[:500])

                response.raise_for_status()

                result = response.json()
                # OpenAI format: {"data": [{"embedding": [...]}, ...]}
                if "data" in result:
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                # Alternative format: {"embeddings": [[...], ...]}
                elif "embeddings" in result:
                    batch_embeddings = result["embeddings"]
                else:
                    raise ValueError(f"Unexpected TEI response format: {result.keys()}")

                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    "  Batch %d completed (%d embeddings)",
                    batch_num,
                    len(batch_embeddings),
                )

            except Exception as e:
                logger.error("  Error in batch %d: %s", batch_num, e)
                raise

        logger.debug("All batches completed (%d total embeddings)", len(all_embeddings))
        return all_embeddings


class AnsibleErrorEmbedder:
    """
    Handles embedding generation and FAISS index creation for Ansible errors.

    Uses TEI (text-embeddings-inference) service for embeddings.
    Model is hardcoded to nomic-ai/nomic-embed-text-v1.5.
    Service URL defaults to http://alm-embedding:8080 (can be overridden via EMBEDDINGS_LLM_URL).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Model name (defaults to hardcoded nomic-ai/nomic-embed-text-v1.5)
            api_url: API endpoint URL (defaults to config, which defaults to http://alm-embedding:8080)
            index_path: Path to save FAISS index (defaults to config)
            metadata_path: Path to save metadata (defaults to config)
        """
        # Use config values as defaults (model is hardcoded in config)
        self.model_name = model_name or config.embeddings.model_name
        self.api_url = api_url or config.embeddings.api_url
        self.index_path = index_path or config.storage.index_path
        self.metadata_path = metadata_path or config.storage.metadata_path

        # Validate configuration
        if not self.api_url:
            raise ValueError(
                "API URL is required. Please configure EMBEDDINGS_LLM_URL as an environment variable or in your .env file."
            )

        # Initialize embedding client (no API key needed for TEI)
        self.client = EmbeddingClient(model_name=self.model_name, api_url=self.api_url)
        self.embedding_dim = self.client.embedding_dim

        self.index = None
        self.error_store = {}
        self.index_to_error_id = {}
        self._embeddings_array = None  # Store embeddings for PostgreSQL saving

        logger.debug("Embedder initialized")
        logger.debug("  Mode: TEI Service")

    def group_chunks_by_error(
        self, chunks: List[Document]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group chunks by error_id and organize into structured format.

        Args:
            chunks: List of Document chunks from parser

        Returns:
            Dictionary mapping error_id to complete error data
        """
        logger.debug("=" * 60)
        logger.debug("STEP:INGESTION - Grouping chunks by error_id")
        logger.debug("=" * 60)

        errors_by_id = defaultdict(
            lambda: {
                "error_id": None,
                "error_title": None,
                "sections": {},
                "metadata": {},
            }
        )

        # Track statistics per file
        file_stats = defaultdict(
            lambda: {"errors": set(), "sections": defaultdict(int)}
        )

        for chunk in chunks:
            error_id = chunk.metadata.get("error_id")
            section_type = chunk.metadata.get("section_type")
            source_file = chunk.metadata.get("source_file", "unknown")

            if not error_id or not section_type:
                continue

            # Track per-file statistics
            file_stats[source_file]["errors"].add(error_id)
            file_stats[source_file]["sections"][section_type] += 1

            # Initialize error entry
            if errors_by_id[error_id]["error_id"] is None:
                errors_by_id[error_id]["error_id"] = error_id
                errors_by_id[error_id]["error_title"] = chunk.metadata.get(
                    "error_title"
                )
                errors_by_id[error_id]["metadata"] = {
                    "source_file": chunk.metadata.get("source_file"),
                    "page": chunk.metadata.get("page"),
                }

            # Extract content (remove the header added by chunking)
            content = chunk.page_content
            # Remove "Error: X\n\nSection: Y\n\n" prefix
            lines = content.split("\n\n", 2)
            if len(lines) >= 3:
                content = lines[2]
            else:
                content = lines[-1]

            errors_by_id[error_id]["sections"][section_type] = content

        logger.debug(
            "Grouped %d chunks into %d unique errors", len(chunks), len(errors_by_id)
        )

        # Print per-file statistics
        logger.debug("-" * 60)
        logger.debug("Section distribution per file:")
        logger.debug("-" * 60)
        for source_file in sorted(file_stats.keys()):
            stats = file_stats[source_file]
            num_errors = len(stats["errors"])
            logger.debug("%s:", Path(source_file).name)
            logger.debug("   Total errors: %d", num_errors)
            logger.debug("   Sections:")
            for section, count in sorted(stats["sections"].items()):
                logger.debug("     %s: %d errors", section, count)

        # Overall statistics
        section_counts = defaultdict(int)
        for error in errors_by_id.values():
            for section in error["sections"].keys():
                section_counts[section] += 1

        logger.debug("-" * 60)
        logger.debug("Overall section distribution:")
        logger.debug("-" * 60)
        for section, count in sorted(section_counts.items()):
            logger.debug("  %s: %d errors", section, count)

        return dict(errors_by_id)

    def create_composite_embeddings(
        self, error_store: Dict[str, Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create composite embeddings from description + symptoms for each error.

        Args:
            error_store: Dictionary of errors grouped by error_id

        Returns:
            Tuple of (embedding_matrix, error_ids)
        """
        logger.debug("=" * 60)
        logger.debug("GENERATING COMPOSITE EMBEDDINGS")
        logger.debug("=" * 60)

        composite_texts = []
        error_ids = []
        skipped = 0

        # Determine if we should use task prefixes (for Nomic models)
        use_task_prefix = "nomic" in self.model_name.lower()

        for error_id, error_data in error_store.items():
            sections = error_data["sections"]

            # Extract description and symptoms
            description = sections.get("description", "").strip()
            symptoms = sections.get("symptoms", "").strip()

            # Skip errors without description or symptoms
            if not description and not symptoms:
                logger.warning(
                    "Skipping error %s: No description or symptoms",
                    error_data["error_title"],
                )
                skipped += 1
                continue

            # Create composite text
            composite_parts = []
            if description:
                composite_parts.append(description)
            if symptoms:
                composite_parts.append(symptoms)

            composite_text = "\n\n".join(composite_parts)

            # Add task prefix for Nomic models
            if use_task_prefix:
                prefixed_text = f"search_document: {composite_text}"
            else:
                prefixed_text = composite_text

            # Store composite text in error_store for reference (without prefix)
            error_data["composite_text"] = composite_text

            composite_texts.append(prefixed_text)
            error_ids.append(error_id)

        logger.debug("Created %d composite texts", len(composite_texts))
        if skipped > 0:
            logger.warning(
                "Skipped %d errors (missing description and symptoms)", skipped
            )

        if use_task_prefix:
            logger.debug("Using task prefix: 'search_document:'")

        # Generate embeddings
        logger.debug("Generating embeddings using %s...", self.model_name)

        embeddings = self.client.encode(
            composite_texts, normalize_embeddings=True, show_progress_bar=True
        )

        logger.debug("Generated embeddings: shape=%s", embeddings.shape)

        return embeddings, error_ids

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        error_ids: List[str],
        error_store: Dict[str, Dict[str, Any]],
    ):
        """Build FAISS index from embeddings."""
        logger.debug("=" * 60)
        logger.debug("STEP:CREATING FAISS INDEX")
        logger.debug("=" * 60)

        # Store embeddings array for PostgreSQL saving
        self._embeddings_array = embeddings.copy()

        # Verify embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        logger.debug(
            "Embedding norms: min=%.4f, max=%.4f, mean=%.4f",
            norms.min(),
            norms.max(),
            norms.mean(),
        )

        # Create FAISS index
        logger.debug(
            "Building FAISS IndexFlatIP with dimension %d...", self.embedding_dim
        )
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Add vectors to index
        self.index.add(embeddings)

        logger.debug("Index created with %d vectors", self.index.ntotal)

        # Create mapping from index position to error_id
        self.index_to_error_id = {i: error_id for i, error_id in enumerate(error_ids)}

        # Store only errors that have embeddings
        self.error_store = {error_id: error_store[error_id] for error_id in error_ids}

        logger.debug("Stored metadata for %d errors", len(self.error_store))

    def save_index(self):
        """
        Persist FAISS index and metadata to disk.

        DEPRECATED: This method is kept for backward compatibility with tests.
        Production code should use save_to_minio() instead.
        """
        logger.debug("=" * 60)
        logger.debug("SAVING INDEX AND METADATA")
        logger.debug("=" * 60)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024)
        logger.debug("FAISS index saved to: %s", self.index_path)
        logger.debug("  Index size: %.2f MB", index_size_mb)

        # Save metadata
        metadata = {
            "error_store": self.error_store,
            "index_to_error_id": self.index_to_error_id,
            "model_name": self.model_name,
            "api_url": self.api_url,
            "embedding_dim": self.embedding_dim,
            "total_errors": len(self.error_store),
        }

        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        metadata_size_mb = os.path.getsize(self.metadata_path) / (1024 * 1024)
        logger.debug("Metadata saved to: %s", self.metadata_path)
        logger.debug("  Metadata size: %.2f MB", metadata_size_mb)
        logger.debug("  Total storage: %.2f MB", index_size_mb + metadata_size_mb)

    def load_index(self):
        """
        Load FAISS index and metadata from disk.

        DEPRECATED: This method is kept for backward compatibility with tests.
        Production code should use RAGIndexLoader (from MinIO) instead.
        """
        logger.debug("=" * 60)
        logger.debug("LOADING INDEX AND METADATA")
        logger.debug("=" * 60)

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}")

        self.index = faiss.read_index(self.index_path)
        logger.debug("FAISS index loaded: %d vectors", self.index.ntotal)

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.error_store = metadata["error_store"]
        self.index_to_error_id = metadata["index_to_error_id"]

        logger.debug("Metadata loaded: %d errors", len(self.error_store))
        logger.debug("  Model: %s", metadata["model_name"])

        if metadata["model_name"] != self.model_name:
            logger.warning("Model mismatch!")
            logger.warning("  Index: %s", metadata["model_name"])
            logger.warning("  Current: %s", self.model_name)

    def ingest_and_index(self, chunks: List[Document]):
        """
        Complete ingestion and indexing pipeline.

        DEPRECATED: This method is kept for backward compatibility with tests.
        Production code should use ingest_and_index_to_minio() instead.
        """
        logger.info("=" * 70)
        logger.info("ANSIBLE ERROR RAG SYSTEM - INGESTION AND INDEXING")
        logger.info("=" * 70)

        error_store = self.group_chunks_by_error(chunks)
        embeddings, error_ids = self.create_composite_embeddings(error_store)
        self.build_faiss_index(embeddings, error_ids, error_store)
        self.save_index()

        logger.info("=" * 70)
        logger.info("INGESTION AND INDEXING COMPLETE")
        logger.info("=" * 70)

    async def ingest_and_index_to_minio(self, chunks: List[Document]):
        """
        Complete ingestion and indexing pipeline, saving to MinIO.

        This is the async version that saves to MinIO instead of disk or PostgreSQL.
        """
        logger.info("=" * 70)
        logger.info("ANSIBLE ERROR RAG SYSTEM - INGESTION AND INDEXING (MinIO)")
        logger.info("=" * 70)

        error_store = self.group_chunks_by_error(chunks)
        embeddings, error_ids = self.create_composite_embeddings(error_store)
        self.build_faiss_index(embeddings, error_ids, error_store)
        await self.save_to_minio()

        logger.info("=" * 70)
        logger.info("INGESTION AND INDEXING COMPLETE (MinIO)")
        logger.info("=" * 70)

    async def save_to_minio(
        self,
        bucket_name: str = "rag-index",
        minio_endpoint: str = None,
        minio_port: str = None,
        minio_access_key: str = None,
        minio_secret_key: str = None,
    ):
        """
        Save FAISS index and metadata to MinIO with status tracking via LATEST.json pointer file.

        Process:
        1. Set status=BUILDING in LATEST.json
        2. Upload index.faiss and metadata.pkl to fixed paths
        3. Set status=READY in LATEST.json
        4. If any step fails, set status=FAILED with error_message

        Args:
            bucket_name: MinIO bucket name (default: "rag-index")
            minio_endpoint: MinIO endpoint (from env if not provided)
            minio_port: MinIO port (from env if not provided)
            minio_access_key: MinIO access key (from env if not provided)
            minio_secret_key: MinIO secret key (from env if not provided)
        """
        if self.index is None:
            raise ValueError("FAISS index must be built before saving to MinIO")

        if not self.error_store:
            raise ValueError("Error store must be populated before saving to MinIO")

        logger.info("=" * 60)
        logger.info("SAVING RAG INDEX TO MINIO")
        logger.info("=" * 60)

        from utils.minio import get_minio_client, ensure_bucket_exists

        minio_client = get_minio_client(
            minio_endpoint, minio_port, minio_access_key, minio_secret_key
        )

        # Ensure bucket exists
        if ensure_bucket_exists(minio_client, bucket_name):
            logger.info(f"Created MinIO bucket: {bucket_name}")

        # Step 1: Set BUILDING status
        pointer = {
            "status": "BUILDING",
            "error_message": None,
        }
        pointer_json = json.dumps(pointer)
        minio_client.put_object(
            bucket_name,
            "LATEST.json",
            io.BytesIO(pointer_json.encode()),
            length=len(pointer_json),
        )
        logger.info("Set status: BUILDING")

        try:
            # Step 2: Save FAISS index to temp file first
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".faiss"
            ) as tmp_index:
                faiss.write_index(self.index, tmp_index.name)
                tmp_index_path = tmp_index.name

            try:
                # Upload FAISS index to fixed path
                minio_client.fput_object(bucket_name, "index.faiss", tmp_index_path)
                index_size_mb = os.path.getsize(tmp_index_path) / (1024 * 1024)
                logger.info(
                    f"Uploaded FAISS index: index.faiss ({index_size_mb:.2f} MB)"
                )

                # Save metadata to temp file
                metadata = {
                    "error_store": self.error_store,
                    "index_to_error_id": self.index_to_error_id,
                    "model_name": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "total_errors": len(self.error_store),
                }

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pkl"
                ) as tmp_metadata:
                    pickle.dump(metadata, tmp_metadata)
                    tmp_metadata_path = tmp_metadata.name

                try:
                    # Upload metadata to fixed path
                    minio_client.fput_object(
                        bucket_name, "metadata.pkl", tmp_metadata_path
                    )
                    metadata_size_mb = os.path.getsize(tmp_metadata_path) / (
                        1024 * 1024
                    )
                    logger.info(
                        f"Uploaded metadata: metadata.pkl ({metadata_size_mb:.2f} MB)"
                    )

                    # Step 3: Set READY status
                    pointer = {
                        "status": "READY",
                        "error_message": None,
                        "total_errors": len(self.error_store),
                        "model_name": self.model_name,
                        "embedding_dim": self.embedding_dim,
                    }
                    pointer_json = json.dumps(pointer)
                    minio_client.put_object(
                        bucket_name,
                        "LATEST.json",
                        io.BytesIO(pointer_json.encode()),
                        length=len(pointer_json),
                    )

                    logger.info("✓ RAG index saved to MinIO")
                    logger.info("  Status: READY")
                    logger.info(f"  Total errors: {len(self.error_store)}")
                    logger.info(
                        f"  Total storage: {index_size_mb + metadata_size_mb:.2f} MB"
                    )

                finally:
                    # Cleanup temp metadata file
                    if os.path.exists(tmp_metadata_path):
                        os.unlink(tmp_metadata_path)

            finally:
                # Cleanup temp index file
                if os.path.exists(tmp_index_path):
                    os.unlink(tmp_index_path)

        except Exception as e:
            # Step 4: Set FAILED status on error
            error_msg = str(e)[:500]  # Limit error message length
            logger.error(f"Failed to save RAG index to MinIO: {e}")

            pointer = {
                "status": "FAILED",
                "error_message": error_msg,
            }
            pointer_json = json.dumps(pointer)
            minio_client.put_object(
                bucket_name,
                "LATEST.json",
                io.BytesIO(pointer_json.encode()),
                length=len(pointer_json),
            )

            logger.error(f"Set status: FAILED (error: {error_msg})")
            raise


# main() function removed - use rag_init_pipeline.py for production index building
# This file is now a library module, not a standalone script
