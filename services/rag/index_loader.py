"""
Load RAG index from MinIO (FAISS index and metadata).
"""

import os
import json
import tempfile
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import faiss
from minio import Minio


class RAGIndexLoader:
    """
    Loads FAISS index and metadata from MinIO.
    Uses temp files for FAISS compatibility (FAISS prefers file paths).
    """

    def __init__(
        self,
        minio_endpoint: str = None,
        minio_port: str = None,
        minio_access_key: str = None,
        minio_secret_key: str = None,
        bucket_name: str = "rag-index",
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    ):
        """
        Initialize the index loader.

        Args:
            minio_endpoint: MinIO endpoint (from env if not provided)
            minio_port: MinIO port (from env if not provided)
            minio_access_key: MinIO access key (from env if not provided)
            minio_secret_key: MinIO secret key (from env if not provided)
            bucket_name: MinIO bucket name (default: "rag-index")
            model_name: Name of the embedding model (for validation)
        """
        # Get MinIO config from environment
        endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT")
        port = minio_port or os.getenv("MINIO_PORT", "9000")
        access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY")
        secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY")

        if not all([endpoint, port, access_key, secret_key]):
            raise ValueError(
                "Missing required MinIO environment variables: "
                "MINIO_ENDPOINT, MINIO_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
            )

        self.minio_client = Minio(
            endpoint=f"{endpoint}:{port}",
            access_key=access_key,
            secret_key=secret_key,
            secure=False,  # Use HTTP for internal OpenShift services
        )
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.embedding_dim = 768  # nomic-embed-text-v1.5 dimension

        self.index: Optional[faiss.Index] = None
        self.error_store: Dict[str, Dict[str, Any]] = {}
        self.index_to_error_id: Dict[int, str] = {}
        self._loaded = False

    def check_index_ready(self) -> bool:
        """
        Check if index is ready by reading LATEST.json pointer file.

        Returns:
            True if status is READY, False otherwise
        """
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                return False

            response = self.minio_client.get_object(self.bucket_name, "LATEST.json")
            pointer = json.loads(response.read().decode())
            return pointer.get("status") == "READY"
        except Exception:
            return False

    async def load_index(
        self,
    ) -> Tuple[faiss.Index, Dict[str, Dict[str, Any]], Dict[int, str]]:
        """
        Load FAISS index and metadata from MinIO.

        Uses temp files for FAISS compatibility (FAISS prefers file paths).

        Returns:
            Tuple of (FAISS index, error_store, index_to_error_id mapping)

        Raises:
            ValueError: If index is not ready or not found
        """
        if self._loaded and self.index is not None:
            return self.index, self.error_store, self.index_to_error_id

        print("Loading RAG index from MinIO...")

        # Check if bucket exists
        if not self.minio_client.bucket_exists(self.bucket_name):
            raise ValueError(
                f"MinIO bucket '{self.bucket_name}' does not exist. "
                "Run init job first to create the index."
            )

        # Check status from LATEST.json
        try:
            response = self.minio_client.get_object(self.bucket_name, "LATEST.json")
            pointer = json.loads(response.read().decode())
            status = pointer.get("status")

            if status == "FAILED":
                error_msg = pointer.get("error_message", "Unknown error")
                raise ValueError(
                    f"RAG index build failed: {error_msg}. "
                    "Run init job again to rebuild the index."
                )

            if status != "READY":
                raise ValueError(
                    f"RAG index is not ready (status: {status}). "
                    "Wait for init job to complete or run init job first."
                )

            # Validate model name if present
            if "model_name" in pointer:
                model_name_db = pointer["model_name"]
                if model_name_db != self.model_name:
                    print(
                        f"Warning: Model mismatch. Index has {model_name_db}, "
                        f"expected {self.model_name}"
                    )

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(
                f"Could not read LATEST.json from MinIO: {e}. "
                "Run init job first to create the index."
            )

        # Create temp directory for artifacts
        temp_dir = Path(tempfile.mkdtemp(prefix="rag-index-"))

        try:
            index_path = temp_dir / "index.faiss"
            metadata_path = temp_dir / "metadata.pkl"

            # Download FAISS index to temp file
            try:
                self.minio_client.fget_object(
                    self.bucket_name, "index.faiss", str(index_path)
                )
                print("Downloaded FAISS index to temp file")
            except Exception as e:
                raise ValueError(
                    f"Could not download index.faiss from MinIO: {e}. "
                    "Run init job first to create the index."
                )

            # Load FAISS index from file (FAISS prefers file paths)
            try:
                self.index = faiss.read_index(str(index_path))
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                raise ValueError(f"Could not load FAISS index: {e}")

            # Download metadata to temp file
            try:
                self.minio_client.fget_object(
                    self.bucket_name, "metadata.pkl", str(metadata_path)
                )
                print("Downloaded metadata to temp file")
            except Exception as e:
                raise ValueError(
                    f"Could not download metadata.pkl from MinIO: {e}. "
                    "Run init job first to create the index."
                )

            # Load metadata from file
            try:
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Could not load metadata: {e}")

            self.error_store = metadata["error_store"]
            self.index_to_error_id = metadata["index_to_error_id"]

            # Validate model name from metadata
            if "model_name" in metadata:
                model_name_meta = metadata["model_name"]
                if model_name_meta != self.model_name:
                    print(
                        f"Warning: Model mismatch in metadata. "
                        f"Metadata has {model_name_meta}, expected {self.model_name}"
                    )

            self._loaded = True

            print("✓ RAG index loaded successfully")
            print(f"  Total errors: {len(self.error_store)}")
            print(f"  Model: {metadata.get('model_name', 'unknown')}")

            return self.index, self.error_store, self.index_to_error_id

        finally:
            # Cleanup temp directory
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def reload_index(self):
        """Force reload of index from MinIO."""
        self._loaded = False
        self.index = None
        self.error_store = {}
        self.index_to_error_id = {}
        return await self.load_index()
