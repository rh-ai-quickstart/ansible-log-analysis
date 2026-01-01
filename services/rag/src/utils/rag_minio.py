"""
RAG-specific MinIO utilities for checking RAG index status.
"""

import json
from typing import Optional, Dict, Any

from utils.minio import get_minio_client


def get_rag_index_status(
    bucket_name: str = "rag-index",
    minio_endpoint: Optional[str] = None,
    minio_port: Optional[str] = None,
    minio_access_key: Optional[str] = None,
    minio_secret_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the status of the RAG index from MinIO LATEST.json file.

    Configuration priority: function params > env vars > defaults (see get_minio_client).

    Args:
        bucket_name: MinIO bucket name (default: "rag-index")
        minio_endpoint: MinIO endpoint (overrides env/default)
        minio_port: MinIO port (overrides env/default)
        minio_access_key: MinIO access key (overrides env/default)
        minio_secret_key: MinIO secret key (overrides env/default)

    Returns:
        Dictionary with status information if LATEST.json exists, None otherwise
    """
    try:
        minio_client = get_minio_client(
            minio_endpoint, minio_port, minio_access_key, minio_secret_key
        )

        if not minio_client.bucket_exists(bucket_name):
            return None

        try:
            response = minio_client.get_object(bucket_name, "LATEST.json")
            pointer = json.loads(response.read().decode())
            return pointer
        except Exception:
            # LATEST.json doesn't exist or can't be read
            return None
    except Exception:
        # MinIO connection failed
        return None


def check_rag_index_exists(
    bucket_name: str = "rag-index",
    minio_endpoint: Optional[str] = None,
    minio_port: Optional[str] = None,
    minio_access_key: Optional[str] = None,
    minio_secret_key: Optional[str] = None,
) -> bool:
    """
    Check if a RAG index exists in MinIO and is in READY status.

    Configuration priority: function params > env vars > defaults (see get_minio_client).

    Args:
        bucket_name: MinIO bucket name (default: "rag-index")
        minio_endpoint: MinIO endpoint (overrides env/default)
        minio_port: MinIO port (overrides env/default)
        minio_access_key: MinIO access key (overrides env/default)
        minio_secret_key: MinIO secret key (overrides env/default)

    Returns:
        True if index exists and status is READY, False otherwise
    """
    status = get_rag_index_status(
        bucket_name, minio_endpoint, minio_port, minio_access_key, minio_secret_key
    )
    return status is not None and status.get("status") == "READY"
