"""
RAG-specific object storage utilities for checking RAG index status.
"""

import json
from typing import Optional, Dict, Any

from utils.s3 import get_s3_client


def get_rag_index_status(
    bucket_name: str = "rag-index",
    s3_endpoint: Optional[str] = None,
    s3_port: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the status of the RAG index from object storage LATEST.json file.

    Configuration priority: function params > env vars > defaults (see get_s3_client).

    Args:
        bucket_name: object storage bucket name (default: "rag-index")
        s3_endpoint: object storage host (overrides env/default)
        s3_port: S3 API port (overrides env/default)
        s3_access_key: S3 access key (overrides env/default)
        s3_secret_key: S3 secret key (overrides env/default)

    Returns:
        Dictionary with status information if LATEST.json exists, None otherwise
    """
    try:
        s3_client = get_s3_client(
            s3_endpoint, s3_port, s3_access_key, s3_secret_key
        )

        if not s3_client.bucket_exists(bucket_name):
            return None

        try:
            response = s3_client.get_object(bucket_name, "LATEST.json")
            pointer = json.loads(response.read().decode())
            return pointer
        except Exception:
            # LATEST.json doesn't exist or can't be read
            return None
    except Exception:
        # object storage connection failed
        return None


def check_rag_index_exists(
    bucket_name: str = "rag-index",
    s3_endpoint: Optional[str] = None,
    s3_port: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
) -> bool:
    """
    Check if a RAG index exists in object storage and is in READY status.

    Configuration priority: function params > env vars > defaults (see get_s3_client).

    Args:
        bucket_name: object storage bucket name (default: "rag-index")
        s3_endpoint: object storage host (overrides env/default)
        s3_port: S3 API port (overrides env/default)
        s3_access_key: S3 access key (overrides env/default)
        s3_secret_key: S3 secret key (overrides env/default)

    Returns:
        True if index exists and status is READY, False otherwise
    """
    status = get_rag_index_status(
        bucket_name, s3_endpoint, s3_port, s3_access_key, s3_secret_key
    )
    return status is not None and status.get("status") == "READY"
