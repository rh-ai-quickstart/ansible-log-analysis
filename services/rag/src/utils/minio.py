"""
Standalone MinIO utilities for RAG service.
No dependencies on backend (alm.*) code.
"""

import os
from typing import Optional
from minio import Minio


def get_minio_client(
    minio_endpoint: Optional[str] = None,
    minio_port: Optional[str] = None,
    minio_access_key: Optional[str] = None,
    minio_secret_key: Optional[str] = None,
) -> Minio:
    """
    Get a MinIO client with configuration from parameters, env vars, or defaults.

    Configuration priority:
    1. Function parameters (if provided)
    2. Environment variables (MINIO_ENDPOINT, MINIO_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    3. Defaults (localhost:9000, minioadmin/minioadmin) - only for local development outside Docker

    Args:
        minio_endpoint: MinIO endpoint (overrides env/default)
        minio_port: MinIO port (overrides env/default)
        minio_access_key: MinIO access key (overrides env/default)
        minio_secret_key: MinIO secret key (overrides env/default)

    Returns:
        MinIO client instance

    Raises:
        ValueError: If required MinIO configuration is missing
    """
    # Priority: function param > env var > default
    # Defaults only used for local development outside Docker
    endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT", "localhost")
    port = minio_port or os.getenv("MINIO_PORT", "9000")
    access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")

    if not all([endpoint, port, access_key, secret_key]):
        raise ValueError(
            "Missing required MinIO environment variables: "
            "MINIO_ENDPOINT, MINIO_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
        )

    return Minio(
        endpoint=f"{endpoint}:{port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=False,  # Use HTTP for internal services
    )


def ensure_bucket_exists(minio_client: Minio, bucket_name: str) -> bool:
    """
    Ensure a MinIO bucket exists, creating it if necessary.

    Args:
        minio_client: MinIO client instance
        bucket_name: Name of the bucket to ensure exists

    Returns:
        True if the bucket was created, False if it already existed
    """
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        return True
    return False
