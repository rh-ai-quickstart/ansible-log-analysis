"""
Standalone object storage utilities for RAG service.
No dependencies on backend (alm.*) code.

Backed by aws-compatible-storage (S4), which exposes an S3 API on port 7480.
The minio-py client is kept because it speaks plain S3 - only the endpoint and
credentials changed, so the MINIO_* variable names are retained for continuity.
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
    Get an S3 client with configuration from parameters, env vars, or defaults.

    Configuration priority:
    1. Function parameters (if provided)
    2. Environment variables (MINIO_ENDPOINT, MINIO_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    3. Defaults (localhost:7480, s4admin/s4secret) - only for local development outside Podman

    Args:
        minio_endpoint: Object storage host (overrides env/default)
        minio_port: S3 API port (overrides env/default)
        minio_access_key: S3 access key (overrides env/default)
        minio_secret_key: S3 secret key (overrides env/default)

    Returns:
        S3 client instance

    Raises:
        ValueError: If required object storage configuration is missing
    """
    # Priority: function param > env var > default
    # Defaults only used for local development outside Podman
    endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT", "localhost")
    port = minio_port or os.getenv("MINIO_PORT", "7480")
    access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY", "s4admin")
    secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY", "s4secret")

    if not all([endpoint, port, access_key, secret_key]):
        raise ValueError(
            "Missing required object storage environment variables: "
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
    Ensure a bucket exists, creating it if necessary.

    Args:
        minio_client: S3 client instance
        bucket_name: Name of the bucket to ensure exists

    Returns:
        True if the bucket was created, False if it already existed
    """
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        return True
    return False
