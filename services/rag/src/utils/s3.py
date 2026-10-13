"""
Standalone object storage utilities for RAG service.
No dependencies on backend (alm.*) code.

Backed by aws-compatible-storage (S4), which exposes an S3 API on port 7480.
Uses the minio-py library which speaks plain S3 protocol.
"""

import os
from typing import Optional
from minio import Minio


def get_s3_client(
    s3_endpoint: Optional[str] = None,
    s3_port: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
) -> Minio:
    """
    Get an S3 client with configuration from parameters, env vars, or defaults.

    Configuration priority:
    1. Function parameters (if provided)
    2. Environment variables (S3_ENDPOINT, S3_PORT, S3_ACCESS_KEY, S3_SECRET_KEY)
    3. Defaults (localhost:7480, s4admin/s4secret) - only for local development outside Podman

    Args:
        s3_endpoint: Object storage host (overrides env/default)
        s3_port: S3 API port (overrides env/default)
        s3_access_key: S3 access key (overrides env/default)
        s3_secret_key: S3 secret key (overrides env/default)

    Returns:
        S3 client instance

    Raises:
        ValueError: If required object storage configuration is missing
    """
    # Priority: function param > env var > default
    # Defaults only used for local development outside Podman
    endpoint = s3_endpoint or os.getenv("S3_ENDPOINT", "localhost")
    port = s3_port or os.getenv("S3_PORT", "7480")
    access_key = s3_access_key or os.getenv("S3_ACCESS_KEY", "s4admin")
    secret_key = s3_secret_key or os.getenv("S3_SECRET_KEY", "s4secret")

    if not all([endpoint, port, access_key, secret_key]):
        raise ValueError(
            "Missing required object storage environment variables: "
            "S3_ENDPOINT, S3_PORT, S3_ACCESS_KEY, S3_SECRET_KEY"
        )

    return Minio(
        endpoint=f"{endpoint}:{port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=False,  # Use HTTP for internal services
    )


def ensure_bucket_exists(s3_client: Minio, bucket_name: str) -> bool:
    """
    Ensure a bucket exists, creating it if necessary.

    Args:
        s3_client: S3 client instance
        bucket_name: Name of the bucket to ensure exists

    Returns:
        True if the bucket was created, False if it already existed
    """
    if not s3_client.bucket_exists(bucket_name):
        s3_client.make_bucket(bucket_name)
        return True
    return False
