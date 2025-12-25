import os
import json
import io
from typing import Optional, Dict, Any
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


def upload_model_to_minio(model, bucket_name: str, file_name: str):
    """
    Upload a sklearn model to MinIO.

    Args:
        model: sklearn model to upload
        bucket_name: MinIO bucket name
        file_name: Name of the file in MinIO
    """
    # Lazy import to avoid requiring sklearn/joblib for RAG init job
    import joblib

    minio_client = get_minio_client()
    ensure_bucket_exists(
        minio_client, bucket_name
    )  # Bucket creation logged by caller if needed

    # Serialize model to BytesIO buffer
    with io.BytesIO() as buffer:
        joblib.dump(model, buffer)
        buffer.seek(0)

        minio_client.put_object(
            bucket_name, file_name, buffer, length=buffer.getbuffer().nbytes
        )
