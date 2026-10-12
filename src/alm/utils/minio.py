"""
Object storage helpers.

Backed by aws-compatible-storage (S4), which exposes an S3 API on port 7480.
The minio-py client is kept because it speaks plain S3 - only the endpoint and
credentials changed, so the MINIO_* variable names are retained for continuity.
"""

import os
import io
from typing import Optional
from minio import Minio
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def get_minio_client(
    minio_endpoint: Optional[str] = None,
    minio_port: Optional[str] = None,
    minio_access_key: Optional[str] = None,
    minio_secret_key: Optional[str] = None,
) -> Minio:
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


def upload_model_to_minio(model, bucket_name: str, file_name: str):
    """
    Upload a sklearn model to object storage.

    Args:
        model: sklearn model to upload
        bucket_name: Target bucket name
        file_name: Name of the object in the bucket
    """
    # Lazy import to avoid requiring sklearn/joblib for RAG init job
    import joblib

    minio_client = get_minio_client()

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} created")
    else:
        logger.info(f"Bucket {bucket_name} already exists")

    # Serialize model to BytesIO buffer
    with io.BytesIO() as buffer:
        joblib.dump(model, buffer)
        buffer.seek(0)

        minio_client.put_object(
            bucket_name, file_name, buffer, length=buffer.getbuffer().nbytes
        )
    logger.info(f"Model {file_name} uploaded to object storage")
