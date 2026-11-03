"""
Object storage helpers.

Backed by aws-compatible-storage (S4), which exposes an S3 API on port 7480.
Uses the minio-py library which speaks plain S3 protocol.
"""

import os
import io
from typing import Optional
from minio import Minio
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def get_s3_client(
    s3_endpoint: Optional[str] = None,
    s3_port: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
) -> Minio:
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


def upload_model_to_s3(model, bucket_name: str, file_name: str):
    """
    Upload a sklearn model to object storage.

    Args:
        model: sklearn model to upload
        bucket_name: Target bucket name
        file_name: Name of the object in the bucket
    """
    # Lazy import to avoid requiring sklearn/joblib for RAG init job
    import joblib

    s3_client = get_s3_client()

    if not s3_client.bucket_exists(bucket_name):
        s3_client.make_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} created")
    else:
        logger.info(f"Bucket {bucket_name} already exists")

    # Serialize model to BytesIO buffer
    with io.BytesIO() as buffer:
        joblib.dump(model, buffer)
        buffer.seek(0)

        s3_client.put_object(
            bucket_name, file_name, buffer, length=buffer.getbuffer().nbytes
        )
    logger.info(f"Model {file_name} uploaded to object storage")
