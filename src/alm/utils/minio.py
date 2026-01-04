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
    logger.info(f"Model {file_name} uploaded to MinIO")
