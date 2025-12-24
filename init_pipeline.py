import asyncio
from alm.pipeline.offline import training_pipeline
from alm.utils.phoenix import register_phoenix
import os
import glob
from pathlib import Path
import httpx


def setup_data_directories():
    """
    Setup data directory structure.
    Knowledge base PDFs should be baked into the container image at /app/data/knowledge_base.
    """
    from alm.config import config

    logger.info("\n" + "=" * 70)
    logger.info("SETTING UP DATA DIRECTORY STRUCTURE")
    logger.info("=" * 70)

    # Get paths from config (uses DATA_DIR env var)
    data_dir = Path(config.storage.data_dir)
    logs_dir = data_dir / "logs" / "failed"

    # Create necessary directories (for logs, etc.)
    print("Creating directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {data_dir}")
    print(f"  ✓ {logs_dir}")

    # Check for knowledge base PDFs
    # Use config path (works for both local and container)
    kb_dir = Path(config.storage.knowledge_base_dir)
    if kb_dir.exists():
        pdfs = list(kb_dir.glob("*.pdf"))
        if pdfs:
            print(f"\n✓ Found {len(pdfs)} PDF file(s) in knowledge base ({kb_dir}):")
            for pdf in pdfs:
                print(f"  - {pdf.name}")
        else:
            print(f"\n⚠ No PDF files found in {kb_dir}")
            print("  Add PDF files to the knowledge base directory to enable RAG")
    else:
        print(f"\n⚠ Knowledge base directory not found at {kb_dir}")
        print("  Create the directory and add PDF files to enable RAG")

    logger.info("=" * 70)


async def build_rag_index():
    """
    Build RAG index from knowledge base PDFs and save to MinIO.
    This runs during the init job to create the FAISS index and save artifacts to MinIO.
    """
    from alm.config import config
    from alm.rag.ingest_and_chunk import AnsibleErrorParser
    from alm.rag.embed_and_index import AnsibleErrorEmbedder
    from minio import Minio
    import json

    # Check if RAG is enabled (consistent with rag_handler.py)
    rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
    rag_enabled = rag_enabled_env in ["true", "1", "yes"]
    if not rag_enabled:
        print(
            f"RAG is disabled (RAG_ENABLED={rag_enabled_env}), skipping RAG index build"
        )
        return

    # Check if index already exists in MinIO (skip rebuild for faster upgrades)
    # Use defaults for local development (when running outside Docker)
    bucket_name = os.getenv("RAG_BUCKET_NAME", "rag-index")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost")
    minio_port = os.getenv("MINIO_PORT", "9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    if all([minio_endpoint, minio_port, minio_access_key, minio_secret_key]):
        try:
            minio_client = Minio(
                endpoint=f"{minio_endpoint}:{minio_port}",
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=False,
            )

            # Check if LATEST.json exists and status is READY
            if minio_client.bucket_exists(bucket_name):
                try:
                    response = minio_client.get_object(bucket_name, "LATEST.json")
                    pointer = json.loads(response.read().decode())
                    if pointer.get("status") == "READY":
                        total_errors = pointer.get("total_errors", 0)
                        print(
                            f"✓ Found existing RAG index in MinIO (status: READY, {total_errors} errors), skipping rebuild"
                        )
                        print(
                            "  To force rebuild, delete index from MinIO or set RAG_FORCE_REBUILD=true"
                        )
                        if os.getenv("RAG_FORCE_REBUILD", "false").lower() != "true":
                            return
                except Exception:
                    # LATEST.json doesn't exist or can't be read, proceed with build
                    pass
        except Exception as e:
            print(f"⚠ Could not check MinIO: {e}")
            print("  Proceeding with index build...")

    print("\n" + "=" * 70)
    print("BUILDING RAG INDEX FROM KNOWLEDGE BASE")
    print("  Storage: MinIO")
    print("=" * 70)

    try:
        # Validate configuration
        config.print_config()
        config.validate()

        # Initialize components
        parser = AnsibleErrorParser()
        embedder = AnsibleErrorEmbedder()

        # Find PDFs in knowledge base
        # Use config path (works for both local and container)
        kb_dir = Path(config.storage.knowledge_base_dir)
        pdf_files = sorted(glob.glob(str(kb_dir / "*.pdf")))

        if not pdf_files:
            print(f"⚠ WARNING: No PDF files found in {kb_dir}")
            print("  RAG index will not be created")
            return

        logger.info(f"\n✓ Found {len(pdf_files)} PDF files in knowledge base:")
        for pdf in pdf_files:
            logger.info(f"  - {Path(pdf).name}")

        # Process all PDFs
        all_chunks = []
        for pdf_path in pdf_files:
            logger.info(f"\n📄 Processing: {Path(pdf_path).name}")
            try:
                chunks = parser.parse_pdf_to_chunks(pdf_path)
                all_chunks.extend(chunks)
                logger.info(f"  ✓ Extracted {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"  ✗ Error processing {Path(pdf_path).name}: {e}")
                continue

        if not all_chunks:
            logger.warning("\n⚠ WARNING: No chunks extracted from PDFs")
            logger.warning("  RAG index will not be created")
            return

        logger.info(f"\n{'=' * 70}")
        logger.info(f"TOTAL: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
        logger.info(f"{'=' * 70}")

        # Build and save index to MinIO
        await embedder.ingest_and_index_to_minio(all_chunks)
        print("\n" + "=" * 70)
        print("✓ RAG INDEX BUILD COMPLETE (MinIO)")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ ERROR building RAG index: {e}")
        logger.error("  The system will continue without RAG functionality")
        import traceback

        traceback.print_exc()


async def wait_for_rag_service(rag_service_url: str, max_wait_time: int = 300):
    """
    Wait for RAG service to be ready before proceeding.

    Args:
        rag_service_url: URL of the RAG service (e.g., http://alm-rag:8002)
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
    """
    # Check if RAG is enabled
    rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
    rag_enabled = rag_enabled_env in ["true", "1", "yes"]
    if not rag_enabled:
        print("RAG is disabled, skipping RAG service wait")
        return

    print("\n" + "=" * 70)
    print("WAITING FOR RAG SERVICE TO BE READY")
    print("=" * 70)

    ready_url = f"{rag_service_url}/ready"
    elapsed = 0
    check_interval = 5

    async with httpx.AsyncClient(timeout=10.0) as client:
        while elapsed < max_wait_time:
            try:
                response = await client.get(ready_url)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        index_size = data.get("index_size", 0)
                        print(f"✓ RAG service is ready (index size: {index_size})")
                        return
                    except (ValueError, TypeError):
                        # Invalid JSON response - treat as not ready
                        print(
                            f"RAG service returned invalid JSON (status: {response.status_code}), waiting..."
                        )
                else:
                    print(
                        f"RAG service not ready yet (status: {response.status_code}), waiting..."
                    )
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                ValueError,
                TypeError,
            ):
                if elapsed == 0:
                    print(
                        f"RAG service not yet available at {rag_service_url}, waiting..."
                    )
                elif elapsed % 30 == 0:  # Print every 30 seconds
                    print(f"Still waiting for RAG service... (elapsed: {elapsed}s)")

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        # Timeout reached
        print(
            f"\n⚠ WARNING: RAG service did not become ready within {max_wait_time} seconds"
        )
        print("  The training pipeline will proceed, but RAG queries may fail")
        print("  This is expected if the RAG service is still starting up")


async def main():
    # Setup and initialization
    logger.info("\n" + "=" * 70)
    logger.info("ANSIBLE LOG MONITOR - INITIALIZATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Setup data directories (create dirs, copy PDFs if needed)
    setup_data_directories()

    # Step 2: Build RAG index
    await build_rag_index()

    # Step 2.5: Wait for RAG service to be ready (if RAG is enabled)
    rag_service_url = os.getenv("RAG_SERVICE_URL", "http://alm-rag:8002")
    await wait_for_rag_service(rag_service_url)

    # Step 3: Run main pipeline (clustering, summarization, etc.)
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING MAIN PIPELINE")
    logger.info("=" * 70)
    await training_pipeline()

    logger.info("\n" + "=" * 70)
    logger.info("✓ INITIALIZATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    register_phoenix()
    asyncio.run(main())
