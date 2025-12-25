import asyncio
import os
import glob
from pathlib import Path
from alm.utils.job_monitor import monitor_other_job_async


def setup_rag_directories():
    """
    Setup RAG data directory structure.
    Creates the directory needed for temporary index storage before uploading to MinIO.
    """
    from alm.config import config

    print("\n" + "=" * 70)
    print("SETTING UP RAG DATA DIRECTORY STRUCTURE")
    print("=" * 70)

    # Get paths from config (uses DATA_DIR env var)
    data_dir = Path(config.storage.data_dir)

    # Create necessary directories for RAG index building
    print("Creating directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {data_dir}")

    # Check for knowledge base PDFs (informational)
    kb_dir = Path(config.storage.knowledge_base_dir)
    if kb_dir.exists():
        pdfs = list(kb_dir.glob("*.pdf"))
        if pdfs:
            print(f"\n✓ Found {len(pdfs)} PDF file(s) in knowledge base ({kb_dir}):")
            for pdf in pdfs:
                print(f"  - {pdf.name}")
        else:
            print(f"\n⚠ No PDF files found in {kb_dir}")
            print("  RAG index will not be created")
    else:
        print(f"\n⚠ Knowledge base directory not found at {kb_dir}")
        print("  RAG index will not be created")

    logger.info("=" * 70)


async def build_rag_index():
    """
    Build RAG index from knowledge base PDFs and save to MinIO.
    This runs during the RAG init job to create the FAISS index and save artifacts to MinIO.
    """
    from alm.config import config
    from alm.rag.ingest_and_chunk import AnsibleErrorParser
    from alm.rag.embed_and_index import AnsibleErrorEmbedder
    from alm.utils.minio import check_rag_index_exists, get_rag_index_status

    # Check if RAG is enabled (consistent with rag_handler.py)
    rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
    rag_enabled = rag_enabled_env in ["true", "1", "yes"]
    if not rag_enabled:
        print(
            f"RAG is disabled (RAG_ENABLED={rag_enabled_env}), skipping RAG index build"
        )
        return

    # Check if index already exists in MinIO (skip rebuild for faster upgrades)
    bucket_name = os.getenv("RAG_BUCKET_NAME", "rag-index")
    try:
        if check_rag_index_exists(bucket_name):
            status = get_rag_index_status(bucket_name)
            total_errors = status.get("total_errors", 0) if status else 0
            print(
                f"✓ Found existing RAG index in MinIO (status: READY, {total_errors} errors), skipping rebuild"
            )
            print(
                "  To force rebuild, delete index from MinIO or set RAG_FORCE_REBUILD=true"
            )
            if os.getenv("RAG_FORCE_REBUILD", "false").lower() != "true":
                return
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
        print(f"\n✗ ERROR building RAG index: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise to fail the job


async def main():
    # Setup and initialization
    print("\n" + "=" * 70)
    print("ANSIBLE LOG MONITOR - RAG INITIALIZATION PIPELINE")
    print("=" * 70)

    # Get job names and namespace for monitoring
    backend_job_name = os.getenv(
        "BACKEND_INIT_JOB_NAME", "ansible-log-monitor-backend-init"
    )
    namespace = os.getenv("NAMESPACE", os.getenv("POD_NAMESPACE", "default"))

    # Start monitoring the backend init job in the background
    monitor_task = None
    try:
        monitor_task = asyncio.create_task(
            monitor_other_job_async(backend_job_name, namespace, check_interval=30)
        )
    except Exception as e:
        print(f"⚠ Warning: Could not start job monitoring: {e}")
        print("  Continuing without monitoring...")

    try:
        # Step 1: Setup RAG data directories (create /app/data/rag)
        setup_rag_directories()

        # Step 2: Build RAG index and save to MinIO
        await build_rag_index()

        print("\n" + "=" * 70)
        print("✓ RAG INITIALIZATION COMPLETE")
        print("  Index saved to MinIO - RAG service will load it on startup")
        print("=" * 70)

    finally:
        # Cancel monitoring task
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
