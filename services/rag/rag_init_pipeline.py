import asyncio
import os
import sys
import glob
from pathlib import Path

# Add services/rag/src to Python path for RAG imports
rag_src_path = Path(__file__).parent / "src"
if rag_src_path.exists():
    sys.path.insert(0, str(rag_src_path))

from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def setup_rag_directories():
    """
    Setup RAG data directory structure.
    Creates the directory needed for temporary index storage before uploading to MinIO.
    """
    from utils.config import config

    logger.info("\n" + "=" * 70)
    logger.info("SETTING UP RAG DATA DIRECTORY STRUCTURE")
    logger.info("=" * 70)

    # Get paths from config (uses DATA_DIR env var)
    data_dir = Path(config.storage.data_dir)

    # Create necessary directories for RAG index building
    logger.info("Creating directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  ✓ {data_dir}")

    # Check for knowledge base PDFs (informational)
    kb_dir = Path(config.storage.knowledge_base_dir)
    if kb_dir.exists():
        pdfs = list(kb_dir.glob("*.pdf"))
        if pdfs:
            logger.info(
                f"\n✓ Found {len(pdfs)} PDF file(s) in knowledge base ({kb_dir}):"
            )
            for pdf in pdfs:
                logger.info(f"  - {pdf.name}")
        else:
            logger.warning(f"\n⚠ No PDF files found in {kb_dir}")
            logger.warning("  RAG index will not be created")
    else:
        logger.warning(f"\n⚠ Knowledge base directory not found at {kb_dir}")
        logger.warning("  RAG index will not be created")

    logger.info("=" * 70)


async def build_rag_index():
    """
    Build RAG index from knowledge base PDFs and save to MinIO.
    This runs during the RAG init job to create the FAISS index and save artifacts to MinIO.
    """
    from utils.config import config
    from rag.ingest_and_chunk import AnsibleErrorParser
    from rag.embed_and_index import AnsibleErrorEmbedder
    from utils.rag_minio import check_rag_index_exists, get_rag_index_status

    # Check if RAG is enabled
    rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
    rag_enabled = rag_enabled_env in ["true", "1", "yes"]
    if not rag_enabled:
        logger.info(
            f"RAG is disabled (RAG_ENABLED={rag_enabled_env}), skipping RAG index build"
        )
        return

    # Check if index already exists in MinIO (skip rebuild for faster upgrades)
    bucket_name = os.getenv("RAG_BUCKET_NAME", "rag-index")
    try:
        if check_rag_index_exists(bucket_name):
            status = get_rag_index_status(bucket_name)
            total_errors = status.get("total_errors", 0) if status else 0
            logger.info(
                f"✓ Found existing RAG index in MinIO (status: READY, {total_errors} errors), skipping rebuild"
            )
            logger.info(
                "  To force rebuild, delete index from MinIO or set RAG_FORCE_REBUILD=true"
            )
            if os.getenv("RAG_FORCE_REBUILD", "false").lower() != "true":
                return
    except Exception as e:
        logger.warning(f"⚠ Could not check MinIO: {e}")
        logger.info("  Proceeding with index build...")

    logger.info("\n" + "=" * 70)
    logger.info("BUILDING RAG INDEX FROM KNOWLEDGE BASE")
    logger.info("  Storage: MinIO")
    logger.info("=" * 70)

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
            logger.warning(f"⚠ WARNING: No PDF files found in {kb_dir}")
            logger.warning("  RAG index will not be created")
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
        logger.info("\n" + "=" * 70)
        logger.info("✓ RAG INDEX BUILD COMPLETE (MinIO)")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ ERROR building RAG index: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise  # Re-raise to fail the job


async def main():
    # Setup and initialization
    logger.info("\n" + "=" * 70)
    logger.info("ANSIBLE LOG MONITOR - RAG INITIALIZATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Setup RAG data directories (create /app/data/rag)
    setup_rag_directories()

    # Step 2: Build RAG index and save to MinIO
    await build_rag_index()

    logger.info("\n" + "=" * 70)
    logger.info("✓ RAG INITIALIZATION COMPLETE")
    logger.info("  Index saved to MinIO - RAG service will load it on startup")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
