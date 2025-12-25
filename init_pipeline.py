import asyncio
from alm.pipeline.offline import training_pipeline
from alm.utils.phoenix import register_phoenix
import os
import glob
import shutil
from pathlib import Path
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def setup_data_directories():
    """
    Setup data directory structure in PVC mount path.
    Creates necessary directories and copies PDFs from image to PVC if needed.
    """
    from src.alm.config import config

    logger.info("\n" + "=" * 70)
    logger.info("SETTING UP DATA DIRECTORY STRUCTURE")
    logger.info("=" * 70)

    # Get paths from config (uses DATA_DIR and KNOWLEDGE_BASE_DIR env vars)
    data_dir = Path(config.storage.data_dir)
    knowledge_base_dir = Path(config.storage.knowledge_base_dir)
    logs_dir = data_dir / "logs" / "failed"

    # Create necessary directories
    logger.info("Creating directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  ✓ {data_dir}")
    logger.info(f"  ✓ {knowledge_base_dir}")
    logger.info(f"  ✓ {logs_dir}")

    # Copy PDFs from image to PVC if PVC knowledge_base is empty
    image_kb_dir = Path("/app/data/knowledge_base")
    pvc_kb_dir = knowledge_base_dir

    # Check if PVC knowledge_base has any PDFs
    pvc_pdfs = list(pvc_kb_dir.glob("*.pdf"))

    if not pvc_pdfs:
        # PVC is empty, copy from image if available
        if image_kb_dir.exists():
            image_pdfs = list(image_kb_dir.glob("*.pdf"))
            if image_pdfs:
                logger.info(
                    f"\nCopying {len(image_pdfs)} PDF file(s) from image to PVC..."
                )
                for pdf_path in image_pdfs:
                    dest_path = pvc_kb_dir / pdf_path.name
                    try:
                        shutil.copy2(pdf_path, dest_path)
                        logger.info(f"  ✓ Copied {pdf_path.name}")
                    except Exception as e:
                        logger.error(f"  ✗ Error copying {pdf_path.name}: {e}")
                logger.info("✓ Knowledge base PDFs copied to PVC")
            else:
                logger.warning(f"\n⚠ No PDFs found in image at {image_kb_dir}")
        else:
            logger.warning(
                f"\n⚠ Image knowledge base directory not found at {image_kb_dir}"
            )
    else:
        logger.info(
            f"\n✓ PVC knowledge base already contains {len(pvc_pdfs)} PDF file(s), skipping copy"
        )

    logger.info("=" * 70)


def build_rag_index():
    """
    Build RAG index from knowledge base PDFs.
    This runs during the init job to create the FAISS index and metadata.
    """
    from src.alm.config import config
    from src.alm.rag.ingest_and_chunk import AnsibleErrorParser
    from src.alm.rag.embed_and_index import AnsibleErrorEmbedder

    # Check if RAG is enabled
    rag_enabled = os.getenv("RAG_ENABLED", "true").lower() == "true"
    if not rag_enabled:
        logger.info("RAG is disabled (RAG_ENABLED=false), skipping RAG index build")
        return

    # Check if index already exists (skip rebuild for faster upgrades)
    index_path = Path(config.storage.index_path)
    metadata_path = Path(config.storage.metadata_path)

    if index_path.exists() and metadata_path.exists():
        logger.info("✓ RAG index already exists, skipping rebuild")
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Metadata: {metadata_path}")
        logger.info("  To force rebuild, delete the PVC or these files")
        return

    logger.info("\n" + "=" * 70)
    logger.info("BUILDING RAG INDEX FROM KNOWLEDGE BASE")
    logger.info("=" * 70)

    try:
        # Validate configuration
        config.print_config()
        config.validate()

        # Initialize components
        parser = AnsibleErrorParser()
        embedder = AnsibleErrorEmbedder()

        # Find PDFs in knowledge base
        kb_dir = config.storage.knowledge_base_dir
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

        # Build and save index
        embedder.ingest_and_index(all_chunks)

        logger.info("\n" + "=" * 70)
        logger.info("✓ RAG INDEX BUILD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Metadata: {metadata_path}")

    except Exception as e:
        logger.error(f"\n✗ ERROR building RAG index: {e}")
        logger.error("  The system will continue without RAG functionality")
        import traceback

        logger.error(traceback.format_exc())


async def main():
    # Setup and initialization
    logger.info("\n" + "=" * 70)
    logger.info("ANSIBLE LOG MONITOR - INITIALIZATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Setup data directories (create dirs, copy PDFs if needed)
    setup_data_directories()

    # Step 2: Build RAG index
    build_rag_index()

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
