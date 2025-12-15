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

    print("\n" + "=" * 70)
    print("SETTING UP DATA DIRECTORY STRUCTURE")
    print("=" * 70)

    # Get paths from config (uses DATA_DIR env var)
    data_dir = Path(config.storage.data_dir)
    logs_dir = data_dir / "logs" / "failed"

    # Create necessary directories (for logs, etc.)
    print("Creating directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {data_dir}")
    print(f"  ✓ {logs_dir}")

    # Check for knowledge base PDFs in image
    image_kb_dir = Path("/app/data/knowledge_base")
    if image_kb_dir.exists():
        image_pdfs = list(image_kb_dir.glob("*.pdf"))
        if image_pdfs:
            print(f"\n✓ Found {len(image_pdfs)} PDF file(s) in container image:")
            for pdf in image_pdfs:
                print(f"  - {pdf.name}")
        else:
            print(f"\n⚠ No PDF files found in image at {image_kb_dir}")
    else:
        print(f"\n⚠ Knowledge base directory not found in image at {image_kb_dir}")
        print(
            "  PDFs should be baked into the container image at /app/data/knowledge_base"
        )

    print("=" * 70)


async def build_rag_index():
    """
    Build RAG index from knowledge base PDFs and save to PostgreSQL.
    This runs during the init job to create the FAISS index and save embeddings to database.
    """
    from alm.config import config
    from alm.rag.ingest_and_chunk import AnsibleErrorParser
    from alm.rag.embed_and_index import AnsibleErrorEmbedder
    from alm.database import init_tables

    # Check if RAG is enabled (consistent with rag_handler.py)
    rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
    rag_enabled = rag_enabled_env in ["true", "1", "yes"]
    if not rag_enabled:
        print(
            f"RAG is disabled (RAG_ENABLED={rag_enabled_env}), skipping RAG index build"
        )
        return

    # Check if embeddings already exist in PostgreSQL (skip rebuild for faster upgrades)
    from alm.database import get_session
    from alm.models import RAGEmbedding
    from sqlmodel import select

    try:
        async with get_session() as session:
            result = await session.exec(select(RAGEmbedding))
            existing = result.first()
            if existing:
                count_result = await session.exec(select(RAGEmbedding))
                count = len(list(count_result.all()))
                print(
                    f"✓ Found {count} existing embeddings in PostgreSQL, skipping rebuild"
                )
                print(
                    "  To force rebuild, delete embeddings from PostgreSQL or set RAG_FORCE_REBUILD=true"
                )
                if os.getenv("RAG_FORCE_REBUILD", "false").lower() != "true":
                    return
    except Exception as e:
        print(f"⚠ Could not check PostgreSQL: {e}")
        print("  Proceeding with index build...")

    print("\n" + "=" * 70)
    print("BUILDING RAG INDEX FROM KNOWLEDGE BASE")
    print("  Storage: PostgreSQL")
    print("=" * 70)

    try:
        # Ensure database tables exist
        await init_tables(delete_tables=False)

        # Validate configuration
        config.print_config()
        config.validate()

        # Initialize components
        parser = AnsibleErrorParser()
        embedder = AnsibleErrorEmbedder()

        # Find PDFs in knowledge base (from container image)
        # PDFs should be baked into the image at /app/data/knowledge_base
        image_kb_dir = Path("/app/data/knowledge_base")
        pdf_files = sorted(glob.glob(str(image_kb_dir / "*.pdf")))

        if not pdf_files:
            print(f"⚠ WARNING: No PDF files found in {image_kb_dir}")
            print("  RAG index will not be created")
            return

        print(f"\n✓ Found {len(pdf_files)} PDF files in knowledge base:")
        for pdf in pdf_files:
            print(f"  - {Path(pdf).name}")

        # Process all PDFs
        all_chunks = []
        for pdf_path in pdf_files:
            print(f"\n📄 Processing: {Path(pdf_path).name}")
            try:
                chunks = parser.parse_pdf_to_chunks(pdf_path)
                all_chunks.extend(chunks)
                print(f"  ✓ Extracted {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ Error processing {Path(pdf_path).name}: {e}")
                continue

        if not all_chunks:
            print("\n⚠ WARNING: No chunks extracted from PDFs")
            print("  RAG index will not be created")
            return

        print(f"\n{'=' * 70}")
        print(f"TOTAL: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
        print(f"{'=' * 70}")

        # Build and save index to PostgreSQL
        await embedder.ingest_and_index_to_postgresql(all_chunks)
        print("\n" + "=" * 70)
        print("✓ RAG INDEX BUILD COMPLETE (PostgreSQL)")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR building RAG index: {e}")
        print("  The system will continue without RAG functionality")
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
                    data = response.json()
                    index_size = data.get("index_size", 0)
                    print(f"✓ RAG service is ready (index size: {index_size})")
                    return
                else:
                    print(
                        f"RAG service not ready yet (status: {response.status_code}), waiting..."
                    )
            except (httpx.RequestError, httpx.HTTPStatusError):
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
    print("\n" + "=" * 70)
    print("ANSIBLE LOG MONITOR - INITIALIZATION PIPELINE")
    print("=" * 70)

    # Step 1: Setup data directories (create dirs, copy PDFs if needed)
    setup_data_directories()

    # Step 2: Build RAG index
    await build_rag_index()

    # Step 2.5: Wait for RAG service to be ready (if RAG is enabled)
    rag_service_url = os.getenv("RAG_SERVICE_URL", "http://alm-rag:8002")
    await wait_for_rag_service(rag_service_url)

    # Step 3: Run main pipeline (clustering, summarization, etc.)
    print("\n" + "=" * 70)
    print("RUNNING MAIN PIPELINE")
    print("=" * 70)
    await training_pipeline()

    print("\n" + "=" * 70)
    print("✓ INITIALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    register_phoenix()
    asyncio.run(main())
