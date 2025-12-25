import asyncio
from alm.pipeline.offline import training_pipeline
from alm.utils.phoenix import register_phoenix
from alm.utils.rag_service import wait_for_rag_service
from alm.utils.job_monitor import monitor_other_job_async, wait_for_job_complete
import os
from pathlib import Path


def setup_data_directories():
    """
    Setup data directory structure for backend processing.
    Creates directories needed for log processing (not RAG-related).
    """
    print("\n" + "=" * 70)
    print("SETTING UP DATA DIRECTORY STRUCTURE")
    print("=" * 70)

    # Create logs directory for the training pipeline
    # The training pipeline uses "data/logs/failed" (relative to working directory /app)
    logs_dir = Path("data/logs/failed")

    # Create necessary directories for backend processing
    print("Creating directories...")
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {logs_dir}")

    print("=" * 70)


async def main():
    # Setup and initialization
    print("\n" + "=" * 70)
    print("ANSIBLE LOG MONITOR - BACKEND INITIALIZATION PIPELINE")
    print("=" * 70)

    # Get job names and namespace for monitoring
    rag_job_name = os.getenv(
        "RAG_INIT_JOB_NAME", "ansible-log-monitor-backend-rag-init"
    )
    namespace = os.getenv("NAMESPACE", os.getenv("POD_NAMESPACE", "default"))

    # Start monitoring the RAG init job in the background
    monitor_task = None
    try:
        monitor_task = asyncio.create_task(
            monitor_other_job_async(rag_job_name, namespace, check_interval=30)
        )
    except Exception as e:
        print(f"⚠ Warning: Could not start job monitoring: {e}")
        print("  Continuing without monitoring...")

    try:
        # Step 1: Setup data directories (create dirs, copy PDFs if needed)
        setup_data_directories()

        # Step 2: Wait for RAG init job to complete first
        # This ensures the RAG index is saved to MinIO before the RAG service tries to load it
        print("\n" + "=" * 70)
        print("WAITING FOR RAG INIT JOB TO COMPLETE")
        print("=" * 70)
        try:
            await wait_for_job_complete(rag_job_name, namespace, max_wait_time=600)
        except (TimeoutError, RuntimeError) as e:
            print(f"\n✗ ERROR: {e}")
            print("  Cannot proceed without RAG index. Exiting...")
            raise

        # Step 3: Wait for RAG service to be ready (required for training pipeline)
        # The RAG service will start after its init container detects the index in MinIO
        rag_service_url = os.getenv("RAG_SERVICE_URL", "http://alm-rag:8002")
        await wait_for_rag_service(rag_service_url)

        # Step 4: Run main pipeline (clustering, summarization, etc.)
        print("\n" + "=" * 70)
        print("RUNNING MAIN PIPELINE")
        print("=" * 70)
        await training_pipeline()

        print("\n" + "=" * 70)
        print("✓ BACKEND INITIALIZATION COMPLETE")
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
    register_phoenix()
    asyncio.run(main())
