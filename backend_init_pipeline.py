import asyncio
import os

from alm.utils.phoenix import register_phoenix  # noqa: E402
from alm.utils.rag_service import wait_for_rag_service  # noqa: E402
from alm.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


async def main():
    # Setup and initialization
    logger.info("\n" + "=" * 70)
    logger.info("ANSIBLE LOG MONITOR - BACKEND INITIALIZATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Run pipeline preparation steps
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING PIPELINE")
    logger.info("=" * 70)

    from alm.pipeline.offline import training_pipeline_prepare

    log_entries, cluster_labels, unique_cluster = await training_pipeline_prepare()

    # Step 2: Wait for RAG service to be ready (required for alert processing)
    # This also implicitly waits for the RAG init job to complete, as the service
    # won't become ready until the index is built and available in object storage
    logger.info("\n" + "=" * 70)
    logger.info("WAITING FOR RAG SERVICE TO BE READY")
    logger.info("=" * 70)

    rag_service_url = os.getenv("RAG_SERVICE_URL", "http://alm-rag:8002")
    wait_for_rag_service(rag_service_url)

    # Step 3: Process alerts (this requires RAG service)
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING ALERTS")
    logger.info("=" * 70)

    from alm.pipeline.offline import training_pipeline_process

    await training_pipeline_process(log_entries, cluster_labels, unique_cluster)

    logger.info("\n" + "=" * 70)
    logger.info("✓ BACKEND INITIALIZATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    register_phoenix()
    asyncio.run(main())
