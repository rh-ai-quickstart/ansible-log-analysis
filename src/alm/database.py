from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from alm.models import GrafanaAlert, RAGEmbedding
from alm.agents.state import GrafanaAlertState
from alm.models import LogEntry
from alm.utils.logger import get_logger

logger = get_logger(__name__)

# Create SQLModel engine
engine = create_async_engine(
    os.getenv("DATABASE_URL")
    .replace("+asyncpg", "")
    .replace("postgresql", "postgresql+asyncpg")
)


# Create tables
async def init_tables(delete_tables=False):
    async with engine.begin() as conn:
        if delete_tables:
            logger.info("Starting to delete tables")
            # Only delete GrafanaAlert table, NOT RAGEmbedding
            # RAG embeddings should persist across training pipeline runs
            await conn.run_sync(GrafanaAlert.metadata.drop_all)
            # RAGEmbedding table is NOT deleted - it persists across runs

        # Ensure pgvector extension is enabled (must be done before creating tables)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension: {e}")
            logger.warning(
                "This is OK if extension is already enabled or not available"
            )

        # Create all tables
        await conn.run_sync(GrafanaAlert.metadata.create_all)
        await conn.run_sync(RAGEmbedding.metadata.create_all)


def get_session():
    session = AsyncSession(engine)
    return session


async def get_session_gen() -> Generator[AsyncSession, None, None]:
    async with get_session() as session:
        yield session


def convert_state_to_grafana_alert(state: dict) -> GrafanaAlert:
    return GrafanaAlert(
        logTimestamp=datetime.fromisoformat(state["log_entry"].timestamp),
        logMessage=state["log_entry"].message,
        logSummary=state["logSummary"],
        expertClassification=state["expertClassification"],
        logCluster=state["logCluster"],
        needMoreContext=state["needMoreContext"],
        stepByStepSolution=state["stepByStepSolution"],
        contextForStepByStepSolution=state["contextForStepByStepSolution"],
        log_labels=state["log_entry"].log_labels,
    )


def convert_grafana_alert_to_grafana_alert_state(
    alert: GrafanaAlert,
) -> GrafanaAlertState:
    return GrafanaAlertState(
        log_entry=LogEntry(
            timestamp=alert.logTimestamp.isoformat(),
            log_labels=alert.log_labels,
            message=alert.logMessage,
        ),
        logSummary=alert.logSummary,
        expertClassification=alert.expertClassification,
        logCluster=alert.logCluster,
        needMoreContext=alert.needMoreContext,
        stepByStepSolution=alert.stepByStepSolution,
        contextForStepByStepSolution=alert.contextForStepByStepSolution,
    )
