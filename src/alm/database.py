from __future__ import annotations

import os
from typing import Generator

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from alm.models import GrafanaAlert
from alm.agents.state import GrafanaAlertState
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
            await conn.run_sync(GrafanaAlert.metadata.drop_all)
        await conn.run_sync(GrafanaAlert.metadata.create_all)


def get_session():
    session = AsyncSession(engine)
    return session


async def get_session_gen() -> Generator[AsyncSession, None, None]:
    async with get_session() as session:
        yield session


def convert_state_to_grafana_alert(state: GrafanaAlertState) -> GrafanaAlert:
    return GrafanaAlert(
        logTimestamp=state.log_entry.timestamp,
        logMessage=state.log_entry.message,
        logSummary=state.logSummary,
        expertClassification=state.expertClassification,
        logCluster=state.logCluster,
        needMoreContext=state.needMoreContext,
        stepByStepSolution=state.stepByStepSolution,
        contextForStepByStepSolution=state.contextForStepByStepSolution,
        log_labels=state.log_entry.log_labels.model_dump(mode="json"),
    )
