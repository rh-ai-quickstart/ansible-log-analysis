from typing import List
from fastapi import APIRouter, Depends, Query, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from alm.database import get_session_gen
from alm.models import GrafanaAlert
from alm.agents.graph import inference_graph
from alm.models import LogEntry, LogLabels, DetectedLevel
from alm.models import LogStatus, LogType
from datetime import datetime
from alm.database import convert_state_to_grafana_alert
from alm.agents.state import GrafanaAlertState

router = APIRouter(prefix="/grafana-alert", tags=["grafana-alert"])


@router.get(
    "/{alert_id}", summary="Get grafana alert by id", response_model=GrafanaAlert
)
async def get_grafana_alert(
    alert_id: int, session: AsyncSession = Depends(get_session_gen)
) -> GrafanaAlert:
    alert = await session.get(GrafanaAlert, alert_id)
    return alert


@router.get("/", summary="Get all grafana alerts", response_model=List[GrafanaAlert])
async def get_grafana_alerts(
    session: AsyncSession = Depends(get_session_gen),
) -> List[GrafanaAlert]:
    alerts = await session.exec(select(GrafanaAlert))
    return alerts.all()


@router.get(
    "/by-expert-class/",
    summary="Get grafana alerts by expert class",
    response_model=List[GrafanaAlert],
)
async def get_grafana_alerts_by_expert_class(
    expert_class: str = Query(..., description="The expert class to filter alerts by"),
    session: AsyncSession = Depends(get_session_gen),
) -> List[GrafanaAlert]:
    query = select(GrafanaAlert).where(
        GrafanaAlert.expertClassification == expert_class
    )
    alerts = await session.exec(query)
    return alerts


@router.get(
    "/unique-clusters/",
    summary="Get unique log clusters for an expert class with representative alerts",
    response_model=List[GrafanaAlert],
)
async def get_unique_clusters_by_expert_class(
    expert_class: str = Query(..., description="The expert class to filter alerts by"),
    session: AsyncSession = Depends(get_session_gen),
) -> List[GrafanaAlert]:
    """Get one representative alert for each unique log cluster within an expert class."""
    from sqlalchemy import func

    # Get one alert per unique cluster within the expert class
    subquery = (
        select(GrafanaAlert.logCluster, func.min(GrafanaAlert.id).label("min_id"))
        .where(GrafanaAlert.expertClassification == expert_class)
        .where(GrafanaAlert.logCluster.is_not(None))
        .group_by(GrafanaAlert.logCluster)
    ).alias("clusters")

    query = (
        select(GrafanaAlert)
        .join(subquery, GrafanaAlert.id == subquery.c.min_id)
        .order_by(GrafanaAlert.logCluster)
    )

    alerts = await session.exec(query)
    return alerts


# TODO feels not efficent
@router.get(
    "/by-expert-class-and-log-cluster/",
    summary="Get grafana alerts by expert class and log cluster",
    response_model=List[GrafanaAlert],
)
async def get_grafana_alerts_by_expert_class_and_log_cluster(
    expert_class: str = Query(..., description="The expert class to filter alerts by"),
    log_cluster: str = Query(..., description="The log cluster to filter alerts by"),
    session: AsyncSession = Depends(get_session_gen),
) -> List[GrafanaAlert]:
    query = select(GrafanaAlert).where(
        GrafanaAlert.logCluster == log_cluster,
        GrafanaAlert.expertClassification == expert_class,
    )
    alerts = await session.exec(query)
    return alerts


@router.post("/", status_code=status.HTTP_202_ACCEPTED, summary="Post log alert")
async def post_log_alert(
    log_alert: str,
    detected_level: DetectedLevel = DetectedLevel.UNKNOWN,
    filename: str = "Unknown filename",
    job: str = "Unknown job",
    service_name: str = "Unknown service name",
    timestamp: datetime = Depends(lambda: datetime.now()),
    status: LogStatus = LogStatus.OK,
    log_type: LogType = LogType.OTHER,
    session: AsyncSession = Depends(get_session_gen),
) -> GrafanaAlert:
    log_labels = LogLabels(
        detected_level=detected_level,
        filename=filename,
        job=job,
        database_timestamp=timestamp,  # TODO think about it
        service_name=service_name,
        status=status,
        log_type=log_type,
    )
    log_entry = LogEntry(
        timestamp=timestamp.isoformat(), log_labels=log_labels, message=log_alert
    )
    state = await inference_graph().ainvoke({"log_entry": log_entry})

    grafana_alert = convert_state_to_grafana_alert(GrafanaAlertState(**state))

    session.add(grafana_alert)
    await session.commit()
    await session.refresh(grafana_alert)
    return grafana_alert
