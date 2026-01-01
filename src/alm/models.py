from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel
from enum import Enum

from pydantic import BaseModel
import pydantic


# DB / API models
class GrafanaAlert(SQLModel, table=True):
    """Grafana alert payload for Loki log alerts."""

    # # Optional ID field
    id: Optional[int] = Field(default=None, primary_key=True)

    # Grouping information
    logTimestamp: Optional[datetime] = Field(
        default=None, description="Timestamp of the log message"
    )

    logMessage: str = Field(description="Original log message that triggered the alert")

    log_labels: dict = Field(
        default_factory=dict, description="Loki log metadata", sa_column=Column(JSON)
    )

    logSummary: str = Field(
        default="No summary available", description="Summary of the log message"
    )

    expertClassification: Optional[str] = Field(
        default=None, description="Classification of the log message"
    )

    logCluster: Optional[str] = Field(
        default=None, description="Cluster of the log message"
    )
    needMoreContext: Optional[bool] = Field(
        default=None, description="Is additional context needed to solve the problem"
    )

    stepByStepSolution: Optional[str] = Field(
        default=None, description="Step by step solution to the problem"
    )

    contextForStepByStepSolution: Optional[str] = Field(
        default=None, description="Context for the step by step solution"
    )


# Input models
class DetectedLevel(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"
    UNKNOWN = "unknown"
    # FATAL = "fatal" # This type of error ask yossi what he thinks to do.


class LogStatus(str, Enum):
    """Status for task logs (entries with log_type="task"),
    non-task logs have empty status value."""

    OK = "ok"
    CHANGED = "changed"
    FAILED = "failed"
    FATAL = "fatal"
    IGNORING = "ignoring"
    SKIPPING = "skipping"
    INCLUDED = "included"


class LogType(str, Enum):
    TASK = "task"
    RECAP = "recap"
    PLAY = "play"
    OTHER = "other"


class LogLabels(BaseModel):
    """Metadata labels for a single log entry from Loki"""

    detected_level: DetectedLevel = pydantic.Field(
        default=DetectedLevel.UNKNOWN, description="Detected level of the log"
    )
    filename: str = pydantic.Field(
        default="Unknown filename", description="Filename of the log"
    )
    job: str = pydantic.Field(default="Unknown job", description="Job of the log")
    log_type: LogType = pydantic.Field(
        default=LogType.OTHER, description="Type of the log"
    )
    service_name: str = pydantic.Field(
        default="Unknown service name", description="Service name of the log"
    )
    database_timestamp: datetime = pydantic.Field(
        default_factory=datetime.now, description="Timestamp of the log in the database"
    )
    status: LogStatus = pydantic.Field(
        default=LogStatus.OK, description="Status of the log"
    )

    @pydantic.field_validator("database_timestamp", mode="before")
    @classmethod
    def convert_to_utc_datetime(cls, v):
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc)
        if isinstance(v, str):
            from alm.tools import timestamp_to_utc_datetime

            return timestamp_to_utc_datetime(v).replace(tzinfo=timezone.utc)
        return v


class LogEntry(BaseModel):
    """Represents a single log entry from Loki"""

    timestamp: Optional[datetime] = pydantic.Field(
        default=None, description="Timestamp of the log"
    )
    log_labels: LogLabels = pydantic.Field(description="Log labels of the log")
    message: str = pydantic.Field(description="Message of the log")

    @pydantic.field_validator("timestamp", mode="before")
    @classmethod
    def convert_to_datetime(cls, v):
        """Parse timestamp (custom or ISO format)."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # ISO format starts with digit: "2025-08-04T08:00:26"
            # Custom format starts with letter: "Tuesday 05 August 2025..."
            if v[0].isdigit():
                from dateutil import parser as date_parser

                return date_parser.parse(v).replace(tzinfo=None)
            else:
                return datetime.strptime(v, "%A %d %B %Y  %H:%M:%S %z").replace(
                    tzinfo=None
                )
        return v


# RAG embeddings are now stored in MinIO, not PostgreSQL
# The RAGEmbedding table has been removed in favor of MinIO artifact storage
