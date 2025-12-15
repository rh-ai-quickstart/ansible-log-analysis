from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel
from pgvector.sqlalchemy import Vector
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
        default_factory=datetime.now, description="Timestamp of the log message"
    )
    logMessage: str = Field(description="Original log message that triggered the alert")
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
    log_labels: dict = Field(
        default={}, description="Loki log metadata", sa_column=Column(JSON)
    )


# Input models
class LogLevel(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"
    UNKNOWN = "unknown"
    # FATAL = "fatal" # This type of error ask yossi what he thinks to do.


class LogLabels(BaseModel):
    """Metadata labels for a single log entry from Loki"""

    detected_level: Optional[LogLevel] = pydantic.Field(
        default=None, description="Detected level of the log"
    )
    filename: Optional[str] = pydantic.Field(
        default=None, description="Filename of the log"
    )
    job: Optional[str] = pydantic.Field(default=None, description="Job of the log")
    service_name: Optional[str] = pydantic.Field(
        default=None, description="Service name of the log"
    )


class LogEntry(BaseModel):
    """Represents a single log entry from Loki"""

    timestamp: str = pydantic.Field(
        default="Unknown timestamp", description="Timestamp of the log"
    )
    log_labels: LogLabels = pydantic.Field(description="Log labels of the log")
    message: str = pydantic.Field(description="Message of the log")

    @pydantic.field_validator("timestamp", mode="before")
    @classmethod
    def convert_datetime_to_str(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v


# RAG Embeddings Model
class RAGEmbedding(SQLModel, table=True):
    """
    Stores RAG embeddings and metadata for knowledge base retrieval.

    This table stores the embeddings generated from knowledge base PDFs,
    along with the complete error metadata needed for RAG queries.
    """

    error_id: str = Field(
        primary_key=True, description="Unique identifier for the error"
    )

    # Embedding vector (stored using pgvector Vector type)
    # Note: pgvector extension must be enabled in PostgreSQL
    # Dimension is 768 for nomic-embed-text-v1.5 model
    embedding: list[float] = Field(
        sa_column=Column(Vector(768)),
        description="Embedding vector (768 dimensions for nomic-embed-text-v1.5)",
    )

    # Error metadata stored as JSONB for flexibility
    error_title: Optional[str] = Field(default=None, description="Title of the error")
    error_metadata: dict = Field(
        default_factory=dict,
        description="Complete error metadata including sections (description, symptoms, resolution, code, benefits) and source information",
        sa_column=Column(JSON),
    )

    # Model information
    model_name: str = Field(description="Name of the embedding model used")
    embedding_dim: int = Field(
        default=768, description="Dimension of the embedding vector"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the embedding was created",
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the embedding was last updated"
    )
