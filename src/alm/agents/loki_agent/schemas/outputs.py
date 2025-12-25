"""
Pydantic schemas for Loki tool outputs.
These schemas define the output structure for each tool to ensure consistency.
"""

from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime
from collections import defaultdict
from langchain_core.messages import ToolMessage

from alm.agents.loki_agent.constants import (
    LOG_CONTEXT_SEPARATOR_WIDTH,
    NANOSECONDS_PER_SECOND,
)
from alm.models import LogEntry, DetectedLevel
from alm.utils.logger import get_logger

logger = get_logger(__name__)


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class LogToolOutput(BaseModel):
    """Output schema for tools that retrieve logs"""

    status: ToolStatus = Field(description="Status of the operation")
    message: Optional[str] = Field(
        default=None, description="Human-readable message, especially for errors"
    )
    query: Optional[str] = Field(
        default=None, description="The LogQL query that was executed"
    )
    execution_time_ms: Optional[int] = Field(
        default=None, description="Query execution time in milliseconds"
    )
    logs: List[LogEntry] = Field(
        default_factory=list, description="List of log entries retrieved"
    )
    number_of_logs: int = Field(
        default=0, description="Total number of log entries returned"
    )

    def build_context(self) -> str:
        """
        Build a context for the step by step solution from the log entries.
        Groups logs by stream and sorts them by timestamp.
        """
        return build_log_context(self.logs)


class LokiAgentOutput(BaseModel):
    """Output schema for the Loki agent"""

    user_request: str = Field(description="User request that was processed")
    status: ToolStatus = Field(description="Status of the operation")
    message: Optional[str] = Field(
        default=None, description="Human-readable message, especially for errors"
    )
    agent_result: LogToolOutput = Field(description="Result of the agent")
    raw_output: str | Any = Field(description="Raw output of the agent")
    tool_messages: List[ToolMessage] = Field(
        default_factory=list, description="Tool messages from the agent execution"
    )


class IdentifyMissingDataSchema(BaseModel):
    missing_data_request: str = Field(
        description="Natural language description of what data/context is missing to fully understand and resolve the issue"
    )


# Helper functions for log context building


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object for sorting"""
    try:
        # Try nanosecond timestamp (common in Loki)
        if timestamp_str.isdigit():
            # Convert nanoseconds to seconds
            timestamp_seconds = int(timestamp_str) / NANOSECONDS_PER_SECOND
            return datetime.fromtimestamp(timestamp_seconds)

        # Try ISO format
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        # Return epoch if parsing fails
        return datetime.fromtimestamp(0)


def build_log_context(logs: List["LogEntry"]) -> str:
    """
    Build a context for the step by step solution from the log entries.
    Groups logs by labels (excluding log level) with inline log level display.
    Preserves chronological order from Loki streams.
    """
    if not logs:
        logger.warning("No logs found to build context from.")
        return ""

    # Group logs by labels (excluding detected_level to keep all logs from same file together)
    logs_by_labels = defaultdict(list)
    for log in logs:
        # Convert labels dict to a string key for grouping, excluding detected_level
        labels_dict = log.log_labels.model_dump(exclude_none=True)
        # Remove detected_level from grouping key
        labels_dict.pop("detected_level", None)
        labels_key = ", ".join([f"{k}={v}" for k, v in sorted(labels_dict.items())])
        logs_by_labels[labels_key].append(log)

    # Build context with grouped logs (preserving natural order from Loki)
    context_parts = []

    for labels_key, label_logs in logs_by_labels.items():
        # Logs are already merged and sorted chronologically per file (oldest to newest)
        # by merge_loki_streams in execute_loki_query
        # Add labels header
        context_parts.append(f"\n{'=' * LOG_CONTEXT_SEPARATOR_WIDTH}")
        context_parts.append(f"Labels: {labels_key}")
        context_parts.append(f"{'=' * LOG_CONTEXT_SEPARATOR_WIDTH}")

        # Add logs for this label group
        for log in label_logs:
            # Add log level inline if available
            log_level = (
                log.log_labels.detected_level.value.upper()
                if log.log_labels.detected_level
                else DetectedLevel.UNKNOWN.value.upper()
            )
            # Abandon timestamp since it's the generated timestamp (ingested timestamp) and used only for loki query,
            # the real timestamp is in the log message itself
            context_parts.append(f"{log_level} - {log.message}")

    return "\n".join(context_parts)
