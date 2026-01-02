"""
Loki agent schemas for inputs and outputs.
"""

from alm.agents.loki_agent.constants import (
    DEFAULT_START_TIME,
    DEFAULT_END_TIME,
    DEFAULT_LIMIT,
    DEFAULT_DIRECTION,
    DEFAULT_LINE_ABOVE,
)
from alm.agents.loki_agent.schemas.inputs import (
    FileLogSchema,
    SearchTextSchema,
    LogLinesAboveSchema,
    PlayRecapSchema,
)
from alm.agents.loki_agent.schemas.outputs import (
    ToolStatus,
    LogToolOutput,
    LokiAgentOutput,
    IdentifyMissingDataSchema,
)
from alm.models import LogLabels, LogEntry, DetectedLevel

__all__ = [
    # Inputs
    "DetectedLevel",
    "FileLogSchema",
    "SearchTextSchema",
    "LogLinesAboveSchema",
    "PlayRecapSchema",
    "DEFAULT_START_TIME",
    "DEFAULT_END_TIME",
    "DEFAULT_LIMIT",
    "DEFAULT_DIRECTION",
    "DEFAULT_LINE_ABOVE",
    # Outputs
    "ToolStatus",
    "LogLabels",
    "LogEntry",
    "LogToolOutput",
    "LokiAgentOutput",
    "IdentifyMissingDataSchema",
]
