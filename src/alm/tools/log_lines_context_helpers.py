"""
Helper functions specifically for the get_log_lines_above tool.

These functions implement the step-by-step logic for retrieving context lines
before a specific log entry using a time window approach.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

from alm.agents.loki_agent.constants import (
    CONTEXT_WINDOW_DAYS_BEFORE,
    CONTEXT_WINDOW_MINUTES_AFTER,
    DIRECTION_BACKWARD,
    MAX_LOGS_PER_QUERY,
)
from alm.agents.loki_agent.schemas import (
    LogToolOutput,
    ToolStatus,
    LightweightToolResponse,
)
from alm.tools.loki_tool_cache import _get_tool_result
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_time_window(
    target_datetime: datetime,
) -> Tuple[str, str]:
    """
    Calculate the time window for querying around the target datetime.

    Args:
        target_datetime: The target datetime (UTC-aware datetime object)

    Returns:
        Tuple of (start_time_rfc3339, end_time_rfc3339)
        - start_time_rfc3339: Start time in RFC3339 UTC format (25 days before target)
        - end_time_rfc3339: End time in RFC3339 UTC format (2 minutes after target)
    """
    from alm.tools.loki_helpers import format_rfc3339_utc

    # Calculate time window: N days before to M minutes after target
    # This ensures we capture the file start and handle fractional second issues
    start_datetime = target_datetime - timedelta(days=CONTEXT_WINDOW_DAYS_BEFORE)
    end_datetime = target_datetime + timedelta(minutes=CONTEXT_WINDOW_MINUTES_AFTER)

    # Format as RFC3339 with Z
    start_time_rfc3339 = format_rfc3339_utc(start_datetime)
    end_time_rfc3339 = format_rfc3339_utc(end_datetime)

    logger.debug("Time window: %s to %s", start_time_rfc3339, end_time_rfc3339)

    return start_time_rfc3339, end_time_rfc3339


async def query_logs_in_time_window(
    file_name: str, start_time_rfc3339: str, end_time_rfc3339: str
) -> Tuple[Optional[LogToolOutput], Optional[str]]:
    """
    Query logs within the specified time window.

    Args:
        file_name: File name to query
        start_time_rfc3339: Start time in RFC3339 UTC format
        end_time_rfc3339: End time in RFC3339 UTC format

    Returns:
        Tuple of (log_output, error_message)
        - log_output: LogToolOutput with the fetched logs if successful
        - error_message: Error description if query fails or returns no logs, None otherwise
    """
    from alm.tools.loki_tools import get_logs_by_file_name

    context_query = {
        "file_name": file_name,
        "start_time": start_time_rfc3339,
        "end_time": end_time_rfc3339,
        "limit": MAX_LOGS_PER_QUERY,  # Max allowed by Loki
        "direction": DIRECTION_BACKWARD,  # Get most recent logs in the window
    }

    # Tool now returns lightweight response, need to get full result from cache
    context_result = await get_logs_by_file_name.ainvoke(context_query)
    lightweight_response = LightweightToolResponse.model_validate_json(context_result)
    context_data = _get_tool_result(lightweight_response.result_id)

    # Handle cache miss
    if context_data is None:
        error_msg = f"Cache miss for result_id {lightweight_response.result_id} in query_logs_in_time_window"
        logger.error(error_msg)
        return None, error_msg

    if context_data.status != ToolStatus.SUCCESS.value or not context_data.logs:
        error_msg = f"Failed to retrieve context logs, Status: {context_data.status}, Logs: {context_data.logs}"
        return None, error_msg

    logger.debug("Fetched %d logs from Loki", len(context_data.logs))
    return context_data, None


def extract_context_lines_above(
    all_logs: list, target_message: str, lines_above: int
) -> Tuple[list, Optional[str]]:
    """
    Extract N lines before the target message from a list of logs.

    Args:
        all_logs: List of LogEntry objects (should be sorted chronologically)
        target_message: The log message to find
        lines_above: Number of lines to return before the target

    Returns:
        Tuple of (context_logs, error_message)
        - context_logs: List containing N lines before target + target itself
        - error_message: Error description if target not found, None otherwise
    """
    # Find the target log in the list
    target_idx = None
    for i, log in enumerate(all_logs):
        if target_message in log.message:
            target_idx = i
            break
    logger.debug("Target log message found at index: %s", target_idx)

    if target_idx is None:
        return [], f"Target log message not found in the {len(all_logs)} fetched logs"

    # Calculate the range of logs to return
    # We want N lines BEFORE the target, plus the target itself
    start_idx = max(0, target_idx - lines_above)
    end_idx = target_idx + 1  # +1 to include the target line

    logger.debug("Start index: %d, End index: %d", start_idx, end_idx)

    context_logs = all_logs[start_idx:end_idx]

    logger.debug("Context logs length: %d", len(context_logs))

    return context_logs, None
