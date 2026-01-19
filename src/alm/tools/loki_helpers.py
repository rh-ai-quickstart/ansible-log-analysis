"""
Helper functions for Loki log querying tools.
"""

import heapq
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict
from dateutil import parser as date_parser

from alm.agents.loki_agent.constants import (
    DEFAULT_DIRECTION,
    DIRECTION_BACKWARD,
    FALLBACK_LOG_SEARCH_DAYS,
    MILLISECOND_THRESHOLD,
    MILLISECONDS_PER_SECOND,
    NANOSECOND_THRESHOLD,
    NANOSECONDS_PER_SECOND,
    RELATIVE_TIME_PATTERN,
    TIME_UNIT_MAP,
    UTC_OFFSET_SUFFIX,
    UTC_TIMEZONE_SUFFIX,
    VALID_TIMESTAMP_MAX_YEAR,
    VALID_TIMESTAMP_MIN_YEAR,
)
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def timestamp_to_utc_datetime(timestamp: str) -> datetime:
    """
    Convert timestamp (any format) to UTC-aware datetime.

    Supports:
    - Nanoseconds (19+ digits)
    - Milliseconds (13+ digits)
    - Seconds (10 digits or less)
    - ISO format strings

    Args:
        timestamp: Timestamp string in any supported format

    Returns:
        UTC-aware datetime object

    Raises:
        ValueError: If timestamp format is invalid
    """
    if timestamp.isdigit():
        ts_int = int(timestamp)

        # Detect format based on number of digits
        if ts_int > NANOSECOND_THRESHOLD:  # 19+ digits = nanoseconds
            ts_seconds = ts_int / NANOSECONDS_PER_SECOND
        elif ts_int > MILLISECOND_THRESHOLD:  # 13+ digits = milliseconds
            ts_seconds = ts_int / MILLISECONDS_PER_SECOND
        else:  # 10 digits or less = seconds
            ts_seconds = float(ts_int)

        # Create UTC-aware datetime
        return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
    else:
        # ISO format or other parseable format
        dt = date_parser.parse(timestamp)

        # Ensure it's UTC-aware
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            dt = dt.astimezone(timezone.utc)

        return dt


def format_rfc3339_utc(dt: datetime) -> str:
    """
    Format UTC datetime as RFC3339 with Z suffix.

    Output format: YYYY-MM-DDTHH:MM:SS.ffffffZ
    Note: Python datetime only supports microsecond precision (6 digits),
    not nanosecond precision (9 digits).

    Args:
        dt: UTC-aware datetime object

    Returns:
        RFC3339 formatted string with Z suffix
    """
    # Format with microseconds and Z suffix
    # isoformat() gives us YYYY-MM-DDTHH:MM:SS.ffffff+00:00
    # We want YYYY-MM-DDTHH:MM:SS.ffffffZ
    iso_str = dt.isoformat()

    # Replace timezone offset with Z
    if iso_str.endswith(UTC_OFFSET_SUFFIX):
        return iso_str[:-6] + UTC_TIMEZONE_SUFFIX
    elif iso_str.endswith(UTC_TIMEZONE_SUFFIX):
        return iso_str
    else:
        # Should not happen if dt is UTC, but handle it
        return dt.astimezone(timezone.utc).isoformat()[:-6] + UTC_TIMEZONE_SUFFIX


def parse_relative_offset(time_str: str) -> timedelta:
    """
    Parse relative time string like '-5m', '+10m', '2h', '-1d' into timedelta.

    Supports both backward (-) and forward (+) relative times:
    - '-5m' or '5m': 5 minutes backward (before reference)
    - '+10m': 10 minutes forward (after reference)

    Args:
        time_str: Relative time string (e.g., "-5m", "+10m", "2h", "-1d")

    Returns:
        timedelta object representing the offset

    Raises:
        ValueError: If the time string format is invalid
    """
    match = re.match(RELATIVE_TIME_PATTERN, time_str.strip())
    if not match:
        raise ValueError(f"Invalid relative time format: {time_str}")

    sign, value, unit = match.groups()
    value = int(value)

    # Handle sign: '-' means negative (backward), '+' or empty means positive (forward)
    if sign == "-":
        value = -value
    # '+' or no sign means positive (forward in time)

    return timedelta(**{TIME_UNIT_MAP[unit]: value})


def parse_time_relative_to_timestamp(time_str: str, reference_timestamp: str) -> str:
    """
    Parse relative time string based on a reference timestamp.

    Args:
        time_str: Relative time string (e.g., "-5m", "2h")
        reference_timestamp: Reference timestamp (milliseconds, nanoseconds, or ISO format)

    Returns:
        RFC3339 UTC formatted string with Z suffix

    Raises:
        ValueError: If timestamp or time format is invalid
    """
    # Convert reference timestamp to UTC datetime
    ref_datetime = timestamp_to_utc_datetime(reference_timestamp)

    # Parse the relative offset and calculate result
    offset = parse_relative_offset(time_str)
    result_datetime = ref_datetime + offset

    # Format as RFC3339 with Z
    return format_rfc3339_utc(result_datetime)


def parse_time_absolute(time_str: str) -> str:
    """
    Parse absolute time string into RFC3339 UTC format.

    Args:
        time_str: Absolute time string (ISO format, human-readable, etc.)

    Returns:
        RFC3339 UTC formatted string with Z suffix, or original string if parsing fails
    """
    try:
        dt = date_parser.parse(time_str)

        # Ensure it's UTC-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return format_rfc3339_utc(dt)
    except Exception:
        # Return as-is and let Loki handle it
        return time_str


def parse_time_input(time_str: str, reference_timestamp: Optional[str] = None) -> str:
    """
    Parse various time input formats into Loki-compatible format.

    When reference_timestamp is provided, all relative times are calculated from it
    and returned as RFC3339 UTC format with Z suffix.
    When reference_timestamp is None, relative times are returned as-is for Loki.

    Args:
        time_str: Time string to parse (relative like "-5m" or absolute like ISO format)
        reference_timestamp: Optional reference timestamp for relative calculations.
                           If None, relative times are passed as-is to Loki.

    Returns:
        Loki-compatible time string (RFC3339 UTC with Z or relative format like "-5m")

    Examples:
        parse_time_input("-5m", "1762414393000000000") → "2025-05-01T14:33:13.000000Z"
        parse_time_input("-5m", None) → "-5m"
        parse_time_input("now", "1762414393000000000") → "2025-05-01T14:38:13.000000Z"
        parse_time_input("now", None) → "now"
        parse_time_input("2024-01-01T10:00:00", ...) → "2024-01-01T10:00:00.000000Z"
    """
    # Validate reference_timestamp if provided
    ref_datetime, is_valid_timestamp = validate_timestamp(reference_timestamp)
    if reference_timestamp and not is_valid_timestamp:
        logger.warning(
            "Invalid reference timestamp '%s'. Treating relative times as relative to 'now' instead.",
            reference_timestamp,
        )
        reference_timestamp = None

    # Handle "now"
    if not time_str or time_str.lower() == "now":
        if ref_datetime:
            # "now" relative to reference timestamp = the reference timestamp itself + small delta to avoid exact match
            return format_rfc3339_utc(ref_datetime + timedelta(seconds=1))
        else:
            # No reference timestamp: pass "now" to Loki
            return "now"

    # Handle relative times like "2h ago", "30m ago", "1d ago"
    if "ago" in time_str.lower():
        time_str = f"-{time_str.replace('ago', '').strip()}"

    # Handle direct relative times like "2h", "30m", "1d", "-5m"
    if any(unit in time_str for unit in ["h", "m", "s", "d"]):
        if reference_timestamp:
            # Calculate relative to reference timestamp
            try:
                return parse_time_relative_to_timestamp(time_str, reference_timestamp)
            except Exception as e:
                logger.warning(
                    "Failed to parse relative time '%s' with reference timestamp: %s",
                    time_str,
                    e,
                )
                # Fallback: return as-is for Loki
                return time_str
        else:
            # No reference timestamp: return as-is for Loki (relative to "now")
            return time_str

    # Try to parse as absolute datetime
    return parse_time_absolute(time_str)


async def find_log_timestamp(
    file_name: str, log_message: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the timestamp of a target log message by searching for it.

    This is a fallback function used when timestamp is not provided or invalid.

    Args:
        file_name: The log file to search in
        log_message: The log message to find

    Returns:
        Tuple of (timestamp, error_message)
        - timestamp: The timestamp string in nanoseconds if found, None otherwise
        - error_message: Error description if not found, None otherwise
    """
    from alm.agents.loki_agent.schemas import LogToolOutput
    from alm.tools import search_logs_by_text

    logger.debug(
        "[find_log_timestamp] Searching for target log message in %s", file_name
    )

    # Use current time for the search (past N days)
    current_time = datetime.now(timezone.utc)
    start_time = current_time - timedelta(
        days=FALLBACK_LOG_SEARCH_DAYS
    )  # Max time range allowed by Loki

    target_result = await search_logs_by_text.ainvoke(
        {
            "text": log_message,
            "file_name": file_name,
            "log_timestamp": str(
                int(current_time.timestamp() * NANOSECONDS_PER_SECOND)
            ),  # Current time in nanoseconds
            "start_time": format_rfc3339_utc(start_time),  # RFC3339 UTC format
            "end_time": format_rfc3339_utc(current_time),  # RFC3339 UTC format
            "limit": 1,
        }
    )
    target_result = LogToolOutput.model_validate_json(target_result)

    if not target_result.logs:
        error_msg = f"Log message '{log_message}' not found in file '{file_name}'"
        return None, error_msg

    # Get the timestamp of the target log line
    target_log = target_result.logs[0]
    target_timestamp_raw = target_log.timestamp

    logger.debug("Target log found with timestamp: %s", target_timestamp_raw)
    return target_timestamp_raw, None


def validate_timestamp(timestamp: Optional[str]) -> Tuple[Optional[datetime], bool]:
    """
    Validate if a timestamp string is valid (milliseconds, nanoseconds, or ISO format).

    Args:
        timestamp: Timestamp string to validate

    Returns:
        Tuple of (UTC-aware datetime object if valid, None otherwise, True if valid, False otherwise)
    """
    if not timestamp:
        return None, False

    try:
        # Try to convert to datetime - if it works, it's valid
        dt = timestamp_to_utc_datetime(timestamp)
        # Check if the timestamp is in a reasonable range
        if (
            datetime(VALID_TIMESTAMP_MIN_YEAR, 1, 1, tzinfo=timezone.utc)
            < dt
            < datetime(VALID_TIMESTAMP_MAX_YEAR, 1, 1, tzinfo=timezone.utc)
        ):
            return dt, True
        else:
            return None, False
    except Exception:
        return None, False


def merge_loki_streams(streams: List[Dict], direction: str = DEFAULT_DIRECTION) -> List:
    """
    Merge multiple Loki streams into a sorted list of LogEntry objects.

    Streams are grouped by labels (excluding detected_level), then merged using
    heapq.merge for efficient O(n log k) performance where k is the number of
    streams per file group (constant in our case).

    Each file's logs are sorted chronologically (oldest to newest), but different
    files can be interleaved in the result.

    Args:
        streams: List of Loki stream objects, each containing:
                 - "stream": Dict of labels (detected_level, filename, service_name, cluster_name, etc.)
                 - "values": List of [timestamp, message] pairs
        direction: Loki query direction:
                   - "backward": Streams contain newest-first logs (will be reversed)
                   - "forward": Streams contain oldest-first logs (used as-is)

    Returns:
        List of LogEntry objects, sorted chronologically per file (oldest to newest)
    """
    from alm.models import LogEntry, LogLabels

    if not streams:
        return []

    # Group streams by labels (excluding detected_level)
    # Key: tuple of (label_key, label_value) pairs, excluding detected_level
    # Value: list of streams with those labels
    streams_by_file: Dict[str, List[Dict]] = defaultdict(list)

    for stream in streams:
        stream_labels = stream.get("stream", {})

        # Create grouping key excluding detected_level
        labels_dict = {k: v for k, v in stream_labels.items() if k != "detected_level"}
        labels_key = ", ".join([f"{k}={v}" for k, v in sorted(labels_dict.items())])

        streams_by_file[labels_key].append(stream)

    # Merge each file group's streams and concatenate results
    all_logs = []

    for labels_key, file_streams in streams_by_file.items():
        # Convert each stream to LogEntry iterator
        def stream_to_log_entries(stream_data: Dict):
            """Convert a Loki stream to LogEntry objects, handling direction"""
            stream_labels = stream_data.get("stream", {})
            values = stream_data.get("values", [])

            # Extract real_timestamp from structured metadata (if available)
            real_timestamp = stream_labels.pop("real_timestamp", None)

            # If direction is backward, reverse the values to get oldest-first
            if direction == DIRECTION_BACKWARD:
                values = reversed(values)

            # Yield LogEntry objects
            for entry in values:
                # Set database_timestamp to Loki's ingestion timestamp
                stream_labels["database_timestamp"] = entry[0]

                yield LogEntry(
                    timestamp=real_timestamp,  # Real log timestamp from content
                    log_labels=LogLabels(**stream_labels),
                    message=entry[1],
                )

        # Create iterators for each stream in this file group
        stream_iterators = [stream_to_log_entries(s) for s in file_streams]

        # Merge streams chronologically (oldest to newest)
        # heapq.merge expects sorted iterables and merges them efficiently
        merged_logs = heapq.merge(
            *stream_iterators,
            key=lambda log: log.log_labels.database_timestamp.timestamp(),  # Sort by database timestamp as float
        )

        # Extend all_logs with this file's merged logs
        all_logs.extend(merged_logs)

    return all_logs


def escape_logql_string(text: str) -> str:
    r"""
    Escape special characters in text for use in LogQL string literals.

    LogQL string literals (used with |= operator) require escaping of:
    - Double quotes (") -> \"
    - Backslashes (\) -> \\

    Args:
        text: The text to escape

    Returns:
        Escaped text safe for use in LogQL string literals
    """
    # Escape backslashes first (must be done before quotes)
    text = text.replace("\\", "\\\\")
    # Escape double quotes
    text = text.replace('"', '\\"')
    return text
