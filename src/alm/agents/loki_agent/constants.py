"""
Constants for Loki agent module.

This module contains business logic constants, configuration values, and thresholds
used throughout the loki_agent module and related tools.
"""

# ==============================================================================
# FILE PATHS
# ==============================================================================

LOKI_AGENT_SYSTEM_PROMPT_PATH = (
    "src/alm/agents/loki_agent/prompts/loki_agent_system_prompt.md"
)
IDENTIFY_MISSING_DATA_PROMPT_PATH = (
    "src/alm/agents/loki_agent/prompts/identify_missing_data.md"
)


# ==============================================================================
# LOGQL QUERY TEMPLATES
# ==============================================================================

LOGQL_FILE_NAME_QUERY_TEMPLATE = '{{filename=~".*{file_name}$"'  # Opening brace only, closing brace added where needed
LOGQL_SERVICE_NAME_WILDCARD_QUERY = '{service_name=~".+"}'
LOGQL_LEVEL_FILTER_TEMPLATE = "| detected_level=`{level}`"
LOGQL_STATUS_FILTER_TEMPLATE = (
    'status=~"{status}"'  # Status is a LABEL (goes in {} selector)
)
LOGQL_LOG_TYPE_FILTER_TEMPLATE = (
    '| log_type=~"{log_type}"'  # log_type is structured metadata (goes after |)
)
LOGQL_TEXT_SEARCH_TEMPLATE = '|= "{text}"'


# ==============================================================================
# TIMESTAMP DETECTION THRESHOLDS
# ==============================================================================

NANOSECOND_THRESHOLD = 1_000_000_000_000_000_000
MILLISECOND_THRESHOLD = 1_000_000_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000


# ==============================================================================
# TIME PARSING
# ==============================================================================

TIME_UNIT_MAP = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}

# Pattern supports both backward (-) and forward (+) relative times
# Examples: "-5m" (5 minutes before), "+10m" (10 minutes after), "1h" (1 hour before)
RELATIVE_TIME_PATTERN = r"([-+]?)(\d+)([smhd])"


# ==============================================================================
# TIME RANGES & WINDOWS
# ==============================================================================

FALLBACK_LOG_SEARCH_DAYS = 30
CONTEXT_WINDOW_DAYS_BEFORE = 25
CONTEXT_WINDOW_MINUTES_AFTER = 2


# ==============================================================================
# TIMESTAMP VALIDATION
# ==============================================================================

VALID_TIMESTAMP_MIN_YEAR = 2000
VALID_TIMESTAMP_MAX_YEAR = 2100


# ==============================================================================
# QUERY LIMITS & DEFAULTS
# ==============================================================================

MAX_LOGS_PER_QUERY = 5000
DEFAULT_START_TIME = "-24h"
DEFAULT_END_TIME = "now"
DEFAULT_LIMIT = 100
DEFAULT_DIRECTION = "backward"
DEFAULT_LINE_ABOVE = 10


# ==============================================================================
# QUERY DIRECTIONS
# ==============================================================================

DIRECTION_BACKWARD = "backward"
DIRECTION_FORWARD = "forward"


# ==============================================================================
# TIMEZONE FORMATTING
# ==============================================================================

UTC_TIMEZONE_SUFFIX = "Z"
UTC_OFFSET_SUFFIX = "+00:00"


# ==============================================================================
# STRING OPERATIONS CONSTANTS
# ==============================================================================

CONTEXT_TRUNCATE_LENGTH = 500
CONTEXT_TRUNCATE_SUFFIX = " [TRUNCATED - LOG MESSAGE CONTINUES]"
LOG_CONTEXT_SEPARATOR_WIDTH = 80
