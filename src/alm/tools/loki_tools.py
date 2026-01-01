"""
LangChain tools for Loki log querying with perfect function matching.
Each tool represents a common log querying pattern with rich descriptions.
"""

import os
import json
from typing import Literal, Optional, List

from langchain_core.tools import tool

from alm.mcp import MCPClient
from alm.agents.loki_agent.constants import (
    DEFAULT_DIRECTION,
    DEFAULT_END_TIME,
    DEFAULT_START_TIME,
    DIRECTION_FORWARD,
    LOGQL_FILE_NAME_QUERY_TEMPLATE,
    LOGQL_SERVICE_NAME_WILDCARD_QUERY,
    LOGQL_STATUS_FILTER_TEMPLATE,
    LOGQL_LOG_TYPE_FILTER_TEMPLATE,
    LOGQL_TEXT_SEARCH_TEMPLATE,
    MAX_LOGS_PER_QUERY,
)
from alm.agents.loki_agent.schemas import (
    DEFAULT_LINE_ABOVE,
    DEFAULT_LIMIT,
    FileLogSchema,
    SearchTextSchema,
    LogLinesAboveSchema,
    PlayRecapSchema,
    LogEntry,
    LogLabels,
    LogToolOutput,
    ToolStatus,
)
from alm.models import LogStatus, LogType
from alm.tools.loki_helpers import escape_logql_string, validate_timestamp
from alm.utils.logger import get_logger

logger = get_logger(__name__)

# MCP Server URL configuration
_mcp_server_url = os.getenv("LOKI_MCP_SERVER_URL")


async def create_mcp_client() -> MCPClient:
    """Create and initialize a new MCP client instance"""
    client = MCPClient(_mcp_server_url)
    await client.__aenter__()
    init_result = await client.initialize()
    if not init_result:
        raise Exception("Failed to initialize MCP session")
    return client


async def execute_loki_query(
    query: str,
    start: str | int = DEFAULT_START_TIME,
    end: str | int = DEFAULT_END_TIME,
    limit: int = DEFAULT_LIMIT,
    reference_timestamp: Optional[str] = None,
    direction: str = DEFAULT_DIRECTION,
) -> str:
    """Execute a LogQL query via MCP client"""
    # Import here to avoid circular dependency
    from alm.tools.loki_helpers import parse_time_input, merge_loki_streams

    client = None
    if limit > MAX_LOGS_PER_QUERY:
        logger.warning(
            "Limit is greater than %d, setting to %d",
            MAX_LOGS_PER_QUERY,
            MAX_LOGS_PER_QUERY,
        )
        limit = MAX_LOGS_PER_QUERY

    try:
        # Create a new MCP client for each query (proper async context management)
        client = await create_mcp_client()

        # Prepare arguments for loki_query tool
        # If it not str its already a timestamp, so we don't need to parse it
        start_parsed = (
            parse_time_input(start, reference_timestamp)
            if isinstance(start, str)
            else start
        )
        end_parsed = (
            parse_time_input(end, reference_timestamp) if isinstance(end, str) else end
        )

        arguments = {
            "query": query,
            "start": start_parsed,
            "end": end_parsed,
            "limit": limit,
            "direction": direction,
            "format": "json",
        }

        logger.debug("Executing MCP query with args: %s", arguments)

        # Call the MCP loki_query tool
        result = await client.call_tool("loki_query", arguments)

        # Parse the result, format should be json as default
        if isinstance(result, str) and result.strip().startswith("{"):
            try:
                parsed_result = json.loads(result)
                logs = []

                # Parse Loki response format and merge streams efficiently
                if "data" in parsed_result and "result" in parsed_result["data"]:
                    # Use heapq.merge to efficiently merge pre-sorted streams
                    # Groups by file (excluding log level) and sorts chronologically
                    logs = merge_loki_streams(
                        parsed_result["data"]["result"], direction=direction
                    )

                # Add helpful message when no logs are found and no message is provided by the tools
                message = parsed_result.get("message", None)
                if len(logs) == 0 and not message:
                    message = "No logs found matching the query. Try using a different search term, simpler keywords, or expanding the time range."

                return LogToolOutput(
                    status=ToolStatus.SUCCESS,
                    message=message,
                    logs=logs,
                    number_of_logs=len(logs),
                    query=query,
                    execution_time_ms=parsed_result.get("stats", {})
                    .get("summary", {})
                    .get("execTime", 0),
                ).model_dump_json(indent=2)
            except json.JSONDecodeError as e:
                logger.error("JSON decode error: %s", e)
                # If not JSON, treat as plain text result
                return LogToolOutput(
                    status=ToolStatus.SUCCESS,
                    logs=[LogEntry(log_labels=LogLabels(), message=result)],
                    number_of_logs=1,
                    query=query,
                ).model_dump_json(indent=2)
        else:
            # Handle non-JSON or error responses
            logger.warning("Non-JSON result: %s", result)
            return LogToolOutput(
                status=ToolStatus.SUCCESS,
                logs=[LogEntry(log_labels=LogLabels(), message=str(result))],
                number_of_logs=1,
                query=query,
            ).model_dump_json(indent=2)

    except Exception as e:
        logger.error("MCP query execution failed: %s", str(e), exc_info=True)
        raise Exception(f"Failed to execute Loki query: {str(e)}")
    finally:
        # Clean up the client
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass


@tool(args_schema=FileLogSchema)
async def get_logs_by_file_name(
    file_name: str,
    log_timestamp: Optional[str] = None,
    start_time: str | int = DEFAULT_START_TIME,
    end_time: str = DEFAULT_END_TIME,
    status_list: Optional[List[LogStatus]] = None,
    log_type_list: Optional[List[LogType]] = None,
    limit: int = DEFAULT_LIMIT,
    direction: Literal["backward", "forward"] = DEFAULT_DIRECTION,
) -> str:
    """
    Get logs for a specific file with time ranges relative to a reference timestamp,
    optionally filtered by status and log type.

    Perfect for queries like:
    - "show me logs from nginx.log 5 minutes before this error"
    - "get failed and fatal logs from job_12345.txt between 1 hour before and 10 minutes before this timestamp"
    - "show me play and recap logs from job_12345.txt"

    Time range examples:
    - start_time="-1h", end_time="-10m": 1 hour before to 10 minutes before the timestamp
    - start_time="now", end_time="+1h": from the timestamp to 1 hour after

    Note: Status only applies to TASK log_type. Other log types have empty status values.
    """
    try:
        # Build base selector with filename
        selector_parts = [LOGQL_FILE_NAME_QUERY_TEMPLATE.format(file_name=file_name)]

        # Convert status list to pipe-separated string and add to selector
        # Status is a LABEL, so it goes in the {} selector
        if status_list:
            status_str = "|".join([s.value for s in status_list])
            selector_parts.append(
                ", " + LOGQL_STATUS_FILTER_TEMPLATE.format(status=status_str)
            )

        # Close the selector
        query_parts = ["".join(selector_parts) + "}"]

        # Convert log_type list to pipe-separated string and add as metadata filter
        # log_type is STRUCTURED METADATA, so it goes after |
        if log_type_list:
            log_type_str = "|".join([lt.value for lt in log_type_list])
            query_parts.append(
                LOGQL_LOG_TYPE_FILTER_TEMPLATE.format(log_type=log_type_str)
            )

        query = "".join(query_parts)

        result = await execute_loki_query(
            query, start_time, end_time, limit, log_timestamp, direction
        )
        return result

    except Exception as e:
        logger.error("Error in get_logs_by_file_name: %s", e)
        output = LogToolOutput(
            status=ToolStatus.ERROR, message=str(e), number_of_logs=0, logs=[]
        )
        return output.model_dump_json(indent=2)


@tool(args_schema=SearchTextSchema)
async def search_logs_by_text(
    text: str,
    log_timestamp: Optional[str] = None,
    start_time: str | int = DEFAULT_START_TIME,
    end_time: str | int = DEFAULT_END_TIME,
    file_name: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    Search for logs containing specific text with time ranges relative to a reference timestamp,
    across all logs or in a specific file.

    Perfect for queries like:
    - "find logs containing 'timeout' 5 minutes before this error"
    - "search for 'user login' between 1 hour before and 30 minutes before this timestamp"
    - "show logs with 'database connection' around this time"
    - "find errors in the 10 minutes after this event"

    Time range examples:
    - start_time="-30m", end_time="now": 30 minutes before to the timestamp
    - start_time="-1h", end_time="-10m": 1 hour before to 10 minutes before the timestamp
    - start_time="now", end_time="+15m": from the timestamp to 15 minutes after

    Note: This is a case-sensitive text search using LogQL's |= operator.
    """
    try:
        # Escape special characters in search text for LogQL
        escaped_text = escape_logql_string(text)

        # Build LogQL query for text search
        if file_name:
            # Search within a specific file
            query = (
                LOGQL_FILE_NAME_QUERY_TEMPLATE.format(file_name=file_name)
                + "}"
                + " "
                + LOGQL_TEXT_SEARCH_TEMPLATE.format(text=escaped_text)
            )
        else:
            # Search across all logs
            # Use service_name=~".+" to match any service with non-empty value (Loki requirement)
            query = (
                LOGQL_SERVICE_NAME_WILDCARD_QUERY
                + " "
                + LOGQL_TEXT_SEARCH_TEMPLATE.format(text=escaped_text)
            )

        result = await execute_loki_query(
            query, start_time, end_time, limit, log_timestamp
        )
        return result

    except Exception as e:
        logger.error("Error in search_logs_by_text: %s", e)
        output = LogToolOutput(
            status=ToolStatus.ERROR, message=str(e), number_of_logs=0, logs=[]
        )
        return output.model_dump_json(indent=2)


@tool(args_schema=PlayRecapSchema)
async def get_play_recap(
    file_name: str,
    log_timestamp: str,
    buffer_time: str = "6h",
) -> str:
    """
    This tool searches forward in time from a target timestamp to find the first
    PLAY RECAP entry, which shows the results of Ansible playbook execution. Useful for
    determining the outcome of a playbook run after encountering an error.

    Perfect for queries like:
    - "show me the play recap after this error timestamp"
    - "get the playbook result for the task that failed at this time"
    - "give me an overview of the tasks in this playbook"

    buffer_time example:
    - "12h" or "+12h": search 12 hours forward
    """
    try:
        # Build LogQL query using regex suffix match
        query = (
            LOGQL_FILE_NAME_QUERY_TEMPLATE.format(file_name=file_name)
            + "}"
            + ' | log_type="recap"'
        )

        if buffer_time.startswith("-"):
            logger.warning("Buffer time starts with '-', setting to default of 6h")
            buffer_time = "6h"

        # Execute query with forward direction and limit=1 to get only the NEXT recap
        result = await execute_loki_query(
            query=query,
            start=log_timestamp,  # Start from the error timestamp
            end=buffer_time,  # Forward time buffer (e.g., "24h")
            limit=1,  # Get only the first/next recap
            reference_timestamp=log_timestamp,  # For relative time calculation
            direction=DIRECTION_FORWARD,  # Search forward in time
        )

        return result

    except Exception as e:
        logger.error("Error in get_play_recap: %s", e)
        output = LogToolOutput(
            status=ToolStatus.ERROR, message=str(e), number_of_logs=0, logs=[]
        )
        return output.model_dump_json(indent=2)


def create_log_lines_above_tool(
    file_name: str,
    log_message: str,
    log_timestamp: str,
):
    """
    Factory function to create get_log_lines_above tool with bound context values.

    This uses Python closures to capture file_name, log_message, and log_timestamp,
    avoiding the need for LLM JSON serialization of these constant values.
    Especially useful for passing complex log messages to the tool, avoiding JSON serialization issues.

    Args:
        file_name: The log file name to search in
        log_message: The target log message to find context for
        log_timestamp: The timestamp of the target log

    Returns:
        A LangChain tool with the context values bound via closure
    """

    @tool(args_schema=LogLinesAboveSchema)
    async def get_log_lines_above(lines_above: int = DEFAULT_LINE_ABOVE) -> str:
        """
        Get log lines that occurred before/above a specific log line in a file.

        This tool has log context (file_name, log_message, log_timestamp) bound
        via closure at creation time. The LLM only needs to specify how many lines
        to retrieve.

        This tool uses a time window approach to retrieve context lines:
        1. Uses the bound timestamp, or finds the target log line to get its timestamp
        2. Queries a wide time window (target - 25 days to target + 2 minutes)
        3. Fetches up to 5000 logs to ensure we have enough context
        4. Filters client-side to extract N lines before the target

        The +10 minute buffer handles cases where Loki ignores fractional seconds
        and multiple logs have the same timestamp.

        Args:
            lines_above: Number of lines to retrieve before the target log (default: 10)

        Perfect for queries like:
        - "get 10 lines above this error"
        - "show me 5 lines before this failure"
        - "get context lines above this specific log entry"
        """
        try:
            # Import helper functions
            from alm.tools.log_lines_context_helpers import (
                calculate_time_window,
                query_logs_in_time_window,
                extract_context_lines_above,
            )
            from alm.agents.loki_agent.constants import CONTEXT_TRUNCATE_SUFFIX

            # Use closure-captured values
            logger.debug("get_log_lines_above invoked with closure-bound context:")
            logger.debug("  - file_name: %s", file_name)
            logger.debug(
                "  - log_message: %s",
                f"{log_message[:100]}..."
                if log_message and len(log_message) > 100
                else log_message,
            )
            logger.debug("  - log_timestamp: %s", log_timestamp)
            logger.debug("  - lines_above: %d", lines_above)

            # Process the log message
            processed_log_message = log_message
            # Truncate the log message at the end only if it ends with the truncate suffix
            if processed_log_message and processed_log_message.endswith(
                CONTEXT_TRUNCATE_SUFFIX
            ):
                processed_log_message = processed_log_message[
                    : -len(CONTEXT_TRUNCATE_SUFFIX)
                ].rstrip()

            # Step 1: Validate and convert the timestamp to a datetime object
            logger.debug(
                "[Step 1] Validating and converting timestamp to datetime object"
            )
            target_datetime, is_valid = validate_timestamp(log_timestamp)
            if not is_valid or not target_datetime:
                return LogToolOutput(
                    status=ToolStatus.ERROR,
                    message=f"Invalid timestamp: {log_timestamp}, please provide a valid timestamp",
                    number_of_logs=0,
                    logs=[],
                ).model_dump_json(indent=2)

            # Step 2: Calculate time window
            logger.debug("[Step 2] Calculating time window around timestamp")
            start_time_rfc3339, end_time_rfc3339 = calculate_time_window(
                target_datetime
            )

            # Step 3: Query logs in the time window
            logger.debug("[Step 3] Querying large context window (limit=5000)")
            context_data, error = await query_logs_in_time_window(
                file_name, start_time_rfc3339, end_time_rfc3339
            )
            if error or not isinstance(context_data, LogToolOutput):
                return LogToolOutput(
                    status=ToolStatus.ERROR,
                    message=f"Failed to query logs in time window. Error: {error}, Context data: {context_data}",
                    number_of_logs=0,
                    logs=[],
                ).model_dump_json(indent=2)

            # Step 4: Extract N lines before the target
            logger.debug("[Step 4] Extracting %d lines before target", lines_above)

            context_logs, error = extract_context_lines_above(
                context_data.logs, processed_log_message, lines_above
            )

            if error:
                return LogToolOutput(
                    status=ToolStatus.ERROR,
                    message=error,
                    query=context_data.query,
                    number_of_logs=0,
                    logs=[],
                ).model_dump_json(indent=2)

            logger.debug(
                "Successfully extracted %d logs (including target)", len(context_logs)
            )
            logger.debug(
                "Requested: %d lines above, Got: %d lines above + target",
                lines_above,
                len(context_logs) - 1,
            )

            # Step 5: Return the context logs
            return LogToolOutput(
                status=ToolStatus.SUCCESS,
                message=f"Retrieved {len(context_logs) - 1} lines above the target log (total {len(context_logs)} logs including target)",
                query=context_data.query,
                number_of_logs=len(context_logs),
                logs=context_logs,
                execution_time_ms=context_data.execution_time_ms,
            ).model_dump_json(indent=2)

        except Exception as e:
            logger.error("Error in get_log_lines_above: %s", e, exc_info=True)

            return LogToolOutput(
                status=ToolStatus.ERROR, message=str(e), number_of_logs=0, logs=[]
            ).model_dump_json(indent=2)

    return get_log_lines_above


# List of static tools (tools that don't need closure-bound context)
# get_log_lines_above is created dynamically via create_log_lines_above_tool()
# TODO: Add fallback_query tool
LOKI_STATIC_TOOLS = [
    get_logs_by_file_name,
    search_logs_by_text,
    get_play_recap,
]
