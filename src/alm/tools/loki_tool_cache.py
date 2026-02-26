"""
Cache for Loki tool results.

This module provides a lightweight caching mechanism for tool results to prevent
bloating the agent's context with verbose log data. Tools store full LogToolOutput
objects here and return lightweight responses to the agent.
"""

from typing import Dict, Optional
import uuid

from alm.agents.loki_agent.schemas import LightweightToolResponse, LogToolOutput


# ==============================================================================
# TOOL RESULTS CACHE
# ==============================================================================
# Cache for storing full LogToolOutput objects while returning lightweight
# responses to the agent. This prevents bloating agent context with verbose logs.
# This module-level cache is shared across all imports of this module.
_TOOL_RESULTS_CACHE: Dict[str, LogToolOutput] = {}


def _store_tool_result(full_output: LogToolOutput) -> str:
    """
    Store full LogToolOutput in cache and return lightweight response.

    Args:
        full_output: Complete LogToolOutput with all logs

    Returns:
        str: JSON string with LightweightToolResponse containing result_id
    """
    # Generate unique ID for this result
    result_id = str(uuid.uuid4())

    # Store full output in cache
    _TOOL_RESULTS_CACHE[result_id] = full_output

    # Create lightweight response (agent sees this)
    lightweight_response = LightweightToolResponse(
        result_id=result_id,
        status=full_output.status,
        number_of_logs=full_output.number_of_logs,
    )

    return lightweight_response.model_dump_json(indent=2)


def _get_tool_result(result_id: str) -> Optional[LogToolOutput]:
    """
    Retrieve and remove full LogToolOutput from cache.

    Args:
        result_id: UUID string identifying the cached result

    Returns:
        LogToolOutput if found, None otherwise
    """
    return _TOOL_RESULTS_CACHE.pop(result_id, None)
