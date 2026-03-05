"""
LangGraph node functions for Loki MCP integration.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
import json

from alm.agents.loki_agent.constants import (
    IDENTIFY_MISSING_DATA_PROMPT_PATH,
    SUMMARIZE_LOKI_LOGS_PROMPT_PATH,
)
from alm.agents.loki_agent.schemas import (
    IdentifyMissingDataSchema,
    LogSummarizationSchema,
)
from alm.models import LogLabels
from alm.utils.logger import get_logger

logger = get_logger(__name__)


async def identify_missing_data(
    log_summary: str,
    log_labels: LogLabels | Dict[str, Any],
    log_timestamp: str,
    llm: ChatOpenAI,
):
    """
    Identify what critical data is missing to fully understand and resolve the issue.

    Args:
        log_summary: Summary of the log to analyze
        log_labels: Log labels of the log (can be LogLabels object or dict)
        log_timestamp: Timestamp of the log
        llm: ChatOpenAI instance to use for generation

    Returns:
        str: Natural language description of missing data needed for investigation
    """
    with open(IDENTIFY_MISSING_DATA_PROMPT_PATH, "r") as f:
        generate_loki_query_request_user_message = f.read()

    # Convert log_labels to LogLabels object if it's a dict to exclude none values
    if isinstance(log_labels, dict):
        log_labels_obj = LogLabels.model_validate(log_labels)
    else:
        log_labels_obj = log_labels
    log_labels_json = log_labels_obj.model_dump_json(indent=2, exclude_none=True)

    llm_identify_missing_data = llm.with_structured_output(IdentifyMissingDataSchema)
    missing_data_result = await llm_identify_missing_data.ainvoke(
        [
            {
                "role": "system",
                "content": "You are an Ansible expert and helpful assistant specializing in log analysis",
            },
            {
                "role": "user",
                "content": generate_loki_query_request_user_message.replace(
                    "{log_summary}", log_summary
                )
                .replace(
                    "{log_labels}",
                    log_labels_json,
                )
                .replace(
                    "{log_timestamp}",
                    log_timestamp,
                ),
            },
        ]
    )
    return missing_data_result.missing_data_request


async def summarize_loki_logs(
    log_summary: str,
    expert_classification: Optional[str],
    log_labels: LogLabels | Dict[str, Any],
    log_timestamp: str,
    raw_log_context: str,
    llm: ChatOpenAI,
) -> str:
    """
    Summarize retrieved log context using LLM to extract most valuable information
    for root cause analysis.

    This function takes the raw log context retrieved from Loki and uses an LLM
    to create a refined, focused summary that preserves critical details while
    reducing verbosity.

    Args:
        log_summary: Summary of the original triggering log
        expert_classification: Classification of the original triggering log
        log_labels: Log labels of the original triggering log
        log_timestamp: Timestamp of the original triggering log
        raw_log_context: Raw log context string built by build_log_context()
        llm: ChatOpenAI instance to use for summarization

    Returns:
        str: Summarized log context focusing on root cause analysis
    """
    try:
        # Load the summarization prompt
        with open(SUMMARIZE_LOKI_LOGS_PROMPT_PATH, "r") as f:
            summarization_prompt = f.read()

        # Convert log_labels to dict and exclude database_timestamp
        log_labels_dict = log_labels.model_dump(exclude={"database_timestamp"})

        # Convert to JSON for prompt
        log_labels_json = json.dumps(log_labels_dict, indent=2)

        # Prepare the prompt with template replacement
        filled_prompt = (
            summarization_prompt.replace("{log_summary}", log_summary)
            .replace(
                "{expert_classification}", expert_classification or "Not classified"
            )
            .replace("{log_labels}", log_labels_json)
            .replace("{log_timestamp}", log_timestamp)
            .replace("{raw_log_context}", raw_log_context)
        )

        # Get structured output from LLM
        llm_with_structure = llm.with_structured_output(LogSummarizationSchema)

        summarization_result = await llm_with_structure.ainvoke(
            [
                {
                    "role": "system",
                    "content": "You are an expert log analyst specializing in Ansible automation and infrastructure diagnostics. You excel at extracting relevant information from verbose logs while preserving critical details.",
                },
                {"role": "user", "content": filled_prompt},
            ]
        )

        logger.info("Successfully summarized Loki log context")
        return summarization_result.summarized_context

    except Exception as e:
        logger.error(f"Error during log summarization: {e}", exc_info=True)
        logger.warning(
            "Summarization failed - returning empty string. Downstream will use raw context as fallback."
        )
        # Return empty string - downstream has fallback to raw context
        return ""
