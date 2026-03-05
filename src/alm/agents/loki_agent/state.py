"""
State definition for the Loki agent subgraph.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from alm.agents.get_more_context_agent.node import LokiRouterSchema
from alm.models import LogEntry


class LokiAgentState(BaseModel):
    """
    State for the Loki log context retrieval subgraph.

    This state contains the minimal fields needed for the Loki agent to:
    1. Identify what log data is missing
    2. Execute Loki queries to retrieve additional context
    3. Return the retrieved context
    """

    # Input fields (required to start the subgraph)
    log_entry: LogEntry = Field(description="The log entry that triggered the alert")
    log_summary: str = Field(description="The summary of Ansible error log")
    expert_classification: Optional[str] = Field(
        default=None, description="Classification of the log message"
    )
    cheat_sheet_context: Optional[str] = Field(
        description="The context from the cheat sheet that will help understand the log error.",
        default=None,
    )
    loki_router_result: Optional[LokiRouterSchema] = Field(
        description="The result from the loki router that will help understand the log error.",
        default=None,
    )

    # Intermediate fields (populated during subgraph execution)
    loki_user_request: Optional[str] = Field(
        default="", description="User's natural language request for additional logs"
    )

    # Output fields (populated at the end of subgraph execution)
    loki_query_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Dictionary representation of LokiAgentOutput object"
    )
    additional_context_from_loki: Optional[str] = Field(
        default="", description="Additional context logs from Loki"
    )
    summarized_loki_context: Optional[str] = Field(
        default="", description="LLM-summarized log context for root cause analysis"
    )
