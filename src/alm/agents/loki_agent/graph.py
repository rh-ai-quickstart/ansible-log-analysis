"""
Loki agent subgraph definition.

This subgraph handles retrieval of additional log context from Loki:
- START → identify_missing_log_data_node → loki_execute_query_node → END
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from alm.agents.loki_agent.state import LokiAgentState
from alm.agents.loki_agent.nodes import identify_missing_data
from alm.agents.loki_agent.agent import create_loki_agent
from alm.agents.loki_agent.schemas import LogToolOutput, LokiAgentOutput, ToolStatus
from alm.llm import get_llm
from alm.utils.logger import get_logger

logger = get_logger(__name__)


async def identify_missing_log_data_node(
    state: LokiAgentState,
) -> Command[Literal["loki_execute_query_node"]]:
    """
    Node that processes a request for additional log context using Loki.
    This node uses an LLM to intelligently identify what data is missing and
    generate a natural language request for additional logs.
    """
    # Get the current state
    log_summary = state.log_summary
    log_labels = state.log_entry.log_labels
    log_timestamp = str(state.log_entry.timestamp)
    # Get LLM instance
    llm = get_llm()

    # Use LLM to identify what data is missing and generate a smart request
    user_request = await identify_missing_data(
        log_summary=log_summary,
        log_labels=log_labels,
        log_timestamp=log_timestamp,
        llm=llm,
    )
    return Command(
        goto="loki_execute_query_node", update={"loki_user_request": user_request}
    )


async def loki_execute_query_node(
    state: LokiAgentState,
) -> Command[Literal[END]]:
    """
    Node that executes the Loki query using the ToolCallingAgent.

    TODO: Add query validation and retry logic
    """
    try:
        user_request = state.loki_user_request
        if not user_request:
            raise ValueError(
                "No user request found in state.loki_user_request. \
                Please use the identify_missing_log_data_node to set the user request."
            )

        # Extract log context for agent creation
        log_message = state.log_entry.message
        log_timestamp = str(state.log_entry.timestamp)
        log_labels = state.log_entry.log_labels
        file_name = log_labels.filename

        if not all([file_name, log_message, log_timestamp]):
            raise ValueError(
                f"One of the log context fields is missing in log_entry, log_enty = {state.log_entry}"
            )

        # Create a fresh agent instance with log context bound via closure
        agent = create_loki_agent(file_name, log_message, log_timestamp)

        # Prepare context from the current state
        context = {
            "logSummary": state.log_summary,
            "expertClassification": state.expert_classification,
            "logMessage": log_message,
            "logLabels": log_labels,
            "timestamp": log_timestamp,
        }

        # Execute the query
        result = await agent.query_logs(user_request, context)

        # Build context from Loki query result
        old_loki_context = state.additional_context_from_loki
        if result.agent_result and isinstance(result.agent_result, LogToolOutput):
            additional_context = result.agent_result.build_context()
        else:
            logger.warning("No logs returned from Loki query.")
            additional_context = ""
        if old_loki_context and additional_context:
            additional_context = old_loki_context + "\n\n" + additional_context

        return Command(
            goto=END,
            update={
                "loki_query_result": result.model_dump(),
                "additional_context_from_loki": additional_context,
            },
        )

    except Exception as e:
        logger.error("Exception in loki_execute_query_node: %s", e)
        logger.warning("Continuing without Loki context due to error.")
        return Command(
            goto=END,
            update={
                "loki_query_result": LokiAgentOutput(
                    status=ToolStatus.ERROR,
                    user_request=user_request
                    if user_request
                    else "No user request found",
                    agent_result=LogToolOutput(
                        status=ToolStatus.ERROR,
                        message=f"Failed to execute Loki query: {str(e)}",
                        logs=[],
                        number_of_logs=0,
                    ),
                    raw_output=str(e),
                    tool_messages=[],
                ).model_dump()
            },
        )


def build_loki_agent_graph():
    """
    Build the Loki agent subgraph.

    Graph flow:
    START → identify_missing_log_data_node → loki_execute_query_node → END
    """
    builder = StateGraph(LokiAgentState)

    # Add edges and nodes
    builder.add_edge(START, "identify_missing_log_data_node")
    builder.add_node("identify_missing_log_data_node", identify_missing_log_data_node)
    builder.add_node("loki_execute_query_node", loki_execute_query_node)

    return builder.compile()


# Export the compiled graph
loki_agent_graph = build_loki_agent_graph()
