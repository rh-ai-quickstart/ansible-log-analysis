from alm.agents.get_more_context_agent.state import ContextAgentState
from alm.llm import get_llm
from alm.agents.state import GrafanaAlertState
from alm.agents.node import (
    summarize_log,
    classify_log,
    suggest_step_by_step_solution,
    router_step_by_step_solution,
    infer_cluster_log,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from alm.agents.get_more_context_agent.graph import more_context_agent_graph

from alm.utils.logger import get_logger

logger = get_logger(__name__)

llm = get_llm()


# Nodes
async def cluster_logs_node(
    state: GrafanaAlertState,
) -> Command:
    logs = state.log_entry.message
    log_cluster = infer_cluster_log(logs)
    return Command(goto="no_clustering_graph_node", update={"logCluster": log_cluster})


async def summarize_log_node(
    state: GrafanaAlertState,
) -> Command:
    log_summary = await summarize_log(state.log_entry.message, llm)
    return Command(goto="classify_log_node", update={"logSummary": log_summary})


async def classify_log_node(
    state: GrafanaAlertState,
) -> Command:
    log_summary = state.logSummary
    log_category = await classify_log(log_summary, llm)
    return Command(
        goto="router_step_by_step_solution_node",
        update={"expertClassification": log_category},
    )


async def suggest_step_by_step_solution_node(
    state: GrafanaAlertState,
) -> Command:
    log_summary = state.logSummary
    log = state.log_entry.message
    context = state.contextForStepByStepSolution
    try:
        step_by_step_solution = await suggest_step_by_step_solution(
            log_summary, log, llm, context
        )
    except Exception as e:
        logger.error("Exception in suggest_step_by_step_solution_node: %s", e)
        logger.warning("Continuing **without context** due to error.")
        step_by_step_solution = await suggest_step_by_step_solution(
            log_summary, log, llm
        )
    return Command(goto=END, update={"stepByStepSolution": step_by_step_solution})


async def router_step_by_step_solution_node(
    state: GrafanaAlertState,
) -> Command:
    log_summary = state.logSummary
    classification = await router_step_by_step_solution(log_summary, llm)
    return Command(
        goto="suggest_step_by_step_solution_node"
        if classification == "No More Context Needed"
        else "get_more_context_node",
        update={"needMoreContext": classification == "Need More Context"},
    )


async def get_more_context_node(
    state: GrafanaAlertState,
) -> Command:
    log_summary = state.logSummary
    subgraph_state = await more_context_agent_graph.ainvoke(
        ContextAgentState(
            log_summary=log_summary,
            log_entry=state.log_entry,
            expert_classification=state.expertClassification,
        )
    )
    context_agent_state = ContextAgentState.model_validate(subgraph_state)
    loki_context = context_agent_state.loki_context
    cheat_sheet_context = (
        f"Context from cheat sheet:\n{context_agent_state.cheat_sheet_context}"
    )
    context = (
        f"Context logs from loki:\n{loki_context}\n\n{cheat_sheet_context}"
        if loki_context
        else cheat_sheet_context
    )
    return Command(
        goto="suggest_step_by_step_solution_node",
        update={"contextForStepByStepSolution": context},
    )


def graph_without_clustering():
    builder = StateGraph(GrafanaAlertState)
    builder.add_edge(START, "summarize_log_node")
    builder.add_node("summarize_log_node", summarize_log_node)
    builder.add_node(classify_log_node)
    builder.add_node(suggest_step_by_step_solution_node)
    builder.add_node(router_step_by_step_solution_node)
    builder.add_node(get_more_context_node)

    return builder.compile()


async def no_clustering_graph_node(state: GrafanaAlertState) -> Command:
    graph = await graph_without_clustering().ainvoke(state)
    return Command(goto=END, update=graph)


# infernece graph
def inference_graph():
    builder = StateGraph(GrafanaAlertState)
    builder.add_edge(START, "cluster_logs_node")
    builder.add_node("cluster_logs_node", cluster_logs_node)
    builder.add_node(no_clustering_graph_node)

    return builder.compile()
