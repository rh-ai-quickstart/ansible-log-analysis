from alm.agents.loki_agent.graph import loki_agent_graph
from alm.agents.loki_agent.state import LokiAgentState
from alm.llm import get_llm

from alm.agents.get_more_context_agent.node import (
    get_cheat_sheet_context,
    loki_router,
)
from alm.agents.get_more_context_agent.state import ContextAgentState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

llm = get_llm()


async def cheat_sheet_context_node(state: ContextAgentState):
    cheat_sheet_context = await get_cheat_sheet_context(state.log_summary)
    return Command(
        goto="loki_router_node", update={"cheat_sheet_context": cheat_sheet_context}
    )


async def loki_router_node(state: ContextAgentState):
    loki_router_result = await loki_router(
        state.log_summary, state.cheat_sheet_context, llm
    )
    return Command(
        goto="loki_sub_agent"
        if loki_router_result.classification == "need_more_context_from_loki_db"
        else END,
        update={"loki_router_result": loki_router_result.model_dump()},
    )


async def loki_sub_agent(state: ContextAgentState):
    loki_state = LokiAgentState(
        log_entry=state.log_entry,
        log_summary=state.log_summary,
        expert_classification=state.expert_classification,
        cheat_sheet_context=state.cheat_sheet_context,
        loki_router_result=state.loki_router_result,
    )
    loki_result = await loki_agent_graph.ainvoke(loki_state)
    loki_result_state = LokiAgentState.model_validate(loki_result)
    return Command(
        goto=END,
        update={"loki_context": loki_result_state.additional_context_from_loki},
    )


def build_graph():
    builder = StateGraph(ContextAgentState)
    builder.add_edge(START, "cheat_sheet_context_node")
    builder.add_node("cheat_sheet_context_node", cheat_sheet_context_node)
    builder.add_node("loki_router_node", loki_router_node)
    builder.add_node("loki_sub_agent", loki_sub_agent)
    return builder.compile()


more_context_agent_graph = build_graph()
