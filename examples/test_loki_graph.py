"""
Simple example to test the get_more_context_agent graph.

This example demonstrates the get_more_context_agent flow:
1. Starts with cheat_sheet_context_node
2. Routes through loki_router_node
3. If needed, invokes the loki_agent subgraph
4. Returns the combined context (cheat sheet + loki logs)
"""

import asyncio
from datetime import datetime

from alm.agents.get_more_context_agent.graph import more_context_agent_graph
from alm.agents.get_more_context_agent.state import ContextAgentState
from alm.agents.loki_agent.schemas import LogEntry, LogLabels, DetectedLevel

# from langchain.globals import set_debug

# set_debug(True)  # Enables LangChain debug mode globally


async def test_get_more_context_graph():
    """
    Run a test of the get_more_context_agent graph.
    """
    print("=" * 80)
    print("🧪 Testing Get More Context Agent Graph")
    print("=" * 80)

    # Create a test log entry
    log_message = r"""TASK [ocp4_workload_rhacm_hypershift : Abort] **********************************
Monday 04 August 2025  20:00:55 +0000 (0:00:00.029)       1:36:38.896 ********* 
fatal: [bastion.6jxd6.internal]: FAILED! => {"changed": false, "msg": "Cluster creation failed. Aborting."}"""
    log_entry = LogEntry(
        log_labels=LogLabels(
            detected_level=DetectedLevel.ERROR,
            filename="/var/log/ansible_logs/failed/job_1461865.txt",
            service_name="failed_logs",
            cluster_name="test_cluster",
        ),
        message=log_message,
        timestamp=datetime.now(),
    )

    # Create a test state with sample data
    test_state = ContextAgentState(
        log_entry=log_entry,
        log_summary="Request failed with HTTP Error 307: Temporary Redirect when downloading argocd binary",
    )

    print("\n📝 Initial State:")
    print(f"  Log Summary: {test_state.log_summary}")
    print(f"  Log Entry (truncated): {test_state.log_entry.message[:150]}...")
    print(f"  Loki Router Result: {test_state.loki_router_result}")

    print("\n🚀 Running Get More Context Agent Graph...")
    print("-" * 80)

    try:
        # Execute the graph
        result_dict = await more_context_agent_graph.ainvoke(test_state)
        # Convert dict back to ContextAgentState object
        result = ContextAgentState.model_validate(result_dict)

        print("\n" + "=" * 80)
        print("✨ Graph Execution Complete!")
        print("=" * 80)

        print("\n📊 Final State Summary:")

        # Cheat sheet context
        if result.cheat_sheet_context:
            print("\n  ✅ Cheat Sheet Context Retrieved:")
            context = result.cheat_sheet_context
            if len(context) > 200:
                print(f"    {context[:200]}...")
                print(f"    (truncated, total length: {len(context)} chars)")
            else:
                print(f"    {context}")
        else:
            print("\n  ❌ No Cheat Sheet Context")

        # Loki router result
        if result.loki_router_result:
            print("\n  🔀 Loki Router Decision:")
            print(f"    Classification: {result.loki_router_result.classification}")
            print(f"    Reasoning: {result.loki_router_result.reasoning}")
        else:
            print("\n  ❌ No Loki Router Result")

        # Loki context (if retrieved)
        if result.loki_context:
            print("\n  ✅ Loki Context Retrieved:")
            print(f"Context logs from loki:\n{result.loki_context}")
        else:
            print("\n  ℹ️  No Loki Context (may not have been needed)")

        return result

    except Exception as e:
        print(f"\n❌ Error during graph execution: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_get_more_context_graph())

    print("\n" + "=" * 80)
    print("🎉 Test Complete!")
    print("=" * 80)
