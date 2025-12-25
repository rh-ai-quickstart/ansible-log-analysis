"""
Utility functions for RAG service interaction.
"""

import asyncio
import os
import httpx


async def wait_for_rag_service(rag_service_url: str, max_wait_time: int = 300):
    """
    Wait for RAG service to be ready before proceeding.

    This function BLOCKS until the RAG service is ready. If the service
    does not become ready within max_wait_time, it raises an exception
    to fail the init job.

    Args:
        rag_service_url: URL of the RAG service (e.g., http://alm-rag:8002)
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)

    Raises:
        TimeoutError: If RAG service does not become ready within max_wait_time
    """
    # Check if RAG is enabled
    rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
    rag_enabled = rag_enabled_env in ["true", "1", "yes"]
    if not rag_enabled:
        print("RAG is disabled, skipping RAG service wait")
        return

    print("\n" + "=" * 70)
    print("WAITING FOR RAG SERVICE TO BE READY")
    print("=" * 70)

    ready_url = f"{rag_service_url}/ready"
    elapsed = 0
    check_interval = 5

    async with httpx.AsyncClient(timeout=10.0) as client:
        while elapsed < max_wait_time:
            try:
                response = await client.get(ready_url)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        index_size = data.get("index_size", 0)
                        print(f"✓ RAG service is ready (index size: {index_size})")
                        return  # Success - service is ready
                    except (ValueError, TypeError):
                        # Invalid JSON response - treat as not ready
                        print(
                            f"RAG service returned invalid JSON (status: {response.status_code}), waiting..."
                        )
                else:
                    print(
                        f"RAG service not ready yet (status: {response.status_code}), waiting..."
                    )
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                ValueError,
                TypeError,
            ):
                if elapsed == 0:
                    print(
                        f"RAG service not yet available at {rag_service_url}, waiting..."
                    )
                elif elapsed % 30 == 0:  # Print every 30 seconds
                    print(f"Still waiting for RAG service... (elapsed: {elapsed}s)")

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        # Timeout reached - FAIL the init job
        error_msg = (
            f"RAG service did not become ready within {max_wait_time} seconds. "
            f"Init job cannot proceed without RAG service."
        )
        print(f"\n✗ ERROR: {error_msg}")
        raise TimeoutError(error_msg)
