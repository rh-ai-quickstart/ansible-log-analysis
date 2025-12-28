import os
from typing import Optional, Dict, Any
import httpx

from alm.utils.logger import get_logger

logger = get_logger(__name__)


class RAGHandler:
    """
    Handles RAG (Retrieval-Augmented Generation) operations for retrieving
    relevant context from the knowledge base.

    Uses HTTP client to communicate with the RAG service.
    """

    _instance: Optional["RAGHandler"] = None
    _enabled: Optional[bool] = None
    _rag_service_url: Optional[str] = None
    _client: Optional[httpx.AsyncClient] = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(RAGHandler, cls).__new__(cls)
        return cls._instance

    async def cleanup(self):
        """
        Cleanup HTTP client resources.

        Should be called during application shutdown to properly close
        the HTTP connection pool and avoid resource leaks.
        """
        if self._client is not None:
            try:
                await self._client.aclose()
                logger.info("RAG service HTTP client closed")
            except Exception as e:
                logger.warning("Error closing RAG service HTTP client: %s", e)
            finally:
                self._client = None

    def _initialize_rag_service(self):
        """
        Initialize RAG service client.

        Returns:
            True if service is available, False otherwise
        """
        # Check if already initialized
        if self._enabled is not None:
            return self._enabled

        # Check if RAG is enabled via environment variable
        rag_enabled_env = os.getenv("RAG_ENABLED", "true").lower()
        if rag_enabled_env not in ["true", "1", "yes"]:
            logger.debug("RAG is disabled (RAG_ENABLED=%s)", rag_enabled_env)
            self._enabled = False
            return False

        # Get RAG service URL
        self._rag_service_url = os.getenv("RAG_SERVICE_URL", "http://alm-rag:8002")

        # Create HTTP client and initialize (wrapped in try-except for error handling)
        try:
            # Create HTTP client
            self._client = httpx.AsyncClient(
                base_url=self._rag_service_url,
                timeout=30.0,
            )

            # We'll do a lazy check on first request instead of blocking here
            self._enabled = True
            logger.info(
                "RAG service client initialized (URL: %s)", self._rag_service_url
            )
            return True
        except Exception as e:
            logger.warning("Failed to initialize RAG service client: %s", e)
            self._enabled = False
            # Clean up client if it was partially created
            self._client = None
            return False

    def _format_rag_results(self, response_data: Dict[str, Any]) -> str:
        """
        Format RAG query results into a structured string for LLM context.

        Args:
            response_data: Response dictionary from RAG service

        Returns:
            Formatted string with error solutions
        """
        results = response_data.get("results", [])
        if not results:
            return "No matching solutions found in knowledge base."

        output = ["## Relevant Error Solutions from Knowledge Base\n"]

        for i, result in enumerate(results, 1):
            error_title = result.get(
                "error_title", result.get("error_id", f"Error {i}")
            )
            similarity_score = result.get("similarity_score", 0.0)
            sections = result.get("sections", {})

            output.append(f"### Error {i}: {error_title}")
            output.append(f"**Confidence Score:** {similarity_score:.2f}\n")

            if sections.get("description"):
                output.append("**Description:**")
                output.append(sections["description"])
                output.append("")

            if sections.get("symptoms"):
                output.append("**Symptoms:**")
                output.append(sections["symptoms"])
                output.append("")

            if sections.get("resolution"):
                output.append("**Resolution:**")
                output.append(sections["resolution"])
                output.append("")

            if sections.get("code"):
                output.append("**Code Example:**")
                output.append(f"```\n{sections['code']}\n```")
                output.append("")

            output.append("---\n")

        return "\n".join(output)

    async def get_cheat_sheet_context(self, log_summary: str) -> str:
        """
        Retrieve relevant context from the RAG knowledge base for solving the error.

        This function:
        1. Initializes the RAG service client (if not already done)
        2. Queries the RAG service with the log summary
        3. Formats the results for LLM consumption
        4. Returns empty string if RAG is disabled or fails

        Args:
            log_summary: Summary of the Ansible error log

        Returns:
            Formatted string with relevant error solutions, or empty string if unavailable
        """
        logger.debug("Retrieving cheat sheet context for log summary")

        # Initialize RAG service client (lazy loading)
        if not self._initialize_rag_service():
            logger.debug("RAG service not available, returning empty context")
            return ""

        if self._client is None:
            logger.debug("RAG service client not initialized, returning empty context")
            return ""

        try:
            # Get configuration from environment variables
            top_k = int(os.getenv("RAG_TOP_K", "10"))
            top_n = int(os.getenv("RAG_TOP_N", "3"))
            similarity_threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.6"))

            # Query the RAG service
            logger.debug(
                "Querying RAG service with log summary: %s...", log_summary[:100]
            )

            response = await self._client.post(
                "/rag/query",
                json={
                    "query": log_summary,
                    "top_k": top_k,
                    "top_n": top_n,
                    "similarity_threshold": similarity_threshold,
                },
            )

            response.raise_for_status()
            response_data = response.json()

            # Format results
            formatted_context = self._format_rag_results(response_data)

            metadata = response_data.get("metadata", {})
            logger.debug(
                "✓ Retrieved %d relevant errors from knowledge base (search time: %.2fms)",
                metadata.get("num_results", 0),
                metadata.get("search_time_ms", 0.0),
            )

            return formatted_context

        except httpx.HTTPStatusError as e:
            logger.error(
                "RAG service returned error status %d: %s",
                e.response.status_code,
                e.response.text,
            )
            logger.warning("Proceeding without cheat sheet context")
            return ""
        except httpx.RequestError as e:
            logger.error("Error connecting to RAG service: %s", e)
            logger.warning("Proceeding without cheat sheet context")
            return ""
        except Exception as e:
            logger.error("Error querying RAG service: %s", e, exc_info=True)
            logger.warning("Proceeding without cheat sheet context")
            return ""
