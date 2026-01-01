#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple API wrapper for the query pipeline.
This provides a clean interface for agent integration.
"""

from typing import Dict, Any, Optional, List
from rag.query_pipeline import AnsibleErrorQueryPipeline

from utils.logger import get_logger

logger = get_logger(__name__)


class AnsibleErrorRAG:
    """
    High-level interface for the Ansible Error RAG system.
    Designed for easy integration with agent systems.
    """

    def __init__(
        self, top_k: int = 10, top_n: int = 3, similarity_threshold: float = 0.6
    ):
        """
        Initialize the RAG system.

        Args:
            top_k: Number of candidates to retrieve
            top_n: Number of final results to return
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.pipeline = AnsibleErrorQueryPipeline(
            top_k=top_k, top_n=top_n, similarity_threshold=similarity_threshold
        )

    def search_errors(
        self,
        log_summary: str,
        max_results: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search for relevant errors based on log summary.

        Args:
            log_summary: Error description from logs (few sentences)
            max_results: Maximum number of results to return
            min_confidence: Minimum confidence score (0-1)

        Returns:
            Dictionary with query results
        """
        response = self.pipeline.query(
            log_summary=log_summary,
            top_n=max_results,
            similarity_threshold=min_confidence,
        )

        return response.to_dict()

    def get_best_match(self, log_summary: str) -> Optional[Dict[str, Any]]:
        """
        Get the single best matching error.

        Args:
            log_summary: Error description from logs

        Returns:
            Single best error result or None if no good match
        """
        response = self.pipeline.query(
            log_summary=log_summary,
            top_n=1,
            similarity_threshold=0.7,  # Higher threshold for single result
        )

        if response.results:
            return response.results[0].to_dict()
        return None

    def get_resolution_steps(self, log_summary: str) -> Optional[str]:
        """
        Get resolution steps for the best matching error.

        Args:
            log_summary: Error description from logs

        Returns:
            Resolution text or None if no match
        """
        best_match = self.get_best_match(log_summary)

        if best_match:
            return best_match["sections"]["resolution"]
        return None

    def get_similar_errors(
        self, log_summary: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get multiple similar errors for comparison.

        Args:
            log_summary: Error description from logs
            num_results: Number of similar errors to return

        Returns:
            List of similar error results
        """
        response = self.pipeline.query(
            log_summary=log_summary,
            top_n=num_results,
            similarity_threshold=0.5,  # Lower threshold for more results
        )

        return [r.to_dict() for r in response.results]

    def batch_search(
        self, log_summaries: List[str], max_results_per_query: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for multiple log summaries in batch.

        Args:
            log_summaries: List of error descriptions
            max_results_per_query: Max results per query

        Returns:
            List of query responses
        """
        results = []

        for log_summary in log_summaries:
            response = self.pipeline.query(
                log_summary=log_summary, top_n=max_results_per_query
            )
            results.append(response.to_dict())

        return results


def main():
    """
    Example usage of the API.
    """
    logger.info("=" * 70)
    logger.info("ANSIBLE ERROR RAG - API EXAMPLE")
    logger.info("=" * 70)

    # Initialize RAG system
    rag = AnsibleErrorRAG(top_n=3, similarity_threshold=0.6)

    # Example 1: Search for errors
    logger.info("1. Search for errors:")
    log_summary = "The role name does not follow the naming convention"
    results = rag.search_errors(log_summary)

    logger.info("Query: %s", log_summary)
    logger.info("Found: %d results", results["metadata"]["num_results"])
    for i, result in enumerate(results["results"], 1):
        logger.info(
            "  %d. %s (score: %.3f)",
            i,
            result["error_title"],
            result["similarity_score"],
        )

    # Example 2: Get best match
    logger.info("2. Get best match:")
    best = rag.get_best_match(log_summary)
    if best:
        logger.info("Best match: %s", best["error_title"])
        logger.info("Confidence: %.3f", best["similarity_score"])
    else:
        logger.info("No confident match found")

    # Example 3: Get resolution steps
    logger.info("3. Get resolution steps:")
    resolution = rag.get_resolution_steps(log_summary)
    if resolution:
        logger.info("Resolution: %s...", resolution[:200])
    else:
        logger.info("No resolution found")

    # Example 4: Get similar errors
    logger.info("4. Get similar errors:")
    similar = rag.get_similar_errors(log_summary, num_results=5)
    logger.info("Found %d similar errors:", len(similar))
    for i, err in enumerate(similar, 1):
        logger.info("  %d. %s...", i, err["error_title"][:50])

    logger.info("=" * 70)
    logger.info("API EXAMPLES COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
