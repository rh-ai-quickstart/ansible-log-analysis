#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive testing tool for the query pipeline.
Uses TEI (text-embeddings-inference) service for embeddings.

Usage (from project root):
    # Interactive mode
    python tests/rag/test_queries.py

    # Batch mode with example queries file
    python tests/rag/test_queries.py data/example_queries.txt

    # Batch mode with custom queries file
    python tests/rag/test_queries.py /path/to/queries.txt

Note: Model is hardcoded to nomic-ai/nomic-embed-text-v1.5.
      EMBEDDINGS_LLM_URL can be set to override default (http://alm-embedding:8080).
"""

import sys
from pathlib import Path

# Load .env file if it exists (needed for AnsibleErrorQueryPipeline config)
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

# Import config to check mode
from utils.config import config  # noqa: E402

from rag.query_pipeline import (  # noqa: E402
    AnsibleErrorQueryPipeline,
    format_response_for_display,
)


def interactive_mode():
    """Run in interactive mode for testing queries."""
    print("=" * 70)
    print("ANSIBLE ERROR RAG - INTERACTIVE QUERY MODE")
    print("=" * 70)

    # Show configuration mode
    print(f"\n✓ Using MAAS API mode: {config.embeddings.api_url}")
    print(f"  Model: {config.embeddings.model_name}")

    print("\nInitializing...")

    # Initialize pipeline
    pipeline = AnsibleErrorQueryPipeline(top_k=10, top_n=3, similarity_threshold=0.5)

    print("\n" + "=" * 70)
    print("Ready! Enter your log summaries below.")
    print("Commands:")
    print("  - Type your query and press Enter")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'threshold X' to change threshold (e.g., 'threshold 0.7')")
    print("  - 'topn X' to change number of results (e.g., 'topn 5')")
    print("=" * 70)

    while True:
        try:
            # Get user input
            print("\n" + "-" * 70)
            query = input("Enter log summary (or command): ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if query.lower().startswith("threshold "):
                try:
                    new_threshold = float(query.split()[1])
                    pipeline.similarity_threshold = new_threshold
                    print(f"✓ Threshold set to {new_threshold}")
                    continue
                except (ValueError, IndexError):
                    print("⚠ Invalid threshold. Use: threshold 0.7")
                    continue

            if query.lower().startswith("topn "):
                try:
                    new_topn = int(query.split()[1])
                    pipeline.top_n = new_topn
                    print(f"✓ Top-N set to {new_topn}")
                    continue
                except (ValueError, IndexError):
                    print("⚠ Invalid top-n. Use: topn 5")
                    continue

            # Execute query
            response = pipeline.query(query)

            # Display results
            print(format_response_for_display(response))

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            import traceback

            traceback.print_exc()


def batch_mode(queries_file: str):
    """Run in batch mode from file."""
    print("=" * 70)
    print("ANSIBLE ERROR RAG - BATCH QUERY MODE")
    print("=" * 70)

    # Show configuration mode
    print(f"\n✓ Using MAAS API mode: {config.embeddings.api_url}")
    print(f"  Model: {config.embeddings.model_name}")

    # Load queries from file
    with open(queries_file, "r") as f:
        queries = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    print(f"\nLoaded {len(queries)} queries from {queries_file}")

    # Initialize pipeline
    pipeline = AnsibleErrorQueryPipeline(top_k=10, top_n=3, similarity_threshold=0.6)

    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n{'*' * 70}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'*' * 70}")

        response = pipeline.query(query)
        print(format_response_for_display(response))

    print("\n" + "=" * 70)
    print(f"✓ PROCESSED {len(queries)} QUERIES")
    print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Batch mode with file
        queries_file = sys.argv[1]

        # If the file doesn't exist, check if it's a relative path from the project root
        if not Path(queries_file).exists():
            # Try relative to project root (script is now in tests/rag/)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            alt_path = project_root / queries_file

            if alt_path.exists():
                queries_file = str(alt_path)
            else:
                print(f"Error: Queries file not found: {queries_file}")
                print(f"Also checked: {alt_path}")
                print("\nExample usage (from project root):")
                print("  python tests/rag/test_queries.py data/example_queries.txt")
                sys.exit(1)

        batch_mode(queries_file)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
