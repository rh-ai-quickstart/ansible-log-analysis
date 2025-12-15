"""
Load RAG embeddings from PostgreSQL and build FAISS index.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import faiss
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

# Import models - we'll need to make these available
# For now, we'll define a simple structure or import from the main codebase
# In production, these should be in a shared package


class RAGIndexLoader:
    """
    Loads embeddings from PostgreSQL and builds FAISS index in memory.
    """

    def __init__(
        self, database_url: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        """
        Initialize the index loader.

        Args:
            database_url: PostgreSQL connection URL
            model_name: Name of the embedding model (for validation)
        """
        self.database_url = database_url.replace("+asyncpg", "").replace(
            "postgresql", "postgresql+asyncpg"
        )
        self.model_name = model_name
        self.embedding_dim = 768  # nomic-embed-text-v1.5 dimension

        self.engine = create_async_engine(self.database_url)
        self.session_factory = sessionmaker(
            self.engine, class_=SQLModelAsyncSession, expire_on_commit=False
        )

        self.index: Optional[faiss.Index] = None
        self.error_store: Dict[str, Dict[str, Any]] = {}
        self.index_to_error_id: Dict[int, str] = {}
        self._loaded = False

    async def load_index(
        self,
    ) -> Tuple[faiss.Index, Dict[str, Dict[str, Any]], Dict[int, str]]:
        """
        Load embeddings from PostgreSQL and build FAISS index.

        Returns:
            Tuple of (FAISS index, error_store, index_to_error_id mapping)
        """
        if self._loaded and self.index is not None:
            return self.index, self.error_store, self.index_to_error_id

        print("Loading embeddings from PostgreSQL...")

        # Define RAGEmbedding model inline (or import from shared package)
        # For now, we'll use raw SQL to avoid circular dependencies
        from sqlalchemy import text

        async with self.engine.begin() as conn:
            # Query all embeddings
            # Note: pgvector Vector type may be returned as string, we'll parse it in Python
            result = await conn.execute(
                text("""
                    SELECT 
                        error_id,
                        embedding,
                        error_title,
                        error_metadata,
                        model_name,
                        embedding_dim
                    FROM ragembedding
                    ORDER BY error_id
                """)
            )
            rows = result.fetchall()

        if not rows:
            raise ValueError("No embeddings found in PostgreSQL. Run init job first.")

        print(f"Found {len(rows)} embeddings in database")

        # Extract data
        embeddings_list = []
        error_ids = []
        error_store = {}
        index_to_error_id = {}

        for idx, row in enumerate(rows):
            error_id = row[0]
            embedding = row[1]  # This is a list/array
            error_title = row[2]
            error_metadata = row[3] if row[3] else {}
            model_name_db = row[4]
            embedding_dim_db = row[5]

            # Validate model
            if model_name_db != self.model_name:
                print(
                    f"Warning: Model mismatch. DB has {model_name_db}, expected {self.model_name}"
                )

            if embedding_dim_db != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: DB has {embedding_dim_db}, "
                    f"expected {self.embedding_dim}"
                )

            # Convert embedding to numpy array
            # Handle both array and string representations from pgvector
            if isinstance(embedding, str):
                # Parse string representation (e.g., "[0.1, 0.2, ...]")
                import json
                import ast

                try:
                    # Try JSON first (safer)
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use ast.literal_eval (safe for literals)
                    # pgvector returns vectors as string like '[0.1,0.2,...]'
                    try:
                        embedding = ast.literal_eval(embedding)
                    except (ValueError, SyntaxError):
                        raise ValueError(
                            f"Could not parse embedding for {error_id}: invalid format"
                        )

            embedding_array = np.array(embedding, dtype=np.float32)

            # Validate embedding shape
            if embedding_array.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Invalid embedding shape for {error_id}: "
                    f"expected {self.embedding_dim}, got {embedding_array.shape[0]}"
                )

            embeddings_list.append(embedding_array)
            error_ids.append(error_id)

            # Build error_store
            error_store[error_id] = {
                "error_id": error_id,
                "error_title": error_title,
                "sections": error_metadata.get("sections", {}),
                "metadata": error_metadata.get("metadata", {}),
            }

            index_to_error_id[idx] = error_id

        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)

        print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")

        # Verify embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(
            f"Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
        )

        # Build FAISS index
        print(f"Building FAISS IndexFlatIP with dimension {self.embedding_dim}...")
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        print(f"FAISS index created with {index.ntotal} vectors")

        # Store for reuse
        self.index = index
        self.error_store = error_store
        self.index_to_error_id = index_to_error_id
        self._loaded = True

        return index, error_store, index_to_error_id

    async def reload_index(self):
        """Force reload of index from database."""
        self._loaded = False
        self.index = None
        self.error_store = {}
        self.index_to_error_id = {}
        return await self.load_index()
