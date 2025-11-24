"""
Vector Store for Semantic Search
PATTERN: Simple in-memory vector store with SQLite persistence
WHY: Fast semantic search without external dependencies
RESILIENCE: Graceful degradation if embeddings fail
"""
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import aiosqlite
from datetime import datetime
import hashlib

from app.logger import db_logger
from app.search.config import HybridSearchConfig, DEFAULT_SEARCH_CONFIG


class VectorStore:
    """
    In-memory vector store with SQLite persistence

    CONCEPT: Stores embeddings in memory for fast similarity search,
    with SQLite backing for persistence and metadata
    """

    def __init__(
        self,
        db_path: str = "learning_captures.db",
        config: HybridSearchConfig = DEFAULT_SEARCH_CONFIG
    ):
        self.db_path = db_path
        self.config = config
        self._initialized = False

        # In-memory storage for fast search
        self._embeddings: Dict[int, np.ndarray] = {}
        self._metadata: Dict[int, Dict] = {}

        # Embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}

    async def initialize(self):
        """Initialize vector store tables and load embeddings into memory"""
        if self._initialized:
            return

        try:
            db_logger.info("vector_store_initialization_started", db_path=self.db_path)

            async with aiosqlite.connect(self.db_path) as db:
                # Create embeddings table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        capture_id INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (capture_id) REFERENCES captures(id)
                    )
                """)

                # Create index for fast lookups
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_capture
                    ON embeddings(capture_id)
                """)

                await db.commit()

            # Load existing embeddings into memory
            await self._load_embeddings()

            self._initialized = True
            db_logger.info(
                "vector_store_initialization_complete",
                db_path=self.db_path,
                embeddings_count=len(self._embeddings)
            )

        except Exception as e:
            db_logger.error(
                "vector_store_initialization_failed",
                db_path=self.db_path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def _load_embeddings(self):
        """Load all embeddings from database into memory"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("""
                    SELECT id, capture_id, embedding, embedding_model
                    FROM embeddings
                """)
                rows = await cursor.fetchall()

                for row in rows:
                    # Deserialize numpy array from blob
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    self._embeddings[row['capture_id']] = embedding
                    self._metadata[row['capture_id']] = {
                        'id': row['id'],
                        'model': row['embedding_model']
                    }

                db_logger.info(
                    "embeddings_loaded",
                    count=len(self._embeddings)
                )

        except Exception as e:
            db_logger.error(
                "load_embeddings_failed",
                error=str(e),
                exc_info=True
            )
            # Continue with empty store

    async def add_embedding(
        self,
        capture_id: int,
        embedding: np.ndarray,
        model: str = "text-embedding-ada-002"
    ) -> int:
        """
        Add or update an embedding for a capture

        Args:
            capture_id: ID of the capture in captures table
            embedding: Embedding vector (numpy array)
            model: Name of embedding model used

        Returns:
            Embedding ID
        """
        try:
            # Normalize embedding for cosine similarity
            normalized_embedding = embedding / np.linalg.norm(embedding)

            # Convert to bytes for storage
            embedding_blob = normalized_embedding.astype(np.float32).tobytes()

            async with aiosqlite.connect(self.db_path) as db:
                # Check if embedding exists
                cursor = await db.execute(
                    "SELECT id FROM embeddings WHERE capture_id = ?",
                    (capture_id,)
                )
                existing = await cursor.fetchone()

                if existing:
                    # Update existing
                    await db.execute(
                        """
                        UPDATE embeddings
                        SET embedding = ?, embedding_model = ?
                        WHERE capture_id = ?
                        """,
                        (embedding_blob, model, capture_id)
                    )
                    embedding_id = existing[0]
                else:
                    # Insert new
                    cursor = await db.execute(
                        """
                        INSERT INTO embeddings (capture_id, embedding, embedding_model)
                        VALUES (?, ?, ?)
                        """,
                        (capture_id, embedding_blob, model)
                    )
                    embedding_id = cursor.lastrowid

                await db.commit()

            # Update in-memory store
            self._embeddings[capture_id] = normalized_embedding
            self._metadata[capture_id] = {
                'id': embedding_id,
                'model': model
            }

            db_logger.debug(
                "embedding_added",
                capture_id=capture_id,
                embedding_id=embedding_id,
                dimensions=len(embedding)
            )

            return embedding_id

        except Exception as e:
            db_logger.error(
                "add_embedding_failed",
                capture_id=capture_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        threshold: float = 0.7
    ) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings using cosine similarity

        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (capture_id, similarity_score) tuples, sorted by score
        """
        try:
            if not self._embeddings:
                db_logger.warning("vector_search_empty_store")
                return []

            # Normalize query embedding
            normalized_query = query_embedding / np.linalg.norm(query_embedding)

            # Calculate cosine similarity with all embeddings
            similarities = []
            for capture_id, embedding in self._embeddings.items():
                similarity = float(np.dot(normalized_query, embedding))
                if similarity >= threshold:
                    similarities.append((capture_id, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = similarities[:limit]

            db_logger.debug(
                "vector_search_complete",
                total_embeddings=len(self._embeddings),
                results_count=len(results),
                threshold=threshold
            )

            return results

        except Exception as e:
            db_logger.error(
                "vector_search_failed",
                error=str(e),
                exc_info=True
            )
            return []

    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding for a text query"""
        if self.config.enable_embedding_cache:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            self._embedding_cache[cache_key] = embedding

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for a text query"""
        if not self.config.enable_embedding_cache:
            return None

        cache_key = hashlib.md5(text.encode()).hexdigest()
        return self._embedding_cache.get(cache_key)

    def clear_cache(self):
        """Clear embedding cache"""
        self._embedding_cache.clear()
        db_logger.info("embedding_cache_cleared")

    async def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "total_embeddings": len(self._embeddings),
            "cache_size": len(self._embedding_cache),
            "embedding_dimensions": self.config.embedding_dimensions,
            "model": self.config.embedding_model
        }


# Global vector store instance
vector_store = VectorStore()
