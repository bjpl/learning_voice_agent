"""
Vector Store Module - ChromaDB Integration for Semantic Search
PATTERN: Repository pattern with async wrapper for vector operations
WHY: Enable semantic similarity search across conversation history

SPECIFICATION:
- Store conversation embeddings in ChromaDB
- Support semantic search with configurable similarity threshold
- Integrate with existing SQLite database for hybrid search
- Handle embedding generation via sentence-transformers

ARCHITECTURE:
[User Query] -> [Embedding Model] -> [ChromaDB Query]
                                          |
                                     [Similar Docs] -> [Reranking] -> [Results]
"""
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import logging
from functools import lru_cache

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for semantic conversation search.

    CONCEPT: Semantic search complements FTS5 keyword search
    WHY: Find conceptually similar conversations, not just keyword matches
    PATTERN: Lazy initialization with connection pooling
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize ChromaDB client and collection.
        Returns True if successful, False if ChromaDB not available.
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Semantic search disabled.")
            return False

        if self._initialized:
            return True

        try:
            # Run ChromaDB initialization in thread pool (it's synchronous)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_sync)
            self._initialized = True
            logger.info("Vector store initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False

    def _initialize_sync(self):
        """Synchronous initialization for ChromaDB."""
        # Use new ChromaDB PersistentClient API (v0.4+)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Create or get collection for conversations
        self._collection = self._client.get_or_create_collection(
            name="conversations",
            metadata={
                "description": "Conversation embeddings for semantic search",
                "created_at": datetime.utcnow().isoformat()
            }
        )

        # Initialize embedding model
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load sentence-transformers model for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            # Use a lightweight model optimized for semantic similarity
            model_name = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
            self._embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available. Using fallback.")
            self._embedding_model = None

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        CONCEPT: Dense vector representation of semantic meaning
        WHY: Enable similarity comparison in vector space
        """
        if self._embedding_model is None:
            # Fallback: use simple hash-based pseudo-embedding
            return self._fallback_embedding(text)

        embedding = self._embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Fallback embedding when sentence-transformers unavailable.
        Uses deterministic hash-based approach for consistency.
        """
        import hashlib

        # Create deterministic pseudo-embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Expand hash to required dimension
        embedding = []
        for i in range(dim):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)

        return embedding

    async def add_conversation(
        self,
        conversation_id: str,
        user_text: str,
        agent_text: str,
        session_id: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a conversation exchange to the vector store.

        PATTERN: Combined embedding for better semantic matching
        WHY: Captures the full context of an exchange
        """
        if not self._initialized:
            await self.initialize()

        if not CHROMADB_AVAILABLE or self._collection is None:
            return False

        try:
            # Combine user and agent text for embedding
            combined_text = f"User: {user_text}\nAssistant: {agent_text}"

            # Generate embedding
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_embedding, combined_text
            )

            # Prepare metadata
            doc_metadata = {
                "session_id": session_id,
                "user_text": user_text[:500],  # Truncate for storage
                "agent_text": agent_text[:500],
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }

            # Add to collection (synchronous operation)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.add(
                    ids=[conversation_id],
                    embeddings=[embedding],
                    documents=[combined_text],
                    metadatas=[doc_metadata]
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add conversation to vector store: {e}")
            return False

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        session_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform semantic similarity search.

        CONCEPT: Find conversations by meaning, not just keywords
        WHY: "What did I say about learning?" finds related discussions

        Args:
            query: Search query text
            limit: Maximum results to return
            similarity_threshold: Minimum similarity score (0-1)
            session_filter: Optional session ID to filter results

        Returns:
            List of matching conversations with similarity scores
        """
        if not self._initialized:
            await self.initialize()

        if not CHROMADB_AVAILABLE or self._collection is None:
            return []

        try:
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_embedding, query
            )

            # Build where clause for filtering
            where_clause = None
            if session_filter:
                where_clause = {"session_id": session_filter}

            # Query ChromaDB
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
            )

            # Process and filter results
            processed_results = []

            if results and results.get('ids') and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Convert distance to similarity (ChromaDB uses L2 distance)
                    distance = results['distances'][0][i]
                    similarity = 1 / (1 + distance)  # Convert to similarity score

                    if similarity >= similarity_threshold:
                        metadata = results['metadatas'][0][i]
                        processed_results.append({
                            "id": doc_id,
                            "similarity": round(similarity, 3),
                            "user_text": metadata.get("user_text", ""),
                            "agent_text": metadata.get("agent_text", ""),
                            "session_id": metadata.get("session_id", ""),
                            "timestamp": metadata.get("timestamp", ""),
                            "document": results['documents'][0][i]
                        })

            return processed_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def find_similar_conversations(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find conversations similar to a given conversation.

        CONCEPT: "More like this" functionality
        WHY: Help users discover related learning moments
        """
        if not self._initialized or not CHROMADB_AVAILABLE:
            return []

        try:
            # Get the target conversation
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.get(
                    ids=[conversation_id],
                    include=["documents"]
                )
            )

            if not result or not result.get('documents'):
                return []

            # Search for similar conversations
            document = result['documents'][0]
            return await self.semantic_search(
                query=document,
                limit=limit + 1  # +1 because it will match itself
            )

        except Exception as e:
            logger.error(f"Find similar conversations failed: {e}")
            return []

    async def get_conversation_clusters(
        self,
        session_id: Optional[str] = None,
        n_clusters: int = 5
    ) -> List[Dict]:
        """
        Group conversations into semantic clusters.

        CONCEPT: Automatic topic discovery
        WHY: Help users see patterns in their learning
        """
        # This would require more sophisticated clustering
        # For now, return top conversations by recency
        return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Remove a conversation from the vector store."""
        if not self._initialized or not CHROMADB_AVAILABLE:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.delete(ids=[conversation_id])
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return False

    async def get_stats(self) -> Dict:
        """Get vector store statistics."""
        if not self._initialized or not CHROMADB_AVAILABLE:
            return {"available": False, "count": 0}

        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.count()
            )
            return {
                "available": True,
                "count": count,
                "embedding_model": getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2'),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {"available": False, "error": str(e)}

    async def close(self):
        """Cleanup vector store resources."""
        # PersistentClient handles persistence automatically, no explicit persist() needed
        self._initialized = False
        self._client = None
        self._collection = None


# Global vector store instance
vector_store = VectorStore()
