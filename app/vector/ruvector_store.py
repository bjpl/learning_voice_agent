"""
RuVector Store - Self-Learning Vector Database Implementation
PATTERN: Adapter pattern wrapping RuVector with async interface
WHY: Provide self-learning vector search with GNN-powered index optimization

SPECIFICATION:
- Implement VectorStoreProtocol for drop-in replacement of ChromaDB
- Add self-learning capabilities (train_positive, train_negative)
- Support adaptive compression tiers
- Prepare for Phase 2 Cypher query support
- Graceful degradation when RuVector unavailable

ARCHITECTURE:
[User Query] -> [Embedding Model] -> [RuVector GNN Index]
                                           |
                                     [Self-Learning] <- [Quality Feedback]
                                           |
                                     [Compressed Storage] -> [Results]
"""
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import logging
from functools import lru_cache

from app.config import settings
from app.vector.protocol import (
    VectorStoreProtocol,
    LearningVectorStoreProtocol,
    GraphVectorStoreProtocol
)

# Conditional import for RuVector
try:
    import ruvector
    RUVECTOR_AVAILABLE = True
except ImportError:
    RUVECTOR_AVAILABLE = False
    ruvector = None

logger = logging.getLogger(__name__)


class RuVectorStore(LearningVectorStoreProtocol, GraphVectorStoreProtocol):
    """
    RuVector-based vector store with self-learning capabilities.

    CONCEPT: Vector search that improves through usage
    WHY: GNN-powered index learns optimal search patterns over time
    PATTERN: Implements both Learning and Graph protocols for full feature set

    Key Features:
    - Self-learning index (GNN with attention mechanisms)
    - Adaptive compression (hot/warm/cold tiers)
    - Native Cypher query support
    - Sub-millisecond query latency
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_dim: int = 384,
        enable_learning: Optional[bool] = None,
        enable_compression: Optional[bool] = None
    ):
        """
        Initialize RuVectorStore.

        Args:
            persist_directory: Path for persistent storage
            embedding_dim: Embedding vector dimension (default: 384 for MiniLM)
            enable_learning: Enable GNN self-learning (default from config)
            enable_compression: Enable adaptive compression (default from config)
        """
        self.persist_directory = persist_directory or settings.ruvector_persist_directory
        self.embedding_dim = embedding_dim
        self.enable_learning = enable_learning if enable_learning is not None else settings.ruvector_enable_learning
        self.enable_compression = enable_compression if enable_compression is not None else settings.ruvector_enable_compression

        self._db = None
        self._embedding_model = None
        self._initialized = False

        # Learning statistics
        self._positive_training_count = 0
        self._negative_training_count = 0
        self._quality_improvement = 0.0

    @property
    def is_available(self) -> bool:
        """Check if RuVector is available."""
        return RUVECTOR_AVAILABLE

    async def initialize(self) -> bool:
        """
        Initialize RuVector database and embedding model.

        PATTERN: Lazy initialization with graceful degradation
        WHY: Allow app to start even if RuVector unavailable

        Returns:
            True if successful, False if RuVector not available
        """
        if not RUVECTOR_AVAILABLE:
            logger.warning("RuVector not available. Self-learning search disabled.")
            return False

        if self._initialized:
            return True

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_sync)
            self._initialized = True
            logger.info(
                f"RuVector initialized: dim={self.embedding_dim}, "
                f"learning={self.enable_learning}, compression={self.enable_compression}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RuVector: {e}")
            return False

    def _initialize_sync(self):
        """Synchronous initialization for RuVector."""
        # Initialize RuVector database
        self._db = ruvector.VectorDB(
            dim=self.embedding_dim,
            persist_path=self.persist_directory
        )

        # Configure GNN learning if enabled
        if self.enable_learning and settings.ruvector_gnn_enabled:
            self._db.configure_gnn(
                enabled=True,
                attention_heads=settings.ruvector_attention_heads,
                learning_rate=0.001
            )

        # Configure compression if enabled
        if self.enable_compression:
            self._configure_compression()

        # Load embedding model
        self._load_embedding_model()

    def _configure_compression(self):
        """Configure adaptive compression tiers."""
        # Parse compression tiers from config
        # Format: "hot:f32:24h,warm:f16:168h,cold:i8"
        tiers_config = settings.ruvector_compression_tiers
        tiers = []

        for tier_str in tiers_config.split(','):
            parts = tier_str.strip().split(':')
            if len(parts) >= 2:
                tier_name = parts[0]
                precision = parts[1]
                max_age = parts[2] if len(parts) > 2 else None
                tiers.append({
                    "name": tier_name,
                    "precision": precision,
                    "max_age": max_age
                })

        if tiers and hasattr(self._db, 'configure_compression'):
            self._db.configure_compression(tiers=tiers)
            logger.debug(f"Configured compression tiers: {tiers}")

    def _load_embedding_model(self):
        """Load sentence-transformers model for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = settings.embedding_model
            self._embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available. Using fallback.")
            self._embedding_model = None

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        CONCEPT: Dense vector representation of semantic meaning
        WHY: Enable similarity comparison in high-dimensional space
        """
        if self._embedding_model is None:
            return self._fallback_embedding(text)

        embedding = self._embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding using deterministic hash."""
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(self.embedding_dim):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)

        return embedding

    async def add_conversation(
        self,
        conversation_id: str,
        user_text: str,
        agent_text: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation exchange to RuVector.

        PATTERN: Combined embedding with rich metadata
        WHY: Capture full context for semantic matching
        """
        if not self._initialized:
            await self.initialize()

        if not RUVECTOR_AVAILABLE or self._db is None:
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
                "user_text": user_text[:500],
                "agent_text": agent_text[:500],
                "timestamp": datetime.utcnow().isoformat(),
                "backend": "ruvector",
                **(metadata or {})
            }

            # Insert into RuVector
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._db.insert(
                    conversation_id,
                    embedding,
                    doc_metadata
                )
            )

            logger.debug(f"Added conversation {conversation_id} to RuVector")
            return True

        except Exception as e:
            logger.error(f"Failed to add conversation to RuVector: {e}")
            return False

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        session_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search with self-learning index.

        CONCEPT: GNN-optimized search that improves over time
        WHY: Search quality increases with usage through neural learning
        """
        if not self._initialized:
            await self.initialize()

        if not RUVECTOR_AVAILABLE or self._db is None:
            return []

        try:
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_embedding, query
            )

            # Build filter if session specified
            filter_dict = None
            if session_filter:
                filter_dict = {"session_id": session_filter}

            # Query RuVector
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._db.search(
                    query_embedding,
                    k=limit,
                    filter=filter_dict
                )
            )

            # Process results
            processed_results = []
            for result in results:
                similarity = result.get('similarity', result.get('score', 0))

                if similarity >= similarity_threshold:
                    metadata = result.get('metadata', {})
                    processed_results.append({
                        "id": result.get('id'),
                        "similarity": round(similarity, 3),
                        "user_text": metadata.get("user_text", ""),
                        "agent_text": metadata.get("agent_text", ""),
                        "session_id": metadata.get("session_id", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "document": f"User: {metadata.get('user_text', '')}\nAssistant: {metadata.get('agent_text', '')}"
                    })

            return processed_results

        except Exception as e:
            logger.error(f"RuVector semantic search failed: {e}")
            return []

    async def find_similar_conversations(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find conversations similar to a given conversation.

        CONCEPT: "More like this" functionality
        WHY: Help users discover related learning moments
        """
        if not self._initialized or not RUVECTOR_AVAILABLE:
            return []

        try:
            # Get the target conversation embedding
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._db.get(conversation_id)
            )

            if not result:
                return []

            # Search using the conversation's embedding
            embedding = result.get('embedding')
            if not embedding:
                return []

            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._db.search(embedding, k=limit + 1)
            )

            # Filter out the original conversation
            return [r for r in results if r.get('id') != conversation_id][:limit]

        except Exception as e:
            logger.error(f"Find similar conversations failed: {e}")
            return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Remove a conversation from RuVector."""
        if not self._initialized or not RUVECTOR_AVAILABLE:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._db.delete(conversation_id)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get RuVector statistics."""
        if not self._initialized or not RUVECTOR_AVAILABLE:
            return {"available": False, "backend": "ruvector", "count": 0}

        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._db.count() if hasattr(self._db, 'count') else 0
            )

            return {
                "available": True,
                "backend": "ruvector",
                "count": count,
                "embedding_model": settings.embedding_model,
                "embedding_dim": self.embedding_dim,
                "persist_directory": self.persist_directory,
                "learning_enabled": self.enable_learning,
                "compression_enabled": self.enable_compression,
                "gnn_enabled": settings.ruvector_gnn_enabled
            }
        except Exception as e:
            logger.error(f"Failed to get RuVector stats: {e}")
            return {"available": False, "backend": "ruvector", "error": str(e)}

    async def close(self) -> None:
        """Cleanup RuVector resources."""
        if self._db and hasattr(self._db, 'close'):
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._db.close
            )
        self._initialized = False
        self._db = None

    # ===== LearningVectorStoreProtocol Methods =====

    async def train_positive(self, conversation_id: str, weight: float = 1.0) -> bool:
        """
        Train the GNN index to boost this conversation and similar items.

        CONCEPT: Positive reinforcement learning
        WHY: User found this helpful, improve retrieval of similar content

        This tells the GNN that this conversation was a good search result,
        causing the index to adjust its attention weights to favor similar
        vector neighborhoods.
        """
        if not self._initialized or not self.enable_learning:
            return False

        try:
            if hasattr(self._db, 'train_positive'):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._db.train_positive(conversation_id, weight)
                )
                self._positive_training_count += 1
                logger.debug(f"Positive training for {conversation_id} (weight={weight})")
                return True
            return False
        except Exception as e:
            logger.error(f"Positive training failed: {e}")
            return False

    async def train_negative(self, conversation_id: str, weight: float = 1.0) -> bool:
        """
        Train the GNN index to reduce weight of this conversation pattern.

        CONCEPT: Negative reinforcement learning
        WHY: User found this unhelpful, reduce retrieval of similar content
        """
        if not self._initialized or not self.enable_learning:
            return False

        try:
            if hasattr(self._db, 'train_negative'):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._db.train_negative(conversation_id, weight)
                )
                self._negative_training_count += 1
                logger.debug(f"Negative training for {conversation_id} (weight={weight})")
                return True
            return False
        except Exception as e:
            logger.error(f"Negative training failed: {e}")
            return False

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        return {
            "learning_enabled": self.enable_learning,
            "positive_training_count": self._positive_training_count,
            "negative_training_count": self._negative_training_count,
            "total_training_signals": self._positive_training_count + self._negative_training_count,
            "gnn_enabled": settings.ruvector_gnn_enabled,
            "attention_heads": settings.ruvector_attention_heads
        }

    # ===== GraphVectorStoreProtocol Methods (Phase 2) =====

    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher graph query.

        CONCEPT: Native graph traversal with vector-aware operations
        WHY: Combine relationship queries with semantic similarity

        Phase 2 Enhancement: Full Cypher support with proper error handling
        """
        if not self._initialized or not RUVECTOR_AVAILABLE:
            logger.warning("RuVector not initialized for graph queries")
            return []

        if not settings.enable_graph_queries:
            logger.warning("Graph queries disabled. Enable with ENABLE_GRAPH_QUERIES=true")
            return []

        try:
            # Execute Cypher query via RuVector's graph layer
            if hasattr(self._db, 'execute_cypher'):
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._db.execute_cypher(query, parameters or {})
                )
                logger.debug(f"Cypher query executed: {len(result)} results")
                return result
            elif hasattr(self._db, 'execute'):
                # Fallback for older API
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._db.execute(query, parameters or {})
                )
                return result
            else:
                logger.warning("RuVector instance does not support Cypher queries")
                return []

        except Exception as e:
            logger.error(f"Cypher query failed: {e}", exc_info=True)
            return []

    async def hybrid_search(
        self,
        query: str,
        cypher_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid vector + graph search.

        CONCEPT: Best of both worlds - semantic similarity + graph relationships
        WHY: "Find conversations about X that are related to concept Y"

        Phase 2 Enhancement: Optimized hybrid search with result merging

        Example:
            # Find conversations about neural networks related to deep learning
            results = await store.hybrid_search(
                query="neural network training",
                cypher_filter="EXISTS {MATCH (conv)-[:ABOUT]->(c:Concept {name: 'deep learning'})}",
                limit=10
            )
        """
        if not settings.enable_graph_queries or not cypher_filter:
            # Fall back to pure vector search
            logger.debug("Falling back to pure vector search (no graph filter)")
            return await self.semantic_search(query, limit=limit)

        if not self._initialized or not RUVECTOR_AVAILABLE:
            logger.warning("RuVector not available for hybrid search")
            return await self.semantic_search(query, limit=limit)

        try:
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_embedding, query
            )

            # Execute hybrid search if supported
            if hasattr(self._db, 'hybrid_search'):
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._db.hybrid_search(
                        embedding=query_embedding,
                        cypher_filter=cypher_filter,
                        limit=limit
                    )
                )
                logger.info(f"Hybrid search returned {len(results)} results")
                return self._process_hybrid_results(results)

            # Manual hybrid: vector search + graph filter
            logger.debug("Using manual hybrid search (vector + filter)")
            vector_results = await self.semantic_search(query, limit=limit * 2)
            filtered_results = await self._filter_by_graph(vector_results, cypher_filter)

            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            return await self.semantic_search(query, limit=limit)

    async def _filter_by_graph(
        self,
        vector_results: List[Dict[str, Any]],
        cypher_filter: str
    ) -> List[Dict[str, Any]]:
        """
        Filter vector results using graph query

        PATTERN: Post-filtering for hybrid search
        WHY: Combine vector and graph when native hybrid unavailable
        """
        if not vector_results:
            return []

        try:
            # Extract conversation IDs
            conv_ids = [r.get("id") for r in vector_results if r.get("id")]

            if not conv_ids:
                return vector_results

            # Build graph query to filter
            query = f"""
            MATCH (conv:Conversation)
            WHERE conv.id IN $conv_ids AND {cypher_filter}
            RETURN conv.id as id
            """

            filtered_ids = await self.execute_cypher(query, {"conv_ids": conv_ids})
            filtered_id_set = {r.get("id") for r in filtered_ids}

            # Filter results
            return [r for r in vector_results if r.get("id") in filtered_id_set]

        except Exception as e:
            logger.error(f"Graph filtering failed: {e}")
            return vector_results

    def _process_hybrid_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and normalize hybrid search results

        CONCEPT: Result normalization
        WHY: Consistent format regardless of backend
        """
        processed = []

        for result in results:
            # Normalize result format
            processed_result = {
                "id": result.get("id"),
                "similarity": result.get("similarity", result.get("score", 0)),
                "user_text": result.get("user_text", ""),
                "agent_text": result.get("agent_text", ""),
                "session_id": result.get("session_id", ""),
                "timestamp": result.get("timestamp", ""),
                "document": result.get("document", ""),
                "graph_score": result.get("graph_score", 0),
                "combined_score": result.get("combined_score", result.get("similarity", 0))
            }

            processed.append(processed_result)

        # Sort by combined score
        processed.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return processed


# Convenience instance for direct import
ruvector_store = RuVectorStore()
