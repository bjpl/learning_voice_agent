"""
Vector Store with ChromaDB
PATTERN: Repository pattern with async operations
WHY: Semantic similarity search for conversation context
RESILIENCE: Persistent storage with automatic recovery
"""
import uuid
from typing import List, Dict, Optional, Any, Union, TYPE_CHECKING
from datetime import datetime
import numpy as np
from app.vector.config import VectorConfig, CollectionConfig
from app.vector.embeddings import EmbeddingGenerator, embedding_generator
from app.logger import db_logger
from app.resilience import with_retry

# Conditional import for chromadb - allows tests to run without the package
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

# Type checking imports
if TYPE_CHECKING:
    import chromadb


class VectorStore:
    """
    PATTERN: Persistent vector database with ChromaDB
    WHY: Enable semantic search beyond keyword matching
    RESILIENCE: Automatic collection creation and recovery

    USAGE:
        store = VectorStore()
        await store.initialize()

        # Add embedding
        doc_id = await store.add_embedding(
            collection_name="conversations",
            text="Hello world",
            metadata={"session_id": "123"}
        )

        # Search similar
        results = await store.search_similar(
            collection_name="conversations",
            query_text="Hi there",
            n_results=5
        )
    """

    def __init__(self, config: Optional[VectorConfig] = None):
        """
        Initialize vector store

        Args:
            config: Vector configuration (uses default if None)

        Raises:
            ImportError: If chromadb package is not installed
        """
        from app.vector.config import vector_config as default_config
        self.config = config or default_config

        self.client: Optional[Any] = None  # chromadb.PersistentClient when available
        self.collections: Dict[str, Any] = {}  # chromadb.Collection when available
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self._initialized = False

        # Check availability but don't fail until initialize()
        # Note: Warning will be logged on initialize() if chromadb not available

    @with_retry(max_attempts=3, min_wait=0.5)
    async def initialize(self) -> None:
        """
        CONCEPT: Lazy initialization with persistent client
        WHY: Defer expensive operations until needed
        RESILIENCE: Retry on initialization failures

        Raises:
            ImportError: If chromadb package is not installed
        """
        if self._initialized:
            return

        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is not installed. "
                "Install it with: pip install chromadb"
            )

        try:
            db_logger.info(
                "vector_store_initialization_started",
                persist_directory=str(self.config.persist_directory)
            )

            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )

            # Initialize embedding generator
            self.embedding_generator = embedding_generator
            await self.embedding_generator.initialize()

            # Create/get configured collections
            for collection_name, collection_config in self.config.collections.items():
                await self._get_or_create_collection(collection_config)

            self._initialized = True

            db_logger.info(
                "vector_store_initialized",
                persist_directory=str(self.config.persist_directory),
                collections=list(self.collections.keys())
            )

        except Exception as e:
            db_logger.error(
                "vector_store_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def _get_or_create_collection(
        self,
        config: CollectionConfig
    ) -> Any:  # chromadb.Collection when available
        """
        Get existing collection or create new one

        PATTERN: Idempotent collection management
        WHY: Safe to call multiple times
        RESILIENCE: Handles existing collections gracefully
        """
        try:
            # Map distance metric to ChromaDB format
            distance_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip"
            }

            collection = self.client.get_or_create_collection(
                name=config.name,
                metadata={
                    "hnsw:space": distance_map.get(config.distance_metric, "cosine"),
                    **config.metadata_schema
                }
            )

            self.collections[config.name] = collection

            db_logger.info(
                "collection_ready",
                collection_name=config.name,
                distance_metric=config.distance_metric
            )

            return collection

        except Exception as e:
            db_logger.error(
                "collection_creation_failed",
                collection_name=config.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    @with_retry(max_attempts=3, min_wait=0.5)
    async def add_embedding(
        self,
        collection_name: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Add embedding to collection

        PATTERN: Upsert with automatic embedding generation
        WHY: Simplify usage - caller doesn't need to generate embeddings
        RESILIENCE: Retry on transient failures

        Args:
            collection_name: Target collection name
            text: Text content to embed and store
            metadata: Optional metadata to attach
            document_id: Optional document ID (auto-generated if None)
            embedding: Optional pre-computed embedding (generated if None)

        Returns:
            Document ID
        """
        if not self._initialized:
            await self.initialize()

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        try:
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self.embedding_generator.generate_embedding(text)

            # Generate ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())

            # Prepare metadata
            meta = metadata or {}
            meta["text_preview"] = text[:200]  # Store preview for debugging
            meta["timestamp"] = datetime.now().isoformat()

            # Add to collection
            collection = self.collections[collection_name]
            collection.add(
                ids=[document_id],
                embeddings=[embedding.tolist()],
                documents=[text],
                metadatas=[meta]
            )

            db_logger.info(
                "embedding_added",
                collection=collection_name,
                document_id=document_id,
                text_length=len(text),
                metadata_keys=list(meta.keys())
            )

            return document_id

        except Exception as e:
            db_logger.error(
                "add_embedding_failed",
                collection=collection_name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    @with_retry(max_attempts=3, min_wait=0.5)
    async def add_batch(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple embeddings in batch

        PATTERN: Batch processing for efficiency
        WHY: Much faster than individual adds
        RESILIENCE: Partial success - continue on individual failures

        Args:
            collection_name: Target collection name
            texts: List of texts to embed
            metadatas: Optional list of metadata dicts
            document_ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if not self._initialized:
            await self.initialize()

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        if not texts:
            return []

        try:
            db_logger.info(
                "batch_add_started",
                collection=collection_name,
                num_texts=len(texts)
            )

            # Generate embeddings in batch
            embeddings = await self.embedding_generator.generate_batch(texts)

            # Generate IDs if not provided
            if document_ids is None:
                document_ids = [str(uuid.uuid4()) for _ in texts]

            # Prepare metadatas
            if metadatas is None:
                metadatas = [{} for _ in texts]

            # Add timestamp and preview to each metadata
            now = datetime.now().isoformat()
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                meta["text_preview"] = text[:200]
                meta["timestamp"] = now

            # Add to collection
            collection = self.collections[collection_name]
            collection.add(
                ids=document_ids,
                embeddings=[emb.tolist() for emb in embeddings],
                documents=texts,
                metadatas=metadatas
            )

            db_logger.info(
                "batch_add_complete",
                collection=collection_name,
                num_added=len(document_ids)
            )

            return document_ids

        except Exception as e:
            db_logger.error(
                "batch_add_failed",
                collection=collection_name,
                num_texts=len(texts),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def search_similar(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        n_results: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings

        PATTERN: Flexible search with text or embedding
        WHY: Support different use cases (user query vs. programmatic)
        RESILIENCE: Return empty list on failures

        Args:
            collection_name: Collection to search
            query_text: Query text (will be embedded)
            query_embedding: Pre-computed query embedding
            n_results: Number of results (uses config default if None)
            metadata_filter: Filter results by metadata
            include_distances: Include similarity distances in results

        Returns:
            List of matching documents with metadata and scores
        """
        if not self._initialized:
            await self.initialize()

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        if query_text is None and query_embedding is None:
            raise ValueError("Either query_text or query_embedding must be provided")

        try:
            # Generate embedding if text provided
            if query_embedding is None:
                query_embedding = await self.embedding_generator.generate_embedding(query_text)

            # Get number of results
            n_results = n_results or self.config.default_n_results
            n_results = min(n_results, self.config.max_n_results)

            db_logger.debug(
                "vector_search_started",
                collection=collection_name,
                n_results=n_results,
                has_filter=metadata_filter is not None
            )

            # Query collection
            collection = self.collections[collection_name]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results and results['ids']:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    }

                    if include_distances:
                        # Convert distance to similarity score (0-1, higher is better)
                        distance = results['distances'][0][i]
                        # For cosine distance: similarity = 1 - distance
                        result['similarity'] = 1.0 - distance
                        result['distance'] = distance

                    # Apply similarity threshold filter
                    if include_distances:
                        if result['similarity'] >= self.config.similarity_threshold:
                            formatted_results.append(result)
                    else:
                        formatted_results.append(result)

            db_logger.info(
                "vector_search_complete",
                collection=collection_name,
                results_found=len(formatted_results),
                n_requested=n_results
            )

            return formatted_results

        except Exception as e:
            db_logger.error(
                "vector_search_failed",
                collection=collection_name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return []

    async def delete_embedding(
        self,
        collection_name: str,
        document_id: str
    ) -> bool:
        """
        Delete embedding from collection

        Args:
            collection_name: Collection name
            document_id: Document ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            await self.initialize()

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        try:
            collection = self.collections[collection_name]
            collection.delete(ids=[document_id])

            db_logger.info(
                "embedding_deleted",
                collection=collection_name,
                document_id=document_id
            )

            return True

        except Exception as e:
            db_logger.error(
                "delete_embedding_failed",
                collection=collection_name,
                document_id=document_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return False

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with collection statistics
        """
        if not self._initialized:
            await self.initialize()

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        try:
            collection = self.collections[collection_name]
            count = collection.count()

            stats = {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }

            db_logger.debug("collection_stats_retrieved", **stats)

            return stats

        except Exception as e:
            db_logger.error(
                "get_collection_stats_failed",
                collection=collection_name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return {
                "name": collection_name,
                "count": 0,
                "error": str(e)
            }

    async def list_collections(self) -> List[str]:
        """List all available collections"""
        if not self._initialized:
            await self.initialize()

        return list(self.collections.keys())

    async def create_collection(
        self,
        name: str,
        metadata_schema: Optional[Dict[str, str]] = None,
        distance_metric: str = "cosine"
    ) -> None:
        """
        Create a new collection

        Args:
            name: Collection name
            metadata_schema: Metadata schema definition
            distance_metric: Distance metric (cosine, l2, or ip)
        """
        if not self._initialized:
            await self.initialize()

        config = CollectionConfig(
            name=name,
            metadata_schema=metadata_schema or {},
            distance_metric=distance_metric
        )

        await self._get_or_create_collection(config)

        db_logger.info(
            "collection_created",
            name=name,
            distance_metric=distance_metric
        )

    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection

        Args:
            name: Collection name

        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.client.delete_collection(name=name)

            if name in self.collections:
                del self.collections[name]

            db_logger.info("collection_deleted", name=name)

            return True

        except Exception as e:
            db_logger.error(
                "delete_collection_failed",
                name=name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return False

    async def close(self) -> None:
        """
        Clean up resources

        CONCEPT: Explicit resource management
        WHY: Ensure proper persistence and cleanup
        """
        if self.embedding_generator:
            await self.embedding_generator.close()

        self.collections.clear()
        self.client = None
        self._initialized = False

        db_logger.info("vector_store_closed")


# Global vector store instance (only create if chromadb available)
if CHROMADB_AVAILABLE:
    try:
        vector_store = VectorStore()
    except Exception as e:
        db_logger.error(
            "vector_store_initialization_failed",
            error=str(e),
            error_type=type(e).__name__
        )
        vector_store = None
else:
    vector_store = None
