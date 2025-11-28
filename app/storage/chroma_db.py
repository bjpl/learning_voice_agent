"""
ChromaDB Vector Storage

SPECIFICATION:
- Vector database for semantic memory
- Embedding generation with OpenAI
- Hybrid search (vector + keyword)
- Conversation history persistence
- Session-scoped and global search

ARCHITECTURE:
- ChromaDB client for vector storage
- OpenAI embeddings integration
- Document metadata management
- Efficient batch operations
- Automatic persistence

CODE:
"""
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

# Conditional imports for optional dependencies
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None
    embedding_functions = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Type checking imports
if TYPE_CHECKING:
    import chromadb
    import openai

from app.config import settings
from app.rag.config import rag_config
from app.logger import get_logger

logger = get_logger(__name__)


class ChromaDBStorage:
    """
    ChromaDB-based vector storage for conversation memory

    PATTERN: Repository pattern for vector storage
    WHY: Abstract ChromaDB operations, provide clean interface

    Features:
    - Automatic embedding generation
    - Hybrid search (vector + metadata filtering)
    - Session-scoped search
    - Batch operations
    - Persistence to disk
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize ChromaDB storage

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
            embedding_model: OpenAI embedding model name
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is not installed. "
                "Install it with: pip install chromadb"
            )

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )

        self.persist_directory = persist_directory or rag_config.chroma_persist_directory
        self.collection_name = collection_name or rag_config.chroma_collection_name
        self.embedding_model = embedding_model or rag_config.embedding_model

        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(
                persist_directory=self.persist_directory if rag_config.chroma_enable_persistence else None,
                anonymized_telemetry=False,
            )
        )

        # Initialize OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=self.embedding_model,
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.info(
            "chromadb_initialized",
            collection=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_model=self.embedding_model,
            document_count=self.collection.count(),
        )

    def add_conversation(
        self,
        session_id: str,
        user_text: str,
        agent_text: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add conversation exchange to vector storage

        PATTERN: Combined user+agent text for better context
        WHY: Retrieve complete exchanges, not just fragments

        Args:
            session_id: Session identifier
            user_text: User's message
            agent_text: Agent's response
            timestamp: Conversation timestamp
            metadata: Additional metadata

        Returns:
            Document ID
        """
        timestamp = timestamp or datetime.now()

        # Combine user and agent text for better semantic matching
        combined_text = f"User: {user_text}\nAssistant: {agent_text}"

        # Generate deterministic document ID
        doc_id = self._generate_doc_id(session_id, timestamp)

        # Prepare metadata
        doc_metadata = {
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
            "user_text": user_text,
            "agent_text": agent_text,
            "type": "conversation",
        }

        if metadata:
            doc_metadata.update(metadata)

        try:
            # Add to collection
            self.collection.add(
                documents=[combined_text],
                ids=[doc_id],
                metadatas=[doc_metadata],
            )

            logger.debug(
                "conversation_added",
                doc_id=doc_id,
                session_id=session_id,
                text_length=len(combined_text),
            )

            return doc_id

        except Exception as e:
            logger.error(
                "conversation_add_error",
                error=str(e),
                session_id=session_id,
                exc_info=True,
            )
            raise

    def add_batch(
        self,
        conversations: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple conversations in batch

        PATTERN: Batch operations for efficiency
        WHY: Reduce API calls, improve throughput

        Args:
            conversations: List of conversation dicts with keys:
                - session_id, user_text, agent_text, timestamp, metadata

        Returns:
            List of document IDs
        """
        documents = []
        ids = []
        metadatas = []

        for conv in conversations:
            session_id = conv["session_id"]
            user_text = conv["user_text"]
            agent_text = conv["agent_text"]
            timestamp = conv.get("timestamp", datetime.now())
            metadata = conv.get("metadata", {})

            # Combined text
            combined_text = f"User: {user_text}\nAssistant: {agent_text}"

            # Document ID
            doc_id = self._generate_doc_id(session_id, timestamp)

            # Metadata
            doc_metadata = {
                "session_id": session_id,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                "user_text": user_text,
                "agent_text": agent_text,
                "type": "conversation",
            }
            doc_metadata.update(metadata)

            documents.append(combined_text)
            ids.append(doc_id)
            metadatas.append(doc_metadata)

        try:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

            logger.info(
                "batch_conversations_added",
                count=len(conversations),
            )

            return ids

        except Exception as e:
            logger.error(
                "batch_add_error",
                error=str(e),
                count=len(conversations),
                exc_info=True,
            )
            raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        session_id: Optional[str] = None,
        min_score: float = 0.0,
        max_age_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant conversations

        PATTERN: Semantic vector search with filters
        WHY: Find contextually relevant past conversations

        Args:
            query: Search query
            n_results: Number of results to return
            session_id: Filter by session (None = search all)
            min_score: Minimum similarity score (0.0-1.0)
            max_age_days: Maximum age of documents in days

        Returns:
            List of matching documents with metadata and scores
        """
        try:
            # Prepare metadata filter
            where_filter = {}

            if session_id:
                where_filter["session_id"] = session_id

            if max_age_days:
                cutoff_date = datetime.now() - timedelta(days=max_age_days)
                where_filter["timestamp"] = {"$gte": cutoff_date.isoformat()}

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )

            # Format results
            documents = []
            for i in range(len(results["ids"][0])):
                # Calculate similarity score (ChromaDB returns distances)
                # Convert L2 distance to similarity score (0-1 range)
                distance = results["distances"][0][i]
                similarity = 1 / (1 + distance)  # Normalize to 0-1

                # Filter by minimum score
                if similarity < min_score:
                    continue

                doc = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": similarity,
                    "distance": distance,
                }
                documents.append(doc)

            logger.debug(
                "search_completed",
                query_length=len(query),
                results_found=len(documents),
                session_id=session_id,
            )

            return documents

        except Exception as e:
            logger.error(
                "search_error",
                error=str(e),
                query=query[:100],
                exc_info=True,
            )
            raise

    def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get all conversations for a session

        Args:
            session_id: Session identifier
            limit: Maximum number of results

        Returns:
            List of conversations ordered by timestamp
        """
        try:
            results = self.collection.get(
                where={"session_id": session_id},
                limit=limit,
            )

            # Format results
            documents = []
            for i in range(len(results["ids"])):
                doc = {
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                documents.append(doc)

            # Sort by timestamp
            documents.sort(
                key=lambda x: x["metadata"].get("timestamp", ""),
                reverse=True,
            )

            return documents

        except Exception as e:
            logger.error(
                "get_by_session_error",
                error=str(e),
                session_id=session_id,
                exc_info=True,
            )
            raise

    def delete_session(self, session_id: str) -> int:
        """
        Delete all conversations for a session

        Args:
            session_id: Session identifier

        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents for session
            results = self.collection.get(
                where={"session_id": session_id},
            )

            if not results["ids"]:
                return 0

            # Delete documents
            self.collection.delete(
                ids=results["ids"],
            )

            count = len(results["ids"])

            logger.info(
                "session_deleted",
                session_id=session_id,
                count=count,
            )

            return count

        except Exception as e:
            logger.error(
                "delete_session_error",
                error=str(e),
                session_id=session_id,
                exc_info=True,
            )
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics

        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.collection.count()

            # Get sample to calculate average metadata
            sample = self.collection.peek(limit=100)

            sessions = set()
            if sample["metadatas"]:
                sessions = set(meta.get("session_id") for meta in sample["metadatas"])

            stats = {
                "collection_name": self.collection_name,
                "total_documents": count,
                "unique_sessions_sample": len(sessions),
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
            }

            return stats

        except Exception as e:
            logger.error(
                "get_stats_error",
                error=str(e),
                exc_info=True,
            )
            return {"error": str(e)}

    def _generate_doc_id(self, session_id: str, timestamp: datetime) -> str:
        """
        Generate deterministic document ID

        PATTERN: Hash-based ID generation
        WHY: Prevent duplicates, enable idempotent operations
        """
        # Create unique string from session and timestamp
        unique_str = f"{session_id}:{timestamp.isoformat()}"

        # Generate hash
        hash_obj = hashlib.sha256(unique_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars

    def clear_collection(self) -> None:
        """
        Clear all documents from collection (DANGEROUS!)

        WARNING: This deletes all data
        """
        try:
            self.client.delete_collection(self.collection_name)

            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

            logger.warning(
                "collection_cleared",
                collection=self.collection_name,
            )

        except Exception as e:
            logger.error(
                "clear_collection_error",
                error=str(e),
                exc_info=True,
            )
            raise


# Create default instance (only if dependencies are available)
if CHROMADB_AVAILABLE and OPENAI_AVAILABLE:
    try:
        chroma_storage = ChromaDBStorage()
    except Exception as e:
        logger.warning(f"Failed to initialize default ChromaDB storage: {e}")
        chroma_storage = None
else:
    logger.warning("ChromaDB or OpenAI not available. Default instance not created.")
    chroma_storage = None
