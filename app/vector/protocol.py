"""
Vector Store Protocol - Abstract Interface for Vector Database Backends
PATTERN: Strategy pattern with runtime protocol checking
WHY: Enable seamless switching between ChromaDB and RuVector backends

SPECIFICATION:
- Define common interface for all vector store implementations
- Support async operations throughout
- Enable runtime type checking with @runtime_checkable
- Provide clear contract for new backend implementations
"""
from typing import Protocol, List, Dict, Optional, Any, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Abstract protocol for vector store implementations.

    CONCEPT: Dependency inversion for vector storage backends
    WHY: Allow A/B testing and migration between ChromaDB and RuVector
    PATTERN: Protocol with runtime_checkable for duck typing support

    All implementations must provide these async methods:
    - initialize(): Setup database connection
    - add_conversation(): Index a conversation exchange
    - semantic_search(): Find similar conversations
    - find_similar_conversations(): "More like this" search
    - delete_conversation(): Remove from index
    - get_stats(): Return backend statistics
    - close(): Cleanup resources
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the vector store connection.

        Returns:
            True if initialization successful, False otherwise
        """
        ...

    @abstractmethod
    async def add_conversation(
        self,
        conversation_id: str,
        user_text: str,
        agent_text: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation exchange to the vector store.

        Args:
            conversation_id: Unique identifier for the conversation
            user_text: User's message text
            agent_text: Agent's response text
            session_id: Session identifier for filtering
            metadata: Optional additional metadata

        Returns:
            True if successfully added, False otherwise
        """
        ...

    @abstractmethod
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        session_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query text
            limit: Maximum results to return
            similarity_threshold: Minimum similarity score (0-1)
            session_filter: Optional session ID filter

        Returns:
            List of matching conversations with similarity scores
        """
        ...

    @abstractmethod
    async def find_similar_conversations(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find conversations similar to a given conversation.

        Args:
            conversation_id: Reference conversation ID
            limit: Maximum results to return

        Returns:
            List of similar conversations
        """
        ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Remove a conversation from the vector store.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        ...

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Dictionary with stats (available, count, backend info, etc.)
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Cleanup vector store resources."""
        ...


@runtime_checkable
class LearningVectorStoreProtocol(VectorStoreProtocol, Protocol):
    """
    Extended protocol for self-learning vector stores (RuVector).

    CONCEPT: Self-improving search through feedback
    WHY: RuVector's GNN can learn from user interactions
    PATTERN: Extended protocol for advanced features
    """

    @abstractmethod
    async def train_positive(self, conversation_id: str, weight: float = 1.0) -> bool:
        """
        Train the index to boost this conversation and similar items.

        CONCEPT: Positive reinforcement for good search results
        WHY: User found this result helpful, promote similar content

        Args:
            conversation_id: ID of positively-rated conversation
            weight: Training weight (higher = stronger signal)

        Returns:
            True if training successful
        """
        ...

    @abstractmethod
    async def train_negative(self, conversation_id: str, weight: float = 1.0) -> bool:
        """
        Train the index to reduce weight of this conversation pattern.

        CONCEPT: Negative reinforcement for poor results
        WHY: User found this result unhelpful, demote similar content

        Args:
            conversation_id: ID of negatively-rated conversation
            weight: Training weight (higher = stronger signal)

        Returns:
            True if training successful
        """
        ...

    @abstractmethod
    async def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning process.

        Returns:
            Dictionary with learning metrics (training count, quality improvement, etc.)
        """
        ...


@runtime_checkable
class GraphVectorStoreProtocol(VectorStoreProtocol, Protocol):
    """
    Extended protocol for graph+vector stores (RuVector Phase 2).

    CONCEPT: Unified graph and vector queries
    WHY: Combine semantic similarity with relationship traversal
    PATTERN: Cypher query support for knowledge graph operations
    """

    @abstractmethod
    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher graph query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results
        """
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        cypher_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid vector + graph search.

        Args:
            query: Semantic search query
            cypher_filter: Optional Cypher WHERE clause
            limit: Maximum results

        Returns:
            Combined search results
        """
        ...
