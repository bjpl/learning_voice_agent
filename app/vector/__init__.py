"""
Vector Database Layer for Learning Voice Agent
PATTERN: Semantic search with pluggable backends (ChromaDB, RuVector)
WHY: Enable similarity-based retrieval beyond keyword matching
RESILIENCE: Persistent storage with automatic recovery and backend flexibility

EXPORTS:
- VectorConfig, EmbeddingGenerator: Core utilities
- VectorStore: ChromaDB-based implementation (legacy)
- VectorStoreProtocol: Abstract interface for backends
- RuVectorStore: Self-learning vector store with GNN
- VectorStoreFactory: Factory for backend selection
- get_vector_store: Convenience function
- GraphQueryAdapter: Knowledge graph integration (Phase 2)
- Schema components: Node/relationship types and builders
"""
from .config import VectorConfig
from .embeddings import EmbeddingGenerator

# Protocol exports (always available)
from .protocol import (
    VectorStoreProtocol,
    LearningVectorStoreProtocol,
    GraphVectorStoreProtocol
)

# Factory exports (always available)
from .factory import (
    VectorStoreFactory,
    VectorBackend,
    DualWriteVectorStore
)

# Phase 2: Graph integration exports
from .graph_adapter import GraphQueryAdapter
from .schema import (
    NodeType,
    RelationshipType,
    ConversationNode,
    ConceptNode,
    TopicNode,
    EntityNode,
    Relationship,
    CypherQueryBuilder
)

# Base exports
__all__ = [
    'VectorConfig',
    'EmbeddingGenerator',
    'VectorStoreProtocol',
    'LearningVectorStoreProtocol',
    'GraphVectorStoreProtocol',
    'VectorStoreFactory',
    'VectorBackend',
    'DualWriteVectorStore',
    'get_vector_store',
    # Phase 2 exports
    'GraphQueryAdapter',
    'NodeType',
    'RelationshipType',
    'ConversationNode',
    'ConceptNode',
    'TopicNode',
    'EntityNode',
    'Relationship',
    'CypherQueryBuilder',
]

# Conditional import for ChromaDB VectorStore
try:
    from .vector_store import VectorStore
    __all__.append('VectorStore')
except ImportError:
    VectorStore = None

# Conditional import for RuVectorStore
try:
    from .ruvector_store import RuVectorStore, ruvector_store, RUVECTOR_AVAILABLE
    __all__.extend(['RuVectorStore', 'ruvector_store', 'RUVECTOR_AVAILABLE'])
except ImportError:
    RuVectorStore = None
    ruvector_store = None
    RUVECTOR_AVAILABLE = False


def get_vector_store(
    backend: str = "auto",
    session_id: str = None
) -> VectorStoreProtocol:
    """
    Get a vector store instance.

    Convenience function that wraps VectorStoreFactory.

    Args:
        backend: Backend type ("auto", "chromadb", "ruvector")
        session_id: Optional session ID for A/B testing

    Returns:
        Configured VectorStoreProtocol implementation

    Example:
        from app.vector import get_vector_store

        store = get_vector_store()  # Auto-selects best backend
        await store.initialize()
        await store.add_conversation(...)
    """
    from app.config import settings

    if settings.vector_ab_test_enabled and session_id:
        return VectorStoreFactory.create_with_ab_test(session_id=session_id)

    return VectorStoreFactory.create(backend=backend)
