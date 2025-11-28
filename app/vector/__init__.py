"""
Vector Database Layer for Learning Voice Agent
PATTERN: Semantic search with ChromaDB and sentence-transformers
WHY: Enable similarity-based retrieval beyond keyword matching
RESILIENCE: Persistent storage with automatic recovery
"""
from .config import VectorConfig
from .embeddings import EmbeddingGenerator

# Conditional import for VectorStore - requires chromadb
try:
    from .vector_store import VectorStore
    __all__ = ['VectorStore', 'EmbeddingGenerator', 'VectorConfig']
except ImportError:
    # chromadb not available, VectorStore will not be available
    __all__ = ['EmbeddingGenerator', 'VectorConfig']
